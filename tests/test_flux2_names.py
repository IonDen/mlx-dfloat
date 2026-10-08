"""Klein naming: the tables from mflux's mapping, the group check (offline); the real mapping (mflux lane)."""

from types import SimpleNamespace

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux._mapping import name_map_from_mapping
from mlx_dfloat.mflux.flux2.names import (
    DOUBLE,
    MATRIX_SUBS,
    NONBLOCK_GROUPS,
    SINGLE,
    check_klein_groups,
)


def _t(to, frm):
    return SimpleNamespace(to_pattern=to, from_pattern=[frm], transform=None)


def _same(p):
    return _t(p, p)


# A copy of Flux2WeightMapping.get_transformer_mapping() (mflux 0.20.0 models/flux2/weights/flux2_weight_mapping.py
# :10-126): the renames are attn.to_out.0 -> attn.to_out in double blocks (:59-62) and the two timestep_embedder
# linears (:19-26).
_D = "transformer_blocks.{block}."
_S = "single_transformer_blocks.{block}."
TARGETS = [
    _same("x_embedder.weight"),
    _same("context_embedder.weight"),
    _t(
        "time_guidance_embed.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
    ),
    _t(
        "time_guidance_embed.linear_2.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
    ),
    _same("double_stream_modulation_img.linear.weight"),
    _same("double_stream_modulation_txt.linear.weight"),
    _same("single_stream_modulation.linear.weight"),
    _same("norm_out.linear.weight"),
    _same("proj_out.weight"),
    *[_same(f"{_D}attn.{m}.weight") for m in ("to_q", "to_k", "to_v")],
    _t(f"{_D}attn.to_out.weight", f"{_D}attn.to_out.0.weight"),
    *[
        _same(f"{_D}attn.{m}.weight")
        for m in (
            "norm_q",
            "norm_k",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "norm_added_q",
            "norm_added_k",
            "to_add_out",
        )
    ],
    *[
        _same(f"{_D}{m}.weight")
        for m in ("ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out")
    ],
    *[_same(f"{_S}attn.{m}.weight") for m in ("to_qkv_mlp_proj", "norm_q", "norm_k", "to_out")],
]


def test_double_blocks_place_to_out_0_on_to_out_and_keep_the_pattern_order():
    # Bug caught: the DF11 sub "attn.to_out.0" placed on a module path that does not exist in mflux (the double
    # block's out projection is attn.to_out, flux2_weight_mapping.py:59-62), or the table order not the pattern's
    # (the DF11 pattern_dict of both 4B checkpoints, read 2026-10-08).
    m = name_map_from_mapping(TARGETS, matrix_subs=MATRIX_SUBS)
    assert m.attrs_of(DOUBLE) == (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
    )
    assert m.attrs_of(SINGLE) == ("attn.to_qkv_mlp_proj", "attn.to_out")
    assert m.place("transformer_blocks.3.attn.to_out.0.weight").attr == "attn.to_out"


def test_the_timestep_embedder_linears_are_renamed_and_block_norms_are_not():
    # Bug caught: the two non-block renames dropped (the coverage would refuse two extras), or a block norm renamed.
    m = name_map_from_mapping(TARGETS, matrix_subs=MATRIX_SUBS)
    assert (
        m.param_name("time_guidance_embed.timestep_embedder.linear_1.weight")
        == "time_guidance_embed.linear_1.weight"
    )
    assert (
        m.param_name("time_guidance_embed.timestep_embedder.linear_2.weight")
        == "time_guidance_embed.linear_2.weight"
    )
    assert (
        m.param_name("transformer_blocks.4.attn.norm_added_k.weight")
        == "transformer_blocks.4.attn.norm_added_k.weight"
    )
    assert m.param_name("context_embedder.weight") == "context_embedder.weight"


def test_the_nonblock_groups_are_the_five_one_matrix_modules_of_the_checkpoint():
    # Bug caught: a non-block group missing from the table (its DF11 group would be refused as "not a Klein block
    # group"), or its matrix named other than <group>.weight (an empty pattern list means the module itself). The five
    # names are the DF11 pattern_dict's empty-list keys (both 4B checkpoints, read 2026-10-08).
    assert NONBLOCK_GROUPS == {
        "double_stream_modulation_img.linear": ("double_stream_modulation_img.linear.weight",),
        "double_stream_modulation_txt.linear": ("double_stream_modulation_txt.linear.weight",),
        "single_stream_modulation.linear": ("single_stream_modulation.linear.weight",),
        "context_embedder": ("context_embedder.weight",),
        "norm_out.linear": ("norm_out.linear.weight",),
    }


def _ckpt(
    tmp_path,
    *,
    doubles=(0, 1),
    singles=(0, 1, 2),
    nonblock=tuple(NONBLOCK_GROUPS),
    double_subs=None,
):
    rng = np.random.default_rng(1)
    dsubs = MATRIX_SUBS[DOUBLE] if double_subs is None else double_subs
    groups = {f"{DOUBLE}.{i}": [random_bf16(rng, (2, 2)) for _ in dsubs] for i in doubles}
    groups |= {
        f"{SINGLE}.{i}": [random_bf16(rng, (2, 2)) for _ in MATRIX_SUBS[SINGLE]] for i in singles
    }
    groups |= {g: [random_bf16(rng, (2, 2))] for g in nonblock}
    patterns = {
        r"transformer_blocks\.\d+": dsubs,
        r"single_transformer_blocks\.\d+": MATRIX_SUBS[SINGLE],
    } | dict.fromkeys(NONBLOCK_GROUPS, ())
    return open_checkpoint(
        write_checkpoint(tmp_path, groups=groups, patterns=patterns, single_file=True)
    )


def test_check_counts_both_kinds_and_accepts_the_five_nonblock_groups(tmp_path):
    # Bug caught: a non-block group taken for a block of an unknown kind, or the counts per kind swapped.
    assert check_klein_groups(_ckpt(tmp_path)) == {DOUBLE: 2, SINGLE: 3}


def test_check_accepts_a_checkpoint_without_nonblock_groups(tmp_path):
    # Bug caught: a file that keeps the modulations as BF16 extras refused as malformed.
    assert check_klein_groups(_ckpt(tmp_path, nonblock=())) == {DOUBLE: 2, SINGLE: 3}


def test_check_refuses_a_hole_in_the_single_blocks(tmp_path):
    # Bug caught: a missing single block run as placeholders (zero-size matmul at that depth).
    with pytest.raises(DFloatFormatError, match="not contiguous"):
        check_klein_groups(_ckpt(tmp_path, singles=(0, 2)))


def test_check_refuses_a_double_block_with_other_matrices(tmp_path):
    # Bug caught: a checkpoint compressed with a different pattern accepted (one matrix would never be decoded).
    with pytest.raises(DFloatFormatError, match=r"^transformer_blocks\.0"):
        check_klein_groups(_ckpt(tmp_path, double_subs=MATRIX_SUBS[DOUBLE][:-1]))


def test_check_refuses_a_missing_kind(tmp_path):
    # Bug caught: a checkpoint with no single blocks treated as zero valid blocks (mflux would build 20 empty ones).
    with pytest.raises(DFloatFormatError, match="no single_transformer_blocks groups"):
        check_klein_groups(_ckpt(tmp_path, singles=()))


def test_check_refuses_a_nonblock_group_with_other_matrices(tmp_path):
    # Bug caught: a context_embedder group holding a sub-module's matrix accepted and decoded into the wrong weight.
    rng = np.random.default_rng(1)
    groups = {
        f"{DOUBLE}.0": [random_bf16(rng, (2, 2)) for _ in MATRIX_SUBS[DOUBLE]],
        f"{SINGLE}.0": [random_bf16(rng, (2, 2)) for _ in MATRIX_SUBS[SINGLE]],
        "context_embedder": [random_bf16(rng, (2, 2))],
    }
    patterns = {
        r"transformer_blocks\.\d+": MATRIX_SUBS[DOUBLE],
        r"single_transformer_blocks\.\d+": MATRIX_SUBS[SINGLE],
        "context_embedder": ("1",),
    }
    ckpt = open_checkpoint(
        write_checkpoint(tmp_path, groups=groups, patterns=patterns, single_file=True)
    )
    with pytest.raises(DFloatFormatError, match="context_embedder"):
        check_klein_groups(ckpt)


def test_check_refuses_a_stray_group(tmp_path):
    # Bug caught: a group of another model (layers.0, a Z-Image block) passed through as a Klein checkpoint.
    rng = np.random.default_rng(1)
    groups = {
        f"{DOUBLE}.0": [random_bf16(rng, (2, 2)) for _ in MATRIX_SUBS[DOUBLE]],
        f"{SINGLE}.0": [random_bf16(rng, (2, 2)) for _ in MATRIX_SUBS[SINGLE]],
        "layers.0": [random_bf16(rng, (2, 2))],
    }
    patterns = {
        r"transformer_blocks\.\d+": MATRIX_SUBS[DOUBLE],
        r"single_transformer_blocks\.\d+": MATRIX_SUBS[SINGLE],
        r"layers\.\d+": ("a",),
    }
    ckpt = open_checkpoint(
        write_checkpoint(tmp_path, groups=groups, patterns=patterns, single_file=True)
    )
    with pytest.raises(DFloatFormatError, match=r"not a FLUX\.2 Klein block group") as info:
        check_klein_groups(ckpt)
    # Bug caught (review 2026-10-08): a checkpoint that compresses another non-block module (x_embedder, proj_out)
    # refused without saying which non-block groups this path accepts (the five of the Klein DF11 pattern_dict).
    for accepted in (
        "context_embedder",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "norm_out.linear",
    ):
        assert accepted in str(info.value)


@pytest.mark.mflux
def test_the_real_mapping_derives_the_same_map_and_every_matrix_lands_on_a_linear():
    # Bug caught: mflux's real mapping differing from the copy above (a rename added upstream), or a DF11 matrix with
    # no nn.Linear to land on in a real (tiny) Flux2Transformer.
    import mlx.nn as nn
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    from mlx_dfloat.integrate.placeholders import get_attr_path
    from mlx_dfloat.mflux.flux2.names import klein_name_map

    m = klein_name_map()
    copy = name_map_from_mapping(TARGETS, matrix_subs=MATRIX_SUBS)
    assert m.attrs_of(DOUBLE) == copy.attrs_of(DOUBLE)
    assert m.attrs_of(SINGLE) == copy.attrs_of(SINGLE)
    for source in (
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "proj_out.weight",
        "x_embedder.weight",
    ):
        assert m.param_name(source) == copy.param_name(source), source
    tf = Flux2Transformer(
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        in_channels=8,
        timestep_guidance_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
    )
    for kind in (DOUBLE, SINGLE):
        for attr in m.attrs_of(kind):
            assert isinstance(get_attr_path(getattr(tf, kind)[0], attr), nn.Linear), (kind, attr)
    for group, (matrix,) in NONBLOCK_GROUPS.items():
        module = get_attr_path(tf, m.param_name(matrix).removesuffix(".weight"))
        assert isinstance(module, nn.Linear), group
