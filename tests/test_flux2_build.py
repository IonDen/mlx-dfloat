"""The FLUX.2 Klein build from a DF11 checkpoint: the pure helpers offline, the real tiny transformer in the mflux lane."""

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._flux2_tiny import TINY_OVERRIDES, write_tiny_checkpoint

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux.flux2.names import NONBLOCK_GROUPS
from mlx_dfloat.mflux.flux2.transformer import (
    base_extras,
    base_transformer_files_index,
    build_transformer,
    size_key,
)

# --- offline -----------------------------------------------------------------------------------------------------


def test_size_key_reads_the_head_count_of_the_transformer_overrides():
    # Bug caught: a 9B model costed with 4B constants (or the reverse). Head counts from mflux 0.20.0
    # common/config/model_config.py:428-433 (Klein 4B: 24) and :450-455 (Klein 9B: 32).
    assert size_key(SimpleNamespace(transformer_overrides={"num_attention_heads": 24})) == "4b"
    assert size_key(SimpleNamespace(transformer_overrides={"num_attention_heads": 32})) == "9b"


def test_size_key_refuses_a_width_it_has_no_constants_for():
    # Bug caught: an unknown Klein size silently costed as one of the two measured ones.
    with pytest.raises(DFloatUnsupportedError, match="16"):
        size_key(SimpleNamespace(transformer_overrides={"num_attention_heads": 16}))
    with pytest.raises(DFloatUnsupportedError):
        size_key(SimpleNamespace(transformer_overrides={}))


def test_base_extras_keeps_the_nonblock_matrices_and_drops_block_matrices():
    # Bug caught: the BF16 side building without context_embedder.weight (coverage refuses) or with block matrices as
    # extras (the whole BF16 transformer resident).
    groups = {
        "transformer_blocks.0": SimpleNamespace(
            matrix_names=("transformer_blocks.0.attn.to_out.0.weight",)
        ),
        "context_embedder": SimpleNamespace(matrix_names=("context_embedder.weight",)),
    }
    index = {
        name: (Path(name), None)
        for name in (
            "transformer_blocks.0.attn.to_out.0.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "context_embedder.weight",
            "x_embedder.weight",
        )
    }
    assert sorted(base_extras(index, SimpleNamespace(groups=groups))) == [
        "context_embedder.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "x_embedder.weight",
    ]


def _save(path, tensors):
    mx.save_safetensors(str(path), {k: mx.zeros(s, dtype=mx.bfloat16) for k, s in tensors.items()})


def test_the_files_index_reads_a_single_file_without_an_index(tmp_path):
    # Bug caught: FLUX.2 base-4B's transformer (one file, no index) unreadable for the identity check.
    _save(tmp_path / "diffusion_pytorch_model.safetensors", {"a.weight": (2, 3)})
    index = base_transformer_files_index(tmp_path)
    assert set(index) == {"a.weight"}
    path, info = index["a.weight"]
    assert path == tmp_path / "diffusion_pytorch_model.safetensors"
    assert info.shape == (2, 3)
    assert info.dtype == "BF16"


def test_the_files_index_refuses_zero_or_several_index_less_files(tmp_path):
    # Bug caught: a directory with two loose safetensors files read as one transformer (the second silently
    # ignored or its names merged), or an empty directory read as an empty transformer.
    with pytest.raises(DFloatFormatError, match="no"):
        base_transformer_files_index(tmp_path)
    _save(tmp_path / "a.safetensors", {"a.weight": (2, 3)})
    _save(tmp_path / "b.safetensors", {"b.weight": (2, 3)})
    with pytest.raises(DFloatFormatError, match="2"):
        base_transformer_files_index(tmp_path)


def test_the_files_index_follows_a_shard_index_when_there_is_one(tmp_path):
    # Bug caught: a sharded base (FLUX.2 base-9B's transformer: two shards and an index) read as "several index-less
    # files" and refused, or only one shard's tensors returned.
    _save(tmp_path / "s-00001-of-00002.safetensors", {"a.weight": (2, 3)})
    _save(tmp_path / "s-00002-of-00002.safetensors", {"b.weight": (4,)})
    weight_map = {
        "a.weight": "s-00001-of-00002.safetensors",
        "b.weight": "s-00002-of-00002.safetensors",
    }
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    index = base_transformer_files_index(tmp_path)
    assert {k: (p.name, i.shape) for k, (p, i) in index.items()} == {
        "a.weight": ("s-00001-of-00002.safetensors", (2, 3)),
        "b.weight": ("s-00002-of-00002.safetensors", (4,)),
    }


# --- mflux lane: the real tiny transformer -----------------------------------------------------------------------

# Non-block matrix shapes at TINY size (inner 2 x 16 = 32, joint_attention_dim 32), from mflux 0.20.0:
# Flux2Modulation Linear(dim, dim * 3 * sets) (flux2_transformer/modulation.py:9) with 2 sets for the double-stream
# modulations and 1 for the single (transformer.py:41-43); context_embedder Linear(joint, inner) (transformer.py:46);
# AdaLayerNormContinuous Linear(cond, embedding * 2) (flux/model/flux_transformer/ada_layer_norm_continuous.py:11).
TINY_NONBLOCK = {
    "double_stream_modulation_img.linear.weight": (192, 32),
    "double_stream_modulation_txt.linear.weight": (192, 32),
    "single_stream_modulation.linear.weight": (96, 32),
    "context_embedder.weight": (32, 32),
    "norm_out.linear.weight": (64, 32),
}


@pytest.mark.mflux
def test_the_build_installs_placeholders_on_every_compressed_matrix_and_loads_every_extra(
    tmp_path,
):
    # Bug caught: a non-block matrix left as a live weight (memory) or an extra renamed onto the wrong parameter
    # (the timestep linears), visible as a constant in the wrong place.
    _groups, constants = write_tiny_checkpoint(tmp_path, np.random.default_rng(3))
    ckpt = open_checkpoint(tmp_path)
    # The checkpoint carries the diffusers spelling (the real DF11 files'), so the build has to rename.
    assert "time_guidance_embed.timestep_embedder.linear_1.weight" in ckpt.extras
    assert "time_guidance_embed.linear_1.weight" not in ckpt.extras
    build = build_transformer(ckpt, transformer_overrides=TINY_OVERRIDES)
    params = dict(tree_flatten(build.transformer.parameters()))
    for g in NONBLOCK_GROUPS:
        assert params[f"{g}.weight"].size == 0, g
    assert build.nonblock == TINY_NONBLOCK
    assert (
        constants["time_guidance_embed.linear_1.weight"]
        != constants["time_guidance_embed.linear_2.weight"]
    )
    for name, value in constants.items():
        assert np.all(np.array(params[name].view(mx.uint16)) == value), name
    for block, per in build.shapes.items():
        for attr in per:
            assert params[f"{block}.{attr}.weight"].size == 0, (block, attr)
    assert build.counts == {"transformer_blocks": 1, "single_transformer_blocks": 2}


@pytest.mark.mflux
def test_nonblock_matrices_stored_as_extras_load_as_plain_weights(tmp_path):
    # Bug caught (a DF11 file that keeps the modulations uncompressed): a checkpoint without the five non-block groups
    # refused, or its modulation weights left as placeholders (zero-size matmul in the first block).
    _groups, constants = write_tiny_checkpoint(
        tmp_path, np.random.default_rng(3), nonblock_as_groups=False
    )
    build = build_transformer(open_checkpoint(tmp_path), transformer_overrides=TINY_OVERRIDES)
    params = dict(tree_flatten(build.transformer.parameters()))
    assert build.nonblock == {}
    for name, shape in TINY_NONBLOCK.items():
        assert params[name].shape == shape, name
        assert np.all(np.array(params[name].view(mx.uint16)) == constants[name]), name


@pytest.mark.mflux
def test_a_checkpoint_with_other_block_counts_is_refused_naming_both(tmp_path):
    # Bug caught: a 9B checkpoint built with 4B overrides failing later with a shape error at the first block (or
    # building at the checkpoint's depth) instead of a refusal that names the sizes.
    write_tiny_checkpoint(tmp_path, np.random.default_rng(3), n_double=2, n_single=3)
    with pytest.raises(
        DFloatFormatError,
        match=r"checkpoint has 2 double / 3 single blocks; this model builds 1 / 2",
    ):
        build_transformer(open_checkpoint(tmp_path), transformer_overrides=TINY_OVERRIDES)


@pytest.mark.mflux
def test_the_count_check_defaults_to_mflux_s_klein_depth(tmp_path):
    # Bug caught: overrides without num_layers / num_single_layers checked against something other than mflux's
    # defaults (5 / 20, flux2_transformer/transformer.py:20-21).
    write_tiny_checkpoint(tmp_path, np.random.default_rng(3))
    no_depth = {
        k: v for k, v in TINY_OVERRIDES.items() if k not in ("num_layers", "num_single_layers")
    }
    with pytest.raises(
        DFloatFormatError, match=r"1 double / 2 single blocks; this model builds 5 / 20"
    ):
        build_transformer(open_checkpoint(tmp_path), transformer_overrides=no_depth)


@pytest.mark.mflux
def test_a_checkpoint_of_another_width_is_refused_by_the_extras_shape_check(tmp_path):
    # Bug caught: same counts but another inner dim (a hand-made checkpoint) accepted; load_extras must refuse.
    write_tiny_checkpoint(tmp_path, np.random.default_rng(3), inner_override=4)
    with pytest.raises(DFloatFormatError, match="shape"):
        build_transformer(open_checkpoint(tmp_path), transformer_overrides=TINY_OVERRIDES)


@pytest.mark.mflux
def test_depth_overrides_build_fewer_blocks_and_are_range_checked(tmp_path):
    # Bug caught: n_double/n_single past the checkpoint accepted (shapes and extras plan disagree), or a depth
    # override that does not reach the class (the one-by-one slow test would build the full depth).
    write_tiny_checkpoint(tmp_path, np.random.default_rng(3))
    ckpt = open_checkpoint(tmp_path)
    one = build_transformer(ckpt, transformer_overrides=TINY_OVERRIDES, n_double=1, n_single=1)
    assert one.counts == {"transformer_blocks": 1, "single_transformer_blocks": 1}
    assert len(one.transformer.single_transformer_blocks) == 1
    assert sorted(one.shapes) == ["single_transformer_blocks.0", "transformer_blocks.0"]
    for kw in ({"n_double": 2}, {"n_single": 3}, {"n_single": -1}):
        with pytest.raises(DFloatIntegrationError, match="the checkpoint has 1 / 2"):
            build_transformer(ckpt, transformer_overrides=TINY_OVERRIDES, **kw)


@pytest.mark.mflux
def test_the_nonblock_matrices_load_from_base_extras_when_asked(tmp_path):
    # Bug caught: nonblock_from_extras still putting placeholders on the five matrices (the BF16 side of an identity
    # check would run on empty weights), or base_extras dropping them.
    write_tiny_checkpoint(tmp_path / "ext", np.random.default_rng(3), nonblock_as_groups=False)
    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(3))
    extras = base_extras(
        open_checkpoint(tmp_path / "ext").extras, open_checkpoint(tmp_path / "df11")
    )
    build = build_transformer(
        open_checkpoint(tmp_path / "df11"),
        transformer_overrides=TINY_OVERRIDES,
        extras=extras,
        nonblock_from_extras=True,
    )
    params = dict(tree_flatten(build.transformer.parameters()))
    assert build.nonblock == {}
    for name, shape in TINY_NONBLOCK.items():
        assert params[name].shape == shape, name


@pytest.mark.mflux
@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("flux2-klein-base-4b", "4b"),
        ("flux2-klein-4b", "4b"),
        ("flux2-klein-base-9b", "9b"),
        ("flux2-klein-9b", "9b"),
    ],
)
def test_size_key_on_mflux_s_own_klein_configs(name, key):
    # Bug caught: size_key reading a field the real ModelConfig does not carry (every Klein refused, or all costed as
    # one size). Literals: model_config.py:417-541 (4B entries 24 heads, 9B entries 32).
    from mflux.models.common.config.model_config import ModelConfig

    assert size_key(ModelConfig.from_name(model_name=name, base_model=None)) == key
