"""The Qwen-Image 2.1 build from a config-less DF11 checkpoint: base extras offline, the real tiny build (mflux lane)."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._df11_fixtures import write_bf16_original
from tests._qwen21_tiny import TINY, write_tiny_checkpoint

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux.qwen21.transformer import base_extras, build_transformer

# The seven block matrices and their shapes at TINY size (inner 2 x 16 = 32, mlp_ratio 3), from mflux 0.20.0:
# Qwen21Attention's four Linear(32, 32) (qwen21_attention.py:13-16) and Qwen21SwiGLUFeedForward's
# proj/gate_layer Linear(32, 96) and out Linear(96, 32) (qwen21_feed_forward.py:8-10); the weights are (out, in).
SEVEN = {
    "attn.to_q": (32, 32),
    "attn.to_k": (32, 32),
    "attn.to_v": (32, 32),
    "attn.to_out.0": (32, 32),
    "img_mlp.gate_layer": (96, 32),
    "img_mlp.proj": (96, 32),
    "img_mlp.out": (32, 96),
}
# The shared modulation: nn.Linear(inner, 4 * inner) (qwen21_transformer.py:37), so (128, 32).
MODULATION = {"modulation.layers.1.weight": (128, 32)}


def test_base_extras_drop_the_expanded_block_matrices_and_keep_modulation():
    # Bug caught: the stored gate_up names compared (nothing dropped, so the BF16 side would load its block matrices
    # as extras), or modulation.1 dropped (the BF16 side's coverage would refuse it).
    groups = {
        "transformer_blocks.0": SimpleNamespace(
            matrix_names=(
                "transformer_blocks.0.img_mlp.gate_layer.weight",
                "transformer_blocks.0.img_mlp.proj.weight",
            )
        ),
        "modulation.1": SimpleNamespace(matrix_names=("modulation.1.weight",)),
    }
    index = {
        name: (Path(name), None)
        for name in (
            "transformer_blocks.0.img_mlp.gate_layer.weight",
            "transformer_blocks.0.img_mlp.proj.weight",
            "modulation.1.weight",
            "img_in.weight",
        )
    }
    assert sorted(base_extras(index, SimpleNamespace(groups=groups))) == [
        "img_in.weight",
        "modulation.1.weight",
    ]


def _open(tmp_path, **kwargs):
    root = tmp_path / "df11"
    matrices, constants, layout = write_tiny_checkpoint(root, np.random.default_rng(3), **kwargs)
    return open_checkpoint(root, layouts=(layout,)), matrices, constants


def _params(build):
    return dict(tree_flatten(build.transformer.parameters()))


@pytest.mark.mflux
def test_the_build_installs_placeholders_and_loads_every_extra_by_name(tmp_path):
    # Bug caught: the modulation rename lost (an extra-coverage refusal), a compressed matrix left live (memory),
    # or an extra loaded onto the wrong parameter (a constant in the wrong place).
    ckpt, _matrices, constants = _open(tmp_path)
    assert "modulation.1" in ckpt.groups
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    params = _params(build)
    assert build.nonblock == MODULATION
    assert params["modulation.layers.1.weight"].size == 0
    assert build.shapes == {f"transformer_blocks.{i}": SEVEN for i in range(2)}
    for block in build.shapes:
        for attr in SEVEN:
            assert params[f"{block}.{attr}.weight"].size == 0, (block, attr)
    # Every parameter a checkpoint holds outside the compressed groups (S0: 72 extras for 32 blocks = 8 + 2 x 32).
    assert sorted(constants) == [
        "img_in.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.attn.norm_k.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.1.attn.norm_k.weight",
        "transformer_blocks.1.attn.norm_q.weight",
        "txt_in.in_layer.weight",
        "txt_in.out_layer.weight",
        "txt_in.text_norm.weight",
    ]
    for name, value in constants.items():
        assert np.all(np.array(params[name].view(mx.uint16)) == value), name
    assert build.counts == {"transformer_blocks": 2}


@pytest.mark.mflux
def test_the_computed_tables_are_left_as_mflux_built_them(tmp_path):
    # Bug caught: the coverage exemption replaced by loading something into the computed tables (a zero or stale
    # RoPE table would wreck the positions while every check passes).
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_rope import Qwen21Rope
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import (
        Qwen21Timesteps,
    )

    ckpt, _m, _c = _open(tmp_path)
    params = _params(build_transformer(ckpt, transformer_kwargs=TINY))
    rope = Qwen21Rope(axes_dim=[4, 6, 6])
    for i in range(3):
        for table, fresh in (("cos", rope.cos_tables[i]), ("sin", rope.sin_tables[i])):
            got = params[f"pos_embed.{table}_tables.{i}"]
            assert mx.array_equal(got.view(mx.uint32), fresh.view(mx.uint32)).item(), (table, i)
    freqs = params["time_text_embed.time_proj.freqs"]
    assert mx.array_equal(freqs.view(mx.uint32), Qwen21Timesteps().freqs.view(mx.uint32)).item()


@pytest.mark.mflux
def test_modulation_stored_as_an_extra_loads_as_a_plain_weight(tmp_path):
    # Bug caught: a checkpoint without the non-block group refused, or its modulation left on a placeholder (a
    # zero-size matmul before the first block).
    ckpt, _m, constants = _open(tmp_path, modulation_as_group=False)
    assert "modulation.1.weight" in ckpt.extras
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    weight = _params(build)["modulation.layers.1.weight"]
    assert build.nonblock == {}
    assert weight.shape == (128, 32)
    assert np.all(np.array(weight.view(mx.uint16)) == constants["modulation.layers.1.weight"])


@pytest.mark.mflux
def test_a_checkpoint_with_other_block_counts_is_refused_naming_both(tmp_path):
    # Bug caught: a 3-block checkpoint built as 2 (a shape error, or block 2 silently dropped) instead of a refusal.
    ckpt, _m, _c = _open(tmp_path, n_layers=3)
    with pytest.raises(DFloatFormatError, match=r"checkpoint has 3 blocks; this model builds 2\b"):
        build_transformer(ckpt, transformer_kwargs=TINY)


@pytest.mark.mflux
def test_the_count_check_defaults_to_mflux_s_depth(tmp_path):
    # Bug caught: kwargs without num_layers checked against something other than mflux's default 32
    # (qwen21_transformer.py:21): the real model passes no overrides.
    ckpt, _m, _c = _open(tmp_path)
    no_depth = {k: v for k, v in TINY.items() if k != "num_layers"}
    with pytest.raises(DFloatFormatError, match=r"checkpoint has 2 blocks; this model builds 32\b"):
        build_transformer(ckpt, transformer_kwargs=no_depth)


@pytest.mark.mflux
def test_n_layers_builds_fewer_blocks_and_is_range_checked(tmp_path):
    # Bug caught: a partial build loading block 1's norms into a transformer without block 1 (an extra without a
    # parameter), or a request beyond the checkpoint built from nothing.
    ckpt, _m, _c = _open(tmp_path)
    build = build_transformer(ckpt, transformer_kwargs=TINY, n_layers=1)
    assert build.counts == {"transformer_blocks": 1}
    assert len(build.transformer.transformer_blocks) == 1
    with pytest.raises(DFloatIntegrationError, match=r"asked for 3 blocks; the checkpoint has 2"):
        build_transformer(ckpt, transformer_kwargs=TINY, n_layers=3)


@pytest.mark.mflux
def test_a_checkpoint_of_another_width_is_refused_by_the_extras_shape_check(tmp_path):
    # Bug caught: the same block count at another inner width (a hand-made checkpoint) accepted.
    ckpt, _m, _c = _open(tmp_path, inner_override=4)
    with pytest.raises(DFloatFormatError, match="shape"):
        build_transformer(ckpt, transformer_kwargs=TINY)


@pytest.mark.mflux
def test_the_decoded_halves_land_on_gate_layer_and_proj(tmp_path):
    # Bug caught: the halves swapped anywhere between the layout, the reader and the map (the model would run
    # silu(proj) * gate), or a decoded matrix cut at the wrong row.
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.integrate.providers import DF11Provider
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    ckpt, matrices, _c = _open(tmp_path)
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    blocks = [g for g in ckpt.groups if g != "modulation.1"]
    provider = DF11Provider(
        load_resident_set(ckpt, blocks),
        {g: ckpt.groups[g].matrix_names for g in blocks},
        qwen21_name_map(),
        decode=partial(decode_group, backend="reference"),
    )
    got = provider.weights_for("transformer_blocks.1", build.shapes["transformer_blocks.1"])
    provider.verify()
    for attr in SEVEN:
        want = matrices["transformer_blocks.1"][attr]
        assert np.array_equal(np.array(got[attr].view(mx.uint16)), want), attr


@pytest.mark.mflux
def test_the_bf16_side_builds_from_a_sharded_base_with_modulation_as_a_plain_weight(tmp_path):
    # Bug caught (the BF16 side of an identity check): the sharded base's index not followed, the block matrices
    # loaded as extras, or modulation.1 left on a placeholder instead of loading as a plain weight.
    from mlx_dfloat.mflux.qwen21.transformer import base_transformer_files_index

    ckpt, matrices, _constants = _open(tmp_path)
    originals = {}
    for name, (_path, info) in ckpt.extras.items():
        originals[name] = np.full(
            info.shape, 0x4000, dtype=np.uint16
        )  # 2.0: distinct from the DF11 extras
    originals["modulation.1.weight"] = matrices["modulation.1"]["weight"]
    for block in ("transformer_blocks.0", "transformer_blocks.1"):
        for attr, mat in matrices[block].items():
            originals[f"{block}.{attr}.weight"] = mat
    root = write_bf16_original(tmp_path / "base", originals)
    index = base_transformer_files_index(root)
    assert {p.name for p, _i in index.values()} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    extras = base_extras(index, ckpt)
    assert not any(".img_mlp." in n or ".attn.to_" in n for n in extras)
    build = build_transformer(
        ckpt, transformer_kwargs=TINY, extras=extras, nonblock_from_extras=True
    )
    params = _params(build)
    assert build.nonblock == {}
    modulation = np.array(params["modulation.layers.1.weight"].view(mx.uint16))
    assert np.array_equal(modulation, matrices["modulation.1"]["weight"])
    assert np.all(np.array(params["img_in.weight"].view(mx.uint16)) == 0x4000)
    assert params["transformer_blocks.0.img_mlp.proj.weight"].size == 0


@pytest.mark.mflux
def test_the_build_refuses_one_that_adds_too_much_active_memory(tmp_path, monkeypatch):
    # Bug caught: the active-memory guard not wired (a build that materialised the decoded set would pass).
    from mlx_dfloat.mflux.qwen21 import transformer as module

    ckpt, _m, _c = _open(tmp_path)
    monkeypatch.setattr(module, "MAX_BUILD_ACTIVE_BYTES", 0)
    with pytest.raises(DFloatIntegrationError, match="GiB of active memory"):
        build_transformer(ckpt, transformer_kwargs=TINY)
