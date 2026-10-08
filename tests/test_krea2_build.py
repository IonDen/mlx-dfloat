"""The Krea 2 build from a config-less DF11 checkpoint: base extras offline, the real tiny build (mflux lane)."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._krea2_tiny import TINY, write_tiny_checkpoint

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux.krea2.transformer import base_extras, build_transformer

# The eight block matrices at TINY size, from mflux 0.20.0 (weights are (out, in)): Krea2Attention's wq / gate / wo
# Linear(32, 32) and wk / wv Linear(32, 16 x 1 kv head) (attention.py:16-21); Krea2SwiGLU's gate / up
# Linear(32, 128) and down Linear(128, 32) (feed_forward.py:7-12, width 84 rounded up to 128).
EIGHT = {
    "attn.wq": (32, 32),
    "attn.wk": (16, 32),
    "attn.wv": (16, 32),
    "attn.gate": (32, 32),
    "attn.wo": (32, 32),
    "mlp.gate": (128, 32),
    "mlp.up": (128, 32),
    "mlp.down": (32, 128),
}
# A text-fusion block at txtdim 32, 2 heads and 2 kv heads: five Linear(32, 32), the same SwiGLU (text_fusion.py).
TEXT_BLOCK = {
    "attn.wq": (32, 32),
    "attn.wk": (32, 32),
    "attn.wv": (32, 32),
    "attn.gate": (32, 32),
    "attn.wo": (32, 32),
    "mlp.gate": (128, 32),
    "mlp.up": (128, 32),
    "mlp.down": (32, 128),
}
# The non-block matrices as mflux names them: tmlp Linear(tdim 32, 32) and Linear(32, 32); tproj Linear(32, 6 x 32);
# txtmlp Linear(txtdim 32, 32) and Linear(32, 32) (timestep_embedder.py, text_mlp.py).
NONBLOCK = {
    "tmlp.linear_in.weight": (32, 32),
    "tmlp.linear_out.weight": (32, 32),
    "tproj.linear.weight": (192, 32),
    "txtmlp.linear_in.weight": (32, 32),
    "txtmlp.linear_out.weight": (32, 32),
    **{
        f"txtfusion.{kind}.{i}.{attr}.weight": shape
        for kind in ("layerwise_blocks", "refiner_blocks")
        for i in (0, 1)
        for attr, shape in TEXT_BLOCK.items()
    },
}


def test_base_extras_drop_the_block_matrices_and_keep_the_nonblock_ones():
    # Bug caught: the block matrices loaded from the base as extras (a coverage refusal), or a non-block matrix
    # dropped (the BF16 side of an identity check would leave tproj on a placeholder).
    groups = {
        "blocks.0": SimpleNamespace(
            matrix_names=("blocks.0.attn.wq.weight", "blocks.0.mlp.down.weight")
        ),
        "tproj": SimpleNamespace(matrix_names=("tproj.1.weight",)),
        "txtfusion.refiner_blocks.1": SimpleNamespace(
            matrix_names=("txtfusion.refiner_blocks.1.attn.wq.weight",)
        ),
    }
    index = {
        name: (Path(name), None)
        for name in (
            "blocks.0.attn.wq.weight",
            "blocks.0.mlp.down.weight",
            "blocks.0.mod.lin",
            "tproj.1.weight",
            "txtfusion.refiner_blocks.1.attn.wq.weight",
            "first.weight",
        )
    }
    assert sorted(base_extras(index, SimpleNamespace(groups=groups))) == [
        "blocks.0.mod.lin",
        "first.weight",
        "tproj.1.weight",
        "txtfusion.refiner_blocks.1.attn.wq.weight",
    ]


def _open(tmp_path, **kwargs):
    root = tmp_path / "df11"
    matrices, constants, layout = write_tiny_checkpoint(root, np.random.default_rng(3), **kwargs)
    return open_checkpoint(root, layouts=(layout,)), matrices, constants


def _params(build):
    return dict(tree_flatten(build.transformer.parameters()))


@pytest.mark.mflux
def test_the_build_installs_placeholders_and_loads_every_extra_by_name(tmp_path):
    # Bug caught: a rename lost (txtmlp.0.scale loaded nowhere: a coverage refusal) or an extra on the wrong parameter
    # (a constant in the wrong place), a compressed matrix left live (memory), or a non-block matrix not on a
    # placeholder (it would load nothing and keep mflux's random init).
    ckpt, _matrices, constants = _open(tmp_path)
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    params = _params(build)
    assert build.shapes == {f"blocks.{i}": EIGHT for i in range(2)}
    assert build.nonblock == NONBLOCK
    for block in build.shapes:
        for attr in EIGHT:
            assert params[f"{block}.{attr}.weight"].size == 0, (block, attr)
    for name in NONBLOCK:
        assert params[name].size == 0, name
    assert "tproj.linear.bias" in constants
    assert "blocks.1.mod.lin" in constants
    assert "txtmlp.norm.scale" in constants
    for name, value in constants.items():
        assert np.all(np.array(params[name].view(mx.uint16)) == value), name
    assert build.counts == {"blocks": 2}


@pytest.mark.mflux
def test_nonblock_groups_stored_as_extras_load_as_plain_weights(tmp_path):
    # Bug caught: a checkpoint without the non-block groups refused, or tproj left on a placeholder (a zero-size matmul
    # before the first block).
    ckpt, _m, constants = _open(tmp_path, nonblock_as_groups=False)
    assert set(ckpt.groups) == {"blocks.0", "blocks.1"}
    assert "tproj.1.weight" in ckpt.extras
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    weight = _params(build)["tproj.linear.weight"]
    assert build.nonblock == {}
    assert weight.shape == (192, 32)
    assert np.all(np.array(weight.view(mx.uint16)) == constants["tproj.linear.weight"])


@pytest.mark.mflux
def test_a_checkpoint_with_other_block_counts_is_refused_naming_both(tmp_path):
    # Bug caught: a 3-block checkpoint built as 2 (block 2 silently dropped) instead of a refusal.
    ckpt, _m, _c = _open(tmp_path, n_layers=3)
    with pytest.raises(DFloatFormatError, match=r"^checkpoint has 3 blocks; this model builds 2$"):
        build_transformer(ckpt, transformer_kwargs=TINY)


@pytest.mark.mflux
def test_the_count_check_defaults_to_mfluxs_depth_under_mfluxs_parameter_name(tmp_path):
    # Bug caught: the depth read from `num_layers` (Qwen's and ERNIE's name) instead of Krea's `layers`, so the
    # published model (no kwargs) or the tiny one would be checked against the wrong depth.
    ckpt, _m, _c = _open(tmp_path)
    no_depth = {k: v for k, v in TINY.items() if k != "layers"}
    with pytest.raises(DFloatFormatError, match=r"^checkpoint has 2 blocks; this model builds 28$"):
        build_transformer(ckpt, transformer_kwargs=no_depth)


@pytest.mark.mflux
def test_n_layers_builds_fewer_blocks_and_is_range_checked(tmp_path):
    # Bug caught: a partial build loading block 1's norms into a transformer without block 1 (an extra without a
    # parameter), or a request beyond the checkpoint built from nothing.
    ckpt, _m, _c = _open(tmp_path)
    build = build_transformer(ckpt, transformer_kwargs=TINY, n_layers=1)
    assert build.counts == {"blocks": 1}
    assert len(build.transformer.blocks) == 1
    with pytest.raises(DFloatIntegrationError, match=r"asked for 3 blocks; the checkpoint has 2"):
        build_transformer(ckpt, transformer_kwargs=TINY, n_layers=3)


@pytest.mark.mflux
def test_the_decoded_block_matrices_land_in_pattern_order(tmp_path):
    # Bug caught: the order swapped between the layout, the reader and the map (attn.gate and attn.wo, or mlp.up and
    # mlp.down: equal shapes or element counts, so only the values show it).
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.integrate.providers import DF11Provider
    from mlx_dfloat.mflux.krea2.names import krea2_name_map

    ckpt, matrices, _c = _open(tmp_path)
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    blocks = ["blocks.0", "blocks.1"]
    provider = DF11Provider(
        load_resident_set(ckpt, blocks),
        {g: ckpt.groups[g].matrix_names for g in blocks},
        krea2_name_map(),
        decode=partial(decode_group, backend="reference"),
    )
    got = provider.weights_for("blocks.1", build.shapes["blocks.1"])
    provider.verify()
    for attr in EIGHT:
        want = matrices["blocks.1"][attr]
        assert np.array_equal(np.array(got[attr].view(mx.uint16)), want), attr


@pytest.mark.mflux
def test_the_raw_model_refuses_the_turbo_layouts_checkpoint_at_build(tmp_path):
    # Bug caught: Turbo's distilled weights built and run under Raw's 25-step default (or the card's
    # 52 steps at 3.5), and a row labelled Raw measuring Turbo. The refusal comes before any module is built: the
    # transformer class is never called (MLX builds modules lazily, so active memory cannot show a stray build).
    from mlx_dfloat.mflux.krea2.transformer import seam_transformer_class

    ckpt, _m, _c = _open(tmp_path, key="krea-2-turbo-comfyui")
    built = []

    def recording(**kwargs):
        built.append(kwargs)
        return seam_transformer_class()(**kwargs)

    with pytest.raises(
        DFloatFormatError, match=r"Krea 2 Turbo DF11 checkpoint .*run it with --model krea-2$"
    ):
        build_transformer(
            ckpt, model="krea-2-raw", transformer_kwargs=TINY, transformer_class=recording
        )
    assert built == []
    build = build_transformer(
        ckpt, model="krea-2", transformer_kwargs=TINY, transformer_class=recording
    )
    assert build.counts == {"blocks": 2}
    assert len(built) == 1


@pytest.mark.mflux
def test_the_build_refuses_one_that_adds_too_much_active_memory(tmp_path, monkeypatch):
    # Bug caught: the active-memory guard not wired (a build that materialised the decoded set would pass).
    from mlx_dfloat.mflux.krea2 import transformer as module

    ckpt, _m, _c = _open(tmp_path)
    monkeypatch.setattr(module, "MAX_BUILD_ACTIVE_BYTES", 0)
    with pytest.raises(DFloatIntegrationError, match="GiB of active memory"):
        build_transformer(ckpt, transformer_kwargs=TINY)


@pytest.mark.mflux
def test_the_bf16_side_builds_with_the_nonblock_matrices_as_plain_weights(tmp_path):
    # Bug caught (the BF16 side of an identity check): the block matrices loaded as extras, or a non-block matrix left
    # on a placeholder instead of loading from the base.
    from tests._df11_fixtures import write_bf16_original

    from mlx_dfloat.mflux.krea2.transformer import base_transformer_files_index

    ckpt, matrices, _constants = _open(tmp_path)
    originals = {
        name: np.full(info.shape, 0x4000, dtype=np.uint16)
        for name, (_p, info) in ckpt.extras.items()
    }
    originals.update(
        (name, mat)
        for group, g in ckpt.groups.items()
        for name, mat in zip(g.matrix_names, matrices[group].values(), strict=True)
    )
    root = write_bf16_original(tmp_path / "base", originals)
    extras = base_extras(base_transformer_files_index(root), ckpt)
    assert not any(n.startswith("blocks.") and n.endswith(".weight") for n in extras)
    build = build_transformer(
        ckpt, transformer_kwargs=TINY, extras=extras, nonblock_from_extras=True
    )
    params = _params(build)
    assert build.nonblock == {}
    tproj = np.array(params["tproj.linear.weight"].view(mx.uint16))
    assert np.array_equal(tproj, matrices["tproj"]["tproj.linear.weight"])
    assert np.all(np.array(params["first.weight"].view(mx.uint16)) == 0x4000)
    assert params["blocks.0.mlp.up.weight"].size == 0


# --- the BF16/FP32 base file as the identity check's reference side -----------------------------------------------

# FP32 bit patterns and their BF16 by round-to-nearest-even, by hand (the patterns of the parity check's tests):
# a tie with an even neighbour stays, a tie with an odd neighbour rounds up, just above half rounds up, just below
# half rounds down, and the largest finite float32 rounds to +inf. Truncation would give 0x3F80, 0x3F81, 0x3F80,
# 0x3F80, 0x7F7F.
F32_PATTERNS = (0x3F808000, 0x3F818000, 0x3F808001, 0x3F807FFF, 0x7F7FFFFF)
RNE_BF16 = (0x3F80, 0x3F82, 0x3F81, 0x3F80, 0x7F80)


def test_mlx_astype_rounds_fp32_to_bf16_nearest_even():
    # Bug caught: if MLX truncated (or rounded half away from zero), the identity's reference side, which casts the
    # base's FP32 tensors as mflux does at load (weight_loader.py:448-449), would differ from the checkpoint's own
    # BF16 (rounded nearest even when it was made) and report a false mismatch on the five FP32 groups and the extras.
    f32 = mx.array(np.array(F32_PATTERNS, dtype=np.uint32)).view(mx.float32)
    assert np.array(f32.astype(mx.bfloat16).view(mx.uint16)).tolist() == list(RNE_BF16)


def _base_file(path, tensors):
    """A safetensors file of ``{name: (numpy bits, mx dtype)}``, written as that dtype."""
    mx.save_safetensors(
        str(path), {n: mx.array(bits).view(dtype) for n, (bits, dtype) in tensors.items()}
    )
    return path


def test_base_bf16_weights_reads_bf16_as_is_and_rounds_fp32_to_nearest_even(tmp_path):
    # Bug caught: an FP32 base tensor truncated (two of the five patterns differ) or read as two BF16 halves, a BF16
    # one converted at all, or a name read from another tensor.
    from mlx_dfloat.mflux.krea2.transformer import base_bf16_weights

    bf16 = np.array([0x3F80, 0xBF80, 0x0001], dtype=np.uint16)
    path = _base_file(
        tmp_path / "raw.safetensors",
        {
            "blocks.0.attn.wq.weight": (bf16, mx.bfloat16),
            "tproj.1.bias": (np.array(F32_PATTERNS, dtype=np.uint32), mx.float32),
            "unused": (np.array([7], dtype=np.uint16), mx.bfloat16),
        },
    )
    got = base_bf16_weights(path, ["tproj.1.bias", "blocks.0.attn.wq.weight"])
    assert sorted(got) == ["blocks.0.attn.wq.weight", "tproj.1.bias"]
    assert {n: a.dtype for n, a in got.items()} == {
        "blocks.0.attn.wq.weight": mx.bfloat16,
        "tproj.1.bias": mx.bfloat16,
    }
    assert np.array(got["blocks.0.attn.wq.weight"].view(mx.uint16)).tolist() == [
        0x3F80,
        0xBF80,
        0x0001,
    ]
    assert np.array(got["tproj.1.bias"].view(mx.uint16)).tolist() == list(RNE_BF16)


def test_base_bf16_weights_evaluates_each_cast_as_it_goes(tmp_path, monkeypatch):
    # Bug caught: the FP32 casts left lazy, so every FP32 source array (twice its BF16 size) stays alive until the
    # caller's first eval: on the real file 174 FP32 tensors held at once instead of one. One eval per FP32 tensor,
    # each on its own cast, in read order; a BF16 tensor needs none.
    from mlx_dfloat.mflux.krea2 import transformer as ktf

    seen = []
    real = ktf._eval
    monkeypatch.setattr(ktf, "_eval", lambda *a: (seen.append(tuple(x.dtype for x in a)), real(*a)))
    f32 = (np.array(F32_PATTERNS, dtype=np.uint32), mx.float32)
    path = _base_file(
        tmp_path / "raw.safetensors",
        {
            "a.bias": f32,
            "b.weight": (np.array([0x3F80], dtype=np.uint16), mx.bfloat16),
            "c.bias": f32,
        },
    )
    got = ktf.base_bf16_weights(path, ["a.bias", "b.weight", "c.bias"])
    assert seen == [(mx.bfloat16,), (mx.bfloat16,)]
    assert np.array(got["c.bias"].view(mx.uint16)).tolist() == list(RNE_BF16)


def test_base_bf16_weights_refuses_an_f16_tensor_and_a_missing_name(tmp_path):
    # Bug caught: an F16 tensor read as BF16 bits (a silent wrong reference), or a tensor the base lacks skipped (the
    # reference side would keep the DF11 file's value for it and agree by construction).
    from mlx_dfloat.mflux.krea2.transformer import base_bf16_weights

    path = _base_file(
        tmp_path / "raw.safetensors",
        {"first.bias": (np.array([0x3C00], dtype=np.uint16), mx.float16)},
    )
    with pytest.raises(
        DFloatFormatError, match=r"first\.bias is F16; the base must hold BF16 or F32"
    ):
        base_bf16_weights(path, ["first.bias"])
    with pytest.raises(
        DFloatFormatError, match=r"raw\.safetensors has no tensor 'last\.linear\.bias'"
    ):
        base_bf16_weights(path, ["last.linear.bias"])


@pytest.mark.mflux
def test_install_base_weights_replaces_every_extra_and_nonblock_matrix_with_the_bases(tmp_path):
    # Bug caught: the reference side keeping any DF11 value (an extra or a non-block matrix not replaced: the identity
    # would then compare the checkpoint with itself on it), an FP32 base tensor not rounded as mflux loads it, a
    # rename lost (tmlp.0 vs tmlp.linear_in), or a shape mismatch accepted. The base holds every tensor of the tiny
    # model under the checkpoint's names; extras and non-block matrices as FP32 (random bits), blocks as BF16.
    from scripts.verify_remote_group import fp32_to_bf16_rne
    from tests._krea2_tiny import write_tiny_checkpoint

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS, krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import install_base_weights

    rng = np.random.default_rng(13)
    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", rng)
    ckpt = open_checkpoint(tmp_path / "df11", layouts=(layout,))
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    names = krea2_name_map()
    nonblock = [m for g in NONBLOCK_GROUPS for m in ckpt.groups[g].matrix_names]
    tensors, want = {}, {}
    built = dict(tree_flatten(build.transformer.parameters()))
    for name in [*ckpt.extras, *nonblock]:
        param = names.param_name(name)
        shape = build.nonblock[param] if param in build.nonblock else tuple(built[param].shape)
        bits = rng.integers(0x3F000000, 0x40000000, size=shape, dtype=np.uint64).astype(np.uint32)
        tensors[name] = (bits, mx.float32)
        want[param] = fp32_to_bf16_rne(bits)  # an independent RNE (the parity check's NumPy one)
    path = _base_file(tmp_path / "raw.safetensors", tensors)
    replaced = install_base_weights(build.transformer, ckpt, path, names, build.nonblock)
    assert sorted(replaced) == sorted([*ckpt.extras, *nonblock])
    params = dict(tree_flatten(build.transformer.parameters()))
    for param, bits in want.items():
        got = params[param]
        assert got.dtype == mx.bfloat16, param
        assert np.array_equal(np.array(got.view(mx.uint16)), bits), param
    assert params["blocks.0.attn.wq.weight"].size == 0  # the blocks still stream


@pytest.mark.mflux
def test_install_base_weights_refuses_a_tensor_of_another_shape(tmp_path):
    # Bug caught: a base of another width loaded onto the tiny transformer without a check (a silent wrong reference
    # or a matmul error minutes into the run).
    from tests._krea2_tiny import write_tiny_checkpoint

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS, krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import install_base_weights

    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(1))
    ckpt = open_checkpoint(tmp_path / "df11", layouts=(layout,))
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    names = krea2_name_map()
    tensors = {
        n: (np.zeros(info.shape, dtype=np.uint16), mx.bfloat16)
        for n, (_p, info) in ckpt.extras.items()
    }
    for g in NONBLOCK_GROUPS:
        for m in ckpt.groups[g].matrix_names:
            tensors[m] = (
                np.zeros(build.nonblock[names.param_name(m)], dtype=np.uint16),
                mx.bfloat16,
            )
    tensors["tproj.1.weight"] = (np.zeros((4, 4), dtype=np.uint16), mx.bfloat16)
    path = _base_file(tmp_path / "raw.safetensors", tensors)
    with pytest.raises(
        DFloatFormatError, match=r"tproj\.1\.weight has shape \(4, 4\); the model needs \(192, 32\)"
    ):
        install_base_weights(build.transformer, ckpt, path, names, build.nonblock)
