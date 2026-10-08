"""The real ERNIE-Image DF11 checkpoint: its groups, Metal decode against the BF16 base, and a one-block build run.

Env-gated: every test skips unless the snapshot directories are named by ``MLX_DFLOAT_ERNIE_DF11`` (the
``mingyi456/ERNIE-Image-DF11`` snapshot: ``config.json`` and ``model.safetensors``) and, for the base comparisons,
``MLX_DFLOAT_ERNIE_BASE`` (the ``baidu/ERNIE-Image`` snapshot with its ``transformer/`` index and two shards). Expected
counts are literals from the checkpoint's header (39 groups: ``layers.0..35`` and the three non-block groups; 153 BF16
extras, read 2026-10-08) and from mflux 0.20.0 (``ErnieTransformer`` builds 36 blocks, transformer.py:36-76); the
expected bits come from the base's tensors, never from the code under test.
"""

import gc
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_dfloat.decode import check, decode_group, split_matrices
from mlx_dfloat.format import DF11Checkpoint, load_group_mx, open_checkpoint
from mlx_dfloat.integrate.coverage import load_resident_set
from mlx_dfloat.integrate.providers import DF11Provider, StreamingBF16Provider, read_bf16
from mlx_dfloat.integrate.resident import decode_nonblock, install_nonblock
from mlx_dfloat.mflux.ernie.names import KIND, NONBLOCK_GROUPS, check_ernie_groups

DF11_ENV, BASE_ENV = "MLX_DFLOAT_ERNIE_DF11", "MLX_DFLOAT_ERNIE_BASE"
# Every block's split positions, by hand: to_q, to_k, to_v, to_out.0 (4096 x 4096 = 16777216 each), then gate_proj and
# up_proj (12288 x 4096 = 50331648 each); linear_fc2 (4096 x 12288) ends the group at 218103808.
BLOCK_SPLITS = (16777216, 33554432, 50331648, 67108864, 117440512, 167772160)
SEVEN = (
    "self_attention.to_q",
    "self_attention.to_k",
    "self_attention.to_v",
    "self_attention.to_out.0",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.linear_fc2",
)
# ModelConfig.ernie_image().transformer_overrides (mflux 0.20.0 model_config.py:698-731).
OVERRIDES = {"rope_axes_dim": [32, 48, 48]}


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return Path(value)


def _bits(array: mx.array) -> np.ndarray:
    """The array's raw bits through a dtype-sized unsigned view (equality of bits, never of floats)."""
    view = {2: mx.uint16, 4: mx.uint32}[array.dtype.size]
    return np.array(array.view(view))


def _base_index(base: Path) -> dict[str, Any]:
    from mlx_dfloat.mflux.ernie.transformer import base_transformer_files_index

    return base_transformer_files_index(base / "transformer")


@pytest.mark.slow
def test_the_real_checkpoint_opens_with_ernies_groups() -> None:
    # Bug caught: a group or extra the adapter does not expect (the build would refuse at the first user's call): a
    # group count other than the header's 39, a block holding other than the seven matrices, a non-block group
    # missing, or an extra count other than 153 (four per block plus nine non-block tensors).
    ckpt = open_checkpoint(_env(DF11_ENV))
    assert ckpt.config_source == "config.json"
    assert check_ernie_groups(ckpt) == {KIND: 36}
    assert len(ckpt.groups) == 39
    assert sorted(g for g in ckpt.groups if not g.startswith("layers.")) == [
        "adaLN_modulation.1",
        "final_norm.linear",
        "time_embedding",
    ]
    assert tuple(ckpt.groups["layers.35"].matrix_names) == tuple(
        f"layers.35.{sub}.weight" for sub in SEVEN
    )
    assert len(ckpt.extras) == 153
    assert sorted({info.dtype for _path, info in ckpt.extras.values()}) == ["BF16"]


@pytest.mark.slow
@pytest.mark.metal
def test_four_real_groups_decode_on_metal_bit_exact_against_the_base() -> None:
    # Bug caught: a group decoded against the wrong base tensor (a stored matrix order off in the first or last
    # block: gate_proj and up_proj have the same shape, so only their values show it), or a non-block group mis-split
    # (time_embedding's two matrices cut at the wrong element, adaLN_modulation.1 (24576 x 4096) in the wrong layout).
    ckpt = open_checkpoint(_env(DF11_ENV))
    index = _base_index(_env(BASE_ENV))
    for name in ("layers.0", "layers.35", "adaLN_modulation.1", "time_embedding"):
        group = ckpt.groups[name]
        loaded = load_group_mx(group)
        if name.startswith("layers."):
            assert loaded.split_positions == BLOCK_SPLITS, name
        result = decode_group(loaded, backend="metal")
        check(result, name=name)
        parts = split_matrices(result.bits, loaded.split_positions)
        assert len(parts) == len(group.matrix_names), name
        for matrix, part in zip(group.matrix_names, parts, strict=True):
            path, info = index[matrix]
            want = np.array(read_bf16(path, info).view(mx.uint16)).ravel()
            assert np.array_equal(np.array(part.view(mx.uint16)).ravel(), want), matrix
        del loaded, result, parts
        gc.collect()
        mx.clear_cache()


def _inputs(*, size: int, text_tokens: int, seed: int = 1) -> dict[str, Any]:
    """One call's keyword inputs: a (1, 128, size/16, size/16) latent, ``text_tokens`` of width 3072, one timestep."""
    side = size // 16
    keys = mx.random.split(mx.random.key(seed), 2)
    return {
        "hidden_states": mx.random.normal((1, 128, side, side), key=keys[0]).astype(mx.bfloat16),
        "timestep": mx.array([500.0]),
        "text_bth": mx.random.normal((1, text_tokens, 3072), key=keys[1]).astype(mx.bfloat16),
        "text_lens": mx.array([text_tokens], dtype=mx.int32),
    }


def _run(transformer: Any, inputs: Mapping[str, Any]) -> mx.array:
    out = transformer(**inputs)
    mx.eval(out)
    return out


def df11_one_block(ckpt: DF11Checkpoint, inputs: Mapping[str, Any]) -> mx.array:
    """Build one block from the checkpoint, decode and install the non-block groups, attach the DF11 decode, run."""
    from mlx_dfloat.mflux.ernie.names import ernie_name_map
    from mlx_dfloat.mflux.ernie.transformer import build_transformer

    names = ernie_name_map()
    build = build_transformer(ckpt, transformer_overrides=OVERRIDES, n_layers=1)
    blocks = list(build.shapes)
    assert blocks == ["layers.0"]
    groups = load_resident_set(ckpt, [*blocks, *NONBLOCK_GROUPS])
    matrix_names = {n: ckpt.groups[n].matrix_names for n in groups}
    nonblock = decode_nonblock(
        {g: groups[g] for g in NONBLOCK_GROUPS},
        {g: matrix_names[g] for g in NONBLOCK_GROUPS},
        build.nonblock,
        names,
    )
    assert sorted(nonblock) == [
        "adaln_modulation.weight",
        "final_norm.linear.weight",
        "time_embedding.linear_1.weight",
        "time_embedding.linear_2.weight",
    ]
    install_nonblock(build.transformer, nonblock)
    provider = DF11Provider(
        {n: groups[n] for n in blocks}, {n: matrix_names[n] for n in blocks}, names
    )
    build.transformer.attach(provider, build.shapes, verify_in_call=True)
    return _run(build.transformer, inputs)


def bf16_one_block(
    ckpt: DF11Checkpoint, index: Mapping[str, Any], inputs: Mapping[str, Any]
) -> mx.array:
    """The same one-block build with the base's BF16 extras, the non-block matrices plain weights, blocks streamed."""
    from mlx_dfloat.mflux.ernie.names import ernie_name_map
    from mlx_dfloat.mflux.ernie.transformer import base_extras, build_transformer

    names = ernie_name_map()
    build = build_transformer(
        ckpt,
        transformer_overrides=OVERRIDES,
        n_layers=1,
        extras=base_extras(index, ckpt),
        nonblock_from_extras=True,
    )
    assert build.nonblock == {}
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in build.shapes}, names
    )
    build.transformer.attach(provider, build.shapes, verify_in_call=True)
    return _run(build.transformer, inputs)


@pytest.mark.slow
@pytest.mark.metal
@pytest.mark.mflux
def test_a_one_block_build_matches_the_bf16_stream_bit_for_bit() -> None:
    # Bug caught: a non-block rename or a block matrix misplaced on one side (the adaLN_modulation.1 rename, the patch
    # conv's transpose, a non-block group left on its placeholder: a zero-size matmul before the first block), or any
    # block-seam difference between the DF11 and the streamed BF16 paths (the decoder itself is the test above's).
    # 1024²: a 64 x 64 latent of 128 channels, 32 text tokens of mflux's 3072-wide text input.
    ckpt = open_checkpoint(_env(DF11_ENV))
    index = _base_index(_env(BASE_ENV))
    inputs = _inputs(size=1024, text_tokens=32)
    out_a = df11_one_block(ckpt, inputs)
    out_b = bf16_one_block(ckpt, index, inputs)
    assert out_a.shape == out_b.shape == (1, 128, 64, 64)
    assert out_a.dtype == out_b.dtype
    assert np.array_equal(_bits(out_a), _bits(out_b))
