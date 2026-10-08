"""Real Z-Image checkpoints: group layout, Metal decode against the base shards, and a reduced build run.

Env-gated: every test skips unless the snapshot directories are named by ``MLX_DFLOAT_ZIMAGE_DF11`` (the Z-Image
DF11 checkpoint), ``MLX_DFLOAT_ZIMAGE_TURBO_DF11`` (the Turbo one) and ``MLX_DFLOAT_ZIMAGE_BASE`` (the BF16
``Tongyi-MAI/Z-Image`` snapshot). Expected layouts are literals from mflux 0.20.0 (``ZImageTransformer`` defaults
``n_layers=30``, ``n_refiner_layers=2``, transformer.py:20-21) and the checkpoints' own ``pattern_dict``; the
expected bits come from the base shards, never from the code under test.
"""

import os
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
from mlx_dfloat.mflux.flux1.transformer import base_transformer_index
from mlx_dfloat.mflux.zimage.names import check_zimage_groups, zimage_name_map
from mlx_dfloat.mflux.zimage.transformer import (
    ZIMAGE_BASE_INDEX,
    base_extras,
    build_transformer,
)


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
    return base_transformer_index(base / "transformer", index_file=ZIMAGE_BASE_INDEX)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["MLX_DFLOAT_ZIMAGE_DF11", "MLX_DFLOAT_ZIMAGE_TURBO_DF11"])
def test_both_checkpoints_have_zimage_groups_and_bf16_extras(env: str) -> None:
    # Bug caught: a checkpoint layout other than the pattern_dict promises (counts from mflux's defaults n_layers=30,
    # n_refiner_layers=2, transformer.py:20-21), or a Turbo extra left in FP32 (read_extra would refuse it mid-load).
    ckpt = open_checkpoint(_env(env))
    assert check_zimage_groups(ckpt) == {"noise_refiner": 2, "context_refiner": 2, "layers": 30}
    assert sorted({info.dtype for _path, info in ckpt.extras.values()}) == ["BF16"]


@pytest.mark.slow
@pytest.mark.metal
def test_four_real_groups_decode_on_metal_bit_exact_against_the_base_shards() -> None:
    # Bug caught: a real-file decode or split error on a kind the FLUX checkpoints never had (context blocks without
    # modulation; the one-matrix cap_embedder group).
    ckpt = open_checkpoint(_env("MLX_DFLOAT_ZIMAGE_DF11"))
    index = _base_index(_env("MLX_DFLOAT_ZIMAGE_BASE"))
    for name in ("noise_refiner.0", "context_refiner.0", "layers.0", "cap_embedder"):
        group = ckpt.groups[name]
        loaded = load_group_mx(group)
        result = decode_group(loaded, backend="metal")
        check(result, name=name)
        parts = split_matrices(result.bits, loaded.split_positions)
        assert len(parts) == len(group.matrix_names), name
        for matrix, part in zip(group.matrix_names, parts, strict=True):
            path, info = index[matrix]
            want = np.array(read_bf16(path, info).view(mx.uint16)).ravel()
            assert np.array_equal(np.array(part.view(mx.uint16)).ravel(), want), matrix


def _run(transformer: Any) -> mx.array:
    keys = mx.random.split(mx.random.key(1), 2)
    out = transformer(
        x=mx.random.normal((16, 1, 8, 8), key=keys[0]),
        timestep=mx.array([0.5]),
        sigmas=mx.array([1.0, 0.0]),
        cap_feats=mx.random.normal((16, 2560), key=keys[1]),
    )
    mx.eval(out)
    return out


@pytest.mark.slow
@pytest.mark.metal
@pytest.mark.mflux
def test_a_one_by_one_by_one_build_from_real_extras_matches_the_bf16_stream_bit_for_bit() -> None:
    # Bug caught: a real extra renamed wrongly (coverage passes on fakes but a real name differs), the cap_embedder
    # weight decoded to a wrong layout, or any block-seam difference between the DF11 and the streamed BF16 paths.
    ckpt: DF11Checkpoint = open_checkpoint(_env("MLX_DFLOAT_ZIMAGE_DF11"))
    index = _base_index(_env("MLX_DFLOAT_ZIMAGE_BASE"))
    names = zimage_name_map()

    df11 = build_transformer(ckpt, name_map=names, n_refiner=1, n_layers=1)
    blocks = list(df11.shapes)
    groups = load_resident_set(ckpt, [*blocks, "cap_embedder"])
    matrix_names = {n: ckpt.groups[n].matrix_names for n in groups}
    nonblock = decode_nonblock(
        {"cap_embedder": groups["cap_embedder"]},
        {"cap_embedder": matrix_names["cap_embedder"]},
        df11.nonblock,
        names,
    )
    install_nonblock(df11.transformer, nonblock)
    provider = DF11Provider(
        {n: groups[n] for n in blocks}, {n: matrix_names[n] for n in blocks}, names
    )
    df11.transformer.attach(provider, df11.shapes, verify_in_call=True)
    out_a = _run(df11.transformer)

    bf16 = build_transformer(
        ckpt,
        name_map=names,
        n_refiner=1,
        n_layers=1,
        extras=base_extras(index, ckpt),
        nonblock_from_extras=True,
    )
    bf16.transformer.attach(
        StreamingBF16Provider(index, {n: ckpt.groups[n].matrix_names for n in bf16.shapes}, names),
        bf16.shapes,
        verify_in_call=True,
    )
    out_b = _run(bf16.transformer)

    assert out_a.shape == out_b.shape == (16, 1, 8, 8)
    assert out_a.dtype == out_b.dtype
    assert np.array_equal(_bits(out_a), _bits(out_b))
