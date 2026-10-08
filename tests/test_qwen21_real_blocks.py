"""The real Qwen-Image 2.1 DF11 file: its layout, Metal decode against the BF16 base, and a one-block build run.

Env-gated: every test skips unless the snapshot directories are named by ``MLX_DFLOAT_QWEN21_DF11`` (the
``mingyi456/Qwen-Image-2.1-DF11-ComfyUI`` snapshot holding the single config-less file) and, for the base comparisons,
``MLX_DFLOAT_QWEN21_BASE`` (the ``Qwen/Qwen-Image-2.1`` snapshot with its ``transformer/`` index and shards). Expected
counts are literals from the file's own header (33 groups: ``modulation.1`` and ``transformer_blocks.0..31``; 72 BF16
extras) and from mflux 0.20.0 (``Qwen21Transformer`` builds 32 blocks, qwen21_transformer.py:17-28); the expected bits
come from the base's tensors, never from the code under test.
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
from mlx_dfloat.mflux.qwen21.names import KIND, check_qwen21_groups

DF11_ENV, BASE_ENV = "MLX_DFLOAT_QWEN21_DF11", "MLX_DFLOAT_QWEN21_BASE"
MODULATION = "modulation.1"
# Every block's expanded split positions: the stored (16777216, 33554432, 50331648, 67108864, 167772160) with the
# gate/up seam inserted at 67108864 + 12288 x 4096 = 117440512 (the fused img_mlp.gate_up cut into its halves).
BLOCK_SPLITS = (16777216, 33554432, 50331648, 67108864, 117440512, 167772160)
SEVEN = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
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
    from mlx_dfloat.mflux.qwen21.transformer import base_transformer_files_index

    return base_transformer_files_index(base / "transformer")


@pytest.mark.slow
def test_the_real_file_opens_through_its_layout() -> None:
    # Bug caught: the published file not matched by its pinned layout (a header or spot-check drift: the file would
    # be refused, or worse, read with another order), a group count other than the header's 33, a block holding
    # other than the seven expanded matrices (the fused gate_up left whole), or an extra count other than 72 (two
    # RMSNorm weights per block plus eight non-block tensors; a stray tensor the coverage check would refuse).
    ckpt = open_checkpoint(_env(DF11_ENV))
    assert check_qwen21_groups(ckpt) == {KIND: 32}
    assert len(ckpt.groups) == 33
    assert MODULATION in ckpt.groups
    assert tuple(ckpt.groups["transformer_blocks.31"].matrix_names) == tuple(
        f"transformer_blocks.31.{sub}.weight" for sub in SEVEN
    )
    assert len(ckpt.extras) == 72
    assert sorted({info.dtype for _path, info in ckpt.extras.values()}) == ["BF16"]
    assert ckpt.config_source.startswith(
        "header and spot checks match layout qwen-image-2.1-comfyui "
        "(mingyi456/Qwen-Image-2.1-DF11-ComfyUI@1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc)"
    )


@pytest.mark.slow
@pytest.mark.metal
def test_four_real_groups_decode_on_metal_bit_exact_against_the_base() -> None:
    # Bug caught: the exponent bits of this file differing from the base (only sign/mantissa bytes were compared
    # before), the gate_up seam cut at the wrong row or its halves swapped (img_mlp.gate_layer and img_mlp.proj are
    # each compared with the base tensor of their own name), a stored matrix order off in a middle or the last block,
    # or modulation.1 (16384 x 4096, a non-block group) decoded to the wrong layout.
    ckpt = open_checkpoint(_env(DF11_ENV))
    index = _base_index(_env(BASE_ENV))
    for name in (
        "transformer_blocks.0",
        "transformer_blocks.15",
        "transformer_blocks.31",
        MODULATION,
    ):
        group = ckpt.groups[name]
        loaded = load_group_mx(group)
        if name != MODULATION:
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


def _config(size: int) -> Any:
    """mflux's ``Config`` for Qwen-Image 2.1 at ``size`` pixels square (the scheduler's sigma shift reads it)."""
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    return Config(
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=40,
        height=size,
        width=size,
        guidance=1.0,
        scheduler="linear",
    )


def _inputs(*, size: int, text_tokens: int, context_dim: int, seed: int = 1) -> dict[str, Any]:
    """One call's keyword inputs: (size/16)² image tokens of 64 channels, ``text_tokens`` of ``context_dim``, t=0."""
    tokens = (size // 16) ** 2
    keys = mx.random.split(mx.random.key(seed), 2)
    return {
        "t": 0,
        "config": _config(size),
        "hidden_states": mx.random.normal((1, tokens, 64), key=keys[0]).astype(mx.bfloat16),
        "encoder_hidden_states": mx.random.normal(
            (1, text_tokens, context_dim), key=keys[1]
        ).astype(mx.bfloat16),
        "encoder_hidden_states_mask": mx.ones((1, text_tokens), dtype=mx.int32),
    }


def _run(transformer: Any, inputs: Mapping[str, Any]) -> mx.array:
    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    out = transformer(**inputs)
    mx.eval(out)
    assert is_eager(transformer)  # the uncompiled forward ran, as on the model's path
    return out


def df11_one_block(
    ckpt: DF11Checkpoint, inputs: Mapping[str, Any], **build_kwargs: Any
) -> mx.array:
    """Build one block from the checkpoint, decode and install ``modulation.1``, attach the DF11 decode, run once."""
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map
    from mlx_dfloat.mflux.qwen21.transformer import build_transformer

    names = qwen21_name_map()
    build = build_transformer(ckpt, n_layers=1, **build_kwargs)
    blocks = list(build.shapes)
    assert blocks == ["transformer_blocks.0"]
    groups = load_resident_set(ckpt, [*blocks, MODULATION])
    matrix_names = {n: ckpt.groups[n].matrix_names for n in groups}
    nonblock = decode_nonblock(
        {MODULATION: groups[MODULATION]},
        {MODULATION: matrix_names[MODULATION]},
        build.nonblock,
        names,
    )
    assert sorted(nonblock) == ["modulation.layers.1.weight"]
    install_nonblock(build.transformer, nonblock)
    provider = DF11Provider(
        {n: groups[n] for n in blocks}, {n: matrix_names[n] for n in blocks}, names
    )
    build.transformer.attach(provider, build.shapes, verify_in_call=True)
    return _run(build.transformer, inputs)


def bf16_one_block(
    ckpt: DF11Checkpoint,
    index: Mapping[str, Any],
    inputs: Mapping[str, Any],
    **build_kwargs: Any,
) -> mx.array:
    """The same one-block build with the base's BF16 extras, ``modulation.1`` a plain weight, blocks streamed."""
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map
    from mlx_dfloat.mflux.qwen21.transformer import base_extras, build_transformer

    names = qwen21_name_map()
    build = build_transformer(
        ckpt, n_layers=1, extras=base_extras(index, ckpt), nonblock_from_extras=True, **build_kwargs
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
    # Bug caught: a real extra name the coverage or the renames do not handle (the build refuses it on the real
    # files; a rename applied the same way to both builds is not caught here, the two sides share it), modulation.1
    # decoded to a wrong layout or left on its placeholder (a zero-size matmul before the first block), the forward
    # pass compiled on one side, or any block-seam difference between the DF11 and the streamed BF16 paths. 1024²:
    # 4096 image tokens, 32 text tokens of mflux's 4096-wide context (qwen21_transformer.py:17-28).
    ckpt = open_checkpoint(_env(DF11_ENV))
    index = _base_index(_env(BASE_ENV))
    inputs = _inputs(size=1024, text_tokens=32, context_dim=4096)
    out_a = df11_one_block(ckpt, inputs)
    out_b = bf16_one_block(ckpt, index, inputs)
    assert out_a.shape == out_b.shape == (1, 4096, 64)
    assert out_a.dtype == out_b.dtype
    assert np.array_equal(_bits(out_a), _bits(out_b))
