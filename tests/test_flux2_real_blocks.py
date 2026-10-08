"""Real FLUX.2 Klein 4B checkpoints: group layout, Metal decode against the base file, and a reduced build run.

Env-gated: every test skips unless the snapshot directories are named by ``MLX_DFLOAT_KLEIN_B4_DF11`` (the
FLUX.2-klein-base-4B DF11 checkpoint), ``MLX_DFLOAT_KLEIN_D4_DF11`` (the distilled FLUX.2-klein-4B one) and
``MLX_DFLOAT_KLEIN_B4_BASE`` (the BF16 ``black-forest-labs/FLUX.2-klein-base-4B`` snapshot). Expected layouts are
literals from mflux 0.20.0 (``Flux2Transformer`` defaults ``num_layers=5``, ``num_single_layers=20``,
flux2_transformer/transformer.py:20-21; the Klein 4B ``transformer_overrides``, common/config/model_config.py:428-433
and :508-513) and the checkpoints' own ``pattern_dict``; the expected bits come from the base file, never from the
code under test.
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
from mlx_dfloat.mflux.flux2.names import check_klein_groups

# The five one-matrix non-block groups of the mingyi456 Klein DF11 checkpoints (their pattern_dict, read 2026-10-08).
NONBLOCK = (
    "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.linear",
    "single_stream_modulation.linear",
    "context_embedder",
    "norm_out.linear",
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
    from mlx_dfloat.mflux.flux2.transformer import base_transformer_files_index

    return base_transformer_files_index(base / "transformer")


@pytest.mark.slow
@pytest.mark.parametrize("env", ["MLX_DFLOAT_KLEIN_B4_DF11", "MLX_DFLOAT_KLEIN_D4_DF11"])
def test_both_4b_checkpoints_have_klein_groups_and_bf16_extras(env: str) -> None:
    # Bug caught: a checkpoint layout other than the pattern_dict promises (counts from mflux's defaults
    # num_layers=5, num_single_layers=20, transformer.py:20-21), a non-block group missing (its matrix would load as
    # nothing), or an extra count other than the arithmetic's 4 + 5 x 4 + 20 x 2 = 64 (a guidance embedder or a
    # stray tensor the coverage check would refuse mid-load), or an extra stored in another dtype.
    ckpt = open_checkpoint(_env(env))
    assert check_klein_groups(ckpt) == {"transformer_blocks": 5, "single_transformer_blocks": 20}
    assert set(NONBLOCK) <= set(ckpt.groups)
    assert len(ckpt.groups) == 30
    assert len(ckpt.extras) == 64
    assert sorted({info.dtype for _path, info in ckpt.extras.values()}) == ["BF16"]


@pytest.mark.slow
@pytest.mark.metal
def test_four_real_groups_decode_on_metal_bit_exact_against_the_base_file() -> None:
    # Bug caught: a real-file decode or split error on a Klein kind (12-matrix double blocks with the to_out.0
    # sub-path, fused qkv+mlp single blocks, one-matrix empty-list groups), or a single-file base the index reader
    # enumerates wrongly.
    ckpt = open_checkpoint(_env("MLX_DFLOAT_KLEIN_B4_DF11"))
    index = _base_index(_env("MLX_DFLOAT_KLEIN_B4_BASE"))
    for name in (
        "transformer_blocks.0",
        "single_transformer_blocks.0",
        "context_embedder",
        "double_stream_modulation_img.linear",
    ):
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
    """One forward at 64 image tokens (an 8 x 8 packed latent of 128 channels) and 16 text tokens of 7680."""
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

    keys = mx.random.split(mx.random.key(1), 2)
    hidden = mx.random.normal((1, 64, 128), key=keys[0]).astype(mx.bfloat16)
    text = mx.random.normal((1, 16, 7680), key=keys[1]).astype(mx.bfloat16)
    out = transformer(
        hidden_states=hidden,
        encoder_hidden_states=text,
        timestep=mx.array(0.5),
        img_ids=Flux2LatentCreator.prepare_grid_ids(mx.zeros((1, 128, 8, 8)), t_coord=0),
        txt_ids=Flux2PromptEncoder.prepare_text_ids(text),
    )
    mx.eval(out)
    return out


@pytest.mark.slow
@pytest.mark.metal
@pytest.mark.mflux
def test_a_one_by_one_build_from_real_extras_matches_the_bf16_stream_bit_for_bit() -> None:
    # Bug caught: a real extra name the renames do not cover (the build's coverage check refuses it on the real
    # files; a rename applied the same way to both builds is not caught here, the two sides share it), a non-block
    # weight decoded to a wrong layout or left on its placeholder, or any block-seam difference between the DF11 and
    # the streamed BF16 paths.
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.mflux.flux2.names import klein_name_map
    from mlx_dfloat.mflux.flux2.transformer import base_extras, build_transformer

    ckpt: DF11Checkpoint = open_checkpoint(_env("MLX_DFLOAT_KLEIN_B4_DF11"))
    index = _base_index(_env("MLX_DFLOAT_KLEIN_B4_BASE"))
    overrides = ModelConfig.flux2_klein_base_4b().transformer_overrides
    names = klein_name_map()

    df11 = build_transformer(
        ckpt, transformer_overrides=overrides, name_map=names, n_double=1, n_single=1
    )
    blocks = list(df11.shapes)
    assert blocks == ["transformer_blocks.0", "single_transformer_blocks.0"]
    groups = load_resident_set(ckpt, [*blocks, *NONBLOCK])
    matrix_names = {n: ckpt.groups[n].matrix_names for n in groups}
    nonblock = decode_nonblock(
        {g: groups[g] for g in NONBLOCK},
        {g: matrix_names[g] for g in NONBLOCK},
        df11.nonblock,
        names,
    )
    assert sorted(nonblock) == sorted(f"{g}.weight" for g in NONBLOCK)
    install_nonblock(df11.transformer, nonblock)
    provider = DF11Provider(
        {n: groups[n] for n in blocks}, {n: matrix_names[n] for n in blocks}, names
    )
    df11.transformer.attach(provider, df11.shapes, verify_in_call=True)
    out_a = _run(df11.transformer)

    bf16 = build_transformer(
        ckpt,
        transformer_overrides=overrides,
        name_map=names,
        n_double=1,
        n_single=1,
        extras=base_extras(index, ckpt),
        nonblock_from_extras=True,
    )
    assert bf16.nonblock == {}
    bf16.transformer.attach(
        StreamingBF16Provider(index, {n: ckpt.groups[n].matrix_names for n in bf16.shapes}, names),
        bf16.shapes,
        verify_in_call=True,
    )
    out_b = _run(bf16.transformer)

    assert out_a.shape == out_b.shape == (1, 64, 128)
    assert out_a.dtype == out_b.dtype
    assert np.array_equal(_bits(out_a), _bits(out_b))
