"""The real Krea 2 Raw DF11 checkpoint: its layout and groups, Metal decode against the CPU reference, a one-block build.

Env-gated: every test skips unless ``MLX_DFLOAT_KREA_DF11`` names the ``mingyi456/Krea-2-Raw-DF11-ComfyUI`` snapshot
(one config-less file, ``krea2_raw_bf16-DF11.safetensors``); the one-block comparison also needs
``MLX_DFLOAT_KREA_RAW_FILE``, a local ``raw.safetensors`` of ``krea/Krea-2-Raw`` (26 GB, kept only for the image
identity check). Expected counts are literals from the checkpoint's header (35 groups: ``blocks.0..27``, four
text-fusion blocks, ``tmlp``, ``tproj``, ``txtmlp``; 169 BF16 extras; read 2026-10-08) and from mflux 0.20.0
(``Krea2Transformer`` builds 28 blocks, transformer.py:21); the expected bits come from the CPU reference decoder or the
base's tensors, never from the code under test. Krea 2 Turbo's file shares the layout and is not tested here.
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
from mlx_dfloat.mflux.krea2.names import KIND, NONBLOCK_GROUPS, check_krea2_groups

DF11_ENV, RAW_ENV = "MLX_DFLOAT_KREA_DF11", "MLX_DFLOAT_KREA_RAW_FILE"
# The split positions, by hand from the matrix shapes in the published header: a block stores wq 6144^2, wk and
# wv 1536 x 6144, gate and wo 6144^2, mlp.gate and mlp.up 16384 x 6144, mlp.down 6144 x 16384 (434_110_464 in all); a
# text-fusion block five 2560^2 and three 6912 x 2560 (85_852_160); tmlp 256 x 6144 then 6144^2; tproj one 36864 x
# 6144; txtmlp 2560 x 6144 then 6144^2.
SPLITS = {
    "blocks": (37748736, 47185920, 56623104, 94371840, 132120576, 232783872, 333447168),
    "txtfusion": (6553600, 13107200, 19660800, 26214400, 32768000, 50462720, 68157440),
    "tmlp": (1572864,),
    "tproj": (),
    "txtmlp": (15728640,),
}


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return Path(value)


def _bits(array: mx.array) -> np.ndarray:
    """The array's raw bits through a dtype-sized unsigned view (equality of bits, never of floats)."""
    view = {2: mx.uint16, 4: mx.uint32}[array.dtype.size]
    return np.array(array.view(view))


@pytest.mark.slow
def test_the_real_checkpoint_opens_through_its_layout() -> None:
    # Bug caught: the published file not matched by its pinned layout (every user's first call refused), a group or
    # extra the adapter does not expect (a group count other than the header's 35, a non-block group missing, an
    # extra count other than 169), or a split table other than the shapes give.
    ckpt = open_checkpoint(_env(DF11_ENV))
    assert ckpt.config_source.startswith(
        "header and spot checks match layout krea-2-raw-comfyui "
        "(mingyi456/Krea-2-Raw-DF11-ComfyUI@8320616b25ac9340a830a7fb21f1b0237e160e66)"
    )
    assert check_krea2_groups(ckpt) == {KIND: 28}
    assert len(ckpt.groups) == 35
    assert sorted(g for g in ckpt.groups if not g.startswith("blocks.")) == sorted(NONBLOCK_GROUPS)
    assert len(ckpt.extras) == 169
    assert sorted({info.dtype for _path, info in ckpt.extras.values()}) == ["BF16"]
    for name in (
        "blocks.0",
        "blocks.27",
        "txtfusion.layerwise_blocks.0",
        "tmlp",
        "tproj",
        "txtmlp",
    ):
        key = name.split(".")[0]
        assert load_group_mx(ckpt.groups[name]).split_positions == SPLITS[key], name


@pytest.mark.slow
@pytest.mark.metal
def test_four_real_groups_decode_on_metal_equal_to_the_cpu_reference() -> None:
    # Bug caught: the Metal decoder differing from the reference on this file's codecs (the first and last block, a
    # text-fusion block, the largest non-block group tproj at 226_492_416 elements), or a group cut into matrices at
    # other points than its split table. The reference decoder is the oracle; equality is by bits.
    ckpt = open_checkpoint(_env(DF11_ENV))
    for name in ("blocks.0", "blocks.27", "txtfusion.refiner_blocks.1", "tproj"):
        group = ckpt.groups[name]
        loaded = load_group_mx(group)
        metal = decode_group(loaded, backend="metal")
        check(metal, name=name)
        reference = decode_group(loaded, backend="reference")
        check(reference, name=name)
        assert np.array_equal(_bits(metal.bits), _bits(reference.bits)), name
        parts = split_matrices(metal.bits, loaded.split_positions)
        assert len(parts) == len(group.matrix_names), name
        del loaded, metal, reference, parts
        gc.collect()
        mx.clear_cache()


def _inputs(*, seed: int = 1) -> dict[str, Any]:
    """One call's inputs at 512²: float32 latents (1, 16, 64, 64) as create_noise makes them, 16 text tokens of the
    30720-wide context (12 taps x 2560), bf16 as the text encoder returns them, one sigma."""
    keys = mx.random.split(mx.random.key(seed), 2)
    return {
        "latents": mx.random.normal((1, 16, 64, 64), key=keys[0]),
        "timestep": mx.array([0.5]),
        "embeds": mx.random.normal((1, 16, 30720), key=keys[1]).astype(mx.bfloat16),
    }


def _run(transformer: Any, inputs: Mapping[str, Any]) -> mx.array:
    out = transformer(inputs["latents"], inputs["timestep"], inputs["embeds"])
    mx.eval(out)
    return out


def df11_one_block(ckpt: DF11Checkpoint, inputs: Mapping[str, Any]) -> mx.array:
    """Build one block the way the model does (non-block groups decoded per call), attach the DF11 decode, run."""
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.integrate.providers import DF11Provider
    from mlx_dfloat.mflux.krea2.names import krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import build_transformer, seam_transformer_class

    names = krea2_name_map()
    build = build_transformer(
        ckpt,
        model="krea-2-raw",
        n_layers=1,
        transformer_class=seam_transformer_class(per_call=True),
    )
    blocks = list(build.shapes)
    assert blocks == ["blocks.0"]
    groups = load_resident_set(ckpt, [*blocks, *NONBLOCK_GROUPS])
    matrix_names = {n: ckpt.groups[n].matrix_names for n in groups}
    build.transformer.bind_nonblock({g: groups[g] for g in NONBLOCK_GROUPS}, build.nonblock, names)
    provider = DF11Provider(
        {n: groups[n] for n in blocks}, {n: matrix_names[n] for n in blocks}, names
    )
    build.transformer.attach(provider, build.shapes, verify_in_call=True)
    return _run(build.transformer, inputs)


def bf16_one_block(ckpt: DF11Checkpoint, raw_file: Path, inputs: Mapping[str, Any]) -> mx.array:
    """The same one-block build reading nothing of the DF11 file but its shapes: every extra and every non-block
    matrix replaced by the base's (FP32 cast to BF16 as mflux casts at load), block 0 streamed from the base."""
    from mlx_dfloat._safetensors import read_header
    from mlx_dfloat.integrate.providers import StreamingBF16Provider
    from mlx_dfloat.mflux.krea2.names import krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import build_transformer, install_base_weights

    names = krea2_name_map()
    build = build_transformer(ckpt, model="krea-2-raw", n_layers=1)
    replaced = install_base_weights(build.transformer, ckpt, raw_file, names, build.nonblock)
    # The built block's five extras, the 49 non-block extras and the 37 non-block matrices: 169 - 27 x 5 + 37.
    assert len(replaced) == 169 - 27 * 5 + 37
    index = {name: (raw_file, info) for name, info in read_header(raw_file).items()}
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in build.shapes}, names
    )
    build.transformer.attach(provider, build.shapes, verify_in_call=True)
    return _run(build.transformer, inputs)


@pytest.mark.slow
@pytest.mark.metal
@pytest.mark.mflux
def test_a_one_block_build_matches_the_bf16_stream_bit_for_bit() -> None:
    # Bug caught: a non-block rename or a block matrix misplaced on one side (tmlp.0 -> tmlp.linear_in, a text-fusion
    # matrix on another's parameter, a non-block group left on its placeholder: a zero-size matmul before the first
    # block), the per-call weights released before the forward starts, or any seam difference between the DF11 and
    # the streamed BF16 paths (the decoder itself is the test above's). The five FP32 base matrices are
    # BF16-representable, so the cast cannot show a rounding difference here. The output is float32 (the residual stream follows the float32 latents). Needs the 26 GB
    # raw.safetensors of krea/Krea-2-Raw @ 6b0ece7f (MLX_DFLOAT_KREA_RAW_FILE).
    ckpt = open_checkpoint(_env(DF11_ENV))
    raw_file = _env(RAW_ENV)
    inputs = _inputs()
    out_a = df11_one_block(ckpt, inputs)
    out_b = bf16_one_block(ckpt, raw_file, inputs)
    assert out_a.shape == out_b.shape == (1, 16, 64, 64)
    assert out_a.dtype == out_b.dtype == mx.float32
    assert np.array_equal(_bits(out_a), _bits(out_b))
