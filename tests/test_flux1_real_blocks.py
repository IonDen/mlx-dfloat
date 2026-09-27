"""Real schnell blocks: bit-for-bit parity between the package's decode and the base BF16 shards.

The slow, mflux-gated test below decodes the checkpoint's first double and single DF11 groups
through the package's own seam-facing surface (``build_transformer`` / ``DF11Provider``) and
compares every decoded matrix, on the attribute a hand table says it belongs to, against the base
BF16 shards bit for bit -- the one check the offline, reduced-width fakes used elsewhere in the
suite cannot make at FLUX.1's real width (3072). It does not run a block. Both mflux and
``ModelConfig`` are imported inside the test function so the module collects cleanly without the
mflux extra installed.

The ungated test above it proves the selection helper the slow test relies on
(``_block_matrix_shards``) picks exactly the checkpoint's own recorded matrix names, using header
reads only. It too needs the local snapshots and skips without them.
"""

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from tests._flux_fakes import EXPECTED_PATHS

from mlx_dfloat._safetensors import read_array, read_header
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.coverage import load_resident_set
from mlx_dfloat.integrate.providers import DF11Provider
from mlx_dfloat.mflux.flux1.names import (
    DOUBLE_PREFIX,
    MATRICES_PER_KIND,
    SINGLE_PREFIX,
    flux_name_map,
)
from mlx_dfloat.mflux.flux1.transformer import build_transformer


def _at_block_0(name: str) -> str:
    kind, _idx, rest = name.split(".", 2)
    return f"{kind}.0.{rest}"


# The 20 FLUX.1 matrices of the first double and single block and the attribute each lands on,
# from the hand-copied run-record table (re-indexed to block 0), never from the map under test.
_EXPECTED_ATTR_AT_0 = {
    _at_block_0(name): (f"{block.partition('.')[0]}.0", attr)
    for name, (block, attr) in EXPECTED_PATHS
}


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return Path(value)


def _block_matrix_shards(base: Path, block: str) -> list[tuple[str, str]]:
    """``(matrix name, shard file)`` for every DF11 matrix of ``block`` in the base BF16 index.

    Filters to ``.weight`` entries under ``block`` and drops RMSNorm scales (``attn.norm_q.weight``
    and its siblings): they end in ``.weight`` too, but they are not DF11 matrices -- the DF11
    checkpoint only compresses ``nn.Linear`` weights, and ``StaticNameMap.place()`` refuses a norm
    scale the same way ``flux_name_map()`` itself excludes them (``_is_matrix``).
    """
    index = json.loads(
        (base / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    out: list[tuple[str, str]] = []
    for name, shard in index["weight_map"].items():
        if not (name.startswith(block + ".") and name.endswith(".weight")):
            continue
        sub = name.removeprefix(block + ".").removesuffix(".weight")
        if any(part.startswith("norm_") for part in sub.split(".")):
            continue
        out.append((name, shard))
    return out


def _bf16_block(base: Path, block: str) -> dict[str, np.ndarray]:
    """The block's DF11 matrices as raw BF16 bit patterns (uint16), read from the base shards."""
    out: dict[str, np.ndarray] = {}
    for name, shard in _block_matrix_shards(base, block):
        path = base / "transformer" / shard
        info = read_header(path)[name]
        out[name] = np.ascontiguousarray(read_array(path, info))  # uint16 bits
    return out


def test_bf16_block_selects_exactly_the_checkpoints_own_matrix_names() -> None:
    # Runs only where the schnell DF11 and base snapshots are present (the two env vars); skips
    # elsewhere.
    # Bug caught: a filter that lets an RMSNorm scale (`attn.norm_q.weight`, `attn.norm_k.weight`,
    # ...) through, drops a real matrix, or otherwise disagrees with the DF11 group's own recorded
    # `matrix_names`, would make the slow test below compare the wrong set of names -- or raise a
    # `DFloatIntegrationError` out of `names.place()` before it gets that far. Header reads only:
    # the base index JSON and each shard's safetensors header, never `read_array`.
    df11 = _env("MLX_DFLOAT_SCHNELL_DF11")
    base = _env("MLX_DFLOAT_SCHNELL_BASE")
    ckpt = open_checkpoint(df11)
    for block, expected_count in (
        (f"{DOUBLE_PREFIX}.0", MATRICES_PER_KIND[DOUBLE_PREFIX]),
        (f"{SINGLE_PREFIX}.0", MATRICES_PER_KIND[SINGLE_PREFIX]),
    ):
        selected = _block_matrix_shards(base, block)
        names = {name for name, _shard in selected}
        assert len(selected) == len(names) == expected_count, selected
        assert names == set(ckpt.groups[block].matrix_names)
        for name, shard in selected:
            info = read_header(base / "transformer" / shard)[name]
            assert info.dtype == "BF16", name


@pytest.mark.slow
@pytest.mark.mflux
def test_real_first_blocks_decode_bit_identically_to_the_bf16_shard() -> None:
    # Bug caught: a name map that swaps two same-shaped matrices (to_q/to_k, or linear1/linear2 of
    # a transposed pair) or a group cut at the wrong split, so a real matrix lands on the wrong
    # attribute of block 0; the expected attribute comes from the hand table, not from the map.
    from mflux.models.common.config.model_config import ModelConfig

    df11 = _env("MLX_DFLOAT_SCHNELL_DF11")
    base = _env("MLX_DFLOAT_SCHNELL_BASE")
    ckpt = open_checkpoint(df11)
    names = flux_name_map()
    _tf, shapes = build_transformer(ModelConfig.schnell(), ckpt, n_double=1, n_single=1)
    resident = load_resident_set(ckpt, ["transformer_blocks.0", "single_transformer_blocks.0"])
    provider = DF11Provider(resident, {n: ckpt.groups[n].matrix_names for n in resident}, names)
    for block in resident:
        decoded = provider.weights_for(block, shapes[block])
        bf16 = _bf16_block(base, block)
        assert set(bf16) == {n for n, (b, _a) in _EXPECTED_ATTR_AT_0.items() if b == block}
        for matrix_name, bits in bf16.items():
            block_name, attr = _EXPECTED_ATTR_AT_0[matrix_name]
            assert block_name == block
            assert np.array_equal(np.array(decoded[attr].view(mx.uint16)), bits), matrix_name
    provider.verify()
    assert provider.launches == 2
