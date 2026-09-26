"""Run one decode scenario for tests/test_decode_bounds.py under Metal shader validation.

Usage: ``PYTHONPATH=<repo> python tests/_shader_validation_repro.py {control|mutant|kernel}``. The caller sets
``MTL_SHADER_VALIDATION=1`` and ``MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`` and reads stderr: the validator
reports an out-of-bounds access there and does not abort, so every mode exits 0 and prints ``ok``.

- ``control``: a deliberately out-of-bounds kernel, proving the validator is live on this machine.
- ``mutant``: our kernel with the second ``gaps`` read unguarded, on groups whose last launched thread reads
  exactly one byte past ``gaps`` (full form, ``gaps`` under 16 KiB so MLX allocates it at its exact size).
- ``kernel``: the real kernel on the same groups plus the short-form group, both paths, with ``check``.
"""

import sys

import mlx.core as mx
import numpy as np
from tests._decode_fixtures import encoder_group, h1, short_form_group

from mlx_dfloat import _metal_decode
from mlx_dfloat.decode import check

PATHS = (True, False)


def _control():
    kernel = mx.fast.metal_kernel(
        name="oob",
        input_names=["a"],
        output_names=["o"],
        source="const uint i = thread_position_in_grid.x; o[i] = a[i + 4096];",
    )
    a = mx.arange(1024, dtype=mx.uint32)
    (o,) = kernel(
        inputs=[a],
        grid=(1024, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1024,)],
        output_dtypes=[mx.uint32],
    )
    mx.eval(o)


def _full_form_groups():
    # A one-block and a two-block group, one launched thread-group per block: the last thread (g = 512 * n_blocks - 1) has
    # its 5-bit field at bit offset 3 of the last gaps byte, so only the guard stops a read of gaps[320 * n].
    bits = np.random.default_rng(7).integers(0x3000, 0x4000, 4096 * 3, dtype=np.uint16)
    groups = [h1(), (encoder_group(bits), bits)]
    for arrays, _ in groups:
        n_launch = arrays.output_positions.size - 1
        assert n_launch == arrays.n_blocks, "short form: the last block launches no thread-group"
        assert arrays.gaps.size == 320 * arrays.n_blocks < 16384, "padded or page-rounded gaps"
    return groups


def _decode_all(groups, *, unguarded):
    for arrays, expected in groups:
        for force_direct in PATHS:
            res = _metal_decode.decode(
                arrays.to_mx(), force_direct=force_direct, _unguarded_gap_read=unguarded
            )
            mx.eval(res.bits, res.status)
            check(res)
            assert np.array_equal(np.array(res.bits), np.asarray(expected, np.uint16))


def main(mode):
    if mode == "control":
        _control()
    elif mode == "mutant":
        _decode_all(_full_form_groups(), unguarded=True)
    elif mode == "kernel":
        _decode_all([*_full_form_groups(), short_form_group()], unguarded=False)
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1])
