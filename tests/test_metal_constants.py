import pytest

from mlx_dfloat import _metal_decode as md


@pytest.mark.parametrize(
    ("declared", "reserved"),
    [(1, 16), (192, 192), (196, 208), (198, 208), (32580, 32592)],
)
def test_metal_static_bytes_rounds_up_to_16(declared, reserved):
    # Measured with PyObjC on an M1 Max (mlx 0.32.2, macOS 27.0): staticThreadgroupMemoryLength
    # is reported at 16-byte granularity. Bug caught: rounding down, to 8, or bumping a multiple.
    assert md._metal_static_bytes(declared) == reserved


def test_threadgroup_bytes_are_what_the_compiler_reserves():
    # The values scripts/regpressure.py read back from the compiled pipelines on 2026-09-27.
    # Bug caught: the constants reporting the unrounded declarations (198 / 32,580) again.
    assert md.THREADGROUP_BYTES_DIRECT == 208
    assert md.THREADGROUP_BYTES_STAGED == 32592
    assert md.THREADGROUP_BYTES_STAGED <= 32 * 1024  # still within the threadgroup memory limit
