import numpy as np
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


def test_warmup_group_binds_like_a_real_group_and_stages_every_block():
    # MLX binds a read-only input of fewer than 8 entries in the `constant` address space, which compiles a
    # different pipeline from the `device`-bound one real groups get. Bug caught: a regression to a tiny warm-up
    # group (the old 3-element H1 one) so readiness never compiles the production signature; or a group whose
    # blocks exceed CAP, so the staged instantiation's warm-up falls back to direct and never runs staging.
    group = md._warmup_group()
    for name in ("encoded_exponent", "sign_mantissa", "luts", "gaps", "positions"):
        assert getattr(group, name).size >= 8, name
    assert group.n_launch >= 2
    assert group.max_elements_per_block <= md.CAP


def test_warmup_stream_decodes_to_its_hand_derived_bits_through_the_reference():
    # Hand derivation (H1 tables: 0xxxxxxx -> exponent 127, length 1; 11xxxxxx -> exponent 128, length 3):
    # every thread's 8 bytes FF FF FF FF FF FF FF FE are 21 codes `111` (exponent 128) then one `0` (127),
    # so each thread decodes 22 elements; sign_mantissa[i] = i & 0xFF. bf16 = ((sm & 0x80) << 8) | (exp << 7)
    # | (sm & 0x7F). Bug caught: a warm-up expectation that disagrees with the oracle, so readiness would refuse
    # a correct kernel or pass a wrong one.
    from mlx_dfloat import reference

    arrays = md._warmup_arrays()
    bits = reference.decode_group(arrays, name="warmup")
    assert bits.size == 7 * 512 * 22
    assert int(bits[0]) == 0x4000  # sm 0x00, exponent 128
    assert int(bits[21]) == 0x3F95  # sm 0x15, exponent 127 (the thread's last code)
    assert int(bits[22]) == 0x4016  # sm 0x16, exponent 128 (next thread's first code)
    assert int(bits[128]) == 0xC000  # sm 0x80: sign set, exponent 128
    assert int(bits[241]) == 0xBFF1  # sm 0xF1, 241 % 22 == 21: sign set, exponent 127
    assert np.array_equal(bits, md._warmup_expected())
