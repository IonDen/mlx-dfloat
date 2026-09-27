import numpy as np
import pytest
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16

from mlx_dfloat import _metal_decode
from mlx_dfloat._metal_decode import CAP, THREADGROUP_BYTES_DIRECT, THREADGROUP_BYTES_STAGED
from mlx_dfloat.decode import STATUS_PATH_DIRECT, check

pytestmark = pytest.mark.metal


def one_symbol(n):
    """One-symbol codebook: 1-bit codes, 32,768 codes per 4096-byte block (verified through the encoder)."""
    return encoder_group(np.full(n, 0x3F80, np.uint16)), np.full(n, 0x3F80, np.uint16)


@pytest.mark.parametrize(("n", "expect_direct"), [(CAP, 0), (CAP + 1, 1)])
def test_kernel_reports_the_path_it_took_at_the_limit(n, expect_direct):
    # Bug caught: an off-by-one in the kernel's interval <= CAP choice (the kernel reports the path, not the host).
    arrays, bits = one_symbol(n)
    g = arrays.to_mx()
    assert (g.n_launch, g.max_elements_per_block) == (1, n)
    res = _metal_decode.decode(g, force_direct=False, _init_value=0xDEAD, _poison_buf=True)
    check(res)
    status = [s & STATUS_PATH_DIRECT for s in np.array(res.status).tolist()]
    assert status == [STATUS_PATH_DIRECT * expect_direct]
    assert res.direct_blocks == expect_direct
    assert res.threadgroup_bytes == THREADGROUP_BYTES_STAGED
    assert np.array_equal(np.array(res.bits), bits)


@pytest.mark.parametrize("force_direct", [True, False])
def test_every_element_is_written_on_both_paths(force_direct):
    # Bug caught: a hole in the output (a recycled buffer would hide it; verified) or in the threadgroup buffer.
    bits = random_bf16(np.random.default_rng(11), (30_000,))
    res = _metal_decode.decode(
        encoder_group(bits).to_mx(),
        force_direct=force_direct,
        _init_value=0xDEAD,
        _poison_buf=True,
    )
    check(res)
    out = np.array(res.bits)
    assert not np.any(out == 0xDEAD)
    assert np.array_equal(out, bits)
    assert res.threadgroup_bytes == (
        THREADGROUP_BYTES_DIRECT if force_direct else THREADGROUP_BYTES_STAGED
    )


def test_mixed_group_uses_both_paths_in_one_launch():
    # Bug caught: a divergence between the staged and direct paths inside one dispatch.
    n = 32768 * 2 + 1000  # positions [0, 32768, 65536, 66536]: two direct blocks, one staged
    arrays, bits = one_symbol(n)
    g = arrays.to_mx()
    assert g.n_launch == 3
    assert g.intervals.tolist() == [32768, 32768, 1000]
    res = _metal_decode.decode(g, force_direct=False, _init_value=0xDEAD, _poison_buf=True)
    check(res)
    status = [s & STATUS_PATH_DIRECT for s in np.array(res.status).tolist()]
    assert status == [8, 8, 0]
    assert res.direct_blocks == 2
    assert np.array_equal(np.array(res.bits), bits)
