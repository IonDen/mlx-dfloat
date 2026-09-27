import mlx.core as mx
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from tests._decode_fixtures import (
    encoder_group,
    fibonacci_group,
    hand_fixtures,
    short_form_group,
    slice_group,
)
from tests._df11_fixtures import random_bf16

from mlx_dfloat import _metal_decode
from mlx_dfloat._metal_decode import CAP
from mlx_dfloat.decode import STATUS_PATH_DIRECT, available_backends, check, decode_group
from mlx_dfloat.errors import DFloatBackendError

pytestmark = pytest.mark.metal
PATHS = [pytest.param(True, id="direct"), pytest.param(False, id="staged")]


def poison_for(expected):
    """A uint16 value absent from the expected output, so a hole cannot masquerade as a correct element."""
    return int(np.setdiff1d(np.arange(65536, dtype=np.uint16), np.asarray(expected, np.uint16))[0])


def metal_bits(arrays, expected, *, force_direct, name="<group>"):
    """Decode with a poisoned output and threadgroup buffer; check status; return (bits, status)."""
    res = _metal_decode.decode(
        arrays.to_mx(name=name),
        force_direct=force_direct,
        _init_value=poison_for(expected),
        _poison_buf=True,
    )
    check(res, name=name)
    return np.array(res.bits), np.array(res.status)


@pytest.mark.parametrize(
    ("name", "arrays", "expected"),
    hand_fixtures(),
    ids=lambda v: v if isinstance(v, str) else "",
)
@pytest.mark.parametrize("force_direct", PATHS)
def test_hand_fixture_decodes_bit_exact(name, arrays, expected, force_direct):
    # Bug caught: any deviation in bit windows, gap read, LUT walk, scan or write from the hand-derived values.
    bits, status = metal_bits(arrays, expected, force_direct=force_direct, name=name)
    assert bits.tolist() == expected
    assert all(
        (s & STATUS_PATH_DIRECT) == (STATUS_PATH_DIRECT if force_direct else 0)
        for s in status.tolist()
    )


def test_the_reference_passes_the_same_fixtures_through_decode_group():
    # Bug caught: a parity fixture that only the Metal path can satisfy (an unfair fixture).
    for name, arrays, expected in hand_fixtures():
        res = decode_group(arrays.to_mx(name=name), backend="reference")
        assert np.array(res.bits).tolist() == expected


def test_metal_backend_is_reported_available_on_this_machine():
    assert available_backends() == ("reference", "metal")


def test_warmup_failure_becomes_a_backend_error(monkeypatch):
    # Bug caught: a compile error or a sub-512 pipeline ceiling surfacing as a raw RuntimeError at the caller's eval.
    monkeypatch.setattr(_metal_decode, "_PIPELINES", {})
    monkeypatch.setattr(
        _metal_decode, "_dispatch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(DFloatBackendError, match="boom"):
        _metal_decode.ensure_pipeline(force_direct=True)


def test_warmup_prefills_its_output_so_an_unwritten_element_cannot_pass(monkeypatch):
    # Bug caught: the warm-up dispatching with init_value=None, so a recycled buffer that already holds the
    # expected bits (the allocator reuses the warm-up buffer between warm-ups) hides a kernel that skips a write.
    monkeypatch.setattr(_metal_decode, "_PIPELINES", {})
    real = _metal_decode._dispatch
    seen = []

    def recording(group, **kwargs):
        seen.append(kwargs["init_value"])
        return real(group, **kwargs)

    monkeypatch.setattr(_metal_decode, "_dispatch", recording)
    _metal_decode.ensure_pipeline(force_direct=True)
    _metal_decode.ensure_pipeline(force_direct=False)
    assert seen == [0, 0]


def _fake_warmup(monkeypatch, status_words):
    # The warm-up's own expected bits with a chosen status word per block: only the status check can refuse it.
    monkeypatch.setattr(_metal_decode, "_PIPELINES", {})
    good = mx.array(_metal_decode._warmup_expected())
    monkeypatch.setattr(
        _metal_decode,
        "_dispatch",
        lambda *a, **k: (good, mx.array(status_words, dtype=mx.uint32)),
    )


N_WARMUP_BLOCKS = 7


@pytest.mark.parametrize(
    ("force_direct", "status_words"),
    [
        pytest.param(False, [8] * N_WARMUP_BLOCKS, id="staged-all-direct"),
        pytest.param(True, [0] * N_WARMUP_BLOCKS, id="direct-all-staged"),
        pytest.param(False, [0] * (N_WARMUP_BLOCKS - 1) + [8], id="staged-last-block-direct"),
        pytest.param(True, [8] * (N_WARMUP_BLOCKS - 1) + [0], id="direct-last-block-staged"),
    ],
)
def test_warmup_refuses_the_wrong_write_path(monkeypatch, force_direct, status_words):
    # Bug caught: a staged instantiation silently running direct (or the reverse) passing its warm-up because
    # only the error bits, or only the first block's word, are checked. Every warm-up block fits under CAP, so
    # every block must run staged unless forced direct.
    _fake_warmup(monkeypatch, status_words)
    with pytest.raises(DFloatBackendError, match="path"):
        _metal_decode.ensure_pipeline(force_direct=force_direct)


@pytest.mark.parametrize("error_bit", [1, 2, 4])
def test_warmup_refuses_an_error_bit_on_any_block(monkeypatch, error_bit):
    # Bug caught: the warm-up reading only status[0], so a block after the first that reports an invalid code, a
    # count mismatch or a broken chain still passes readiness.
    _fake_warmup(monkeypatch, [0] * (N_WARMUP_BLOCKS - 1) + [error_bit])
    with pytest.raises(DFloatBackendError, match="wrong bits"):
        _metal_decode.ensure_pipeline(force_direct=False)


def test_the_real_warmup_passes_on_both_paths(monkeypatch):
    # Bug caught: a warm-up expectation or path check the real kernel cannot meet (readiness always False).
    monkeypatch.setattr(_metal_decode, "_PIPELINES", {})
    _metal_decode.ensure_pipeline(force_direct=True)
    _metal_decode.ensure_pipeline(force_direct=False)
    assert _metal_decode._warmup_group().n_launch == N_WARMUP_BLOCKS


@pytest.mark.parametrize("force_direct", PATHS)
@pytest.mark.parametrize(
    ("n", "low", "width"), [(70_000, 100, 30), (5, 127, 1), (4096 * 3, 90, 60)]
)
def test_encoder_roundtrip_matches_input_bits(n, low, width, force_direct):
    # Bug caught: any divergence over many blocks or on a single-byte stream.
    bits = random_bf16(np.random.default_rng(n), (n,), exponent_low=low, exponent_high=low + width)
    out, _ = metal_bits(encoder_group(bits), bits, force_direct=force_direct)
    assert np.array_equal(out, bits)


@pytest.mark.parametrize("force_direct", PATHS)
def test_fibonacci_codebook_reaches_32_bit_codes_and_four_level_chains(force_direct):
    # Bug caught: a wrong pointer-row walk on real multi-level tables (the round-trips above use 2-row tables).
    arrays, bits = fibonacci_group()
    assert arrays.luts.shape[0] >= 4
    out, _ = metal_bits(arrays, bits, force_direct=force_direct)
    assert np.array_equal(out, bits)


@pytest.mark.parametrize("force_direct", PATHS)
def test_short_form_group_decodes_with_one_fewer_launch(force_direct):
    # Bug caught: launching or indexing a threadgroup for the code-free tail block.
    arrays, bits = short_form_group()
    assert arrays.to_mx().n_launch == arrays.n_blocks - 1 == 1
    out, status = metal_bits(arrays, bits, force_direct=force_direct)
    assert status.shape == (1,)
    assert np.array_equal(out, bits)


@pytest.mark.parametrize("force_direct", PATHS)
def test_real_qwen_slice_is_bit_exact(force_direct):
    # Bug caught: anything the synthetic codebooks miss on real 27-bit codes, 5 LUT rows and real gaps.
    arrays, expected = slice_group()
    assert (
        arrays.n_bytes,
        arrays.n_blocks,
        arrays.output_positions.size,
        arrays.luts.shape[0],
    ) == (16392, 5, 5, 5)
    out, status = metal_bits(arrays, expected, force_direct=force_direct, name="qwen-slice")
    assert status.shape == (4,)
    assert np.array_equal(out, expected)


@settings(
    max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)
@given(
    seed=st.integers(0, 2**31),
    n=st.integers(1, 20_000),
    low=st.integers(60, 200),
    ratio=st.floats(1.05, 3.0),
)
@example(seed=1, n=70_000, low=100, ratio=1.5)  # many blocks
@example(seed=2, n=CAP + 5000, low=127, ratio=1.0)  # one block over CAP
def test_metal_equals_reference_on_random_geometric_groups(seed, n, low, ratio):
    # Bug caught: a staged/direct divergence, a per-block count error or a deep-chain slip on random shapes.
    rng = np.random.default_rng(seed)
    k = rng.geometric(1 / ratio, size=n)  # skewed exponent frequencies -> long codes
    exponent = np.clip(low + k, 0, 239).astype(
        np.uint16
    )  # >= 240 are LUT pointers, never exponents
    bits = (
        (rng.integers(0, 2, n, dtype=np.uint16) << 15)
        | (exponent << 7)
        | rng.integers(0, 128, n, dtype=np.uint16)
    ).astype(np.uint16)
    arrays = encoder_group(bits)
    for force_direct in (True, False):
        out, _ = metal_bits(arrays, bits, force_direct=force_direct)
        assert np.array_equal(out, bits)


def _compiled_kernel_name(group, *, force_direct):
    import re

    from scripts.regpressure import capture_msl

    return re.search(r"void (custom_kernel_\w+)\(", capture_msl(group, force_direct=force_direct))[
        1
    ]


@pytest.mark.parametrize("force_direct", PATHS)
def test_warmup_compiles_the_same_pipeline_a_real_group_uses(force_direct):
    # MLX names the kernel after its input binding (a `constant`-bound input adds a `c` to the type suffix) and
    # compiles one pipeline per name. Bug caught: a warm-up group small enough to bind `constant`, so readiness
    # proves a pipeline no real decode reuses and the production one first compiles at the caller's eval.
    bits = random_bf16(np.random.default_rng(1), (70_000,), exponent_low=100, exponent_high=130)
    real = encoder_group(bits).to_mx(name="real")
    warm = _metal_decode._warmup_group()
    assert _compiled_kernel_name(warm, force_direct=force_direct) == _compiled_kernel_name(
        real, force_direct=force_direct
    )
