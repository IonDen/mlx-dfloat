import mlx.core as mx
import numpy as np
import pytest
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16

from mlx_dfloat import _metal_decode
from mlx_dfloat.decode import DecodeResult, available_backends, check, decode_group, split_matrices
from mlx_dfloat.errors import DFloatBackendError, DFloatFormatError


def test_reference_backend_returns_the_input_bits():
    # Bug caught: a conversion error between MxGroup and GroupArrays inside the backend.
    bits = random_bf16(np.random.default_rng(0), (4000,))
    res = decode_group(encoder_group(bits).to_mx(), backend="reference")
    assert isinstance(res, DecodeResult)
    assert res.backend == "reference"
    assert res.bits.dtype == mx.uint16
    assert np.array_equal(np.array(res.bits), bits)
    assert res.status.shape == (1,)
    assert check(res) is None


def test_unknown_backend_is_a_backend_error():
    g = encoder_group(random_bf16(np.random.default_rng(1), (10,))).to_mx()
    with pytest.raises(DFloatBackendError):
        decode_group(g, backend="nope")  # type: ignore[arg-type]


def test_metal_is_never_a_silent_fallback(monkeypatch):
    # Bug caught: the reference quietly running when Metal is unavailable.
    monkeypatch.setattr(_metal_decode, "_PIPELINES", {})
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert available_backends() == ("reference",)
    g = encoder_group(random_bf16(np.random.default_rng(1), (10,))).to_mx()
    with pytest.raises(DFloatBackendError, match="Metal"):
        decode_group(g, backend="metal")


def test_split_matrices_is_copy_free_and_cuts_at_the_stored_positions():
    # Bug caught: a split that copies (memory) or cuts at the wrong offsets (parity).
    bits = random_bf16(np.random.default_rng(2), (6000,))
    g = encoder_group(bits, 1000, 3500).to_mx()
    res = decode_group(g, backend="reference")
    mx.eval(res.bits)
    before = mx.get_active_memory()
    parts = split_matrices(res.bits, g.split_positions)
    mx.eval(*parts)
    assert mx.get_active_memory() - before < 4096
    assert [p.size for p in parts] == [1000, 2500, 2500]
    assert np.array_equal(np.concatenate([np.array(p) for p in parts]), bits)


@pytest.mark.parametrize(
    ("word", "text"),
    [(1, "invalid code"), (2, "count"), (4, "chain"), (5, "invalid code.*chain")],
)
def test_check_names_the_block_and_the_reason(word, text):
    # Bug caught: check() ignoring the status, masking the wrong bits, or naming only the first of two reasons.
    res = decode_group(
        encoder_group(random_bf16(np.random.default_rng(3), (100,))).to_mx(), backend="reference"
    )
    bad = DecodeResult(
        bits=res.bits,
        status=mx.array([0, word], dtype=mx.uint32),
        backend="reference",
        direct_blocks=0,
        threadgroup_bytes=0,
    )
    with pytest.raises(DFloatFormatError, match=f"block 1.*{text}"):
        check(bad, name="blocks.7")


def test_check_ignores_the_informational_path_bit():
    res = decode_group(
        encoder_group(random_bf16(np.random.default_rng(3), (100,))).to_mx(), backend="reference"
    )
    ok = DecodeResult(
        bits=res.bits,
        status=mx.array([8], dtype=mx.uint32),
        backend="reference",
        direct_blocks=1,
        threadgroup_bytes=0,
    )
    check(ok)
