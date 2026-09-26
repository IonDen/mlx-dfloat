import numpy as np
import pytest
from tests._decode_fixtures import hand_fixtures

from mlx_dfloat import _metal_decode
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
