import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from tests._decode_fixtures import encoder_group, fibonacci_group, h3, h4
from tests._df11_fixtures import random_bf16

from mlx_dfloat.format import GroupArrays

pytestmark = pytest.mark.metal
REPO = Path(__file__).parents[1]

_RUNNER = textwrap.dedent("""
    import json, sys, numpy as np, mlx.core as mx
    from mlx_dfloat.format import GroupArrays
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.decode import check
    from mlx_dfloat.errors import DFloatFormatError
    d = np.load(sys.argv[1])
    arrays = GroupArrays(encoded_exponent=d["e"], sign_mantissa=d["s"], luts=d["l"], gaps=d["g"],
                         output_positions=d["p"], split_positions=d["x"])
    try:
        g = arrays.to_mx()
    except DFloatFormatError as exc:
        print(json.dumps({"where": "load", "msg": str(exc)})); sys.exit(0)
    res = _metal_decode.decode(g, force_direct=bool(int(sys.argv[2])))
    mx.eval(res.bits, res.status)
    status = np.array(res.status).tolist()
    equal = bool(np.array_equal(np.array(res.bits), d["expected"])) if "expected" in d else None
    try:
        check(res); print(json.dumps({"where": "clean", "status": status, "equal": equal}))
    except DFloatFormatError as exc:
        print(json.dumps({"where": "check", "msg": str(exc), "status": status, "equal": equal}))
""")


def _run(tmp_path, arrays, force_direct, expected=None):
    path = tmp_path / "g.npz"
    extra = {"expected": np.asarray(expected, np.uint16)} if expected is not None else {}
    np.savez(
        path,
        e=arrays.encoded_exponent,
        s=arrays.sign_mantissa,
        l=arrays.luts,
        g=arrays.gaps,
        p=arrays.output_positions.astype(np.uint32),
        x=arrays.split_positions,
        **extra,
    )
    proc = subprocess.run(  # a hang is a TimeoutExpired error
        [sys.executable, "-c", _RUNNER, str(path), str(int(force_direct))],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _mutate(arrays, **fields):
    base = {
        f: getattr(arrays, f)
        for f in (
            "encoded_exponent",
            "sign_mantissa",
            "luts",
            "gaps",
            "output_positions",
            "split_positions",
        )
    }
    return GroupArrays(**{**base, **fields})


@pytest.fixture
def good():
    bits = random_bf16(np.random.default_rng(6), (20_000,))
    arrays = encoder_group(bits)
    assert arrays.output_positions.size >= 4, "need >= 3 blocks"
    return arrays, bits


@pytest.mark.parametrize("force_direct", [True, False])
def test_zero_length_symbol_mid_stream_terminates_and_flags(tmp_path, force_direct):
    # Bug caught: a phase-1 loop that never advances on a zero code length (the GPU-hang case).
    arrays, _expected = h3()  # 23 x exponent 128 across two threads
    luts = arrays.luts.copy()
    luts[-1, 128] = 0  # every code now has length 0
    out = _run(tmp_path, _mutate(arrays, luts=luts), force_direct)
    assert out["where"] == "check"
    assert "invalid code" in out["msg"]


@pytest.mark.parametrize("force_direct", [True, False])
def test_pointer_cycle_terminates_and_flags(tmp_path, force_direct):
    # Bug caught: an unbounded LUT walk on a row pointing to itself.
    arrays, _expected = h4()
    luts = arrays.luts.copy()
    luts[1, :] = 255  # row 1 points to row 1 for every byte
    out = _run(tmp_path, _mutate(arrays, luts=luts), force_direct)
    assert out["where"] == "check"


@pytest.mark.parametrize("force_direct", [True, False])
def test_invalid_code_in_a_non_last_lane_sets_the_invalid_bit(tmp_path, force_direct):
    # Bug caught: a simd vote taken inside a lane-31 branch (only lane 31 seen).
    arrays, _bits = fibonacci_group()
    assert arrays.n_threads >= 64
    from mlx_dfloat.reference import _lookup, _peek32

    stream = np.zeros(arrays.n_threads * 8 + 8, np.uint8)
    stream[: arrays.n_bytes] = arrays.encoded_exponent
    sym, _, _ = _lookup(
        np.ascontiguousarray(arrays.luts), _peek32(stream, np.array([0]))
    )  # thread 0's first symbol
    luts = arrays.luts.copy()
    luts[-1, int(sym[0])] = 0
    out = _run(tmp_path, _mutate(arrays, luts=luts), force_direct)
    assert out["where"] == "check"
    assert "invalid code" in out["msg"]


@pytest.mark.parametrize("force_direct", [True, False])
def test_gap_corruption_that_keeps_the_count_sets_broken_chain(tmp_path, good, force_direct):
    # Bug caught: a corrupt gap that resynchronises with the same count decoding silently wrong.
    from mlx_dfloat.reference import decode_group

    arrays, bits = good
    for byte in range(1, 40):  # find a flip the reference calls "corrupt gaps"
        gaps = arrays.gaps.copy()
        gaps[byte] ^= 0x08
        try:
            decode_group(_mutate(arrays, gaps=gaps), check_stream_end=False)
        except Exception as exc:
            if "corrupt gaps" in str(exc):
                out = _run(tmp_path, _mutate(arrays, gaps=gaps), force_direct, expected=bits)
                assert out["where"] == "check"
                assert "chain" in out["msg"]
                return
    pytest.fail("no gap flip produced the reference's corrupt-gaps error in 40 tries")


@pytest.mark.parametrize("force_direct", [True, False])
def test_dropped_positions_entry_is_flagged(tmp_path, good, force_direct):
    # Bug caught: a block that starts codes but has no positions entry passing as the short form.
    arrays, bits = good
    positions = arrays.output_positions.copy()[:-1]
    positions[-1] = arrays.n_elements
    out = _run(tmp_path, _mutate(arrays, output_positions=positions), force_direct, expected=bits)
    assert out["where"] in ("load", "check")
    assert out.get("equal") is not True
