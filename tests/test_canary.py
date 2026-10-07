"""The load-time canary: real and long-code DF11 groups, decoded on both write paths before any real decode."""

import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest
from tests._decode_fixtures import long_code_canary_group
from tests._flux_fakes import FLUX_TABLE as _NAME_MAP

from mlx_dfloat import _canary, _metal_decode, cli, reference
from mlx_dfloat.decode import available_backends, decode_group, selftest
from mlx_dfloat.errors import DFloatBackendError, DFloatFormatError
from mlx_dfloat.integrate.providers import DF11Provider

SLICE = "canary-qwen3-4b-slice"
LONG = "canary-long-codes"
SLICE_ELEMENTS = 46096  # the committed slice's provenance record: n_elements
SLICE_SHA256 = "48f3a1fed75736441184fe10547d505ebea7de94621eee706ff3363c8cdf667a"


def _by_name():
    return {c.name: c for c in _canary.canary_groups()}


# --- the data -------------------------------------------------------------------------------------------------


def test_the_packaged_slice_is_the_committed_upstream_slice():
    # Bug caught: a truncated or swapped slice file shipping in the wheel, so the canary checks the wrong bits.
    path = _canary.data_file("qwen3_4b_layer0_4blocks.npz")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SLICE_SHA256
    provenance = json.loads(_canary.data_file("qwen3_4b_layer0_4blocks.json").read_text())
    assert provenance["npz_sha256"] == SLICE_SHA256
    assert provenance["df11_repo"] == "DFloat11/Qwen3-4B-DF11"


def test_the_slice_canary_exercises_what_the_warmup_does_not():
    # Bug caught: the canary replaced by easy data (one short code, zero gaps, whole blocks), which a decoder with
    # a broken LUT walk, gap read or partial-block write would still pass.
    c = _by_name()[SLICE]
    lengths = c.arrays.luts[-1]
    assert int(lengths.max()) == 27  # the provenance record's max_code_length
    assert c.arrays.luts.shape[0] == 5  # lut_rows
    launched = c.arrays.output_positions.size - 1
    assert (
        launched == 4
    )  # the record's n_blocks: four launched blocks (the fifth holds lookahead bytes)
    gaps = reference.thread_gaps(c.arrays.gaps, launched * 512)
    assert (
        int(gaps.max()) == 11
    )  # the record's max_gap: real per-thread offsets, unpacked from 5 bits
    assert c.arrays.n_bytes == 16392
    assert c.arrays.n_bytes % 4096 != 0  # cut mid-block
    assert c.expected.size == SLICE_ELEMENTS


def test_the_long_code_canary_has_32_bit_codes_a_four_level_chain_and_the_all_device_signature():
    # Bug caught: the long-code group losing the 29-32-bit codes and the deep LUT chain it exists to cover, or
    # shrinking under eight output_positions entries (MLX then binds `positions` as constant and the all-device
    # pipeline real groups use is never checked on real-shaped data), or having no block over CAP (the staged
    # path's per-block direct fallback unchecked), or no partial last block.
    c = _by_name()[LONG]
    assert int(c.arrays.luts[-1].max()) == 32
    assert c.arrays.luts.shape[0] >= 4
    assert c.arrays.output_positions.size >= 8
    intervals = np.diff(
        c.arrays.output_positions.astype(np.int64)
    )  # the stored form ends at n_elements
    assert intervals.max() > 16192
    assert c.arrays.n_bytes % 4096 != 0  # the last block is partial


def test_the_canary_set_covers_both_input_binding_signatures():
    # Bug caught: every canary group landing on one side of MLX's 8-entry constant/device binding switch, so
    # one of the two pipelines real groups use is checked only by the synthetic warm-up.
    sizes = [c.arrays.output_positions.size for c in _canary.canary_groups()]
    assert any(size >= 8 for size in sizes)
    assert any(size < 8 for size in sizes)


def test_the_long_code_canary_is_what_the_vendored_upstream_encoder_produces():
    # Bug caught: hand-edited or stale canary arrays; the expected bits must be the encoder's input, not a decode.
    arrays, bits = long_code_canary_group()
    c = _by_name()[LONG]
    for field in (
        "encoded_exponent",
        "sign_mantissa",
        "luts",
        "gaps",
        "output_positions",
        "split_positions",
    ):
        assert np.array_equal(getattr(c.arrays, field), getattr(arrays, field)), field
    assert np.array_equal(c.expected, np.asarray(bits, np.uint16))


def test_the_expected_bits_are_also_what_the_reference_decoder_gives():
    # Bug caught: a canary expectation the CPU oracle disagrees with (the canary would then refuse correct GPUs).
    from mlx_dfloat import reference

    for c in _canary.canary_groups():
        got = reference.decode_group(c.arrays, name=c.name, check_stream_end=False)
        assert np.array_equal(got, c.expected), c.name


# --- the gate -------------------------------------------------------------------------------------------------


def _faulty_dispatch(
    monkeypatch, *, group_name, index=None, status_word=None, path_blocks=None, only_direct=None
):
    """Real dispatch, except one canary group comes back with one flipped bit, one error bit, or a flipped write-path
    bit (bit 8 of the status word) on the last block, on every block, or on the block at an integer index.

    `only_direct` restricts the fault to dispatches on one write path (True direct, False staged).
    """
    real = _metal_decode._dispatch
    dispatched = []

    def faulty(group, **kwargs):
        dispatched.append(group.name)
        out, status = real(group, **kwargs)
        if group.name != group_name:
            return out, status
        if only_direct is not None and kwargs.get("force_direct") != only_direct:
            return out, status
        if index is not None:
            bits = np.array(out)
            bits[index] ^= 1
            out = mx.array(bits)
        if status_word is not None or path_blocks is not None:
            words = np.array(status)
            if status_word is not None:
                words[-1] |= status_word
            if path_blocks == "last":
                words[-1] ^= 8
            elif path_blocks == "all":
                words ^= 8
            elif path_blocks is not None:
                words[path_blocks] ^= 8
            status = mx.array(words)
        return out, status

    monkeypatch.setattr(_metal_decode, "_dispatch", faulty)
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    return dispatched


@pytest.mark.metal
@pytest.mark.parametrize("force_direct", [True, False], ids=["direct", "staged"])
@pytest.mark.parametrize(
    ("group_name", "index"),
    [(LONG, 123), (LONG, 30000), (SLICE, 0), (SLICE, SLICE_ELEMENTS - 1)],
    ids=["long-codes", "long-codes-over-cap-block", "slice-first", "slice-last-partial-block"],
)
def test_a_one_bit_gpu_fault_on_a_canary_group_refuses_the_backend(
    monkeypatch, force_direct, group_name, index
):
    # Bug caught: a canary decoded but compared on one path only, on one group only, or not to its last element.
    _faulty_dispatch(monkeypatch, group_name=group_name, index=index)
    with pytest.raises(DFloatBackendError, match=group_name):
        _metal_decode.ensure_canary(force_direct=force_direct)


@pytest.mark.metal
@pytest.mark.parametrize("error_bit", [1, 2, 4])
def test_an_error_bit_on_a_canary_block_refuses_the_backend(monkeypatch, error_bit):
    # Bug caught: only the bits compared, so a block that reports an invalid code still passes the canary.
    _faulty_dispatch(monkeypatch, group_name=SLICE, status_word=error_bit)
    with pytest.raises(DFloatBackendError, match=SLICE):
        _metal_decode.ensure_canary(force_direct=False)


@pytest.mark.metal
def test_a_failed_canary_takes_metal_out_of_the_available_backends(monkeypatch):
    # Bug caught: readiness reporting metal after a canary failure, so callers that check it pick a broken GPU.
    _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    assert available_backends() == ("reference",)


@pytest.mark.metal
def test_a_failed_canary_refuses_the_first_real_decode_before_dispatching_it(monkeypatch):
    # Bug caught: the canary checked after the caller's group was already decoded (or not at all on the decode
    # path), or a silent fallback to the CPU reference.
    dispatched = _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    real = _by_name()[SLICE].arrays.to_mx(name="the-callers-group")
    with pytest.raises(DFloatBackendError, match=LONG):
        _metal_decode.decode(real)
    assert "the-callers-group" not in dispatched


@pytest.mark.metal
def test_the_real_canary_passes_on_both_paths_here(monkeypatch):
    # Bug caught: a canary expectation or path check the real kernel cannot meet (metal never available).
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    _metal_decode.ensure_canary(force_direct=True)
    _metal_decode.ensure_canary(force_direct=False)
    assert _metal_decode._CANARY == {True: True, False: True}


@pytest.mark.metal
def test_the_refusal_names_the_group_the_device_the_versions_and_the_next_step(monkeypatch):
    # Bug caught: a refusal a user cannot act on: no group, no device, no versions, no pointer to the selftest
    # report and the issue tracker, or no word that the CPU reference still works.
    from mlx_dfloat._version import __version__

    _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"device_name": "Apple M9 Max", "architecture": "applegpu_g99s"},
    )
    with pytest.raises(DFloatBackendError) as caught:
        _metal_decode.ensure_canary(force_direct=False)
    text = str(caught.value)
    assert LONG in text
    assert "wrong bits at element 7" in text
    assert "Apple M9 Max" in text
    assert "applegpu_g99s" in text
    assert f"mlx {mx.__version__}" in text
    assert f"mlx-dfloat {__version__}" in text
    assert "mlx-dfloat selftest --json" in text
    assert "https://github.com/IonDen/mlx-dfloat/issues" in text
    assert "reference" in text


@pytest.mark.metal
def test_the_refusal_survives_a_device_with_no_description(monkeypatch):
    # Bug caught: the diagnostic text itself raising (KeyError on a missing device key) and hiding the refusal.
    _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    monkeypatch.setattr(mx, "device_info", dict)
    with pytest.raises(DFloatBackendError, match="mlx-dfloat selftest"):
        _metal_decode.ensure_canary(force_direct=False)


# --- the FLUX.1 provider fails fast ---------------------------------------------------------------------------


@pytest.mark.metal
def test_building_the_metal_provider_refuses_a_failed_canary_before_any_real_decode(monkeypatch):
    # Bug caught: the canary left to the first decode, so DFloatFlux1 and `generate` load the text encoders (and
    # minutes of setup) before telling the user the GPU decoder is broken.
    dispatched = _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    with pytest.raises(DFloatBackendError, match=LONG):
        DF11Provider({}, {}, _NAME_MAP)
    assert set(dispatched) <= {SLICE, LONG} | {
        f"warmup-{k}" for k in ("7-block", "1-block", "tiny")
    }


@pytest.mark.metal
def test_building_the_metal_provider_succeeds_with_the_real_decoder(monkeypatch):
    # Bug caught: the fail-fast guard rejecting a healthy GPU (wrong path asked, or the cache not honoured).
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    provider = DF11Provider({}, {}, _NAME_MAP)
    assert provider.launches == 0
    assert _metal_decode._CANARY[False] is True


def test_an_injected_decode_skips_the_gpu_canary(monkeypatch):
    # Bug caught: a test double (or a reference decode) paying for, or being refused by, a GPU check it never uses.
    def boom(**_kwargs):
        raise AssertionError("the canary ran for an injected decode")

    monkeypatch.setattr(_metal_decode, "ensure_canary", boom)
    DF11Provider({}, {}, _NAME_MAP, decode=lambda g: decode_group(g, backend="reference"))


# --- the write-path check, the caching rules ------------------------------------------------------------------


@pytest.mark.metal
@pytest.mark.parametrize("force_direct", [True, False], ids=["direct", "staged"])
@pytest.mark.parametrize(
    ("group_name", "path_blocks"),
    [(LONG, "last"), (LONG, "all"), (LONG, 8), (SLICE, "last")],
    ids=["long-last-block", "long-every-block", "long-over-cap-block", "slice-last-block"],
)
def test_a_flipped_write_path_bit_refuses_the_backend(
    monkeypatch, force_direct, group_name, path_blocks
):
    # Bug caught: the write-path comparison dropped or loosened in canary_failure (bits and error bits pass, a
    # block that took the other path is accepted), including the over-CAP block reported as staged and a
    # direct-forced block reported as staged.
    _faulty_dispatch(monkeypatch, group_name=group_name, path_blocks=path_blocks)
    with pytest.raises(DFloatBackendError, match=r"wrong write path"):
        _metal_decode.ensure_canary(force_direct=force_direct)


@pytest.mark.metal
def test_a_failed_canary_is_never_cached(monkeypatch):
    # Bug caught: a failure (or the attempt) recorded in the cache, so after a fault clears the process
    # keeps refusing, or after one pass a later fault is never looked for.
    real = _metal_decode._dispatch
    _faulty_dispatch(monkeypatch, group_name=LONG, index=7)
    with pytest.raises(DFloatBackendError, match=LONG):
        _metal_decode.ensure_canary(force_direct=False)
    assert _metal_decode._CANARY == {}
    monkeypatch.setattr(_metal_decode, "_dispatch", real)
    _metal_decode.ensure_canary(force_direct=False)
    assert _metal_decode._CANARY == {False: True}


@pytest.mark.metal
def test_the_canary_cache_is_per_write_path(monkeypatch):
    # Bug caught: one cache flag for both paths, so a staged pass lets a faulty direct path through unchecked.
    _faulty_dispatch(monkeypatch, group_name=SLICE, index=5, only_direct=True)
    _metal_decode.ensure_canary(force_direct=False)
    with pytest.raises(DFloatBackendError, match=SLICE):
        _metal_decode.ensure_canary(force_direct=True)


@pytest.mark.metal
def test_every_canary_output_is_prefilled_with_a_value_its_group_never_decodes_to(monkeypatch):
    # Bug caught: the canary output prefilled with 0 (or left recycled), so an element the kernel never writes
    # can equal its expected word and pass.
    real = _metal_decode._dispatch
    fills = []

    def recording(group, **kwargs):
        fills.append((group.name, kwargs.get("init_value")))
        return real(group, **kwargs)

    monkeypatch.setattr(_metal_decode, "_dispatch", recording)
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    _metal_decode.ensure_canary(force_direct=True)
    _metal_decode.ensure_canary(force_direct=False)
    seen = [(name, fill) for name, fill in fills if name in (SLICE, LONG)]
    assert {name for name, _ in seen} == {SLICE, LONG}
    for name, fill in seen:
        assert isinstance(fill, int)
        assert fill not in set(_by_name()[name].expected.tolist()), name


# --- missing or corrupt canary data ---------------------------------------------------------------------------


@pytest.fixture
def fresh_canary_cache(monkeypatch):
    """Forget loaded canaries and the pass cache now, and again after the test's patches are undone."""
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    _canary.canary_groups.cache_clear()
    yield
    monkeypatch.undo()
    _canary.canary_groups.cache_clear()


def _bad_long_codes(tmp_path, monkeypatch, defect):
    """Point the packaged long-code file at a broken copy (or nowhere); the slice stays real."""
    real = _canary.data_file
    target = tmp_path / "long_codes.npz"
    if defect != "missing":
        with real("long_codes.npz").open("rb") as handle, np.load(handle) as data:
            fields = {k: data[k] for k in data.files}
        if defect == "wrong-dtype":
            fields["luts"] = fields["luts"].astype(np.float32)
        elif defect == "no-poison-value":
            fields["expected_bf16"] = np.arange(65536, dtype=np.uint16)
            fields["sign_mantissa"] = np.zeros(65536, np.uint8)
        elif defect == "size-mismatch":
            fields["expected_bf16"] = fields["expected_bf16"][:-1]
        elif defect == "missing-field":
            del fields["gaps"]
        np.savez(target, **fields)
    monkeypatch.setattr(
        _canary,
        "data_file",
        lambda name: target if name == "long_codes.npz" else real(name),
    )


DEFECTS = ["missing", "wrong-dtype", "no-poison-value", "size-mismatch", "missing-field"]


@pytest.mark.parametrize("defect", DEFECTS)
def test_unreadable_canary_data_names_the_file_and_is_a_backend_error(
    tmp_path, monkeypatch, fresh_canary_cache, defect
):
    # Bug caught: a missing or malformed packaged file surfacing as a bare FileNotFoundError, KeyError or
    # numpy error (or loading and being trusted), instead of a refusal that names the canary.
    _bad_long_codes(tmp_path, monkeypatch, defect)
    with pytest.raises(DFloatBackendError, match=rf"canary data {LONG} is unreadable or malformed"):
        _canary.canary_groups()


@pytest.mark.metal
@pytest.mark.parametrize("defect", DEFECTS)
def test_unreadable_canary_data_takes_metal_out_and_refuses_decode(
    tmp_path, monkeypatch, fresh_canary_cache, defect
):
    # Bug caught: readiness still True with the canary data gone (the check silently skipped), or decode()
    # running its caller's group anyway.
    _bad_long_codes(tmp_path, monkeypatch, defect)
    assert _metal_decode.metal_ready() is False
    assert available_backends() == ("reference",)
    group = _the_real_slice().arrays.to_mx(name="the-callers-group")
    with pytest.raises(DFloatBackendError, match="unreadable or malformed"):
        _metal_decode.decode(group)


def _the_real_slice():
    return _canary._load("qwen3_4b_layer0_4blocks.npz", SLICE)


@pytest.mark.metal
@pytest.mark.parametrize("defect", ["missing", "wrong-dtype"])
def test_unreadable_canary_data_gives_a_selftest_report_with_no_checks_and_exit_2(
    tmp_path, monkeypatch, fresh_canary_cache, capsys, defect
):
    # Bug caught: selftest raising out of the data load (a traceback instead of a report), or reporting ok
    # with no checks, or the CLI exiting 0 or 1 for "the check could not run".
    _bad_long_codes(tmp_path, monkeypatch, defect)
    report = selftest()
    assert report.ok is False
    assert report.checks == ()
    assert "unreadable or malformed" in (report.reason or "")
    assert cli.main(["selftest"]) == 2


# --- selftest -------------------------------------------------------------------------------------------------


@pytest.mark.metal
def test_selftest_passes_here_and_reports_every_group_on_every_path():
    # Bug caught: a check skipped (say the direct path or the CPU reference) while the report still says ok.
    report = selftest()
    assert report.ok is True
    assert report.reason is None
    seen = {(c.group, c.path) for c in report.checks}
    assert seen == {(g, p) for g in (SLICE, LONG) for p in ("staged", "direct", "reference")}
    assert all(c.ok for c in report.checks)
    assert report.mlx_version == mx.__version__


@pytest.mark.metal
def test_selftest_names_the_failing_check(monkeypatch):
    # Bug caught: a selftest that raises on the first fault (no report) or reports ok with a failing check.
    _faulty_dispatch(monkeypatch, group_name=LONG, index=99)
    report = selftest()
    assert report.ok is False
    failed = {(c.group, c.path) for c in report.checks if not c.ok}
    assert failed == {(LONG, "staged"), (LONG, "direct")}


def test_selftest_without_metal_reports_why(monkeypatch):
    # Bug caught: a selftest that crashes, or reports success, on a machine with no Metal device.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    report = selftest()
    assert report.ok is False
    assert "Metal" in (report.reason or "")


@pytest.mark.metal
def test_the_selftest_command_exits_0_and_prints_pass(capsys):
    # Bug caught: the subcommand missing, or exiting 0 without saying what it checked.
    assert cli.main(["selftest"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert SLICE in out
    assert LONG in out


@pytest.mark.metal
def test_the_selftest_command_exits_1_on_wrong_bits(monkeypatch, capsys):
    # Bug caught: a failing canary reported with exit 0, which CI and scripts would read as a pass.
    _faulty_dispatch(monkeypatch, group_name=SLICE, index=5)
    assert cli.main(["selftest"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_selftest_command_exits_2_without_metal(monkeypatch, capsys):
    # Bug caught: "no Metal here" reported as wrong bits (exit 1), which reads as a broken GPU kernel.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert cli.main(["selftest"]) == 2


@pytest.mark.metal
def test_the_selftest_command_writes_json(capsys):
    # Bug caught: --json output that is not JSON, or drops the per-check results a script would read.
    assert cli.main(["selftest", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["ok"] is True
    assert {(c["group"], c["path"]) for c in doc["checks"]} == {
        (g, p) for g in (SLICE, LONG) for p in ("staged", "direct", "reference")
    }
    assert {"device", "mlx_version", "package_version", "seconds"} <= doc.keys()


# --- selftest robustness --------------------------------------------------------------------------------------


@pytest.mark.metal
@pytest.mark.parametrize("how", ["flips-a-bit", "raises"])
def test_a_reference_decode_fault_is_a_failed_reference_check_not_an_exception(monkeypatch, how):
    # Bug caught: the CPU reference raising (or disagreeing) turning the whole selftest into a traceback, or
    # failing every check of the group instead of the one that is wrong.
    real = reference.decode_group

    def faulty(arrays, *, name, check_stream_end):
        if name != LONG:
            return real(arrays, name=name, check_stream_end=check_stream_end)
        if how == "raises":
            raise DFloatFormatError(f"{name}: block 3: invalid code")
        bits = real(arrays, name=name, check_stream_end=check_stream_end).copy()
        bits[11] ^= 1
        return bits

    monkeypatch.setattr("mlx_dfloat.decode.reference.decode_group", faulty)
    report = selftest()
    assert report.ok is False
    assert {(c.group, c.path) for c in report.checks if not c.ok} == {(LONG, "reference")}
    assert len(report.checks) == 6


@pytest.mark.metal
def test_a_pipeline_that_cannot_be_warmed_is_a_report_with_no_checks(monkeypatch, capsys):
    # Bug caught: a pipeline refusal escaping selftest as an exception, or the CLI calling it a failed check (1).
    def refuse(**_kwargs):
        raise DFloatBackendError("the Metal decode kernel cannot run here (warmup-7-block): boom")

    monkeypatch.setattr(_metal_decode, "ensure_pipeline", refuse)
    report = selftest()
    assert report.ok is False
    assert report.checks == ()
    assert "boom" in (report.reason or "")
    assert cli.main(["selftest"]) == 2


def test_the_selftest_command_turns_any_exception_into_exit_2(monkeypatch, capsys):
    # Bug caught: an unexpected error in selftest leaving as a traceback with exit 1, which CI and scripts read
    # as "a check failed", not "the tool broke".
    def broken():
        raise RuntimeError("driver exploded")

    monkeypatch.setattr("mlx_dfloat.decode.selftest", broken)
    assert cli.main(["selftest"]) == 2
    assert "driver exploded" in capsys.readouterr().err


def test_the_selftest_command_prints_the_reason_when_no_check_ran(monkeypatch, capsys):
    # Bug caught: "selftest 0 of 0 checks failed" on stdout for a machine that could not run anything, which
    # reads like a pass.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    assert cli.main(["selftest"]) == 2
    captured = capsys.readouterr()
    assert "Metal" in captured.err
    assert "Metal" in captured.out
    assert "0 of 0" not in captured.out


def test_the_device_summary_names_the_chip(monkeypatch):
    # Bug caught: a report that says only "applegpu_g13s", which a user filing an issue cannot tell from
    # another M1 variant, or a KeyError on a device that omits a key.
    from mlx_dfloat.decode import _device_summary

    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "device_name": "Apple M9 Max",
            "architecture": "applegpu_g99s",
            "memory_size": 32 * 2**30,
        },
    )
    assert _device_summary() == "Apple M9 Max, applegpu_g99s, 32 GiB"
    monkeypatch.setattr(mx, "device_info", dict)
    assert _device_summary() == "unknown device"
