import json
import re

import mlx.core as mx
import pytest
from scripts import _bench_common as bc
from scripts._bench_common import (
    Timing,
    bench_exit_code,
    calibration_rate,
    crosscheck_band,
    gbps,
    kill_equivalent_throughput,
    overhead,
    parity_conditions,
    parse_pmset,
    per_dispatch_guard,
    projected_bytes,
    provenance,
    ramp_next_k,
    ramp_should_stop,
    resume_key_diff,
    time_first_then_min,
    write_json_atomic,
)


@pytest.mark.parametrize("bytes_out", [24_000_000, 25_000_000])
def test_a_dispatch_projected_at_or_under_the_limit_passes(bytes_out):
    # Bug caught: `>=` in place of `>`, or a guard that trips on every call.
    per_dispatch_guard(bytes_out, 100e6)  # 0.24 s and exactly 0.25 s against a 0.25 s limit


def test_a_dispatch_projected_over_the_limit_raises_naming_seconds_and_limit():
    # Bug caught: a guard that never trips, letting one dispatch run past the GPU watchdog window.
    with pytest.raises(RuntimeError, match=r"0\.26.*0\.25"):
        per_dispatch_guard(26_000_000, 100e6)


# --- Timing -------------------------------------------------------------------------------------


def test_timing_median_is_the_middle_value_not_the_mean():
    # Bug caught: `mean` in place of `median` (the 10.0 outlier would drag it to 4.0).
    assert Timing(reps=(3.0, 1.0, 2.0, 10.0)).median == pytest.approx(2.5)


def test_timing_spread_is_range_over_median():
    # Bug caught: dividing by the mean or the min, or reporting max - min unnormalised.
    assert Timing(reps=(1.0, 2.0, 4.0)).spread == pytest.approx(1.5)


# --- overhead, gbps -----------------------------------------------------------------------------


def test_overhead_is_the_fraction_df11_adds_over_the_control():
    # Bug caught: returning the plain ratio (1.25) or dividing by t_df11 (0.2).
    assert overhead(1.25, 1.0) == pytest.approx(0.25)


@pytest.mark.parametrize("t_control", [0.0, -1.0])
def test_overhead_refuses_a_non_positive_control(t_control):
    # Bug caught: `< 0` in place of `<= 0` (0 would reach a ZeroDivisionError, not ValueError),
    # or a negative control turning into a meaningless negative overhead.
    with pytest.raises(ValueError, match="control"):
        overhead(1, t_control)


def test_gbps_is_decimal_gigabytes_per_second():
    # Bug caught: GiB (2**30) in place of 1e9, or seconds and bytes swapped.
    assert gbps(2_000_000_000, 0.5) == pytest.approx(4.0)


# --- kill line, cross-check ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "want"),
    [({}, 47.6e9), ({"kill_line": 0.5}, 23.8e9)],
)
def test_kill_equivalent_throughput_decodes_a_step_within_the_kill_line(kwargs, want):
    # 23.8 GB per 2.0 s step at a 25 % kill line: the decode must finish in 0.5 s -> 47.6 GB/s.
    # Bug caught: kill_line ignored, multiplied instead of divided, or t_step left out.
    assert kill_equivalent_throughput(23_800_000_000, 2.0, **kwargs) == pytest.approx(want)


@pytest.mark.parametrize(
    ("decode_seconds", "in_band"),
    [(0.99, False), (1.0, True), (2.0, True), (2.01, False)],
)
def test_crosscheck_band_accepts_one_to_two_times_the_isolated_prediction(decode_seconds, in_band):
    # 1 GB at 1 GB/s predicts 1.0 s; the in-step decode may take 1x to 2x that, inclusive.
    # Bug caught: strict `<` at either edge, or the ratio inverted (prediction / measured).
    ratio, ok = crosscheck_band(decode_seconds, 1_000_000_000, 1e9)
    assert ratio == pytest.approx(decode_seconds)
    assert ok is in_band


# --- parity, exit codes -------------------------------------------------------------------------


def test_parity_conditions_names_only_the_failed_checks_in_argument_order():
    # Bug caught: returning the passing names, or sorting instead of keeping argument order.
    assert parity_conditions(a=True, b=False) == ["b"]
    assert parity_conditions(z=False, a=True, m=0) == ["z", "m"]


def test_parity_conditions_skips_a_none_and_still_fails_a_false():
    # Bug caught: None treated as a failure (the q8 mode, which has no compressed set, could never
    # pass its parity check), or `not ok` kept so a None passes but a False is also skipped.
    assert parity_conditions(compressed_loaded=None, limits_recorded=True) == []
    assert parity_conditions(compressed_loaded=None, finite=False) == ["finite"]
    assert parity_conditions(compressed_loaded=False, finite=True) == ["compressed_loaded"]


@pytest.mark.parametrize(
    ("mismatched", "errors", "want"),
    [(1, 3, 1), (0, 1, 2), (0, 0, 0)],
)
def test_bench_exit_code_mismatch_wins_over_error(mismatched, errors, want):
    # Bug caught: error checked first (a real bit mismatch reported as 2), or 0 on errors.
    assert bench_exit_code(mismatched=mismatched, errors=errors) == want


# --- write_json_atomic --------------------------------------------------------------------------


def test_write_json_atomic_writes_readable_json_and_leaves_no_tmp(tmp_path):
    # Bug caught: the tmp file never renamed into place, or written somewhere else.
    path = tmp_path / "bench.json"
    write_json_atomic(path, {"gbps": 12.5, "groups": {"a": [1, 2]}})
    assert json.loads(path.read_text()) == {"gbps": 12.5, "groups": {"a": [1, 2]}}
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_keeps_the_old_file_when_serialization_fails(tmp_path):
    # Bug caught: streaming json.dump into the tmp file (a partial .tmp left behind) or writing
    # the target in place (the old result truncated by a payload that cannot be serialized).
    path = tmp_path / "bench.json"
    path.write_text('{"old": true}')
    with pytest.raises(TypeError):
        write_json_atomic(path, {"ok": 1, "bad": {1, 2}})
    assert path.read_text() == '{"old": true}'
    assert list(tmp_path.iterdir()) == [path]


# --- power parsing, provenance ------------------------------------------------------------------

_PMSET_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=24576099)\t63%; discharging; 4:12 remaining present: true\n"
)
_PMSET_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=24576099)\t97%; charging; 0:21 remaining present: true\n"
)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (_PMSET_AC, {"ac": True, "battery_pct": 97}),
        (_PMSET_BATTERY, {"ac": False, "battery_pct": 63}),
        ("Now drawing from 'AC Power'\n", {"ac": True, "battery_pct": None}),  # no battery
    ],
)
def test_parse_pmset_reads_the_power_source_and_the_first_percentage(text, want):
    # Bug caught: `ac` true whenever the word "Power" appears, or the id digits read as the %.
    assert parse_pmset(text) == want


class _Done:
    def __init__(self, stdout):
        self.stdout = stdout


def _fake_run(*, head="abc123\n", porcelain="", pmset=_PMSET_AC, fail=()):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        tool = argv[0]
        if tool in fail:
            raise OSError(f"{tool} not found")
        if tool == "pmset":
            return _Done(pmset)
        return _Done(head if "rev-parse" in argv else porcelain)

    return run, calls


PROVENANCE_KEYS = {
    "git",
    "source_hash",
    "mlx",
    "mflux",
    "macos",
    "device_info",
    "memory_caps_gb",
    "cache_limit",
    "cache_memory",
    "power",
    "date",
}


def test_provenance_records_every_listed_key_from_its_source(monkeypatch):
    # Bug caught: a key dropped or renamed, pmset not parsed, or git state not recorded.
    run, calls = _fake_run()
    monkeypatch.setattr(bc.subprocess, "run", run)
    prov = provenance((20, 22))
    assert set(prov) == PROVENANCE_KEYS
    assert prov["git"] == "abc123"
    assert prov["power"] == {"ac": True, "battery_pct": 97}
    assert prov["mlx"] == mx.__version__
    assert prov["device_info"] == dict(mx.device_info())
    assert prov["memory_caps_gb"] == [20, 22]
    assert re.fullmatch(r"[0-9a-f]{64}", prov["source_hash"])
    assert ["pmset", "-g", "batt"] in calls
    json.dumps(prov)  # the record must serialize as-is


def test_provenance_takes_the_caps_instead_of_reinstalling_them(monkeypatch):
    # Bug caught: provenance calling install_memory_caps() again, which puts the host's memory cap
    # back over a smaller tier's limits in the middle of a capped run and records the host caps.
    # The probe limit is not a whole GiB, so no cap install can leave it in place by coincidence.
    import datetime

    class _FrozenDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 1)

    monkeypatch.setattr(bc.datetime, "date", _FrozenDate)
    monkeypatch.setattr(bc.subprocess, "run", _fake_run()[0])
    probe = 3 * 1024**3 + 4096
    before = mx.set_memory_limit(probe)
    try:
        prov = provenance((0, 0))
        after = mx.set_memory_limit(before)
    finally:
        mx.set_memory_limit(before)
    assert after == probe
    assert prov["memory_caps_gb"] == [0, 0]
    assert prov["date"] == "2026-10-01"


def test_provenance_marks_a_dirty_tree(monkeypatch):
    # Bug caught: an edited, uncommitted script's numbers recorded against the clean commit.
    run, _ = _fake_run(porcelain=" M scripts/bench_decode_kernel.py\n")
    monkeypatch.setattr(bc.subprocess, "run", run)
    assert provenance((20, 22))["git"] == "abc123-dirty"


def test_provenance_survives_missing_git_and_pmset(monkeypatch):
    # Bug caught: a machine without git or pmset crashing the bench instead of recording unknowns.
    run, _ = _fake_run(fail=("git", "pmset"))
    monkeypatch.setattr(bc.subprocess, "run", run)
    prov = provenance((20, 22))
    assert prov["git"] == "unknown"
    assert prov["power"] == {"ac": None, "battery_pct": None}


def test_provenance_reads_the_cache_limit_without_changing_it(monkeypatch):
    # Bug caught: the set-and-restore read leaving the probe value installed as the cache limit.
    monkeypatch.setattr(bc.subprocess, "run", _fake_run()[0])
    before = mx.set_cache_limit(1_400_000_000)
    try:
        assert provenance((20, 22))["cache_limit"] == 1_400_000_000
        assert mx.set_cache_limit(before) == 1_400_000_000
    finally:
        mx.set_cache_limit(before)


def test_provenance_records_the_mflux_version_when_installed(monkeypatch):
    # Bug caught: mflux always recorded as None, so a bench cannot tell which mflux it ran.
    monkeypatch.setattr(bc.subprocess, "run", _fake_run()[0])
    real = bc.metadata.version
    monkeypatch.setattr(
        bc.metadata, "version", lambda name: "0.20.1" if name == "mflux" else real(name)
    )
    assert provenance((20, 22))["mflux"] == "0.20.1"


def test_provenance_records_no_mflux_when_absent(monkeypatch):
    # Bug caught: a PackageNotFoundError escaping and killing a bench that never needs mflux.
    monkeypatch.setattr(bc.subprocess, "run", _fake_run()[0])

    def missing(name):
        raise bc.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(bc.metadata, "version", missing)
    assert provenance((20, 22))["mflux"] is None


# --- calibration ramp ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n_launch", "want"),
    [(1, 27, 2), (2, 27, 4), (16, 27, 27), (27, 27, None), (1, 1, None)],
)
def test_ramp_next_k_doubles_and_stops_at_n_launch(k, n_launch, want):
    # Bug caught: a ramp that overshoots n_launch (32 blocks of a 27-block group), steps linearly,
    # or never reports that it is done.
    assert ramp_next_k(k, n_launch) == want


@pytest.mark.parametrize(
    ("seconds", "k", "n_launch", "stop"),
    [
        (0.0099, 4, 27, False),  # just under the minimum step time: keep ramping
        (0.01, 4, 27, True),  # exactly the minimum: long enough to measure throughput
        (0.0011, 27, 27, True),  # the whole group already dispatched
    ],
)
def test_ramp_should_stop_at_the_minimum_time_or_the_whole_group(seconds, k, n_launch, stop):
    # Bug caught: a ramp that never stops (min_seconds ignored or `>` in place of `>=`), or one
    # that keeps ramping after the whole group has been dispatched.
    assert ramp_should_stop(seconds, k, n_launch) is stop


def test_ramp_should_stop_honours_a_custom_minimum():
    # Bug caught: the min_seconds argument ignored in favour of the default.
    assert ramp_should_stop(0.02, 1, 27, min_seconds=0.05) is False


@pytest.mark.parametrize(("k", "want"), [(1, 2 * 7393), (2, 2 * 14780), (3, 2 * 22000)])
def test_projected_bytes_is_two_bytes_per_element_of_the_first_k_blocks(k, want):
    # Bug caught: projecting one block instead of k (a step that skips the guard in effect), an
    # off-by-one (positions[k - 1]), or elements counted as bytes.
    assert projected_bytes([0, 7393, 14780, 22000], k) == want


def test_projected_bytes_counts_from_the_first_position():
    # Bug caught: counting from 0 when the prefix's first position is not 0.
    assert projected_bytes([100, 300, 700], 2) == 2 * 600


# --- resume key ---------------------------------------------------------------------------------

_KEY = {
    "df11": "/models/qwen3-4b-df11",
    "variants": ["direct", "staged"],
    "reps": 5,
    "source_hash": "a" * 64,
    "mlx": "0.32.2",
}


def test_resume_key_diff_is_empty_for_the_same_run():
    # Bug caught: a resume refused even when nothing changed.
    assert resume_key_diff(dict(_KEY), _KEY) == []


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("df11", "/models/flux-krea-df11"),
        ("variants", ["staged"]),
        ("reps", 3),
        ("source_hash", "b" * 64),
        ("mlx", "0.33.0"),
    ],
)
def test_resume_key_diff_names_each_changed_field(field, other):
    # Bug caught: one field left out of the comparison, so results from another checkpoint,
    # variant set, rep count, kernel source or mlx version mix into one file.
    assert resume_key_diff({**_KEY, field: other}, _KEY) == [field]


@pytest.mark.parametrize("stored", [None, "not a key", {}])
def test_resume_key_diff_refuses_a_file_without_a_key(stored):
    # Bug caught: an unkeyed (older or foreign) bench file treated as resumable.
    assert resume_key_diff(stored, _KEY) != []


def test_resume_key_diff_names_a_field_the_current_run_does_not_have():
    # Bug caught: a stored key with an extra field (written by a different bench) accepted.
    assert resume_key_diff({**_KEY, "decoder": "metal"}, _KEY) == ["decoder"]


def test_calibration_rate_is_the_last_steps_rate():
    # The last step is the longest dispatch, the one that measures throughput; an earlier step is
    # launch latency, or paid a pipeline compile. Bug caught: min or mean in place of the last.
    assert calibration_rate([0.038e9, 0.117e9, 0.008e9, 1.3e9]) == 1.3e9


def test_calibration_rate_refuses_an_empty_ramp():
    # Bug caught: an IndexError (or a silent 0) when no step ran, instead of a named refusal.
    with pytest.raises(ValueError, match="ramp"):
        calibration_rate([])


def _fake_clock(durations):
    """A clock whose consecutive (start, end) reads are `durations` apart."""
    ticks, now = [], 0.0
    for d in durations:
        ticks += [now, now + d]
        now += d + 1.0  # a gap between runs that must never land in a duration
    it = iter(ticks)
    return lambda: next(it)


def test_time_first_then_min_times_n_runs_after_the_first_and_keeps_the_fastest():
    # First run: an 11 ms one-off; then three timed runs, the middle one fastest.
    calls = []
    first, best = time_first_then_min(
        lambda: calls.append(1), clock=_fake_clock([0.011, 0.0005, 0.0003, 0.0007])
    )
    # Bug caught: mean (0.5 ms) or last (0.7 ms) in place of min, the first run counted, or N wrong.
    assert len(calls) == 1 + 3
    assert first == pytest.approx(0.011)
    assert best == pytest.approx(0.0003)


def test_time_first_then_min_honours_n():
    # Bug caught: N hard-coded to 3.
    calls = []
    _, best = time_first_then_min(
        lambda: calls.append(1), n=5, clock=_fake_clock([0.01, 0.9, 0.8, 0.7, 0.6, 0.2])
    )
    assert len(calls) == 6
    assert best == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("seconds", "k", "rate", "prev_rate", "stop"),
    [
        (0.02, 4, 0.008e9, 0.117e9, False),  # long but the rate collapsed: a transient, re-measure
        (0.02, 4, 1.2e9, 1.0e9, True),  # long and the rate held: throughput measured
        (0.02, 4, 1.0e9, 1.0e9, True),  # equal to the previous rate counts as held
        (0.0001, 27, 0.001e9, 1.0e9, True),  # the whole group dispatched: stop regardless ...
        (0.02, 27, 0.001e9, 1.0e9, True),  # ... even when that last, long step's rate fell
        (0.02, 1, 0.5e9, None, True),  # the first step has no previous rate to compare with
    ],
)
def test_ramp_should_stop_ignores_a_long_step_whose_rate_fell(seconds, k, rate, prev_rate, stop):
    # Bug caught: one transient slow dispatch ending the ramp with a collapsed rate (the guard then
    # refuses the group), `<=` in place of `<`, or the n_launch stop depending on the rate.
    assert ramp_should_stop(seconds, k, 27, rate=rate, prev_rate=prev_rate) is stop


# --- stale abort artifacts ------------------------------------------------------------------------


def test_move_stale_abort_aside_keeps_the_old_artifact_under_another_name(tmp_path):
    # Bug caught: a resumed orchestration finishing next to an old abort.json (read as this run's
    # outcome), or the stale artifact deleted instead of kept.
    (tmp_path / "abort.json").write_text('{"reason": "stale"}')
    bc.move_stale_abort_aside(tmp_path)
    assert not (tmp_path / "abort.json").exists()
    assert (tmp_path / "abort.previous.json").read_text() == '{"reason": "stale"}'


def test_move_stale_abort_aside_without_an_artifact_changes_nothing(tmp_path):
    bc.move_stale_abort_aside(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []
