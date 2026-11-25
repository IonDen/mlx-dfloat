"""The watchdog's raise mode: a cooperative trip at the stop line, a hard exit at the ceiling or past grace.

Every watchdog here samples an injected ``FakeCounters`` and ends the "process" through a recorder,
so no test reads the real allocator or can end pytest. Waits poll an event with a 5 s timeout; the
only fixed waits are the "nothing happened within N s" checks.
"""

import json
import stat
import threading
import time

import pytest
from tests._fake_counters import FakeCounters

from mlx_dfloat._watchdog import Counters, Watchdog

PREFIX = "mlx-dfloat: the memory watchdog ended the process"


class _ExitRecorder:
    """Stands in for the process exit: records each code and lets a test wait for the first."""

    def __init__(self):
        self.codes: list[int] = []
        self.fired = threading.Event()

    def __call__(self, code):
        self.codes.append(code)
        self.fired.set()


@pytest.fixture
def arm(tmp_path):
    """Start a raise-mode watchdog (ceiling 10_000, stop margin 1_000, so a stop line of 9_000).

    Every watchdog started here is stopped at teardown, so none can escalate into a later test.
    """
    started = []

    def _arm(
        fake, *, out_dir=None, ceiling=10_000, stop_margin=1_000, grace=10.0, budget=1e9, **kw
    ):
        record = _ExitRecorder()
        watchdog = Watchdog(
            tmp_path if out_dir is None else out_dir,
            ceiling=ceiling,
            budget=budget,
            interval=0.01,
            counters=fake,
            exit_process=record,
            mode="raise",
            stop_margin=stop_margin,
            grace=grace,
            **kw,
        )
        started.append(watchdog)
        return watchdog.start(), record

    yield _arm
    for watchdog in started:
        watchdog.stop()


def _at(watched):
    fake = FakeCounters()
    fake.raise_to(watched)
    return fake


def _refuse(constant):
    raise ValueError(f"not strict JSON: {constant}")


def test_raise_mode_trip_writes_the_record_and_does_not_exit(arm, tmp_path):
    # Bug caught: _fire still exiting in raise mode, or the trip not writing the record.
    watchdog, record = arm(_at(9_001))
    assert watchdog.tripped.wait(timeout=5)
    assert not record.fired.wait(timeout=0.3)
    assert watchdog.trip["mode"] == "raise"
    assert watchdog.trip["stop_line"] == 9_000
    assert watchdog.trip["reason"] == "memory"
    assert watchdog.trip["escalation"] is None
    assert watchdog.artifact == tmp_path / "abort.json"
    assert json.loads(watchdog.artifact.read_text())["stop_line"] == 9_000


def test_a_trip_below_the_stop_line_does_not_fire(arm):
    # Bug caught: `>=` in place of `>` at the stop line (9_000 would trip).
    fake = _at(9_000)
    watchdog, _ = arm(fake)
    assert not watchdog.tripped.wait(timeout=0.3)
    fake.raise_to(9_001)
    assert watchdog.tripped.wait(timeout=5)


def test_reaching_the_ceiling_after_a_trip_exits_70(arm, tmp_path):
    # Bug caught: no ceiling check after the trip, `>` in place of `>=` at the ceiling, or a
    # sampler that keeps escalating (`[70, 70, ...]`) instead of ending.
    fake = _at(9_001)
    watchdog, record = arm(fake)
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]
    assert json.loads((tmp_path / "abort.json").read_text())["escalation"] == "ceiling"


class _Breakable(FakeCounters):
    """``FakeCounters`` whose reads start raising once ``broken`` is set: a sampler that dies mid-run."""

    def __init__(self):
        super().__init__()
        self.broken = False

    def __call__(self):
        if self.broken:
            raise OSError("rusage")
        return super().__call__()


def _broken():
    raise OSError("rusage")


def test_a_memory_trip_below_the_ceiling_never_exits_by_grace(arm):
    # Bug caught: the grace clock applied to a memory trip, so a stretch with no checkpoint longer
    # than grace (a prompt encode, a VAE decode) ends the process below the ceiling.
    watchdog, record = arm(_at(9_500), grace=0.2)
    assert watchdog.tripped.wait(timeout=5)
    assert not record.fired.wait(timeout=1.0)
    watchdog.consume()
    assert record.codes == []
    assert watchdog.trip["escalation"] is None


def test_an_unconsumed_wall_trip_exits_after_grace(arm, tmp_path):
    # Bug caught: the grace clock dropped for wall trips too, so a wall trip no checkpoint takes
    # runs on unbounded below the ceiling.
    watchdog, record = arm(_at(0), budget=0.05, grace=0.2)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [71]
    assert json.loads((tmp_path / "abort.json").read_text())["escalation"] == "grace"


def test_a_sample_error_trip_exits_after_grace(arm, tmp_path):
    # Bug caught: no grace for a trip whose samples carry no number, so a dead sampler leaves the
    # job unwatched for good.
    watchdog, record = arm(_broken, grace=0.2)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]
    assert json.loads((tmp_path / "abort.json").read_text())["escalation"] == "grace"


def test_a_failed_sample_after_a_memory_trip_starts_the_grace_clock(arm, tmp_path):
    # Bug caught: grace decided by the trip's reason alone, so a sampler that dies after a memory
    # trip leaves the job unwatched (no number for the ceiling, no grace).
    fake = _Breakable()
    fake.raise_to(9_500)
    watchdog, record = arm(fake, grace=0.2)
    assert watchdog.tripped.wait(timeout=5)
    assert not record.fired.wait(timeout=0.5)
    fake.broken = True
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]
    on_disk = json.loads((tmp_path / "abort.json").read_text())
    assert (on_disk["reason"], on_disk["escalation"]) == ("memory", "grace")


def test_after_a_memory_trip_the_grace_clock_starts_at_the_first_failed_sample(arm, tmp_path):
    # Bug caught: the clock counted from the memory trip, so a sampler that dies long after a trip
    # with live samples ends the process on its first failed sample, with no grace at all.
    fake = _Breakable()
    fake.raise_to(9_500)
    watchdog, record = arm(fake, grace=0.3)
    assert watchdog.tripped.wait(timeout=5)
    assert not record.fired.wait(timeout=0.5)  # the trip is now older than grace
    broke_at = time.monotonic()
    fake.broken = True
    assert not record.fired.wait(timeout=0.15)
    assert record.fired.wait(timeout=5)
    assert time.monotonic() - broke_at >= 0.3
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]
    on_disk = json.loads((tmp_path / "abort.json").read_text())
    assert (on_disk["reason"], on_disk["escalation"]) == ("memory", "grace")


def test_a_recovered_sampler_resets_the_grace_clock_after_a_memory_trip(arm):
    # Bug caught: the failed-sample clock kept running across a recovery, so a second short outage
    # ends the process at once, counted from the first outage.
    fake = _Breakable()
    fake.raise_to(9_500)
    watchdog, record = arm(fake, grace=0.5)
    assert watchdog.tripped.wait(timeout=5)
    fake.broken = True
    assert not record.fired.wait(timeout=0.2)
    fake.broken = False
    assert not record.fired.wait(timeout=0.4)
    fake.broken = True
    assert not record.fired.wait(timeout=0.25)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]


def test_a_sample_error_trip_whose_sampler_recovers_has_no_grace(arm):
    # Bug caught: grace decided by the trip's reason alone, so a sample-error trip whose sampler
    # comes back below the ceiling still ends the process by grace.
    fake = _Breakable()
    fake.broken = True
    watchdog, record = arm(fake, grace=1.0)
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(9_500)
    fake.broken = False
    assert not record.fired.wait(timeout=1.5)
    assert watchdog.trip["reason"] == "sample_error"
    assert record.codes == []


def test_consume_stops_the_grace_clock(arm):
    # Bug caught: consume() not stopping the clock, so cleanup after the error is cut off by a
    # hard exit; or the latency never recorded. A wall trip, the kind grace still ends.
    watchdog, record = arm(_at(0), budget=0.05, grace=1.0)
    assert watchdog.tripped.wait(timeout=5)
    watchdog.consume()
    assert not record.fired.wait(timeout=1.5)
    assert watchdog.trip["checkpoint_latency_s"] >= 0


def test_consume_does_not_stop_the_ceiling_check(arm):
    # Bug caught: consume() also turning off the hard line, so cleanup could run past the ceiling.
    fake = _at(9_500)
    watchdog, record = arm(fake)
    assert watchdog.tripped.wait(timeout=5)
    watchdog.consume()
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    assert record.codes == [70]


def test_stop_after_a_trip_never_exits(arm):
    # Bug caught: the sampler outliving stop(), so a caller that already raised and cleaned up is
    # ended later by the backstop.
    fake = _at(9_001)
    watchdog, record = arm(fake)
    assert watchdog.tripped.wait(timeout=5)
    watchdog.stop()
    fake.raise_to(10_000)
    assert not record.fired.wait(timeout=0.3)


def test_wall_trip_in_raise_mode_escalates_with_71(arm):
    # Bug caught: the wall escalation exiting 70, which reads as a memory abort.
    watchdog, record = arm(_at(0), budget=0.05, grace=0.2)
    assert record.fired.wait(timeout=5)
    assert record.codes == [71]
    assert watchdog.trip["reason"] == "wall"


def test_a_wall_trip_that_reaches_the_ceiling_exits_70(arm, tmp_path):
    # Bug caught: the exit code taken from the trip's reason instead of what ended the process,
    # so a wall trip that later reaches the ceiling exits 71 and tells the user it was the clock
    # when memory ended it.
    fake = _at(0)
    watchdog, record = arm(fake, budget=0.05)
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    assert record.codes == [70]
    assert json.loads((tmp_path / "abort.json").read_text())["escalation"] == "ceiling"


def _blocked_while_locked(watchdog, action):
    """Run ``action`` on another thread while the test holds the watchdog's lock.

    Returns whether it finished while the lock was held, and whether it finished after release.
    """
    done = threading.Event()
    with watchdog._lock:
        worker = threading.Thread(target=lambda: (action(), done.set()))
        worker.start()
        finished_while_locked = done.wait(timeout=0.2)
    finished_after = done.wait(timeout=5)
    worker.join(timeout=5)
    return finished_while_locked, finished_after


def test_set_ceiling_tightens_and_returns_the_previous(arm):
    # Bug caught: set_ceiling not reaching the sampler (a stop line fixed at construction), not
    # returning the old value, or writing the ceiling outside the lock.
    fake = _at(0)
    watchdog, _ = arm(fake)
    assert watchdog.set_ceiling(5_000) == 10_000
    assert _blocked_while_locked(watchdog, lambda: watchdog.set_ceiling(6_000)) == (False, True)
    assert watchdog.set_ceiling(5_000) == 6_000
    fake.raise_to(4_001)
    assert watchdog.tripped.wait(timeout=5)
    assert (watchdog.trip["ceiling"], watchdog.trip["stop_line"]) == (5_000, 4_000)


def test_unwritable_artifact_dir_still_raises_with_artifact_none(arm, tmp_path):
    # Bug caught: the write failure killing the sampler thread (no trip, nothing watched).
    blocked = tmp_path / "blocked"
    blocked.write_text("a regular file")
    watchdog, record = arm(_at(9_001), out_dir=blocked / "out")
    assert watchdog.tripped.wait(timeout=5)
    assert watchdog.artifact is None
    assert "artifact_error" in watchdog.trip
    assert not record.fired.wait(timeout=0.3)


def test_no_wall_budget_records_null_and_parses_as_strict_json(arm):
    # Bug caught: None turned into float("inf"), which json.dumps writes as `Infinity`.
    watchdog, _ = arm(_at(9_001), budget=None)
    assert watchdog.tripped.wait(timeout=5)
    record = json.loads(watchdog.artifact.read_text(), parse_constant=_refuse)
    assert record["budget"] is None


def test_the_escalation_line_names_the_record_on_stderr(arm, tmp_path, capfd):
    # Bug caught: a hard exit with nothing on stderr, so the user cannot find the record.
    fake = _at(9_001)
    watchdog, record = arm(fake, live_context={"phase": lambda: "vae"})
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    err = capfd.readouterr().err
    assert f"{PREFIX} (memory, phase vae); record: {tmp_path / 'abort.json'}" in err.splitlines()


def test_the_escalation_line_carries_the_record_when_the_write_failed(arm, tmp_path, capfd):
    # Bug caught: a line pointing at a file that does not exist, leaving no record anywhere.
    blocked = tmp_path / "blocked"
    blocked.write_text("a regular file")
    fake = _at(9_001)
    watchdog, record = arm(fake, out_dir=blocked / "out")
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    (line,) = [x for x in capfd.readouterr().err.splitlines() if x.startswith(PREFIX)]
    assert "record not written" in line
    carried = json.loads(line[line.index("{") :])
    assert (carried["escalation"], carried["stop_line"]) == ("ceiling", 9_000)


def test_a_sample_error_trip_escalates_only_by_grace(arm):
    # Bug caught: the ceiling compare run on a sample with no number (None >= ceiling raises in
    # the sampler thread), or a sample error escalating at once instead of after grace.
    watchdog, record = arm(_broken)
    assert watchdog.tripped.wait(timeout=5)
    assert watchdog.trip["reason"] == "sample_error"
    assert not record.fired.wait(timeout=0.3)


def test_a_wall_trip_ended_at_the_ceiling_names_memory_on_stderr(arm, tmp_path, capfd):
    # Bug caught: the line naming only the trip's reason ("wall") while the process exits 70 because memory reached
    # the ceiling, so the user reads a clock stop where memory ended the run.
    fake = _at(0)
    watchdog, record = arm(fake, budget=0.05, live_context={"phase": lambda: "denoise"})
    assert watchdog.tripped.wait(timeout=5)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    assert record.codes == [70]
    err = capfd.readouterr().err
    line = f"{PREFIX} (memory ceiling after a wall trip, phase denoise); record: {tmp_path / 'abort.json'}"
    assert line in err.splitlines()


def _boom(*_args):
    raise RuntimeError("the trip failed")


def test_a_trip_that_raises_on_failing_samples_ends_70_after_grace(arm, tmp_path):
    # Bug caught: an exception inside the raise-mode loop kills the sampler thread, so a call whose samples fail runs
    # unwatched: no trip for a checkpoint to raise, and no exit however long it runs.
    fake = _Breakable()
    watchdog, record = arm(fake, grace=1.0)
    watchdog._trip = _boom
    fake.broken = True
    assert watchdog.tripped.wait(timeout=5)
    latched_at = time.monotonic()
    assert not record.fired.wait(timeout=0.5)
    assert record.fired.wait(timeout=5)
    assert time.monotonic() - latched_at >= 0.9
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]
    on_disk = json.loads((tmp_path / "abort.json").read_text())
    assert (on_disk["reason"], on_disk["escalation"]) == ("sample_error", "grace")


def test_a_trip_that_raises_on_a_live_sample_still_trips_and_ends_at_the_ceiling(arm):
    # Bug caught: the same dead sampler with live readings: the call passes the stop line and then the ceiling with
    # no checkpoint error and no exit (the probe returned normally at 29 GiB against a 28 GiB ceiling).
    fake = _at(0)
    watchdog, record = arm(fake)
    watchdog._trip = _boom
    fake.raise_to(9_500)
    assert watchdog.tripped.wait(timeout=5)
    assert not record.fired.wait(timeout=0.3)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    assert record.codes == [70]


def test_a_sample_in_flight_when_stop_returns_writes_no_record(arm, tmp_path):
    # Bug caught: _trip not checking the stop flag under the lock, so a reading taken before stop() lands a trip and
    # a record after the caller stopped the watchdog and already owns the outcome.
    entered, release = threading.Event(), threading.Event()

    def reader():
        entered.set()
        release.wait(timeout=10)
        return Counters(footprint=9_500, rss=0, mlx_active=0, mlx_cache=0)

    watchdog, record = arm(reader)
    assert entered.wait(timeout=5)
    watchdog.stop()  # the sampler is blocked in the read: stop() returns after its join times out
    release.set()
    watchdog._thread.join(timeout=5)
    assert not watchdog._thread.is_alive()
    assert not watchdog.tripped.is_set()
    assert watchdog.trip is None
    assert list(tmp_path.iterdir()) == []
    assert record.codes == []


def _plant_symlink(at, target):
    """Put a symlink to ``target`` at ``at``, replacing whatever file is there."""
    planted = at.with_name(at.name + ".planted")
    planted.symlink_to(target)
    planted.replace(at)


def test_the_trip_record_replaces_a_planted_symlink_and_is_private(arm, tmp_path):
    # Bug caught: the first write follows a symlink at the record name and truncates its target, or leaves the
    # record 0644, or leaves the temp file behind next to it.
    target = tmp_path / "target.txt"
    target.write_text("keep")
    (tmp_path / "abort.json").symlink_to(target)
    watchdog, _ = arm(_at(9_001))
    assert watchdog.tripped.wait(timeout=5)
    written = tmp_path / "abort.json"
    assert not written.is_symlink()
    assert json.loads(written.read_text())["stop_line"] == 9_000
    assert target.read_text() == "keep"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abort.json", "target.txt"]


def test_the_escalation_rewrite_replaces_a_symlink_planted_after_the_trip(arm, tmp_path):
    # Bug caught: the rewrite at the ceiling follows a symlink swapped in after the first write (a local user who owns
    # the folder) and truncates its target.
    fake = _at(9_001)
    watchdog, record = arm(fake)
    assert watchdog.tripped.wait(timeout=5)
    target = tmp_path / "target.txt"
    target.write_text("keep")
    _plant_symlink(tmp_path / "abort.json", target)
    fake.raise_to(10_000)
    assert record.fired.wait(timeout=5)
    watchdog._thread.join(timeout=5)
    written = tmp_path / "abort.json"
    assert not written.is_symlink()
    assert json.loads(written.read_text())["escalation"] == "ceiling"
    assert target.read_text() == "keep"
    assert stat.S_IMODE(written.stat().st_mode) == 0o600
