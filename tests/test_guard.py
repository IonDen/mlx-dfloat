"""The guard: one watchdog per call scope, checkpoints that turn a trip into a typed error.

Every watchdog here is a real raise-mode ``Watchdog`` that samples a ``FakeCounters`` and ends the
"process" through a recorder, so no test reads the real allocator or can end pytest. No test here
imports mflux.
"""

import ast
import contextlib
from pathlib import Path

import pytest
from tests._fake_counters import FakeCounters

from mlx_dfloat import _guard
from mlx_dfloat._watchdog import Watchdog
from mlx_dfloat.errors import DFloatWatchdogError

GUARD_SOURCE = Path(_guard.__file__)


class _ExitRecorder:
    """Stands in for the process exit: records each code."""

    def __init__(self):
        self.codes: list[int] = []

    def __call__(self, code):
        self.codes.append(code)


@pytest.fixture
def armed(tmp_path):
    """Start raise-mode watchdogs (ceiling 10_000, stop margin 1_000: a stop line of 9_000).

    Each is stopped at teardown, so none can escalate into a later test.
    """
    started = []

    def _arm(*, ceiling=10_000, live_context=None, out_dir=None):
        fake, exits = FakeCounters(), _ExitRecorder()
        watchdog = Watchdog(
            tmp_path if out_dir is None else out_dir,
            ceiling=ceiling,
            budget=None,
            interval=0.01,
            counters=fake,
            exit_process=exits,
            mode="raise",
            stop_margin=1_000,
            live_context=live_context,
        )
        started.append(watchdog)
        return watchdog.start(), fake, exits

    yield _arm
    for watchdog in started:
        watchdog.stop()


def _trip(watchdog, fake, watched=9_500):
    fake.raise_to(watched)
    assert watchdog.tripped.wait(timeout=5)


def test_checkpoint_without_a_scope_is_a_no_op(armed):
    # Bug caught: checkpoint reaching a watchdog that no scope armed (a global set at start(), or
    # a scope left set by an earlier call), so an unwatched call raises.
    watchdog, fake, exits = armed()
    _trip(watchdog, fake)
    assert _guard.active() is None
    _guard.checkpoint("block_end")
    assert watchdog.trip["checkpoint_latency_s"] is None
    assert exits.codes == []


def test_checkpoint_raises_once_tripped_and_consumes(armed):
    # Bug caught: no consume() before the raise, so the grace clock keeps running through the
    # cleanup and can end the process; or the error carrying the wrong reason or site.
    watchdog, fake, exits = armed()
    with _guard.scope(watchdog):
        _guard.checkpoint("block_start")  # not tripped yet: no raise
        _trip(watchdog, fake)
        with pytest.raises(DFloatWatchdogError) as caught:
            _guard.checkpoint("block_end")
    assert (caught.value.reason, caught.value.site) == ("memory", "block_end")
    assert watchdog.trip["checkpoint_latency_s"] >= 0
    assert caught.value.record["checkpoint_latency_s"] >= 0
    assert exits.codes == []


def test_scope_resets_on_exit_even_after_a_raise(armed):
    # Bug caught: the ContextVar not reset in a finally, so a raised call leaks its watchdog into
    # every later call (and test) in this thread.
    watchdog, _, _ = armed()
    with contextlib.suppress(RuntimeError), _guard.scope(watchdog):
        assert _guard.active() is watchdog
        raise RuntimeError("boom")
    assert _guard.active() is None


def test_scope_nested_restores_the_outer_watchdog(armed):
    # Bug caught: the reset done with set(None) instead of the token, so an inner scope's exit
    # leaves the outer call unwatched.
    outer, _, _ = armed()
    inner, _, _ = armed()
    with _guard.scope(outer):
        with _guard.scope(inner):
            assert _guard.active() is inner
        assert _guard.active() is outer
    assert _guard.active() is None


def test_tightened_lowers_and_restores(armed):
    # Bug caught: the min rule inverted (a higher inner ceiling raising the outer one), `>=` written
    # as `>` (an equal ceiling still swapped in, caught by the spy below), or the old ceiling not
    # restored after the inner call (also after a raise).
    watchdog, _, _ = armed(ceiling=10_000)
    calls = []
    real_set_ceiling = watchdog.set_ceiling

    def spy(ceiling):
        calls.append(ceiling)
        return real_set_ceiling(ceiling)

    watchdog.set_ceiling = spy
    with _guard.scope(watchdog), _guard.tightened(10_000):
        assert watchdog.ceiling == 10_000
    assert calls == []
    with _guard.scope(watchdog):
        with _guard.tightened(5_000):
            assert watchdog.ceiling == 5_000
        assert watchdog.ceiling == 10_000
        with contextlib.suppress(RuntimeError), _guard.tightened(9_999):
            assert watchdog.ceiling == 9_999
            raise RuntimeError("inner call failed")
        assert watchdog.ceiling == 10_000
        for no_change in (20_000, 10_000, None):
            with _guard.tightened(no_change):
                assert watchdog.ceiling == 10_000
    with _guard.tightened(5_000):  # no active watchdog: nothing to lower
        assert _guard.active() is None


def test_error_message_names_phase_artifact_and_site(armed, tmp_path):
    # Bug caught: a message that drops the phase, the record's path or the site, so the user
    # cannot tell where the call stopped or find the record.
    watchdog, fake, _ = armed(live_context={"phase": lambda: "denoise"})
    _trip(watchdog, fake)
    err = _guard.error_for(watchdog, "set_group")
    message = str(err)
    assert "(memory, phase denoise)" in message
    assert "set_group" in message
    assert str(tmp_path / "abort.json") in message
    assert (err.reason, err.phase, err.site, err.artifact) == (
        "memory",
        "denoise",
        "set_group",
        tmp_path / "abort.json",
    )


def test_error_message_gives_the_watched_number_and_the_stop_line_in_gib(tmp_path):
    # Bug caught: bytes printed raw, or the watched number and the stop line swapped.
    fake = FakeCounters()
    gib = 1024**3
    watchdog = Watchdog(
        tmp_path,
        ceiling=28 * gib,
        budget=None,
        interval=0.01,
        counters=fake,
        exit_process=_ExitRecorder(),
        mode="raise",
        stop_margin=gib,
    )
    watchdog.start()
    try:
        _trip(watchdog, fake, watched=27 * gib + gib // 2)
        message = str(_guard.error_for(watchdog, "block_end"))
    finally:
        watchdog.stop()
    assert "27.50 GiB" in message
    assert "27.00 GiB" in message
    assert message.index("27.50 GiB") < message.index("27.00 GiB")


def test_guard_imports_neither_the_watchdog_nor_mlx():
    # Bug caught: `from mlx_dfloat._watchdog import Watchdog` in _guard (an import cycle once the
    # watchdog or the seam imports the guard).
    tree = ast.parse(GUARD_SOURCE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"contextvars", "contextlib", "typing", "mlx_dfloat.errors"}


def test_a_string_folder_still_raises_the_typed_error(armed, tmp_path):
    # Bug caught: the record path joined outside the write's try, so a str folder raises TypeError in the sampler at
    # the first trip: no trip is set, the sampler is dead, and the call runs on with no error.
    watchdog, fake, exits = armed(out_dir=str(tmp_path))
    with _guard.scope(watchdog):
        _trip(watchdog, fake)
        with pytest.raises(DFloatWatchdogError) as caught:
            _guard.checkpoint("block_end")
    assert caught.value.reason == "memory"
    assert caught.value.artifact is None
    assert "artifact_error" in caught.value.record
    assert exits.codes == []


class _WallTripped:
    """A tripped watchdog as the guard reads it: a wall trip 12.34 s into a 10 s budget."""

    def __init__(self):
        self.tripped = _Set()
        self.trip = {
            "reason": "wall",
            "elapsed": 12.34,
            "budget": 10.0,
            "verdict_memory": 5 * 1024**3,
            "stop_line": 27 * 1024**3,
            "context": {"phase": "denoise"},
        }
        self.artifact = None
        self.ceiling = 28 * 1024**3

    def consume(self):
        pass

    def set_ceiling(self, ceiling):
        previous, self.ceiling = self.ceiling, ceiling
        return previous


class _Set:
    def is_set(self):
        return True


def test_a_wall_trip_message_gives_the_elapsed_time_against_the_budget():
    # Bug caught: a wall trip worded as a memory comparison ("watched 5.00 GiB against the stop line 27.00 GiB"),
    # so the user reads a memory stop where the clock stopped the call.
    message = str(_guard.error_for(_WallTripped(), "block_end"))
    assert "(wall, phase denoise)" in message
    assert "12.3 s against the wall budget of 10.0 s" in message
    assert "GiB" not in message
