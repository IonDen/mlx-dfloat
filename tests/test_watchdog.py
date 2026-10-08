import builtins
import json
import sys
import threading

import psutil
import pytest

import mlx_dfloat._watchdog as wd
from mlx_dfloat._watchdog import Watchdog, verdict, watched_memory
from mlx_dfloat.errors import DFloatDependencyError


def test_memory_verdict_uses_the_watched_number():
    # Bug caught: a verdict comparing something other than the watched number to the ceiling
    # (flip `memory > ceiling` to `<`, or read a different field).
    assert verdict(memory=11, ceiling=10, elapsed=1.0, budget=100.0) == "memory"


def test_wall_verdict():
    # Bug caught: dropping the `elapsed > budget` branch.
    assert verdict(memory=1, ceiling=10, elapsed=101.0, budget=100.0) == "wall"


def test_no_verdict_under_both_limits():
    # Bug caught: `>=` instead of `>` at either limit.
    assert verdict(memory=10, ceiling=10, elapsed=100.0, budget=100.0) is None


def test_watched_memory_is_the_max_of_footprint_and_mlx_active_plus_cache():
    # Bug caught: a footprint-only watchdog (misses an overrun held in MLX's cache pool) or a sum
    # (double counts); and a tie resolving to "mlx" instead of "footprint".
    assert watched_memory(footprint=10, mlx_active=3, mlx_cache=2) == (10, "footprint")
    assert watched_memory(footprint=4, mlx_active=3, mlx_cache=2) == (5, "mlx")
    assert watched_memory(footprint=5, mlx_active=3, mlx_cache=2) == (5, "footprint")  # tie


class _FixedRssProcess:
    """Stands in for psutil.Process(): a process whose RSS is a fixed number of bytes."""

    def __init__(self, rss):
        self._rss = rss

    def __call__(self):
        return self

    def memory_info(self):
        return type("MemoryInfo", (), {"rss": self._rss})()


def _stub_memory(monkeypatch, *, rss, mlx_active, mlx_cache):
    # `psutil` is imported inside the functions that need it (never at module top level), so
    # there is no `wd.psutil` attribute to patch; patching the real, shared `psutil` module here
    # still reaches the watchdog's own `import psutil`, since Python caches modules by name.
    monkeypatch.setattr(psutil, "Process", _FixedRssProcess(rss))
    monkeypatch.setattr(wd.mx, "get_active_memory", lambda: mlx_active)
    monkeypatch.setattr(wd.mx, "get_cache_memory", lambda: mlx_cache)


def _run_watchdog_to_abort(tmp_path, monkeypatch, *, ceiling, budget):
    """Start a real Watchdog with its `_exit` alias patched to record instead of exiting.

    Only the module's own alias is patched, never `os._exit` process-wide, so a watchdog thread
    that outlives its test cannot take pytest down with it.
    """
    exit_codes: list[int] = []
    fired = threading.Event()

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    monkeypatch.setattr(wd, "_exit", fake_exit)
    watchdog = Watchdog(tmp_path, ceiling=ceiling, budget=budget, interval=0.01).start()
    fired.wait(timeout=5)
    watchdog.stop()
    return exit_codes


def test_watchdog_fires_memory_abort_and_writes_the_artifact(tmp_path, monkeypatch):
    # Bug caught: a Watchdog whose thread silently swallows the abort (e.g. `pass` instead of
    # `os._exit`) never fires and this suite would stay green with the job running unwatched.
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=0, budget=1e9)
    assert exit_codes
    assert exit_codes[0] == 70
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["reason"] == "memory"
    assert set(artifact) >= {
        "reason",
        "footprint",
        "peak_footprint",
        "ceiling",
        "elapsed",
        "budget",
        "rss",
        "mlx_active",
        "mlx_cache",
        "verdict_memory",
        "verdict_counter",
        "peak_watched",
        "peak_mlx",
    }


def test_watchdog_fires_wall_abort(tmp_path, monkeypatch):
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=10**15, budget=0.0)
    assert exit_codes
    assert exit_codes[0] == 71
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["reason"] == "wall"


def test_sample_records_rss_mlx_active_and_cache_alongside_footprint(tmp_path, monkeypatch):
    # Bug caught: switching the enforced ceiling number to the OS footprint and dropping rss/
    # mlx_active/mlx_cache from the sample would lose the diagnostic fields the parity scripts'
    # own abort artifacts rely on, even though those fields no longer drive the verdict.
    monkeypatch.setattr(wd, "phys_footprint", lambda: 10**6)
    _stub_memory(monkeypatch, rss=10**12, mlx_active=2 * 10**9, mlx_cache=3 * 10**9)
    reason, sample = wd.Watchdog(tmp_path, ceiling=10**15, budget=1e9)._sample()
    assert reason is None
    assert sample["footprint"] == 10**6
    assert sample["rss"] == 10**12
    assert sample["mlx_active"] == 2 * 10**9
    assert sample["mlx_cache"] == 3 * 10**9


def test_a_stopped_watchdog_never_writes_an_abort_or_exits(tmp_path, monkeypatch):
    # Bug caught: a verdict reached just as the script stops the watchdog (after its summary is
    # written) still writes abort.json and exits 70, contradicting the recorded result.
    exit_codes: list[int] = []
    monkeypatch.setattr(wd, "_exit", exit_codes.append)
    watchdog = Watchdog(tmp_path, ceiling=0, budget=1e9, interval=3600).start()
    watchdog.stop()
    watchdog._fire(
        "memory",
        {
            "footprint": 1,
            "rss": 1,
            "mlx_active": 0,
            "mlx_cache": 0,
            "elapsed": 0.0,
            "verdict_memory": 1,
            "verdict_counter": "footprint",
        },
    )
    assert exit_codes == []
    assert not (tmp_path / "abort.json").exists()


def test_watchdog_sample_error_still_aborts(tmp_path, monkeypatch):
    # Bug caught: an unhandled exception in the sampler thread (psutil, MLX, or disk I/O) kills
    # the daemon thread silently, and the heavy job runs on completely unwatched.
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(psutil, "Process", boom)
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=10**15, budget=1e9)
    assert exit_codes
    assert exit_codes[0] == 70
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["reason"] == "sample_error"
    # Bug caught: the artifact carrying the footprint defaults (counter "footprint", memory 0), which
    # reads as a footprint verdict when no number was ever compared with the ceiling.
    assert (artifact["verdict_counter"], artifact["verdict_memory"]) == ("none", None)


# Mocking `os._exit` (required so the test process itself doesn't exit) means the exception that
# triggered the `finally: os._exit(code)` in the first place resumes propagating once the mocked
# call returns normally, escaping the daemon thread; that residual propagation is an artifact of
# the mock, not of the code under test (in production `os._exit` never returns at all), so it is
# deliberately ignored here rather than silenced project-wide.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_watchdog_still_exits_when_the_artifact_write_itself_fails(tmp_path, monkeypatch):
    # Bug caught: removing the `finally:` around `os._exit(code)` (so a write failure in the abort
    # artifact path propagates up and kills the sampler thread) leaves a genuine ceiling breach
    # completely unenforced -- exit_codes stays empty and the job keeps running.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")  # mkdir(blocked, exist_ok=True) raises FileExistsError
    exit_codes = _run_watchdog_to_abort(blocked, monkeypatch, ceiling=0, budget=1e9)
    assert exit_codes
    assert exit_codes[0] == 70


def test_a_sample_over_the_ceiling_only_through_mlx_cache_aborts_naming_the_counter(
    tmp_path, monkeypatch
):
    # Bug caught: the sampler feeding the footprint alone to the verdict (the 2026-09-27 state),
    # so an overrun held in MLX's cache pool never trips; or the artifact not naming the counter.
    monkeypatch.setattr(wd, "phys_footprint", lambda: 4)
    _stub_memory(monkeypatch, rss=1, mlx_active=3, mlx_cache=2)
    watchdog = wd.Watchdog(tmp_path, ceiling=4, budget=1e9)
    reason, sample = watchdog._sample()
    assert reason == "memory"
    assert sample["verdict_memory"] == 5
    assert sample["verdict_counter"] == "mlx"
    exit_codes: list[int] = []
    monkeypatch.setattr(wd, "_exit", exit_codes.append)
    watchdog._fire("memory", sample)
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["verdict_counter"] == "mlx"
    assert artifact["verdict_memory"] == 5
    # Bug caught: _fire writing a constant or the wrong peak (footprint 4 differs from 5).
    assert artifact["peak_footprint"] == 4
    assert artifact["peak_watched"] == 5
    assert artifact["peak_mlx"] == 5
    assert exit_codes == [70]


def test_the_watched_number_is_a_maximum_not_a_sum(tmp_path, monkeypatch):
    # Bug caught: summing RSS or the footprint with MLX active + cache. An `mx.load`ed array lands
    # in both RSS and MLX active, so a sum double counts it; the maximum does not.
    monkeypatch.setattr(wd, "phys_footprint", lambda: 3 * 10**9)
    _stub_memory(monkeypatch, rss=10**12, mlx_active=3 * 10**9, mlx_cache=0)
    reason, sample = wd.Watchdog(tmp_path, ceiling=5 * 10**9, budget=1e9)._sample()
    assert reason is None
    assert sample["verdict_memory"] == 3 * 10**9


def test_the_watchdog_tracks_the_peaks_of_all_three_numbers_and_resets_them(tmp_path, monkeypatch):
    # Bug caught: peak_watched tracking the footprint only (the tier rows and the proof cap would
    # use a number the ceiling is not enforced on), or reset_peak leaving one of them.
    watchdog = wd.Watchdog(tmp_path, ceiling=10**15, budget=1e9)
    monkeypatch.setattr(wd, "phys_footprint", lambda: 4)
    _stub_memory(monkeypatch, rss=0, mlx_active=3, mlx_cache=2)
    watchdog._sample()
    monkeypatch.setattr(wd, "phys_footprint", lambda: 6)
    _stub_memory(monkeypatch, rss=0, mlx_active=1, mlx_cache=0)
    watchdog._sample()
    assert (watchdog.peak_footprint, watchdog.peak_mlx, watchdog.peak_watched) == (6, 5, 6)
    watchdog.reset_peak()
    assert (watchdog.peak_footprint, watchdog.peak_mlx, watchdog.peak_watched) == (0, None, 0)


def test_footprint_over_the_ceiling_aborts_with_70(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "phys_footprint", lambda: 10**12)
    exits = []
    monkeypatch.setattr(wd, "_exit", lambda code: exits.append(code))
    w = wd.Watchdog(tmp_path, ceiling=10**11, budget=60)
    reason, sample = w._sample()
    w._fire(reason, sample)
    assert reason == "memory"
    assert exits == [70]
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["footprint"] == 10**12
    assert artifact["peak_footprint"] == 10**12


def test_reset_peak_starts_the_footprint_peak_over(monkeypatch, tmp_path):
    # Bug caught: a peak that spans the process lifetime, so the model-load spike hides the timed
    # steps' own peak; or a reset that also forgets to track the next sample.
    values = iter([10**9, 10**6])
    monkeypatch.setattr(wd, "phys_footprint", lambda: next(values))
    w = wd.Watchdog(tmp_path, ceiling=10**15, budget=60)
    w._sample()
    assert w.peak_footprint == 10**9
    w.reset_peak()
    assert w.peak_footprint == 0
    w._sample()
    assert w.peak_footprint == 10**6


def test_a_failing_footprint_read_aborts_as_a_sample_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "phys_footprint", lambda: (_ for _ in ()).throw(OSError("rusage")))
    reason, _ = wd.Watchdog(tmp_path, ceiling=10**11, budget=60)._sample()
    assert reason == "sample_error"


def test_the_scripts_watchdog_module_is_the_package_one():
    # Bug caught: the compatibility shim re-implementing or re-defining Watchdog instead of
    # re-exporting the package's own class, which would let the two drift apart silently.
    import scripts._watchdog as script_side

    from mlx_dfloat import _watchdog as package_side

    assert script_side.Watchdog is package_side.Watchdog


_FOOTPRINT_RUNNER = """
import json, sys
import mlx.core as mx, psutil
from scripts._watchdog import phys_footprint
before, rss_before = phys_footprint(), psutil.Process().memory_info().rss
a = mx.zeros((256 * 1024 * 1024,), dtype=mx.uint8)
mx.eval(a)
after, rss_after = phys_footprint(), psutil.Process().memory_info().rss
print(json.dumps({"footprint": after - before, "rss": rss_after - rss_before}))
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="libproc footprint is macOS-only")
def test_phys_footprint_sees_a_gpu_allocation_that_rss_misses():
    # Bug caught: a wrong struct field (resident size or wired size would not move with a Metal
    # allocation). Runs in a cold subprocess: in the test process the Metal driver hands a freed
    # region of the same size straight back (the footprint then moves by nothing), and it reclaims
    # freed regions lazily, so an in-process before/after pair depends on what ran earlier.
    import json
    import subprocess
    from pathlib import Path

    proc = subprocess.run(
        [sys.executable, "-c", _FOOTPRINT_RUNNER],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    deltas = json.loads(proc.stdout.strip().splitlines()[-1])
    assert deltas["footprint"] > 200 * 1024**2
    # RSS may pick up a little of a Metal allocation; the point is that it does not grow by the
    # allocation while the footprint does.
    assert deltas["rss"] < 200 * 1024**2


def _without_psutil(monkeypatch):
    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)


def test_a_missing_psutil_refuses_the_watchdog_up_front(tmp_path, monkeypatch):
    # Bug caught: psutil imported only inside the sampling try, so a venv without it starts the
    # watchdog, turns the ImportError into `sample_error`, and exits 70 fifty milliseconds later
    # with an abort artifact that never names psutil; or default_ceiling() leaking a bare
    # ImportError instead of the package's dependency error.
    _without_psutil(monkeypatch)
    with pytest.raises(DFloatDependencyError, match="psutil"):
        Watchdog(tmp_path, ceiling=10**15, budget=60)
    with pytest.raises(DFloatDependencyError, match="psutil"):
        wd.default_ceiling()
    assert not (tmp_path / "abort.json").exists()


_SAMPLE = {
    "footprint": 1,
    "rss": 1,
    "mlx_active": 0,
    "mlx_cache": 0,
    "elapsed": 0.0,
    "verdict_memory": 1,
    "verdict_counter": "footprint",
}


def test_the_abort_artifact_carries_the_run_context_verbatim(tmp_path, monkeypatch):
    # Bug caught: the context dropped or reshaped on the way to abort.json, so the harness proof
    # cannot read the aborted run's size from its own record (the table used to assume 1024).
    context = {"model": "schnell", "height": 1024, "width": 768, "seed": 42, "steps": 4}
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    Watchdog(tmp_path, ceiling=0, budget=1e9, context=context)._fire("memory", dict(_SAMPLE))
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["context"] == {
        "model": "schnell",
        "height": 1024,
        "width": 768,
        "seed": 42,
        "steps": 4,
    }


def test_the_abort_artifact_has_no_context_by_default(tmp_path, monkeypatch):
    # Bug caught: an empty or invented context written for callers that passed none (the verify
    # scripts), which a reader would take as a record of the run.
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    Watchdog(tmp_path, ceiling=0, budget=1e9)._fire("memory", dict(_SAMPLE))
    assert "context" not in json.loads((tmp_path / "abort.json").read_text())


def test_live_context_values_are_read_when_the_abort_fires_not_when_the_watchdog_starts(
    tmp_path, monkeypatch
):
    # Bug caught: the phase frozen into the context at start ("build" for every abort, whatever ran when the
    # ceiling tripped), or a raising reader leaving the run without its abort artifact.
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    state = {"phase": "build"}
    live = {"phase": lambda: state["phase"], "broken": lambda: 1 / 0}
    watchdog = Watchdog(
        tmp_path, ceiling=0, budget=1e9, context={"model": "z-image"}, live_context=live
    )
    state["phase"] = "encode"
    watchdog._fire("memory", dict(_SAMPLE))
    context = json.loads((tmp_path / "abort.json").read_text())["context"]
    assert context == {
        "model": "z-image",
        "phase": "encode",
        "broken": "unreadable: ZeroDivisionError",
    }


def test_live_context_alone_still_writes_a_context(tmp_path, monkeypatch):
    # Bug caught: live values dropped when the caller passed no static context.
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    Watchdog(tmp_path, ceiling=0, budget=1e9, live_context={"phase": lambda: None})._fire(
        "memory", dict(_SAMPLE)
    )
    assert json.loads((tmp_path / "abort.json").read_text())["context"] == {"phase": None}


def test_an_unsampled_watchdog_has_no_mlx_peak(tmp_path, monkeypatch):
    # Bug caught: an MLX peak of 0 for a watchdog that never sampled (a real run would report a
    # 0-byte MLX peak as a number instead of "not sampled"); the first sample error leaves it None.
    watchdog = Watchdog(tmp_path, ceiling=10**15, budget=1e9)
    assert watchdog.peak_mlx is None
    monkeypatch.setattr(wd, "phys_footprint", lambda: (_ for _ in ()).throw(OSError("rusage")))
    reason, sample = watchdog._sample()
    assert reason == "sample_error"
    assert watchdog.peak_mlx is None
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    watchdog._fire(reason, sample)
    assert json.loads((tmp_path / "abort.json").read_text())["peak_mlx"] is None


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


def test_reset_peak_takes_the_lock(tmp_path):
    # Bug caught: reset_peak writing the peaks without the lock, so a reset can interleave with the
    # sampler's read-max-write and a timed window inherits the warm-up's peak.
    watchdog = Watchdog(tmp_path, ceiling=10**15, budget=1e9)
    watchdog.peak_footprint = 9
    assert _blocked_while_locked(watchdog, watchdog.reset_peak) == (False, True)
    assert watchdog.peak_footprint == 0


def test_the_sampler_updates_the_peaks_under_the_lock(tmp_path, monkeypatch):
    # Bug caught: _sample updating the peaks without the lock (the race reset_peak's lock guards).
    monkeypatch.setattr(wd, "phys_footprint", lambda: 7)
    _stub_memory(monkeypatch, rss=0, mlx_active=3, mlx_cache=2)
    watchdog = Watchdog(tmp_path, ceiling=10**15, budget=1e9)
    assert _blocked_while_locked(watchdog, watchdog._sample) == (False, True)
    assert (watchdog.peak_footprint, watchdog.peak_mlx, watchdog.peak_watched) == (7, 5, 7)
