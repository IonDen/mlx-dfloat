import builtins
import json
import sys
import threading

import psutil
import pytest

import mlx_dfloat._watchdog as wd
from mlx_dfloat._watchdog import Watchdog, verdict
from mlx_dfloat.errors import DFloatDependencyError


def test_memory_verdict_uses_process_rss():
    # Bug caught: a watchdog that only counts MLX memory would miss a NumPy blow-up.
    assert verdict(rss=11, ceiling=10, elapsed=1.0, budget=100.0) == "memory"


def test_wall_verdict():
    assert verdict(rss=1, ceiling=10, elapsed=101.0, budget=100.0) == "wall"


def test_no_verdict_under_both_limits():
    assert verdict(rss=10, ceiling=10, elapsed=100.0, budget=100.0) is None


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
    watchdog._fire("memory", {"rss": 1, "mlx_active": 0, "mlx_cache": 0, "elapsed": 0.0})
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


def test_verdict_uses_the_os_footprint_not_rss_plus_mlx(monkeypatch, tmp_path):
    # Bug caught: counting mx.load'ed arrays twice (RSS and MLX active) and false-aborting at half
    # the real ceiling.
    monkeypatch.setattr(wd, "phys_footprint", lambda: 10**9)
    monkeypatch.setattr(psutil, "Process", _FixedRssProcess(rss=10**12))
    monkeypatch.setattr(wd.mx, "get_active_memory", lambda: 10**12)
    monkeypatch.setattr(wd.mx, "get_cache_memory", lambda: 0)
    reason, sample = wd.Watchdog(tmp_path, ceiling=5 * 10**9, budget=60)._sample()
    assert reason is None
    assert sample["footprint"] == 10**9
    assert sample["rss"] == 10**12


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
