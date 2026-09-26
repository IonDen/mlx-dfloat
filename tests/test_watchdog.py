import json
import sys
import threading

import pytest
import scripts._watchdog as wd
from scripts._watchdog import Watchdog, verdict


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
    monkeypatch.setattr(wd.psutil, "Process", _FixedRssProcess(rss))
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

    monkeypatch.setattr(wd.psutil, "Process", boom)
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
    monkeypatch.setattr(wd.psutil, "Process", _FixedRssProcess(rss=10**12))
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


def test_a_failing_footprint_read_aborts_as_a_sample_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "phys_footprint", lambda: (_ for _ in ()).throw(OSError("rusage")))
    reason, _ = wd.Watchdog(tmp_path, ceiling=10**11, budget=60)._sample()
    assert reason == "sample_error"


@pytest.mark.skipif(sys.platform != "darwin", reason="libproc footprint is macOS-only")
def test_phys_footprint_sees_a_gpu_allocation_that_rss_misses():
    # Bug caught: a wrong struct field (resident size or wired size would not move with a Metal
    # allocation).
    import mlx.core as mx
    import psutil

    before, rss_before = wd.phys_footprint(), psutil.Process().memory_info().rss
    a = mx.zeros((256 * 1024 * 1024,), dtype=mx.uint8)
    mx.eval(a)
    after, rss_after = wd.phys_footprint(), psutil.Process().memory_info().rss
    assert after - before > 200 * 1024**2
    assert rss_after - rss_before < 64 * 1024**2
    del a
