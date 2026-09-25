import json
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
        "rss",
        "peak_rss",
        "ceiling",
        "elapsed",
        "budget",
        "mlx_active",
        "mlx_cache",
    }


def test_watchdog_fires_wall_abort(tmp_path, monkeypatch):
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=10**15, budget=0.0)
    assert exit_codes
    assert exit_codes[0] == 71
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["reason"] == "wall"


def test_memory_verdict_includes_mlx_active_and_cache(tmp_path, monkeypatch):
    # Bug caught: a watchdog that feeds `verdict` bare process RSS would miss MLX buffers that
    # never land in RSS on this platform (measured: ~1 GB active, ~0 RSS delta). Either MLX term
    # alone (1e9) stays under the 1.5e9 ceiling; only active + cache crosses it.
    _stub_memory(monkeypatch, rss=0, mlx_active=10**9, mlx_cache=10**9)
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=15 * 10**8, budget=1e9)
    assert exit_codes == [70]
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["mlx_active"] == 10**9
    assert artifact["mlx_cache"] == 10**9


def test_memory_verdict_includes_process_rss(tmp_path, monkeypatch):
    # Bug caught: dropping the `rss` term leaves the NumPy-only parity scripts (their memory is
    # invisible to MLX's counters) with no working ceiling at all.
    _stub_memory(monkeypatch, rss=10**12, mlx_active=0, mlx_cache=0)
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=10**11, budget=1e9)
    assert exit_codes == [70]
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["reason"] == "memory"
    assert artifact["rss"] == 10**12


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
