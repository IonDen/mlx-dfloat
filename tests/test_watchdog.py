import json
import threading

import psutil
import scripts._watchdog as wd
from scripts._watchdog import Watchdog, verdict


def test_memory_verdict_uses_process_rss():
    # Bug caught: a watchdog that only counts MLX memory would miss a NumPy blow-up.
    assert verdict(rss=11, ceiling=10, elapsed=1.0, budget=100.0) == "memory"


def test_wall_verdict():
    assert verdict(rss=1, ceiling=10, elapsed=101.0, budget=100.0) == "wall"


def test_no_verdict_under_both_limits():
    assert verdict(rss=10, ceiling=10, elapsed=100.0, budget=100.0) is None


def _run_watchdog_to_abort(tmp_path, monkeypatch, *, ceiling, budget):
    """Start a real Watchdog with `os._exit` monkeypatched to record instead of exiting."""
    exit_codes: list[int] = []
    fired = threading.Event()

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    monkeypatch.setattr(wd.os, "_exit", fake_exit)
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
    # never land in RSS on this platform (measured: ~1 GB active, ~0 RSS delta).
    monkeypatch.setattr(wd.mx, "get_active_memory", lambda: 10**9)
    monkeypatch.setattr(wd.mx, "get_cache_memory", lambda: 10**9)
    # Comfortably above this process's real RSS alone, but well below rss + the two mocked MLX
    # numbers: only the MLX-inclusive sum can push the verdict over it.
    ceiling = int(psutil.Process().memory_info().rss * 4)
    exit_codes = _run_watchdog_to_abort(tmp_path, monkeypatch, ceiling=ceiling, budget=1e9)
    assert exit_codes
    assert exit_codes[0] == 70
    artifact = json.loads((tmp_path / "abort.json").read_text())
    assert artifact["mlx_active"] == 10**9
    assert artifact["mlx_cache"] == 10**9


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
