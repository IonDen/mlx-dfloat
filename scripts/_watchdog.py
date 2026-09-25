"""Process-level watchdog for heavy scripts: RSS ceiling plus wall-clock backstop.

NumPy allocations are invisible to MLX's counters, so the ceiling is checked against the process
RSS; MLX active + cache is sampled too and recorded in the abort artifact.
"""

import json
import os
import threading
import time
from pathlib import Path

import mlx.core as mx
import psutil

EXIT_MEMORY = 70
EXIT_WALL = 71


def verdict(*, rss: int, ceiling: int, elapsed: float, budget: float) -> str | None:
    """Decide whether to abort: "memory", "wall", or None."""
    if rss > ceiling:
        return "memory"
    if elapsed > budget:
        return "wall"
    return None


def default_ceiling() -> int:
    """Physical memory minus 4 GiB (workspace rule)."""
    return int(psutil.virtual_memory().total) - 4 * 1024**3


class Watchdog:
    """Background sampler that aborts the process with an honest artifact."""

    def __init__(
        self, out_dir: Path, *, ceiling: int, budget: float, interval: float = 0.05
    ) -> None:
        """Configure the ceiling (bytes), wall budget (seconds) and poll interval."""
        self.out_dir, self.ceiling, self.budget, self.interval = out_dir, ceiling, budget, interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = time.monotonic()

    def start(self) -> "Watchdog":
        """Start sampling."""
        self._start = time.monotonic()
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling."""
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop.wait(self.interval):
            rss = int(proc.memory_info().rss)
            self.peak_rss = max(self.peak_rss, rss)
            elapsed = time.monotonic() - self._start
            reason = verdict(rss=rss, ceiling=self.ceiling, elapsed=elapsed, budget=self.budget)
            if reason:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "abort.json").write_text(
                    json.dumps(
                        {
                            "reason": reason,
                            "rss": rss,
                            "peak_rss": self.peak_rss,
                            "ceiling": self.ceiling,
                            "elapsed": elapsed,
                            "budget": self.budget,
                            "mlx_active": int(mx.get_active_memory()),
                            "mlx_cache": int(mx.get_cache_memory()),
                        },
                        indent=1,
                    )
                )
                os._exit(EXIT_MEMORY if reason == "memory" else EXIT_WALL)
