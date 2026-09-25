"""Process-level watchdog for heavy scripts: RSS + MLX memory ceiling plus a wall-clock backstop.

The ceiling is checked against process RSS plus MLX active and cache memory combined, since MLX
buffers are not reliably visible in the process's own RSS on this platform (measured: ~1 GB MLX
active memory with ~0 RSS delta). A sampling failure (psutil, MLX, or the artifact write itself)
still aborts the process instead of leaving the job running unwatched.
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

# The one process exit the watchdog makes; tests patch this alias, never os._exit process-wide.
_exit = os._exit


def verdict(*, rss: int, ceiling: int, elapsed: float, budget: float) -> str | None:
    """Decide whether to abort: "memory", "wall", or None.

    ``rss`` is whatever single memory number the caller is enforcing the ceiling against; the
    caller (``Watchdog``) feeds it process RSS plus MLX active and cache memory combined, since
    the verdict itself stays a pure function of one memory number.
    """
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
        # Peak of (process RSS + MLX active + MLX cache): the verdict's own memory number.
        self.peak_rss = 0
        self._stop = threading.Event()
        # Held while an abort is written: stop() waits for it, so no abort can land after the
        # caller has stopped the watchdog and written its own verdict.
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = time.monotonic()

    def start(self) -> "Watchdog":
        """Start sampling."""
        self._start = time.monotonic()
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling. After this returns the watchdog never writes an abort or exits."""
        with self._lock:
            self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _sample(self) -> tuple[str | None, dict[str, float]]:
        elapsed = time.monotonic() - self._start
        sample: dict[str, float] = {"rss": 0, "mlx_active": 0, "mlx_cache": 0, "elapsed": elapsed}
        try:
            sample["rss"] = int(psutil.Process().memory_info().rss)
            sample["mlx_active"] = int(mx.get_active_memory())
            sample["mlx_cache"] = int(mx.get_cache_memory())
            total = int(sample["rss"] + sample["mlx_active"] + sample["mlx_cache"])
            self.peak_rss = max(self.peak_rss, total)
            reason = verdict(rss=total, ceiling=self.ceiling, elapsed=elapsed, budget=self.budget)
        except Exception:  # a dead sampler must still abort, not run the job unwatched
            reason = "sample_error"
        return reason, sample

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            reason, sample = self._sample()
            if reason is not None:
                self._fire(reason, sample)
                return

    def _fire(self, reason: str, sample: dict[str, float]) -> None:
        with self._lock:
            if self._stop.is_set():
                return  # the caller already stopped us and recorded its own result
            code = EXIT_WALL if reason == "wall" else EXIT_MEMORY
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                (self.out_dir / "abort.json").write_text(
                    json.dumps(
                        {
                            "reason": reason,
                            "rss": sample["rss"],
                            "peak_rss": self.peak_rss,
                            "ceiling": self.ceiling,
                            "elapsed": sample["elapsed"],
                            "budget": self.budget,
                            "mlx_active": sample["mlx_active"],
                            "mlx_cache": sample["mlx_cache"],
                        },
                        indent=1,
                    )
                )
            finally:
                _exit(code)  # always exits, even if the artifact write above raised
