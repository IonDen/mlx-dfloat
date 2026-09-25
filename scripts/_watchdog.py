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
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self._start
            rss = mlx_active = mlx_cache = 0
            try:
                rss = int(psutil.Process().memory_info().rss)
                mlx_active = int(mx.get_active_memory())
                mlx_cache = int(mx.get_cache_memory())
                total = rss + mlx_active + mlx_cache
                self.peak_rss = max(self.peak_rss, total)
                reason = verdict(
                    rss=total, ceiling=self.ceiling, elapsed=elapsed, budget=self.budget
                )
            except Exception:  # a dead sampler must still abort, not run the job unwatched
                reason = "sample_error"
            if reason is None:
                continue
            code = EXIT_WALL if reason == "wall" else EXIT_MEMORY
            try:
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
                            "mlx_active": mlx_active,
                            "mlx_cache": mlx_cache,
                        },
                        indent=1,
                    )
                )
            finally:
                os._exit(code)  # always exits, even if the artifact write above raised
