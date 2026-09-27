"""Process-level watchdog for heavy scripts: an OS-footprint ceiling plus a wall-clock backstop.

The ceiling is checked against the process's OS-accounted memory footprint (``phys_footprint``,
what macOS memory pressure sees), not process RSS plus MLX active and cache memory summed: a
buffer loaded with ``mx.load`` appears in both RSS and MLX active memory (verified: a 1 GiB load
moves both by about 1 GiB), so the old summed total double-counted it and would false-abort at
half the real ceiling. ``rss``, ``mlx_active``, and ``mlx_cache`` still ride along in every sample
and abort artifact as diagnostics. A sampling failure (psutil, MLX, the footprint read, or the
artifact write itself) still aborts the process instead of leaving the job running unwatched.
"""

import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx

EXIT_MEMORY = 70
EXIT_WALL = 71

# The one process exit the watchdog makes; tests patch this alias, never os._exit process-wide.
_exit = os._exit

_RUSAGE_INFO_V2 = 2
# Loaded once at import time, not per 0.05 s sample.
_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
if _LIBPROC is not None:
    _LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _LIBPROC.proc_pid_rusage.restype = ctypes.c_int


class _RusageInfoV2(ctypes.Structure):
    """Mirrors macOS's ``rusage_info_v2`` (``proc_info.h``); only ``ri_phys_footprint`` is read."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


def phys_footprint() -> int:
    """The OS-accounted footprint of this process (what macOS memory pressure sees); RSS on other platforms."""
    if _LIBPROC is None:
        import psutil

        return int(psutil.Process().memory_info().rss)
    info = _RusageInfoV2()
    if _LIBPROC.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return int(info.ri_phys_footprint)


def verdict(*, rss: int, ceiling: int, elapsed: float, budget: float) -> str | None:
    """Decide whether to abort: "memory", "wall", or None.

    ``rss`` is whatever single memory number the caller is enforcing the ceiling against; the
    caller (``Watchdog``) feeds it the process's OS-accounted footprint (``phys_footprint``),
    since the verdict itself stays a pure function of one memory number.
    """
    if rss > ceiling:
        return "memory"
    if elapsed > budget:
        return "wall"
    return None


def default_ceiling() -> int:
    """Physical memory minus a 4 GiB reserve for the OS and other processes."""
    import psutil

    return int(psutil.virtual_memory().total) - 4 * 1024**3


class Watchdog:
    """Background sampler that aborts the process with an honest artifact."""

    def __init__(
        self, out_dir: Path, *, ceiling: int, budget: float, interval: float = 0.05
    ) -> None:
        """Configure the ceiling (bytes), wall budget (seconds) and poll interval."""
        self.out_dir, self.ceiling, self.budget, self.interval = out_dir, ceiling, budget, interval
        # Peak OS-accounted footprint seen so far: the verdict's own memory number.
        self.peak_footprint = 0
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

    def reset_peak(self) -> None:
        """Start ``peak_footprint`` over, so a later window's peak is not hidden by an earlier spike."""
        self.peak_footprint = 0

    def stop(self) -> None:
        """Stop sampling. After this returns the watchdog never writes an abort or exits."""
        with self._lock:
            self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _sample(self) -> tuple[str | None, dict[str, float]]:
        elapsed = time.monotonic() - self._start
        sample: dict[str, float] = {
            "footprint": 0,
            "rss": 0,
            "mlx_active": 0,
            "mlx_cache": 0,
            "elapsed": elapsed,
        }
        try:
            import psutil

            footprint = int(phys_footprint())
            sample["footprint"] = footprint
            sample["rss"] = int(psutil.Process().memory_info().rss)
            sample["mlx_active"] = int(mx.get_active_memory())
            sample["mlx_cache"] = int(mx.get_cache_memory())
            self.peak_footprint = max(self.peak_footprint, footprint)
            reason = verdict(
                rss=footprint, ceiling=self.ceiling, elapsed=elapsed, budget=self.budget
            )
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
                            "footprint": sample["footprint"],
                            "peak_footprint": self.peak_footprint,
                            "ceiling": self.ceiling,
                            "elapsed": sample["elapsed"],
                            "budget": self.budget,
                            "rss": sample["rss"],
                            "mlx_active": sample["mlx_active"],
                            "mlx_cache": sample["mlx_cache"],
                        },
                        indent=1,
                    )
                )
            finally:
                _exit(code)  # always exits, even if the artifact write above raised
