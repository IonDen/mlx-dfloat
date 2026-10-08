"""Process-level watchdog for heavy scripts: a memory ceiling plus a wall-clock backstop.

The ceiling is checked against the larger of the process's OS-accounted memory footprint
(``phys_footprint``, what macOS memory pressure sees) and MLX active + cache memory, never a sum
of RSS and MLX: a buffer loaded with ``mx.load`` appears in both RSS and MLX active memory
(verified: a 1 GiB load moves both by about 1 GiB), so a sum double-counts it and would
false-abort at half the real ceiling, while the maximum still catches an overrun held in MLX's
cache pool. The abort artifact names the counter that tripped (``verdict_counter``) and records
the peaks of the footprint, of MLX active + cache, and of the watched maximum. ``rss``,
``mlx_active``, and ``mlx_cache`` ride along in every sample and abort artifact as diagnostics. A
sampling failure (psutil, MLX, the footprint read, or the artifact write itself) still aborts the
process instead of leaving the job running unwatched; its artifact names no counter
(``verdict_counter: "none"``, ``verdict_memory: null``). A caller may pass a ``context`` mapping
(``generate`` passes its model, size, seed and steps); it is written verbatim under ``"context"``
in the abort artifact, so the artifact names the run it stopped. ``live_context`` maps keys to
readers called when the abort fires (``generate`` reads the phase open at that moment); their
values join ``"context"``. Without either there is no ``"context"`` key. ``peak_mlx`` is None
until a sample has read MLX's counters.
"""

import ctypes
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_dfloat.errors import DFloatDependencyError

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


def _psutil() -> Any:
    """The ``psutil`` module, or a package-rooted error naming it (it ships in the mflux extra)."""
    try:
        import psutil
    except ImportError as exc:
        raise DFloatDependencyError(
            "the watchdog needs psutil: install mlx-dfloat[mflux] or `pip install psutil`"
        ) from exc
    return psutil


def phys_footprint() -> int:
    """The OS-accounted footprint of this process (what macOS memory pressure sees); RSS on other platforms."""
    if _LIBPROC is None:
        return int(_psutil().Process().memory_info().rss)
    info = _RusageInfoV2()
    if _LIBPROC.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V2, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return int(info.ri_phys_footprint)


def watched_memory(*, footprint: int, mlx_active: int, mlx_cache: int) -> tuple[int, str]:
    """The number the ceiling is enforced against and which counter produced it.

    The number is max(OS footprint, MLX active + cache). A maximum cannot double count the way a
    sum of RSS and MLX active did (an ``mx.load``ed array lands in both), and it catches an
    overrun that sits in MLX's cache pool, which the footprint alone reports late. A tie names
    the footprint.
    """
    mlx = mlx_active + mlx_cache
    return (footprint, "footprint") if footprint >= mlx else (mlx, "mlx")


def verdict(*, memory: int, ceiling: int, elapsed: float, budget: float) -> str | None:
    """Decide whether to abort: "memory", "wall", or None. ``memory`` is ``watched_memory``'s number."""
    if memory > ceiling:
        return "memory"
    if elapsed > budget:
        return "wall"
    return None


def default_ceiling() -> int:
    """Physical memory minus a 4 GiB reserve for the OS and other processes.

    Raises:
        DFloatDependencyError: ``psutil`` is not installed.
    """
    return int(_psutil().virtual_memory().total) - 4 * 1024**3


class Watchdog:
    """Background sampler that aborts the process with an honest artifact."""

    def __init__(
        self,
        out_dir: Path,
        *,
        ceiling: int,
        budget: float,
        interval: float = 0.05,
        context: Mapping[str, Any] | None = None,
        live_context: Mapping[str, Callable[[], Any]] | None = None,
    ) -> None:
        """Configure the ceiling (bytes), wall budget (seconds), poll interval and run context.

        ``context`` (JSON-serialisable) is written verbatim under ``"context"`` in the abort
        artifact; None writes no ``"context"`` key. Each ``live_context`` reader is called when
        the abort fires and its value joins ``"context"`` under its key (a reader that raises
        records ``"unreadable: <exception name>"``, and the artifact is still written).

        Raises:
            DFloatDependencyError: ``psutil`` (the RSS diagnostic) is not installed; refused here,
                before any sampling, so a missing module never reads as a sampling failure.
        """
        self._psutil = _psutil()
        self.out_dir, self.ceiling, self.budget, self.interval = out_dir, ceiling, budget, interval
        self.context = None if context is None else dict(context)
        self.live_context = None if live_context is None else dict(live_context)
        # Peaks seen so far: the OS footprint, MLX active + cache (None until a sample read MLX's
        # counters), and the watched maximum the ceiling is enforced on. Read and written under
        # ``_lock``, so a reset cannot interleave with a sample's update.
        self.peak_footprint = 0
        self.peak_mlx: int | None = None
        self.peak_watched = 0
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
        """Start all three peaks over, so a later window's peak is not hidden by an earlier high point."""
        with self._lock:
            self.peak_footprint = 0
            self.peak_mlx = None
            self.peak_watched = 0

    def stop(self) -> None:
        """Stop sampling. After this returns the watchdog never writes an abort or exits."""
        with self._lock:
            self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def _sample(self) -> tuple[str | None, dict[str, float | str | None]]:
        elapsed = time.monotonic() - self._start
        # The verdict fields stay "none" / None until a number was compared with the ceiling, so a
        # sample error's artifact does not read as a footprint verdict.
        sample: dict[str, float | str | None] = {
            "footprint": 0,
            "rss": 0,
            "mlx_active": 0,
            "mlx_cache": 0,
            "elapsed": elapsed,
            "verdict_memory": None,
            "verdict_counter": "none",
        }
        try:
            footprint = int(phys_footprint())
            sample["footprint"] = footprint
            sample["rss"] = int(self._psutil.Process().memory_info().rss)
            active = int(mx.get_active_memory())
            cache = int(mx.get_cache_memory())
            sample["mlx_active"] = active
            sample["mlx_cache"] = cache
            memory, counter = watched_memory(
                footprint=footprint, mlx_active=active, mlx_cache=cache
            )
            sample["verdict_memory"] = memory
            sample["verdict_counter"] = counter
            with self._lock:
                self.peak_footprint = max(self.peak_footprint, footprint)
                self.peak_mlx = max(self.peak_mlx or 0, active + cache)
                self.peak_watched = max(self.peak_watched, memory)
            reason = verdict(
                memory=memory, ceiling=self.ceiling, elapsed=elapsed, budget=self.budget
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

    def _read_live(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, read in (self.live_context or {}).items():
            try:
                values[key] = read()
            except Exception as exc:  # a broken reader must not cost the run its abort artifact
                values[key] = f"unreadable: {type(exc).__name__}"
        return values

    def _fire(self, reason: str, sample: dict[str, float | str | None]) -> None:
        with self._lock:
            if self._stop.is_set():
                return  # the caller already stopped us and recorded its own result
            code = EXIT_WALL if reason == "wall" else EXIT_MEMORY
            try:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                artifact: dict[str, Any] = {
                    "reason": reason,
                    "footprint": sample["footprint"],
                    "peak_footprint": self.peak_footprint,
                    "ceiling": self.ceiling,
                    "elapsed": sample["elapsed"],
                    "budget": self.budget,
                    "rss": sample["rss"],
                    "mlx_active": sample["mlx_active"],
                    "mlx_cache": sample["mlx_cache"],
                    "verdict_memory": sample.get("verdict_memory"),
                    "verdict_counter": sample.get("verdict_counter", "none"),
                    "peak_watched": self.peak_watched,
                    "peak_mlx": self.peak_mlx,
                }
                if self.context is not None or self.live_context is not None:
                    artifact["context"] = {**(self.context or {}), **self._read_live()}
                (self.out_dir / "abort.json").write_text(json.dumps(artifact, indent=1))
            finally:
                _exit(code)  # always exits, even if the artifact write above raised
