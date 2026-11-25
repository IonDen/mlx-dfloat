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

import contextlib
import ctypes
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple

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


class Counters(NamedTuple):
    """One reading of the four numbers a sample records, in bytes."""

    footprint: int
    rss: int
    mlx_active: int
    mlx_cache: int


def read_counters() -> Counters:
    """Read this process's footprint, RSS, and MLX active and cache memory.

    Raises:
        DFloatDependencyError: ``psutil`` is not installed.
    """
    return Counters(
        footprint=int(phys_footprint()),
        rss=int(_psutil().Process().memory_info().rss),
        mlx_active=int(mx.get_active_memory()),
        mlx_cache=int(mx.get_cache_memory()),
    )


def watched_memory(*, footprint: int, mlx_active: int, mlx_cache: int) -> tuple[int, str]:
    """The number the ceiling is enforced against and which counter produced it.

    The number is max(OS footprint, MLX active + cache). A maximum cannot double count the way a
    sum of RSS and MLX active did (an ``mx.load``ed array lands in both), and it catches an
    overrun that sits in MLX's cache pool, which the footprint alone reports late. A tie names
    the footprint.
    """
    mlx = mlx_active + mlx_cache
    return (footprint, "footprint") if footprint >= mlx else (mlx, "mlx")


def verdict(*, memory: int, ceiling: int, elapsed: float, budget: float | None) -> str | None:
    """Decide whether to abort: "memory", "wall", or None. ``memory`` is ``watched_memory``'s number.

    A ``budget`` of None has no wall limit.
    """
    if memory > ceiling:
        return "memory"
    if budget is not None and elapsed > budget:
        return "wall"
    return None


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a private temp file in the same folder, then rename it over ``path``.

    The temp file is created 0600 by ``tempfile.mkstemp``, and the rename replaces whatever sits at ``path``
    (a symlink included) without following it, so the write never reaches another file. The temp name starts
    with a dot and ends in ``.tmp``, so it never matches the record's own name pattern.

    Raises:
        OSError: the folder cannot be created, or the write or the rename failed (the temp file is removed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        Path(temp).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temp).unlink()
        raise


def default_ceiling() -> int:
    """Physical memory minus a 4 GiB reserve for the OS and other processes.

    Raises:
        DFloatDependencyError: ``psutil`` is not installed.
    """
    return int(_psutil().virtual_memory().total) - 4 * 1024**3


class Watchdog:
    """Background sampler that aborts the process with an honest artifact.

    In raise mode a trip and an escalation can land on the same sample: one jump past both the
    stop line and the ceiling (or a ``set_ceiling`` below the current number) trips and exits 70
    at once, so no checkpoint gets to raise.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        ceiling: int,
        budget: float | None,
        interval: float = 0.05,
        context: Mapping[str, Any] | None = None,
        live_context: Mapping[str, Callable[[], Any]] | None = None,
        counters: Callable[[], Counters] | None = None,
        exit_process: Callable[[int], object] | None = None,
        artifact_name: str = "abort.json",
        mode: Literal["exit", "raise"] = "exit",
        stop_margin: int = 0,
        grace: float = 10.0,
    ) -> None:
        """Configure the ceiling (bytes), wall budget (seconds), poll interval and run context.

        A ``budget`` of None has no wall limit and is recorded as ``null``.

        ``mode="exit"`` (the command's) ends the process at the first verdict. ``mode="raise"``
        trips instead at the stop line ``ceiling - stop_margin``: it writes the record, sets
        ``trip``, ``artifact`` and ``tripped``, and keeps sampling. It then ends the process when
        the watched number reaches ``ceiling`` (always 70: memory ends it, even after a wall
        trip), with one line on stderr naming the record. ``grace`` is a second backstop for the
        trips where no number holds the line: a wall trip, or any trip whose latest sample failed.
        For those, the process also ends when ``consume()`` has not been called within ``grace``
        seconds (the trip's code: 71 after a wall trip, 70 otherwise). The seconds count from the
        trip, except after a memory trip: there they count from the first failed sample, and start
        over when a later sample reads memory again. A memory trip whose samples still read memory
        never ends by grace, however long the stretch to the next checkpoint takes; it ends only at
        ``ceiling``.

        ``counters`` reads the numbers each sample checks; None means ``read_counters``.
        ``exit_process`` ends the process with the abort's code; None means the module's
        ``_exit`` alias, looked up when the abort fires. The artifact is written to
        ``out_dir / artifact_name``.

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
        self._counters, self._exit_process = counters, exit_process
        self.artifact_name = artifact_name
        self.mode, self.stop_margin, self.grace = mode, stop_margin, grace
        # Raise mode: set once, under the lock, when the stop line is first crossed.
        self.tripped = threading.Event()
        self.trip: dict[str, Any] | None = None
        self.artifact: Path | None = None
        self._trip_time: float | None = None
        # Raise mode, after a trip: when the current run of failed samples began (None while live).
        self._failed_since: float | None = None
        self._consumed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="mlx-dfloat-watchdog")
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
            footprint, rss, active, cache = (self._counters or read_counters)()
            sample["footprint"] = footprint
            sample["rss"] = rss
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
                memory=memory,
                ceiling=self.ceiling - self.stop_margin,
                elapsed=elapsed,
                budget=self.budget,
            )
        except Exception:  # a dead sampler must still abort, not run the job unwatched
            reason = "sample_error"
        return reason, sample

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if self.mode == "exit":
                reason, sample = self._sample()
                if reason is not None:
                    self._fire(reason, sample)
                    return
                continue
            try:
                ended = self._raise_step()
            except Exception as exc:  # a dead sampler must never leave the call running unwatched
                ended = self._fail_closed(exc)
            if ended:
                return

    def _raise_step(self) -> bool:
        """One raise-mode sample: trip at the stop line, then the backstop. True once the process ended."""
        reason, sample = self._sample()
        if self.trip is None and reason is not None:
            self._trip(reason, sample)
        return self.trip is not None and self._backstop(sample)

    def _fail_closed(self, exc: Exception) -> bool:
        """Treat a raise-mode step that raised as a failed sample, after a trip.

        Without a trip yet, a ``sample_error`` trip is set here, with no record written (the record or
        its write may be what raised), so the next checkpoint raises. The sample then counts as failed,
        so the failed-sample grace ends the process (70) if the samples keep failing and no checkpoint
        takes the trip. True once the process ended.
        """
        failed: dict[str, float | str | None] = {
            "footprint": 0,
            "rss": 0,
            "mlx_active": 0,
            "mlx_cache": 0,
            "elapsed": time.monotonic() - self._start,
            "verdict_memory": None,
            "verdict_counter": "none",
        }
        with self._lock:
            if self._stop.is_set():
                return True  # stopped: the caller owns the outcome
            if self.trip is None:
                self.trip = {
                    "reason": "sample_error",
                    **{
                        key: failed[key] for key in ("elapsed", "verdict_memory", "verdict_counter")
                    },
                    "ceiling": self.ceiling,
                    "budget": self.budget,
                    "stop_line": self.ceiling - self.stop_margin,
                    "mode": "raise",
                    "escalation": None,
                    "checkpoint_latency_s": None,
                    "loop_error": f"{type(exc).__name__}: {exc}",
                }
                self._trip_time = time.monotonic()
                self.tripped.set()
        try:
            return self._backstop(failed)
        except Exception:  # the next step tries again; the loop itself never dies
            return False

    def consume(self) -> None:
        """Stop the grace clock: a checkpoint has taken the trip. A no-op before a trip.

        Records the time from the trip to this call as ``trip["checkpoint_latency_s"]``. The
        ceiling check stays on until ``stop()``.
        """
        with self._lock:
            if self.trip is None or self._trip_time is None or self._consumed:
                return
            self._consumed = True
            self.trip["checkpoint_latency_s"] = time.monotonic() - self._trip_time

    def set_ceiling(self, ceiling: int) -> int:
        """Replace the ceiling (bytes), under the lock; returns the previous one."""
        with self._lock:
            previous, self.ceiling = self.ceiling, ceiling
        return previous

    def _record(self, reason: str, sample: dict[str, float | str | None]) -> dict[str, Any]:
        """The abort record (exit mode's whole artifact). Called with ``_lock`` held."""
        record: dict[str, Any] = {
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
            record["context"] = {**(self.context or {}), **self._read_live()}
        return record

    def _write(self, record: dict[str, Any]) -> Path | None:
        """Write ``record`` to the artifact path; on failure note it in the record and return None."""
        try:
            path = self.out_dir / self.artifact_name
            _write_private(path, json.dumps(record, indent=1))
        except Exception as exc:  # a failed write never costs the trip or the exit
            record["artifact_error"] = f"{type(exc).__name__}: {exc}"
            return None
        return path

    def _trip(self, reason: str, sample: dict[str, float | str | None]) -> None:
        with self._lock:
            if self._stop.is_set():
                return  # the caller already stopped us
            record = self._record(reason, sample)
            record.update(
                stop_line=self.ceiling - self.stop_margin,
                mode="raise",
                escalation=None,
                checkpoint_latency_s=None,
            )
            self.artifact = self._write(record)
            self.trip, self._trip_time = record, time.monotonic()
            self.tripped.set()

    def _backstop(self, sample: dict[str, float | str | None]) -> bool:
        """After a trip: escalate at the ceiling, or past grace with no checkpoint. True if ended.

        Grace applies only to a wall trip or when this sample failed: while the samples carry a
        number, a memory trip ends the process at the ceiling and nowhere else. The grace clock
        starts at the trip, except after a memory trip, where it starts at the first failed sample
        and resets when a sample reads a number again.
        """
        memory = sample["verdict_memory"]  # None after a sample error: only grace applies
        if isinstance(memory, (int, float)) and memory >= self.ceiling:
            return self._escalate("ceiling")
        now = time.monotonic()
        if memory is not None:
            self._failed_since = None  # the sampler recovered: the failed-sample clock resets
        elif self._failed_since is None:
            self._failed_since = now
        reason = None if self.trip is None else self.trip["reason"]
        graced = memory is None or reason == "wall"
        # After a memory trip the clock bounds how long a dead sampler runs unwatched, so it starts at
        # the first failed sample; after a wall or sample-error trip it starts at the trip.
        since = self._failed_since if reason == "memory" else self._trip_time
        if graced and not self._consumed and since is not None and now - since > self.grace:
            return self._escalate("grace")
        return False

    def _escalate(self, escalation: str) -> bool:
        with self._lock:
            if self._stop.is_set() or self.trip is None:
                return True  # stopped: the caller owns the outcome
            if escalation == "grace" and self._consumed:
                return False  # a checkpoint took the trip while this sample was read
            record = {k: v for k, v in self.trip.items() if k != "artifact_error"}
            record["escalation"] = escalation
            # Memory ends the process at the ceiling, whatever tripped first; grace keeps the trip's code.
            wall = escalation == "grace" and record["reason"] == "wall"
            try:
                path = self._write(record)
                self.trip = record
                if path is not None:
                    self.artifact = path
                    where = f"record: {path}"
                else:
                    carried = json.dumps(record, default=repr)
                    where = f"record not written ({record['artifact_error']}): {carried}"
                phase = (record.get("context") or {}).get("phase")
                # The line names what ended the process: memory at the ceiling ends it whatever tripped first.
                ended_by = (
                    f"memory ceiling after a {record['reason']} trip"
                    if escalation == "ceiling" and record["reason"] != "memory"
                    else record["reason"]
                )
                print(
                    f"mlx-dfloat: the memory watchdog ended the process ({ended_by}, phase {phase}); {where}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                (self._exit_process or _exit)(EXIT_WALL if wall else EXIT_MEMORY)
        return True

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
                artifact = self._record(reason, sample)
                _write_private(self.out_dir / self.artifact_name, json.dumps(artifact, indent=1))
            finally:
                # Always exits, even if the artifact write above raised.
                (self._exit_process or _exit)(code)
