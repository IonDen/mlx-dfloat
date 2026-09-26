"""The unit-tested helpers behind the bench and parity scripts.

The arithmetic (timings, overhead, throughput, the kill line and the cross-check band, the
per-dispatch guard, the calibration ramp's decisions, the resume-key comparison, exit codes) is
pure, so the numbers these scripts act on can be checked
without a GPU. ``provenance`` and ``write_json_atomic`` are the two helpers with side effects:
one reads the machine state a result is recorded against, the other writes a result so a crash
never leaves a half-written file.

This module imports none of the scripts it serves at module level, so any of them may import it.
"""

import json
import platform
import re
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import mlx.core as mx

from mlx_dfloat._memory_caps import install_memory_caps

_REPO = Path(__file__).resolve().parents[1]
_PERCENT = re.compile(r"(\d+)%")


@dataclass(frozen=True, slots=True, kw_only=True)
class Timing:
    """The timed repetitions of one measurement, in seconds."""

    reps: tuple[float, ...]

    @property
    def median(self) -> float:
        """The median repetition."""
        return float(statistics.median(self.reps))

    @property
    def spread(self) -> float:
        """``(max - min) / median``: how far apart the repetitions are, relative to the median."""
        return (max(self.reps) - min(self.reps)) / self.median


def overhead(t_df11: float, t_control: float) -> float:
    """The fraction of time DF11 adds over the control: ``t_df11 / t_control - 1``.

    Raises:
        ValueError: ``t_control`` is zero or negative.
    """
    if t_control <= 0:
        raise ValueError(f"the control time must be positive, got {t_control}")
    return t_df11 / t_control - 1.0


def gbps(bytes_out: int, seconds: float) -> float:
    """Throughput in decimal gigabytes (1e9 bytes) per second."""
    return bytes_out / seconds / 1e9


def kill_equivalent_throughput(
    bytes_per_step: int, t_step: float, kill_line: float = 0.25
) -> float:
    """The decode throughput (bytes/s) at which a step's decode costs exactly ``kill_line`` of it.

    Decoding ``bytes_per_step`` within ``kill_line * t_step`` seconds needs at least
    ``bytes_per_step / (kill_line * t_step)`` bytes per second.
    """
    return bytes_per_step / (kill_line * t_step)


def crosscheck_band(
    decode_seconds: float, bytes_per_step: int, throughput_bps: float
) -> tuple[float, bool]:
    """Compare an in-step decode time with the isolated bench's prediction.

    The prediction is ``bytes_per_step / throughput_bps`` seconds. Returns the ratio of the
    measured time to it and whether that ratio lies in ``[1.0, 2.0]``, both ends inclusive.
    """
    ratio = decode_seconds / (bytes_per_step / throughput_bps)
    return ratio, 1.0 <= ratio <= 2.0


def per_dispatch_guard(bytes_out: int, measured_bps: float, limit_s: float = 0.25) -> None:
    """Refuse a single kernel dispatch projected to run longer than ``limit_s``.

    One long dispatch can trip the macOS GPU watchdog and stall the display, so a whole group is
    decoded in one dispatch only while its output, at the measured throughput, fits the limit.

    Args:
        bytes_out: Bytes the dispatch writes (2 per BF16 element).
        measured_bps: Measured decode throughput, in bytes per second.
        limit_s: Longest acceptable projected dispatch, in seconds.

    Raises:
        RuntimeError: The projected time ``bytes_out / measured_bps`` exceeds ``limit_s``.
    """
    projected = bytes_out / measured_bps
    if projected > limit_s:
        raise RuntimeError(
            f"one dispatch of {bytes_out} bytes is projected at {projected:.3g} s, over the "
            f"{limit_s:.3g} s per-dispatch limit"
        )


def parity_conditions(**checks: bool) -> list[str]:
    """The names of the checks that failed (falsy values), in argument order."""
    return [name for name, ok in checks.items() if not ok]


def bench_exit_code(*, mismatched: int, errors: int) -> int:
    """1 when any bit mismatched, else 2 when anything errored, else 0."""
    if mismatched:
        return 1
    return 2 if errors else 0


def ramp_next_k(k: int, n_launch: int) -> int | None:
    """The next calibration prefix length: ``k`` doubled, capped at ``n_launch``; None when done."""
    if k >= n_launch:
        return None
    return min(2 * k, n_launch)


def ramp_should_stop(
    seconds: float,
    k: int,
    n_launch: int,
    min_seconds: float = 0.01,
    *,
    rate: float | None = None,
    prev_rate: float | None = None,
) -> bool:
    """Stop the ramp once the whole group was dispatched, or a step measured throughput.

    A step shorter than ``min_seconds`` is dominated by launch latency, so its rate is kept only
    as the guard for the next, larger step. A step that long measures throughput unless its
    ``rate`` fell below the previous step's ``prev_rate``: a larger dispatch should never decode at a lower rate,
    so a drop marks a transient slow dispatch, and the ramp goes on to re-measure.
    """
    if k >= n_launch:
        return True
    if seconds < min_seconds:
        return False
    return rate is None or prev_rate is None or rate >= prev_rate


def calibration_rate(step_rates: Sequence[float]) -> float:
    """The calibration rate: the last ramp step's, the longest dispatch and so the one measuring throughput.

    Raises:
        ValueError: The ramp recorded no step.
    """
    if not step_rates:
        raise ValueError("the calibration ramp recorded no step")
    return step_rates[-1]


def time_first_then_min(
    fn: Callable[[], object], *, n: int = 3, clock: Callable[[], float] = time.perf_counter
) -> tuple[float, float]:
    """Run ``fn`` once untimed, then ``n`` times; return (first run, fastest of the ``n``).

    The first run absorbs a one-off cost such as a Metal pipeline compile, and is returned so it
    stays visible. The minimum filters a transient slow dispatch: a spike only ever adds time.
    """
    t0 = clock()
    fn()
    first = clock() - t0
    runs = []
    for _ in range(n):
        t0 = clock()
        fn()
        runs.append(clock() - t0)
    return first, min(runs)


def projected_bytes(positions: Sequence[int], k: int) -> int:
    """Bytes a dispatch of the first ``k`` blocks writes: 2 per element in ``positions[0:k+1]``."""
    return 2 * (int(positions[k]) - int(positions[0]))


def resume_key_diff(stored: object, current: Mapping[str, object]) -> list[str]:
    """The key fields in which a stored bench file differs from this run; empty when it may resume.

    A stored value that is not a key mapping (absent, or from an older or foreign file) never
    matches: ``["key"]``. Otherwise every field present in either key and unequal is named, sorted
    (an empty stored key names every current field).
    """
    if not isinstance(stored, Mapping):
        return ["key"]
    return sorted(f for f in set(stored) | set(current) if stored.get(f) != current.get(f))


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` as JSON so ``path`` holds either the old content or the complete new one.

    The payload is serialized before any file is touched, then written to a sibling ``.tmp`` and
    moved over ``path`` with ``Path.replace`` (an atomic ``rename``).

    Raises:
        TypeError: ``payload`` holds a value JSON cannot represent; ``path`` is left unchanged.
    """
    text = json.dumps(payload, indent=1)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def parse_pmset(text: str) -> dict[str, bool | int | None]:
    """Power state from ``pmset -g batt`` output: on AC or not, and the first battery percentage.

    ``battery_pct`` is None when no percentage is listed (a Mac without a battery).
    """
    match = _PERCENT.search(text)
    return {"ac": "'AC Power'" in text, "battery_pct": int(match.group(1)) if match else None}


def git_state() -> str:
    """The HEAD commit, suffixed ``-dirty`` when ``src/`` or ``scripts/`` has uncommitted changes.

    Returns "unknown" when git cannot be run or this is not a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO), "status", "--porcelain", "--", "src", "scripts"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def _power() -> dict[str, bool | int | None]:
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"ac": None, "battery_pct": None}
    return parse_pmset(out)


def _mflux_version() -> str | None:
    # The installed distribution's version, read without importing mflux (and its torch stack).
    try:
        return metadata.version("mflux")
    except metadata.PackageNotFoundError:
        return None


def provenance() -> dict[str, object]:
    """The machine and code state a bench result was measured on.

    ``cache_limit`` is read by setting a probe limit and restoring the previous one straight away
    (MLX exposes no getter; ``mx.set_cache_limit`` only swaps the value and returns the old one).
    ``memory_caps_gb`` comes from ``install_memory_caps()``, which is idempotent: the bench scripts
    have already installed the same caps. ``power["ac"]`` is None when ``pmset`` cannot run.
    """
    from scripts.verify_checkpoint import source_hash  # lazy: verify_checkpoint imports this module

    cache_memory = int(mx.get_cache_memory())
    cache_limit = int(mx.set_cache_limit(0))
    mx.set_cache_limit(cache_limit)
    return {
        "git": git_state(),
        "source_hash": source_hash(),
        "mlx": mx.__version__,
        "mflux": _mflux_version(),
        "macos": platform.mac_ver()[0],
        "device_info": dict(mx.device_info()),
        "memory_caps_gb": list(install_memory_caps()),
        "cache_limit": cache_limit,
        "cache_memory": cache_memory,
        "power": _power(),
    }
