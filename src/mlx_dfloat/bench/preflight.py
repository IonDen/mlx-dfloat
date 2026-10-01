"""The launch gate every bench run passes first.

Every probe is one macOS command whose text a pure parser reads; ``check`` is a pure function of
the parsed sample. An unreadable probe parses to None, and ``check`` reports it as a failed gate:
the gate refuses rather than guesses. ``pmset -g therm`` prints no ``CPU_Speed_Limit`` line when
macOS has recorded no limit; that reads as None and passes the speed-limit check.

The busy gate lists heavy processes by pid and executable name only, never their arguments.
Command lines containing a substring from ``MLX_DFLOAT_PREFLIGHT_EXCLUDE`` (comma-separated, empty
by default) are not counted as busy.
"""

import dataclasses
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mlx_dfloat.bench.capped import GIB

_PERCENT = re.compile(r"\t(\d+)%;")
_WATTS = re.compile(r"Wattage\s*=\s*(\d+)W")
_SPEED = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
_CLAMSHELL = re.compile(r'"AppleClamshellState"\s*=\s*(Yes|No)')
_FREE = re.compile(r"free percentage:\s*(\d+)%")
_PS_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(.*)$")
HEAVY_PATTERNS: tuple[str, ...] = (
    "bench_",
    "sweep",
    "calibrat",
    "pytest",
    "mflux",
    "mlx_lm",
    "mlx-guard",
    "generate",
    "verify_checkpoint",
    "verify_remote_group",
    "bench_decode_kernel",
    "verify_image",
    "mlx-dfloat",
)
EXCLUDE_ENV = "MLX_DFLOAT_PREFLIGHT_EXCLUDE"


def preflight_exclude(environ: Mapping[str, str] = os.environ) -> tuple[str, ...]:
    """The substrings ``MLX_DFLOAT_PREFLIGHT_EXCLUDE`` names (comma-separated); empty when unset."""
    return tuple(part.strip() for part in environ.get(EXCLUDE_ENV, "").split(",") if part.strip())


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Preflight:
    """One parsed sample of the machine state before a launch."""

    ac_power: bool | None
    battery_percent: int | None
    charging: str | None
    charger_watts: int | None
    cpu_speed_limit: int | None
    lid_open: bool | None
    free_disk_bytes: int | None
    memory_free_percent: int | None
    busy_processes: tuple[str, ...] | None

    def as_dict(self) -> dict[str, object]:
        """Every field, for the run record."""
        return dataclasses.asdict(self)


def parse_batt(text: str) -> tuple[bool | None, int | None, str | None]:
    """``pmset -g batt``: (on AC or None when unreadable, battery %, charging "yes" / "no" / "full" / None)."""
    if "drawing from" not in text:
        return None, None, None
    ac = "'AC Power'" in text
    match = _PERCENT.search(text)
    pct = int(match.group(1)) if match else None
    if pct is None:
        state = None
    elif "not charging" in text or "discharging" in text:
        state = "no"
    elif "charged" in text:
        state = "full"
    elif "charging" in text:
        state = "yes"
    else:
        state = None
    return ac, pct, state


def parse_ac(text: str) -> int | None:
    """``pmset -g ac``: the adapter wattage, or None (no Wattage line, e.g. "No adapter attached.")."""
    match = _WATTS.search(text)
    return int(match.group(1)) if match else None


def parse_therm(text: str) -> int | None:
    """``pmset -g therm``: the CPU speed limit, or None when none is recorded (or the probe failed)."""
    match = _SPEED.search(text)
    return int(match.group(1)) if match else None


def parse_clamshell(text: str) -> bool | None:
    """``ioreg`` AppleClamshellState: True when the lid is open, False closed, None unreadable."""
    match = _CLAMSHELL.search(text)
    return None if match is None else match.group(1) == "No"


def parse_memory_pressure(text: str) -> int | None:
    """``memory_pressure``: the system-wide free percentage, or None."""
    match = _FREE.search(text)
    return int(match.group(1)) if match else None


def parse_ps(
    text: str,
    *,
    patterns: Sequence[str],
    min_rss_bytes: int = GIB,
    exclude: Sequence[str] = (),
) -> tuple[str, ...]:
    """``ps -Ao pid=,rss=,command=`` (rss in KiB): the matching processes at or above ``min_rss_bytes``.

    Each is listed as ``"<executable name> (pid <pid>)"``; its arguments are never recorded. A
    command line containing any ``exclude`` substring is skipped.
    """
    wanted = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
    out: list[str] = []
    for line in text.splitlines():
        m = _PS_LINE.match(line)
        if not m:
            continue
        rss_bytes, command = int(m.group(2)) * 1024, m.group(3).strip()
        if any(x in command for x in exclude) or not wanted.search(command):
            continue
        if rss_bytes >= min_rss_bytes:
            out.append(f"{Path(command.split()[0]).name} (pid {m.group(1)})")
    return tuple(out)


def check(
    p: Preflight,
    *,
    min_battery: int = 40,
    min_not_charging: int = 50,
    min_free_disk_bytes: int = 20 * GIB,
    min_memory_free_percent: int = 20,
) -> list[str]:
    """The failed gates, by name; empty means go. A None that a gate needs is ``unreadable:<field>``.

    ``not_charging`` (on AC, battery not charging) fires only below ``min_not_charging`` percent:
    macOS Optimized Battery Charging holds a MacBook on AC at 80 % without charging, and a battery
    at half charge or more on AC may run.
    """
    failed: list[str] = []
    pct = p.battery_percent
    if p.ac_power is None:
        failed.append("unreadable:ac_power")
    elif not p.ac_power:
        failed.append("ac_power")
    if pct is not None and pct < min_battery:
        failed.append("battery")
    if pct is not None and p.ac_power and p.charging == "no" and pct < min_not_charging:
        failed.append("not_charging")
    if p.cpu_speed_limit is not None and p.cpu_speed_limit < 100:
        failed.append("cpu_speed_limit")
    if pct is not None:
        if p.lid_open is None:
            failed.append("unreadable:lid_open")
        elif not p.lid_open:
            failed.append("lid")
    if p.free_disk_bytes is None:
        failed.append("unreadable:free_disk_bytes")
    elif p.free_disk_bytes < min_free_disk_bytes:
        failed.append("free_disk")
    if p.memory_free_percent is None:
        failed.append("unreadable:memory_free_percent")
    elif p.memory_free_percent < min_memory_free_percent:
        failed.append("memory_free")
    if p.busy_processes is None:
        failed.append("unreadable:busy_processes")
    elif p.busy_processes:
        failed.append("busy")
    return failed


def _run(*argv: str) -> str:
    try:
        return subprocess.run(list(argv), capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def sample(disk_path: Path = Path("/"), *, run: Callable[..., str] = _run) -> Preflight:
    """Probe the machine once. Never raises; an unreadable probe is None."""
    ac, pct, state = parse_batt(run("pmset", "-g", "batt"))
    try:
        free: int | None = shutil.disk_usage(disk_path).free
    except OSError:
        free = None
    ps_text = run("ps", "-Ao", "pid=,rss=,command=")
    return Preflight(
        ac_power=ac,
        battery_percent=pct,
        charging=state,
        charger_watts=parse_ac(run("pmset", "-g", "ac")),
        cpu_speed_limit=parse_therm(run("pmset", "-g", "therm")),
        lid_open=parse_clamshell(run("ioreg", "-r", "-k", "AppleClamshellState", "-d", "4")),
        free_disk_bytes=free,
        memory_free_percent=parse_memory_pressure(run("memory_pressure")),
        busy_processes=(
            parse_ps(ps_text, patterns=HEAVY_PATTERNS, exclude=preflight_exclude())
            if ps_text
            else None
        ),
    )
