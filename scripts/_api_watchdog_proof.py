"""The pure parts of the API watchdog's hardware proof: per-phase ceilings, unit verdicts and resume.

Standard library only: ``python -m scripts.prove_api_watchdog --help`` and the orchestrator import this
without loading MLX or mflux.

Units: U0 is the baseline generate, which also samples the watched memory per phase. U1-U4 each stop a
call in one phase (encode, set load, denoise, VAE decode) by a raise at a checkpoint; U5 sets the VAE
ceiling 64 MiB above its stop line, so the process ends by escalation (exit 70). U6 stops a call in set
load, reads the watched memory once the stopped call has cleaned up (it must be back under the stop
line, so the set and encoder pages were released), and then runs the baseline call again in the same
process, with its stop line 1 GiB over the baseline's overall peak: the weights reload and the image
completes under that line.

The proof must end with exit 0 (every unit passes) before the release is tagged.
"""

import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

UNITS = ("U0", "U1", "U2", "U3", "U4", "U5", "U6")
PHASE_OF = {
    "U1": "encode",
    "U2": "set_load",
    "U3": "denoise",
    "U4": "vae",
    "U5": "vae",
    "U6": "set_load",
}
"""The phase each stopping unit targets."""

_SET_LOAD_SITES = frozenset({"set_load_start", "set_group"})
_SITES = {
    "U1": frozenset({"encoder_loaded", "prompt_encoded"}),
    "U2": _SET_LOAD_SITES,
    "U3": frozenset({"block_start", "block_end"}),
    "U4": frozenset({"call_end"}),
    "U6": _SET_LOAD_SITES,
}
"""The checkpoints a unit's raise may stop at; U5 passes only by an exit."""

HEADROOM = 410 * 1024**2
"""Bytes kept between a line and the peaks on either side of it: at least 0.4 GiB, since the committed
records of one recipe differ by about 0.3 GiB between runs."""

RETRY_HEADROOM = 1024**3
"""Bytes between U6's retry line and the baseline's overall peak: 1 GiB, wider than the run-to-run spread
of the call-wide peak (up to 0.44 GiB between records), still under the stopped call's set groups (about
2.8 GiB), so a retained set trips it."""

_ORDER = ("encode", "set_load", "denoise", "vae")
_BOUNDARY = ("footprint_start", "mlx_peak", "footprint_end")
_EXIT_MEMORY = 70


def _whole(record: Mapping[str, int]) -> int:
    """Everything seen in or just after a phase: its boundary readings and its sampled peaks."""
    keys = (*_BOUNDARY, "watched_peak", "watched_after")
    return max(record.get(key, 0) for key in keys)


def _own(phase: str, record: Mapping[str, int]) -> int:
    """The phase's own peak.

    Set load leaves out its start and sampled peak: in the baseline it opens on the text encoder's
    footprint, which a pre-encoded, settled unit no longer holds.
    """
    if phase == "set_load":
        return max(record.get("mlx_peak", 0), record.get("footprint_end", 0))
    return max(record.get(key, 0) for key in (*_BOUNDARY, "watched_peak"))


def ceilings_for(
    peaks: Mapping[str, Any],
    *,
    stop_margin: int,
    headroom: int = HEADROOM,
    settled: int | None = None,
) -> dict[str, tuple[int, int] | None]:
    """Per phase, the ``(ceiling, stop_margin)`` that stops a call inside that phase, or None.

    ``peaks`` is a baseline's ``model.report()["peaks"]``, optionally with ``watched_peak`` (the sampled
    maximum of footprint and MLX active + cache while the phase ran) and ``watched_after`` (the same
    between its end and the next phase) per phase; other entries (the ``label``) are ignored.

    ``hi`` is the phase's own peak; ``lo`` is the highest memory the stopping unit holds before the phase:
    the call's start footprint for encode; for set load ``settled`` (the unit's reading after its
    pre-encode settled), else the encode's start footprint as a planning value; for denoise and the VAE
    decode everything seen in the earlier phases the pre-encoded unit runs (set load, then denoise).

    A raise is placed when the margin exceeds twice the headroom: the stop line sits half a margin under
    ``hi``, the ceiling half a margin over it, so the peak crosses the line and stays under the ceiling by
    more than the headroom. Set load is the exception: it has a checkpoint before every group read and then
    a tail with none (the last group and the non-block decode), so its stop line sits mid-climb, halfway
    between ``lo`` and ``hi``, with the ceiling one margin over it; the climb crosses the line with groups
    still to read, and the raise comes at the next group's checkpoint. A smaller margin cannot hold the peak
    under the ceiling, so the line is placed for an exit: the stop line and the ceiling are centred between
    ``lo`` and ``hi``. Either way the stop line must clear ``lo`` by more than the headroom (and an exit's
    ceiling must stay as far under ``hi``), else the phase is not isolable (None). None too when a needed
    record is missing.
    """
    records = {name: peaks[name] for name in _ORDER if isinstance(peaks.get(name), Mapping)}
    ceilings: dict[str, tuple[int, int] | None] = {}
    for phase in _ORDER:
        ceilings[phase] = None
        lo = _low(phase, records, settled)
        if phase not in records or lo is None:
            continue
        hi = _own(phase, records[phase])
        if stop_margin > 2 * headroom:
            stop = (lo + hi) // 2 if phase == "set_load" else hi - stop_margin // 2
            fits = stop > lo + headroom
        else:
            # Centred: the ceiling sits at least as far under hi as the stop line sits over lo.
            stop = (lo + hi - stop_margin) // 2
            fits = stop > lo + headroom
        if fits:
            ceilings[phase] = (stop + stop_margin, stop_margin)
    return ceilings


def retry_line(
    peaks: Mapping[str, Any], *, stop_margin: int, headroom: int = RETRY_HEADROOM
) -> tuple[int, int] | None:
    """The ``(ceiling, stop_margin)`` for U6's second call, or None when ``peaks`` holds no phase record.

    The stop line sits ``headroom`` (1 GiB by default) over the baseline's overall watched peak (every boundary
    reading and sampled field of every phase), the ceiling one margin over it. A second call that still holds
    the stopped call's memory when it reaches its decode peak climbs past that line and trips, so U6 fails.
    """
    records = [peaks[name] for name in _ORDER if isinstance(peaks.get(name), Mapping)]
    if not records:
        return None
    stop = max(_whole(record) for record in records) + headroom
    return stop + stop_margin, stop_margin


def _low(phase: str, records: Mapping[str, Mapping[str, int]], settled: int | None) -> int | None:
    if phase == "set_load" and settled is not None:
        return settled
    if phase in ("encode", "set_load"):
        encode = records.get("encode")
        return None if encode is None else encode["footprint_start"]
    earlier = ("set_load",) if phase == "denoise" else ("set_load", "denoise")
    if any(p not in records for p in earlier):
        return None
    return max(_whole(records[p]) for p in earlier)


class PhasePeaks:
    """The watched memory's maximum per phase, and in the stretch after each phase before the next opens."""

    def __init__(self) -> None:
        """Start with no samples."""
        self.peaks: dict[str, dict[str, int]] = {}
        self._last: str | None = None

    def add(self, phase: str | None, watched: int) -> None:
        """Count one sample taken while ``phase`` was open (None between phases; before any phase it is dropped)."""
        if phase is not None:
            self._last, key = phase, "watched_peak"
        elif self._last is not None:
            phase, key = self._last, "watched_after"
        else:
            return
        record = self.peaks.setdefault(phase, {})
        record[key] = max(record.get(key, 0), watched)

    def into(self, peaks: Mapping[str, Any]) -> dict[str, Any]:
        """``peaks`` with each phase record joined by its sampled fields (a copy)."""
        merged = {k: dict(v) if isinstance(v, Mapping) else v for k, v in peaks.items()}
        for phase, fields in self.peaks.items():
            if isinstance(merged.get(phase), dict):
                merged[phase].update(fields)
        return merged


class PhaseSampler:
    """Samples ``read()`` every ``interval`` seconds on a thread, charging each sample to ``phase()``."""

    def __init__(
        self, read: Callable[[], int], phase: Callable[[], str | None], *, interval: float = 0.05
    ) -> None:
        """Bind the watched-memory reader and the open-phase reader; nothing runs until ``start``."""
        self.sampled = PhasePeaks()
        self._read, self._phase, self._interval = read, phase, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="proof-phase-sampler")

    def sample(self) -> None:
        """Take one sample now."""
        self.sampled.add(self._phase(), self._read())

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()

    def start(self) -> "PhaseSampler":
        """Start sampling."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling and take a last sample."""
        self._stop.set()
        self._thread.join(timeout=1)
        self.sample()


def unit_verdict(unit: str, outcome: Mapping[str, Any]) -> str:
    """``"pass"``, ``"fail"`` or ``"not isolable"`` for one unit's finished record.

    ``outcome`` holds the unit process's ``exit_code``, its own ``result`` (None if it wrote none) and the
    abort ``artifact`` found in its record folder (None if none). U0 passes when it completed and saved its
    image. U1-U4 pass only on a memory-reason ``DFloatWatchdogError`` raised in the unit's phase at one of
    its sites (U4 at ``call_end``: the finished image is discarded). U5 passes only on exit 70 with
    escalation ``"ceiling"`` in the VAE decode. U6 passes on U2's raise, then ``watched_after_abort`` (the
    watched memory once the stopped call cleaned up) under its own stop line, then a completed retry. A
    grace exit is a fail: after a memory trip it means the sampler died. A unit whose ceiling could not be
    derived is ``"not isolable"``, which is never a pass.

    Raises:
        ValueError: ``unit`` is not one of U0-U6.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; the units are {', '.join(UNITS)}")
    result = outcome.get("result") or {}
    code = outcome.get("exit_code")
    if unit == "U0":
        done = code == 0 and result.get("completed") is True and result.get("image_saved") is True
        return "pass" if done else "fail"
    if "ceiling" in result and result["ceiling"] is None:
        return "not isolable"
    phase = PHASE_OF[unit]
    if unit in _SITES:
        raised = result.get("raised") or {}
        stopped = (
            code == 0
            and raised.get("type") == "DFloatWatchdogError"
            and raised.get("reason") == "memory"
            and raised.get("phase") == phase
            and raised.get("site") in _SITES[unit]
        )
        if unit == "U6":
            stopped = (
                stopped
                and _released(result)
                and (result.get("retry") or {}).get("completed") is True
            )
        return "pass" if stopped else "fail"
    artifact = outcome.get("artifact") or {}
    ended_by_ceiling = (
        code == _EXIT_MEMORY
        and artifact.get("reason") == "memory"
        and (artifact.get("context") or {}).get("phase") == phase
        and artifact.get("escalation") == "ceiling"
    )
    return "pass" if ended_by_ceiling else "fail"


def _released(result: Mapping[str, Any]) -> bool:
    """Whether the watched memory after the stopped call's cleanup is under the stop line the unit armed."""
    after = result.get("watched_after_abort")
    line = result.get("ceiling")
    if type(after) is not int or not isinstance(line, list) or len(line) != 2:
        return False
    ceiling, margin = line
    return after < ceiling - margin


def pending_units(out_dir: Path) -> list[str]:
    """The units in order whose finished record ``<out_dir>/<unit>.json`` does not exist yet.

    The record is written atomically when a unit ends, so its presence means the unit finished; a
    leftover temporary file or a unit's own outcome file does not count.
    """
    return [unit for unit in UNITS if not (out_dir / f"{unit}.json").is_file()]
