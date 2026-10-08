"""README fragments rendered from result data, and the marker splice.

Pure rendering: bytes stay ``int`` until a fragment is printed, output is deterministic
(the caption's date is passed in), and every fragment ends with a newline.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mlx_dfloat.bench.capped import GIB
from mlx_dfloat.bench.results import Summary
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux.families import MODELS

_NOT_MEASURED = "not measured"
_NOT_RECORDED = "not recorded"
_STOPPED = "stopped by the watchdog"
# A CAPPED row ran under the wired and memory caps mlx-dfloat installs on a Mac of that size.
_TIER_CAPS_NOTE = "mlx-dfloat caps for the tier"
# A CAPPED row ran under MLX's own defaults for a Mac of that size (no wired limit).
_TIER_DEFAULTS_NOTE = "MLX defaults for the tier"
# The note per ``limits.applied`` path of a generate report ("tier-defaults": runs before the tier caps).
_APPLIED_NOTES = {
    "host-caps": "host caps",
    "tier-caps": _TIER_CAPS_NOTE,
    "tier-defaults": _TIER_DEFAULTS_NOTE,
}


def _model_label(name: str) -> str:
    """The published label of a registered model name; the name itself when it is not registered."""
    return MODELS[name].label if name in MODELS else name


@dataclass(frozen=True, slots=True, kw_only=True)
class TierRow:
    """One row of the tier table."""

    mac_gb: int
    ceiling_bytes: int
    model: str
    df11_bytes: int | None
    watched_peak_bytes: int
    footprint_peak_bytes: int
    mlx_peak_bytes: int | None
    label: str
    status: str
    limits_note: str
    source: str
    stopped_after_s: float | None = None  # a run the watchdog stopped: its peaks are lower bounds


@dataclass(frozen=True, slots=True, kw_only=True)
class ProofRecord:
    """The harness proof: one passing run and one aborted run under the same cap."""

    cap_bytes: int
    pass_size: int
    pass_watched_peak: int
    abort_size: int
    abort_reason: str
    abort_counter: str
    abort_peak: int
    source_dir: str


def _need(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise DFloatFormatError(f"{where} has no {key!r} field")
    return mapping[key]


def _mlx_peak(report: Mapping[str, Any]) -> int:
    """The larger of the watchdog's sampled MLX peak (active + cache) and MLX's exact per-phase peaks.

    The watchdog polls every 0.05 s and can miss a short high point; ``mx.get_peak_memory`` per phase (active only)
    cannot, so whichever is larger is the better lower bound on what MLX held.
    """
    sampled = int(_need(report, "mlx_peak_bytes", "report"))
    phases = report.get("peaks") or {}
    exact = [
        int(v["mlx_peak"]) for v in phases.values() if isinstance(v, Mapping) and "mlx_peak" in v
    ]
    return max([sampled, *exact])


def tier_row_from_generate_report(report: Mapping[str, Any], *, source: str) -> TierRow:
    """Build a tier-table row from an ``mlx-dfloat generate --report`` JSON.

    ``ceiling_bytes`` is ``limits.tier.ceiling_bytes``, the recommended working set minus
    ``reserve_for(tier)``, which the status is judged against. On the host tier (2 GiB reserve) it
    equals the fit budget, and the watchdog's own abort line is higher (RAM minus 4 GiB); on a
    CAPPED tier (1.5 GiB reserve at 16 and 24 GB) it is the watchdog ceiling, and the fit budget
    (``limits.tier.fit_budget_bytes``, 2 GiB reserve) is lower.

    Raises:
        DFloatFormatError: A field is missing, or the report is a harness proof.
    """
    label = _need(report, "label", "report")
    if label == "PROOF":
        raise DFloatFormatError("a PROOF report belongs to the harness proof, not the tier table")
    limits = _need(report, "limits", "report")
    tier = _need(limits, "tier", "limits")
    sizes = _need(report, "sizes", "report")
    ceiling = int(_need(tier, "ceiling_bytes", "limits.tier"))
    watched = int(_need(report, "watched_peak_bytes", "report"))
    return TierRow(
        mac_gb=int(_need(tier, "tier_gb", "limits.tier")),
        ceiling_bytes=ceiling,
        model=str(_need(report, "model", "report")),
        df11_bytes=int(_need(sizes, "compressed", "sizes")) + int(_need(sizes, "extras", "sizes")),
        watched_peak_bytes=watched,
        footprint_peak_bytes=int(_need(report, "footprint_peak_bytes", "report")),
        mlx_peak_bytes=_mlx_peak(report),
        label=str(label),
        status="target" if watched <= ceiling else "over",
        limits_note=_applied_note(_need(limits, "applied", "limits")),
        source=source,
    )


def _applied_note(applied: Any) -> str:
    """The limits note for a report's ``limits.applied`` path.

    Raises:
        DFloatFormatError: The path is not one ``generate`` writes.
    """
    if applied not in _APPLIED_NOTES:
        raise DFloatFormatError(f"limits.applied {applied!r} is not a known limits path")
    return _APPLIED_NOTES[applied]


def _abort_limits_note(label: Any, context: Mapping[str, Any]) -> str:
    """The limits note of an abort row: the host caps off a CAPPED tier, else what ``context.limits`` shows.

    On a CAPPED tier only mlx-dfloat's caps install a wired limit, so a recorded wired limit above
    zero means those caps and zero means MLX's defaults; an artifact without ``limits`` (written
    before the run context carried it) says so.
    """
    if label != "CAPPED":
        return "host caps"
    limits = context.get("limits")
    if not isinstance(limits, Mapping):
        return _NOT_RECORDED
    return (
        _TIER_CAPS_NOTE
        if int(_need(limits, "wired", "context.limits")) > 0
        else _TIER_DEFAULTS_NOTE
    )


def tier_row_from_abort_artifact(artifact: Mapping[str, Any], *, source: str) -> TierRow:
    """Build a tier-table row from the abort artifact of a run the watchdog stopped.

    The tier, label and model come from the artifact's run context (``generate`` hands it to its
    watchdog); the DF11 size was never recorded, and the status says the run was stopped. On a
    CAPPED row the limits note comes from the context's ``limits`` (the MLX limits read back after
    the install): a wired limit above zero is mlx-dfloat's caps for the tier, zero is MLX's
    defaults, and an older artifact without ``limits`` reads "not recorded".

    Raises:
        DFloatFormatError: The artifact has no run context, the context lacks ``model``,
            ``tier_gb`` or ``label``, or the label is PROOF (the harness proof, not a tier row).
    """
    context = artifact.get("context")
    if not isinstance(context, Mapping):
        raise DFloatFormatError("the abort artifact records no run context (model, tier_gb, label)")
    label = _need(context, "label", "abort artifact context")
    if label == "PROOF":
        raise DFloatFormatError("a PROOF abort belongs to the harness proof, not the tier table")
    mlx_peak = artifact.get("peak_mlx")
    return TierRow(
        mac_gb=int(_need(context, "tier_gb", "abort artifact context")),
        ceiling_bytes=int(_need(artifact, "ceiling", "abort artifact")),
        model=str(_need(context, "model", "abort artifact context")),
        df11_bytes=None,
        watched_peak_bytes=int(_need(artifact, "peak_watched", "abort artifact")),
        footprint_peak_bytes=int(_need(artifact, "peak_footprint", "abort artifact")),
        mlx_peak_bytes=None if mlx_peak is None else int(mlx_peak),
        label=str(label),
        status=_STOPPED,
        limits_note=_abort_limits_note(label, context),
        source=source,
        stopped_after_s=float(_need(artifact, "elapsed", "abort artifact")),
    )


def proof_from_files(
    pass_report: Mapping[str, Any],
    abort_artifact: Mapping[str, Any],
    *,
    source_dir: str,
) -> ProofRecord:
    """Pair the passing run with the aborted run of the harness proof.

    The aborted run's size comes from the artifact's own record (``context.height`` and
    ``context.width``, which ``generate`` hands its watchdog), never from a default.

    Raises:
        DFloatFormatError: The pass report is not a PROOF or is not square; the artifact has no
            run context, or its run is not square; the two ran under different caps, for different
            models, or (when both name one) with different seeds.
    """
    if _need(pass_report, "label", "pass report") != "PROOF":
        raise DFloatFormatError("the pass report's label is not PROOF")
    height = _need(pass_report, "height", "pass report")
    if height != _need(pass_report, "width", "pass report"):
        raise DFloatFormatError("the pass report is not square (height != width)")
    cap = int(_need(pass_report, "memory_ceiling_bytes", "pass report"))
    if cap != int(_need(abort_artifact, "ceiling", "abort artifact")):
        raise DFloatFormatError(
            "the proof requires the same cap: pass report and abort artifact differ"
        )
    context = abort_artifact.get("context")
    if not isinstance(context, Mapping):
        raise DFloatFormatError(
            "the abort artifact records no run context (model, height, width): it was written "
            "by a watchdog that was not told the run's size"
        )
    abort_height = _need(context, "height", "abort artifact context")
    if abort_height != _need(context, "width", "abort artifact context"):
        raise DFloatFormatError("the aborted run is not square (height != width)")
    model = _need(pass_report, "model", "pass report")
    if model != _need(context, "model", "abort artifact context"):
        raise DFloatFormatError(
            f"the proof requires the same model: the pass report ran {model!r}, "
            f"the aborted run {context['model']!r}"
        )
    if "seed" in pass_report and "seed" in context and pass_report["seed"] != context["seed"]:
        raise DFloatFormatError(
            f"the proof requires the same seed: the pass report ran {pass_report['seed']!r}, "
            f"the aborted run {context['seed']!r}"
        )
    return ProofRecord(
        cap_bytes=cap,
        pass_size=int(height),
        pass_watched_peak=int(_need(pass_report, "watched_peak_bytes", "pass report")),
        abort_size=int(abort_height),
        abort_reason=str(_need(abort_artifact, "reason", "abort artifact")),
        abort_counter=str(_need(abort_artifact, "verdict_counter", "abort artifact")),
        abort_peak=int(_need(abort_artifact, "peak_watched", "abort artifact")),
        source_dir=source_dir,
    )


def _gib(n: int) -> str:
    return f"{n / GIB:.2f} GiB"


def _gib_or_unrecorded(n: int | None) -> str:
    return _NOT_RECORDED if n is None else _gib(n)


def _pct(x: float | None) -> str:
    return _NOT_MEASURED if x is None else f"{x * 100:+.1f} %"


_TIER_HEADER = (
    "| Mac | Working set − reserve | Model | DF11 size | Peak (watched) | Peak footprint "  # noqa: RUF001
    "| Peak MLX (sampled active + cache, or exact phase peak) | Label | Status | Limits | Result |"
)


def _peaks(r: TierRow) -> tuple[str, str, str]:
    """The three peak cells; a stopped run's are lower bounds, the watched one with the time it ran."""
    watched, footprint = _gib(r.watched_peak_bytes), _gib(r.footprint_peak_bytes)
    mlx = _gib_or_unrecorded(r.mlx_peak_bytes)
    if r.stopped_after_s is None:
        return watched, footprint, mlx
    least = "at least "
    return (
        f"{least}{watched} (stopped after {r.stopped_after_s:.1f} s)",
        least + footprint,
        mlx if r.mlx_peak_bytes is None else least + mlx,
    )


def render_tier_table(rows: Sequence[TierRow]) -> str:
    """Render the tier table: GiB with two decimals, one row per measured tier."""
    lines = [_TIER_HEADER, "|" + "---|" * 11]
    for r in rows:
        watched, footprint, mlx = _peaks(r)
        lines.append(
            f"| {r.mac_gb} GB | {_gib(r.ceiling_bytes)} | {_model_label(r.model)} "
            f"| {_gib_or_unrecorded(r.df11_bytes)} | {watched} "
            f"| {footprint} | {mlx} | {r.label} | {r.status} | {r.limits_note} | `{r.source}` |"
        )
    return "\n".join(lines) + "\n"


def scenario_title(key: str) -> str:
    """``flux1-dev-1024`` as ``FLUX.1-dev, 1024²``; any other key unchanged."""
    m = re.fullmatch(r"flux1-(.+)-(\d+)", key)
    if m is None:
        return key
    return f"{_model_label(m.group(1))}, {m.group(2)}²"


def render_overhead_block(
    summaries: Mapping[str, Summary],
    *,
    caption: str,
    reproducers: Mapping[str, str],
    cache_limit_note: str,
    preflight_skipped: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Render each scenario's overhead line and recorded command, then the caption and cache note.

    ``preflight_skipped`` maps a scenario whose run skipped the launch check to the gates that
    failed; its command line says so.
    """
    skipped = preflight_skipped or {}
    lines = []
    for key, s in summaries.items():
        per_block = _pct(s.overhead.get("per-block"))
        depth2 = _pct(s.overhead.get("depth2"))
        cost = _NOT_MEASURED if s.eval_cost_s is None else f"{s.eval_cost_s:.2f} s/step"
        q8 = _NOT_MEASURED if s.q8_ratio is None else f"{s.q8_ratio:.2f}×"  # noqa: RUF001
        lines.append(
            f"{scenario_title(key)}, per-block evaluation: {per_block} (depth-2: {depth2}); "
            f"eval policy cost {cost}; DF11 (per-block) over the mflux q8 step "
            f"(one eval per step, same cache limit): {q8}"
        )
        lines.append("")
        command = f"Command: `{reproducers.get(key, '')}`"
        if key in skipped:
            command += f" (preflight skipped: {', '.join(skipped[key]) or 'no gate failed'})"
        lines += [command, ""]
    lines += [caption, "", cache_limit_note]
    return "\n".join(lines) + "\n"


def render_proof_paragraph(proof: ProofRecord) -> str:
    """Render the harness-proof paragraph."""
    return (
        f"Harness proof: under one {_gib(proof.cap_bytes)} cap, a {proof.pass_size}² run passed "
        f"with a watched peak of {_gib(proof.pass_watched_peak)}, and a {proof.abort_size}² run was "
        f"stopped by the watchdog ({proof.abort_reason}, counter {proof.abort_counter}) at "
        f"{_gib(proof.abort_peak)}. Records: `{proof.source_dir}`.\n"
    )


def splice(text: str, block: str, fragment: str) -> str:
    """Replace what sits between the ``bench:<block>`` markers with ``fragment``.

    Raises:
        DFloatFormatError: Not exactly one start marker and one end marker, start first.
    """
    start, end = f"<!-- bench:{block} -->", f"<!-- /bench:{block} -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise DFloatFormatError(f"expected exactly one {start} before one {end}")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{fragment}{end}{tail}"


def caption(provenance: Mapping[str, Any], *, date: str) -> str:
    """One-line hardware and version caption; a ``-dirty`` git suffix is kept."""
    info = provenance["device_info"]
    git = str(provenance["git"])
    git = git[:7] + ("-dirty" if git.endswith("-dirty") else "")
    return (
        f"{info['device_name']}, {round(info['memory_size'] / GIB)} GB, macOS {provenance['macos']}, "
        f"mlx {provenance['mlx']}, mflux {provenance['mflux']}, git {git}, {date}"
    )
