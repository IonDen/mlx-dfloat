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

_MODEL_NAMES = {"schnell": "FLUX.1-schnell", "dev": "FLUX.1-dev", "krea-dev": "FLUX.1-Krea-dev"}
_NOT_MEASURED = "not measured"


@dataclass(frozen=True, slots=True, kw_only=True)
class TierRow:
    """One row of the tier table."""

    mac_gb: int
    ceiling_bytes: int
    model: str
    df11_bytes: int
    watched_peak_bytes: int
    footprint_peak_bytes: int
    mlx_peak_bytes: int
    label: str
    status: str
    limits_note: str
    source: str


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


def tier_row_from_generate_report(report: Mapping[str, Any], *, source: str) -> TierRow:
    """Build a tier-table row from an ``mlx-dfloat generate --report`` JSON.

    ``ceiling_bytes`` is the tier's fit budget (``limits.tier.ceiling_bytes``: the recommended
    working set minus the reserve), which the status is judged against. On the host tier the
    watchdog's own abort line is higher (RAM minus 4 GiB); on a CAPPED tier the two are the same.

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
        # The watchdog's MLX peak (active + cache), not a phase's active-only mx.get_peak_memory.
        mlx_peak_bytes=int(_need(report, "mlx_peak_bytes", "report")),
        label=str(label),
        status="target" if watched <= ceiling else "over",
        limits_note="host caps"
        if _need(limits, "applied", "limits") == "host-caps"
        else "tier defaults",
        source=source,
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


def _pct(x: float | None) -> str:
    return _NOT_MEASURED if x is None else f"{x * 100:+.1f} %"


_TIER_HEADER = (
    "| Mac | Fit budget (budget − reserve) | Model | DF11 size | Peak (watched) | Peak footprint "  # noqa: RUF001
    "| Peak MLX (active + cache) | Label | Status | Limits | Result |"
)


def render_tier_table(rows: Sequence[TierRow]) -> str:
    """Render the tier table: GiB with two decimals, one row per measured tier."""
    lines = [_TIER_HEADER, "|" + "---|" * 11]
    lines.extend(
        f"| {r.mac_gb} GB | {_gib(r.ceiling_bytes)} | {_MODEL_NAMES.get(r.model, r.model)} "
        f"| {_gib(r.df11_bytes)} | {_gib(r.watched_peak_bytes)} | {_gib(r.footprint_peak_bytes)} "
        f"| {_gib(r.mlx_peak_bytes)} | {r.label} | {r.status} | {r.limits_note} | `{r.source}` |"
        for r in rows
    )
    return "\n".join(lines) + "\n"


def _scenario_title(key: str) -> str:
    m = re.fullmatch(r"flux1-(.+)-(\d+)", key)
    if m is None:
        return key
    return f"{_MODEL_NAMES.get(m.group(1), m.group(1))}, {m.group(2)}²"


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
            f"{_scenario_title(key)}, per-block evaluation: {per_block} (depth-2: {depth2}); "
            f"eval policy cost {cost}; DF11 (per-block) over mflux q8 as shipped (one eval per step): {q8}"
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
