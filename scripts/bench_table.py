"""Regenerate the README's measured-numbers blocks from the committed bench result files.

    python -m scripts.bench_table            # rewrite the blocks in README.md
    python -m scripts.bench_table --check    # exit 1 when a block differs from the render

Reads ``bench/results/tiers/*.json`` (``mlx-dfloat generate --report`` files), every scenario
directory that holds a ``report.json`` and its child results, and the ``harness-proof/`` pair.
The numbers in the README are never typed by hand: the test suite fails when they differ from
this render. Exit codes: 0 fresh (or written), 1 stale under ``--check``, 2 any error (a
malformed README or result file, an incomplete scenario, a README that cannot be written).
A scenario directory whose runs are not MEASURED (a ``--tier`` run) is skipped with a note on
stderr.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from mlx_dfloat.bench.results import Summary, load_results, summarise  # noqa: E402
from mlx_dfloat.bench.table import (  # noqa: E402
    ProofRecord,
    TierRow,
    caption,
    proof_from_files,
    render_overhead_block,
    render_proof_paragraph,
    render_tier_table,
    scenario_title,
    splice,
    tier_row_from_generate_report,
)
from mlx_dfloat.errors import DFloatFormatError  # noqa: E402

NO_RESULTS = "No result files yet.\n"
_RESULTS_PREFIX = "bench/results"
_SKIP_DIRS = ("tiers", "harness-proof")


@dataclass(frozen=True, slots=True, kw_only=True)
class Collected:
    """Everything the README blocks are rendered from."""

    tier_rows: tuple[TierRow, ...]
    summaries: dict[str, Summary]
    proof: ProofRecord | None
    provenance: dict[str, Any] | None
    reproducers: dict[str, str]
    preflight_skipped: dict[str, tuple[str, ...]]
    provenances: dict[str, dict[str, Any]]
    cache_limits: dict[str, int | None]
    notes: tuple[str, ...]
    date: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise DFloatFormatError(f"{path}: cannot read the result file: {exc}") from exc
    if not isinstance(data, dict):
        raise DFloatFormatError(f"{path}: expected a JSON object")
    return data


def _collect_proof(root: Path) -> ProofRecord | None:
    proof_dir = root / "harness-proof"
    passes = sorted(proof_dir.glob("pass-*.json"))
    abort = proof_dir / "abort.json"
    if not passes or not abort.is_file():
        return None
    return proof_from_files(
        _read_json(passes[0]), _read_json(abort), source_dir=f"{_RESULTS_PREFIX}/harness-proof"
    )


def _refuse_incomplete(name: str, report: dict[str, Any]) -> None:
    for field in ("missing", "stopped"):
        if field not in report:
            raise DFloatFormatError(f"{name}/report.json has no {field!r} field")
    if report["missing"]:
        raise DFloatFormatError(
            f"{name}: the scenario is incomplete (missing {', '.join(map(str, report['missing']))}); "
            "resume it before rendering"
        )
    if report["stopped"] is not None:
        raise DFloatFormatError(
            f"{name}: the scenario stopped at {report['stopped']}; resume it before rendering"
        )


def collect(results_root: Path) -> Collected:
    """Read the result files under ``results_root``.

    A scenario directory whose children are not all ``MEASURED`` (a ``--tier`` run's
    ``<name>-tier<GB>``) is skipped, and ``notes`` says so.

    Raises:
        DFloatFormatError: A result file is unreadable or malformed, or a scenario's report
            records missing runs or a stop.
    """
    root = Path(results_root)
    rows = tuple(
        tier_row_from_generate_report(_read_json(p), source=f"{_RESULTS_PREFIX}/tiers/{p.name}")
        for p in sorted((root / "tiers").glob("*.json"))
    )
    summaries: dict[str, Summary] = {}
    reproducers: dict[str, str] = {}
    skipped: dict[str, tuple[str, ...]] = {}
    provenances: dict[str, dict[str, Any]] = {}
    cache_limits: dict[str, int | None] = {}
    notes: list[str] = []
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    for d in dirs:
        if d.name in _SKIP_DIRS or not (d / "report.json").is_file():
            continue
        report = _read_json(d / "report.json")
        scenario = report.get("scenario")
        if not isinstance(scenario, dict) or "name" not in scenario:
            continue
        results = load_results(d)
        labels = sorted({r.label for r in results})
        if labels != ["MEASURED"]:
            notes.append(
                f"skipped {_RESULTS_PREFIX}/{d.name}: its runs are labelled "
                f"{', '.join(labels) or 'nothing'}, not MEASURED"
            )
            continue
        _refuse_incomplete(d.name, report)
        summaries[d.name] = summarise(results)
        reproducers[d.name] = str(report.get("reproducer", ""))
        if report.get("skipped_preflight"):
            skipped[d.name] = tuple(str(g) for g in report.get("failed_gates", []))
        provenances[d.name] = dict(report["provenance"])
        cache_limits[d.name] = scenario.get("cache_limit_bytes")
    provenance = next(iter(provenances.values()), None)
    return Collected(
        tier_rows=rows,
        summaries=summaries,
        proof=_collect_proof(root),
        provenance=provenance,
        reproducers=reproducers,
        preflight_skipped=skipped,
        provenances=provenances,
        cache_limits=cache_limits,
        notes=tuple(notes),
        date="" if provenance is None else str(provenance.get("date", "")),
    )


def _cache_note(limit: int | None) -> str:
    if limit is None:
        return "The MLX buffer-cache limit for these runs is recorded in each report.json."
    return (
        f"DF11 and the mflux q8 step both run under a {limit / 1e9:.1f} GB MLX buffer-cache limit "
        "(decimal GB)."
    )


def _shared_or_per_scenario(values: dict[str, str]) -> str:
    """One line when every scenario has the same text, else one titled line per scenario."""
    if len(set(values.values())) == 1:
        return next(iter(values.values()))
    return "\n\n".join(f"{scenario_title(k)}: {v}" for k, v in values.items())


def _fragments(collected: Collected, *, date: str) -> dict[str, str]:
    tier = render_tier_table(collected.tier_rows) if collected.tier_rows else NO_RESULTS
    if collected.summaries and collected.provenance is not None:
        own = {k: caption(p, date=str(p.get("date", ""))) for k, p in collected.provenances.items()}
        captions = (
            caption(collected.provenance, date=date)
            if len(set(own.values())) == 1
            else _shared_or_per_scenario(own)
        )
        overhead = render_overhead_block(
            collected.summaries,
            caption=captions,
            reproducers=collected.reproducers,
            cache_limit_note=_shared_or_per_scenario(
                {k: _cache_note(v) for k, v in collected.cache_limits.items()}
            ),
            preflight_skipped=collected.preflight_skipped,
        )
    else:
        overhead = NO_RESULTS
    proof = NO_RESULTS if collected.proof is None else render_proof_paragraph(collected.proof)
    return {"tier-table": tier, "overhead": overhead, "harness-proof": proof}


def render_readme(readme: str, collected: Collected, *, date: str) -> str:
    """Splice the three generated blocks into ``readme``.

    Raises:
        DFloatFormatError: A block's markers are missing or duplicated.
    """
    out = readme
    for block, fragment in _fragments(collected, date=date).items():
        out = splice(out, block, fragment)
    return out


def _stale_blocks(readme: str, collected: Collected) -> list[str]:
    stale = []
    for block, fragment in _fragments(collected, date=collected.date).items():
        if splice(readme, block, fragment) != readme:
            stale.append(block)
    return stale


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # type: ignore[union-attr]
    parser.add_argument("--results-root", type=Path, default=_REPO / "bench" / "results")
    parser.add_argument("--readme", type=Path, default=_REPO / "README.md")
    parser.add_argument("--check", action="store_true", help="exit 1 when a block is stale")
    args = parser.parse_args(argv)
    try:
        readme = args.readme.read_text()
        collected = collect(args.results_root)
        for note in collected.notes:
            print(f"bench_table: {note}", file=sys.stderr)
        rendered = render_readme(readme, collected, date=collected.date)
        stale = _stale_blocks(readme, collected)
        if args.check:
            if stale:
                print(f"stale README blocks: {', '.join(stale)}")
                return 1
            return 0
        _write_atomic(args.readme, rendered)
    except (DFloatFormatError, OSError) as exc:
        print(f"bench_table: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # anything else is a tool error (2), never 1 (stale)
        print(f"bench_table: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
