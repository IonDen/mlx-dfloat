"""Regenerate the README's measured-numbers blocks from the committed bench result files.

    python -m scripts.bench_table            # rewrite the blocks in README.md
    python -m scripts.bench_table --check    # exit 1 when a block differs from the render

Reads ``bench/results/tiers/*.json`` (``mlx-dfloat generate --report`` files), every scenario
directory that holds a ``report.json`` and its child results, and the ``harness-proof/`` pair.
The numbers in the README are never typed by hand: the test suite fails when they differ from
this render. Exit codes: 0 fresh (or written), 1 stale under ``--check``, 2 a malformed README
or result file.
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
    cache_limit_bytes: int | None
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


def collect(results_root: Path) -> Collected:
    """Read the result files under ``results_root``.

    Raises:
        DFloatFormatError: A result file is unreadable or malformed.
    """
    root = Path(results_root)
    rows = tuple(
        tier_row_from_generate_report(_read_json(p), source=f"{_RESULTS_PREFIX}/tiers/{p.name}")
        for p in sorted((root / "tiers").glob("*.json"))
    )
    summaries: dict[str, Summary] = {}
    reproducers: dict[str, str] = {}
    skipped: dict[str, tuple[str, ...]] = {}
    first: dict[str, Any] | None = None
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    for d in dirs:
        if d.name in _SKIP_DIRS or not (d / "report.json").is_file():
            continue
        report = _read_json(d / "report.json")
        scenario = report.get("scenario")
        if not isinstance(scenario, dict) or "name" not in scenario:
            continue
        summaries[d.name] = summarise(load_results(d))
        reproducers[d.name] = str(report.get("reproducer", ""))
        if report.get("skipped_preflight"):
            skipped[d.name] = tuple(str(g) for g in report.get("failed_gates", []))
        first = first or report
    provenance = dict(first["provenance"]) if first else None
    scenario = first["scenario"] if first else {}
    return Collected(
        tier_rows=rows,
        summaries=summaries,
        proof=_collect_proof(root),
        provenance=provenance,
        reproducers=reproducers,
        preflight_skipped=skipped,
        cache_limit_bytes=scenario.get("cache_limit_bytes"),
        date="" if provenance is None else str(provenance.get("date", "")),
    )


def _cache_note(limit: int | None) -> str:
    if limit is None:
        return "The MLX buffer-cache limit for these runs is recorded in each report.json."
    return (
        f"DF11 and the mflux q8 step both run under a {limit / 1e9:.1f} GB MLX buffer-cache limit "
        "(decimal GB)."
    )


def _fragments(collected: Collected, *, date: str) -> dict[str, str]:
    tier = render_tier_table(collected.tier_rows) if collected.tier_rows else NO_RESULTS
    if collected.summaries and collected.provenance is not None:
        overhead = render_overhead_block(
            collected.summaries,
            caption=caption(collected.provenance, date=date),
            reproducers=collected.reproducers,
            cache_limit_note=_cache_note(collected.cache_limit_bytes),
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
        rendered = render_readme(readme, collected, date=collected.date)
        stale = _stale_blocks(readme, collected)
    except (DFloatFormatError, OSError) as exc:
        print(f"bench_table: {exc}", file=sys.stderr)
        return 2
    if args.check:
        if stale:
            print(f"stale README blocks: {', '.join(stale)}")
            return 1
        return 0
    _write_atomic(args.readme, rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
