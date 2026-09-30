"""Read the step bench's per-child result files and pool them into a summary.

Each child writes ``round{r}-{mode}.json``. Every pair pools its two conditions over the rounds in
which both completed; the overhead comes from the pooled medians (``t_df11 / t_control - 1``).
"""

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.bench.scenario import CONDITIONS
from mlx_dfloat.errors import DFloatFormatError

PAIRS: tuple[tuple[str, str, str], ...] = (
    ("per-block", "df11", "control"),
    ("depth2", "df11-depth2", "control-depth2"),
)
EXTRA_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("eval-policy", "control", "control-noeval"),
    ("q8", "df11", "q8"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionResult:
    """One completed child: a condition's timed steps in one round, with its peaks."""

    condition: str
    round: int
    step_s: tuple[float, ...]
    launches_per_step: tuple[int, ...]
    launches_expected: int
    step_footprint_peak: int
    step_mlx_peak: int
    step_watched_peak: int
    footprint_peak: int
    mlx_peak: int
    watched_peak: int
    scenario_hash: str
    label: str
    limits: dict[str, object]

    @property
    def median(self) -> float:
        """Median of the timed steps, in seconds."""
        return statistics.median(self.step_s)

    @property
    def spread(self) -> float:
        """(max - min) / median over the timed steps."""
        return (max(self.step_s) - min(self.step_s)) / self.median


def _field(data: Mapping[str, Any], name: str) -> Any:
    if name not in data:
        raise DFloatFormatError(f"child result: missing field {name!r}")
    return data[name]


def result_from_json(data: Mapping[str, Any]) -> ConditionResult:
    """Build a result from a child JSON.

    Raises:
        DFloatFormatError: A missing field, a nonzero ``exit_code`` or an empty ``step_s``.
    """
    mode = str(_field(data, "mode"))
    exit_code = _field(data, "exit_code")
    if exit_code != 0:
        raise DFloatFormatError(f"{mode}: exit_code {exit_code}: the child did not complete")
    step_s = tuple(float(s) for s in _field(data, "step_s"))
    if not step_s:
        raise DFloatFormatError(f"{mode}: step_s is empty: the child has no timed steps")
    warmup = int(_field(data, "warmup"))
    return ConditionResult(
        condition=mode,
        round=int(_field(data, "round")),
        step_s=step_s,
        launches_per_step=tuple(int(n) for n in _field(data, "launches_per_step")[warmup:]),
        launches_expected=int(_field(data, "launches_expected_per_step")),
        step_footprint_peak=int(_field(data, "step_footprint_peak_bytes")),
        step_mlx_peak=int(_field(data, "step_mlx_peak_bytes")),
        step_watched_peak=int(_field(data, "step_watched_peak_bytes")),
        footprint_peak=int(_field(data, "footprint_peak_bytes")),
        mlx_peak=int(_field(data, "mlx_peak_memory_bytes")),
        watched_peak=int(_field(data, "watched_peak_bytes")),
        scenario_hash=str(_field(data, "scenario_hash")),
        label=str(_field(data, "label")),
        limits=dict(_field(data, "limits")),
    )


def _one_scenario(results: Sequence[ConditionResult]) -> str:
    hashes = sorted({r.scenario_hash for r in results})
    if len(hashes) > 1:
        raise DFloatFormatError(
            f"results carry {len(hashes)} different scenario hashes ({', '.join(h[:12] for h in hashes)}): "
            "a stale child from another recipe is in this set"
        )
    return hashes[0] if hashes else ""


def _order(r: ConditionResult) -> tuple[int, int]:
    idx = CONDITIONS.index(r.condition) if r.condition in CONDITIONS else len(CONDITIONS)
    return (r.round, idx)


def load_results(directory: Path | str) -> list[ConditionResult]:
    """Read the complete ``round*-*.json`` children of ``directory``, sorted by (round, condition).

    Failed or partial children are skipped.

    Raises:
        DFloatFormatError: The complete children carry more than one scenario hash.
    """
    results: list[ConditionResult] = []
    for path in sorted(Path(directory).glob("round*-*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise DFloatFormatError(f"{path}: cannot read the child result: {exc}") from exc
        if not isinstance(data, dict) or data.get("exit_code") != 0 or not data.get("step_s"):
            continue
        results.append(result_from_json(data))
    _one_scenario(results)
    return sorted(results, key=_order)


@dataclass(frozen=True, slots=True, kw_only=True)
class Summary:
    """Pooled results: per-condition stats, pair overheads, the eval-policy cost, the q8 ratio."""

    scenario_hash: str
    rounds_seen: int
    conditions: dict[str, dict[str, float | int]]
    overhead: dict[str, float]
    eval_cost_s: float | None
    q8_ratio: float | None
    paired_rounds: dict[str, list[int]]
    pair_n: dict[str, int]


def summarise(results: Sequence[ConditionResult]) -> Summary:
    """Pool each pair over the rounds where both of its conditions completed.

    Raises:
        DFloatFormatError: The results carry more than one scenario hash.
    """
    scenario = _one_scenario(results)
    by: dict[str, dict[int, ConditionResult]] = {}
    for r in results:
        by.setdefault(r.condition, {})[r.round] = r

    paired: dict[str, list[int]] = {}
    pooled_steps: dict[str, dict[str, list[float]]] = {}
    conditions: dict[str, dict[str, float | int]] = {}
    for label, a, b in (*PAIRS, *EXTRA_PAIRS):
        shared = sorted(set(by.get(a, {})) & set(by.get(b, {})))
        paired[label] = shared
        pooled_steps[label] = {}
        for cond in (a, b):
            rows = [by[cond][rnd] for rnd in shared]
            steps = [s for row in rows for s in row.step_s]
            pooled_steps[label][cond] = steps
            if shared and cond not in conditions:
                m = statistics.median(steps)
                conditions[cond] = {
                    "median": m,
                    "spread": (max(steps) - min(steps)) / m,
                    "n": len(steps),
                    "step_watched_peak": max(row.step_watched_peak for row in rows),
                    "footprint_peak": max(row.footprint_peak for row in rows),
                    "mlx_peak": max(row.mlx_peak for row in rows),
                }

    def med(label: str, cond: str) -> float:
        return statistics.median(pooled_steps[label][cond])

    overhead = {label: med(label, d) / med(label, c) - 1 for label, d, c in PAIRS if paired[label]}
    eval_cost = (
        med("eval-policy", "control") - med("eval-policy", "control-noeval")
        if paired["eval-policy"]
        else None
    )
    q8_ratio = med("q8", "df11") / med("q8", "q8") if paired["q8"] else None
    pair_n = {
        label: sum(len(v) for v in steps.values())
        for label, steps in pooled_steps.items()
        if paired[label]
    }
    return Summary(
        scenario_hash=scenario,
        rounds_seen=len({r.round for r in results}),
        conditions=conditions,
        overhead=overhead,
        eval_cost_s=eval_cost,
        q8_ratio=q8_ratio,
        paired_rounds={k: v for k, v in paired.items() if v},
        pair_n=pair_n,
    )


def expected_missing(
    results: Sequence[ConditionResult], *, conditions: Sequence[str], rounds: int
) -> list[str]:
    """List ``"round r: condition"`` for every absent result, round-major in ``conditions`` order."""
    have = {(r.round, r.condition) for r in results}
    return [
        f"round {rnd}: {c}"
        for rnd in range(1, rounds + 1)
        for c in conditions
        if (rnd, c) not in have
    ]


__all__ = [
    "EXTRA_PAIRS",
    "PAIRS",
    "ConditionResult",
    "Summary",
    "expected_missing",
    "load_results",
    "result_from_json",
    "summarise",
]
