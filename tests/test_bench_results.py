"""Reading child result files and pooling them: medians, overheads, the q8 ratio, worked by hand."""

import json

import pytest

from mlx_dfloat.bench.results import (
    ConditionResult,
    Summary,
    expected_missing,
    load_results,
    result_from_json,
    summarise,
)
from mlx_dfloat.errors import DFloatFormatError

H = "a" * 64


def _r(condition, rnd, step_s, *, launches=0, h=H, fp=10, mp=8):
    return ConditionResult(
        condition=condition,
        round=rnd,
        step_s=tuple(step_s),
        launches_per_step=tuple([launches] * len(step_s)),
        launches_expected=launches,
        step_footprint_peak=fp,
        step_mlx_peak=mp,
        step_watched_peak=fp,
        footprint_peak=fp + 1,
        mlx_peak=mp + 1,
        watched_peak=fp + 1,
        scenario_hash=h,
        label="MEASURED",
        limits={},
    )


# Round 1: df11 1.2/1.3/1.1 (median 1.2), control 1.0x3, df11-depth2 1.1/1.05/1.15 (1.1),
# control-depth2 1.0x3, control-noeval 0.9x3, q8 0.6x3. Round 2: df11 1.5/1.4/1.6 (1.5), control
# 1.2/1.2/1.2. Round 3: df11 only. Pooled df11 over rounds {1,2} (6 reps): 1.35; pooled control:
# 1.0,1.0,1.0,1.2,1.2,1.2 -> 1.1. per-block overhead = 1.35/1.1 - 1 = 0.22727. depth2 over round 1:
# 1.1/1.0 - 1 = 0.1. eval cost over round 1 only: 1.0 - 0.9 = 0.1 (pooling every control round
# would give 1.1 - 0.9 = 0.2). q8 ratio over round 1: 1.2 / 0.6 = 2.0. Round 2's df11 has fp=30
# and mp=20.
FIX = [
    _r("df11", 1, [1.2, 1.3, 1.1], launches=57),
    _r("control", 1, [1.0, 1.0, 1.0]),
    _r("df11-depth2", 1, [1.1, 1.05, 1.15], launches=57),
    _r("control-depth2", 1, [1.0, 1.0, 1.0]),
    _r("control-noeval", 1, [0.9, 0.9, 0.9]),
    _r("q8", 1, [0.6, 0.6, 0.6]),
    _r("df11", 2, [1.5, 1.4, 1.6], launches=57, fp=30, mp=20),
    _r("control", 2, [1.2, 1.2, 1.2]),
    _r("df11", 3, [9.0, 9.0, 9.0], launches=57),
]


def test_summary_pools_pairs_over_their_shared_rounds_only():
    # Bug caught: pooling every round of a condition (round 3's lone df11 would drag the per-block
    # median to 1.5; every control round would make the eval cost 0.2), or computing the overhead
    # from per-round means, or taking the first round's peak instead of the max.
    s = summarise(FIX)
    assert isinstance(s, Summary)
    assert s.overhead["per-block"] == pytest.approx(1.35 / 1.1 - 1)
    assert s.overhead["depth2"] == pytest.approx(0.1)
    assert s.eval_cost_s == pytest.approx(0.1)
    assert s.q8_ratio == pytest.approx(2.0)
    assert s.paired_rounds == {"per-block": [1, 2], "depth2": [1], "eval-policy": [1], "q8": [1]}
    assert s.pair_n == {"per-block": 12, "depth2": 6, "eval-policy": 6, "q8": 6}
    assert s.conditions["df11"]["n"] == 6
    assert s.conditions["df11"]["median"] == pytest.approx(1.35)
    assert s.conditions["df11"]["step_watched_peak"] == 30
    assert s.conditions["df11"]["footprint_peak"] == 31
    # the active-only MLX peak (mlx_peak_memory_bytes) pools as the max over the paired rounds:
    # round 2's mp + 1 = 21, not round 1's 9 and not a sum
    assert s.conditions["df11"]["mlx_peak"] == 21
    assert s.rounds_seen == 3


def test_a_pair_without_a_shared_round_yields_no_overhead():
    # Red if: an overhead were computed from unpaired rounds (df11 round 1 vs control round 2).
    s = summarise([_r("df11", 1, [1.0]), _r("control", 2, [1.0])])
    assert s.overhead == {}
    assert s.q8_ratio is None
    assert s.eval_cost_s is None


def test_two_scenario_hashes_are_refused():
    # Review Focus 4: a stale round1-df11.json from another recipe must not be averaged in.
    with pytest.raises(DFloatFormatError, match="scenario"):
        summarise([_r("df11", 1, [1.0]), _r("control", 1, [1.0], h="b" * 64)])


def test_expected_missing_names_every_absent_round_condition_in_interleaved_order():
    # Red if: the listing were condition-major ("round 2: q8" after "round 3: control") or dropped an entry.
    missing = expected_missing(FIX, conditions=("df11", "control", "q8"), rounds=3)
    assert missing == ["round 2: q8", "round 3: control", "round 3: q8"]


def test_spread_is_max_minus_min_over_the_median():
    # Red if: spread returned max - min without dividing by the median, or used the mean (1.0: (1.5-1.0)/1.0).
    assert _r("df11", 1, [1.0, 1.5, 1.0]).spread == pytest.approx(0.5)


def _json(**over):
    data = {
        "mode": "df11",
        "round": 1,
        "exit_code": 0,
        "warmup": 2,
        "step_s": [1.0, 1.1],
        "launches_per_step": [57, 57, 57, 57],
        "launches_expected_per_step": 57,
        "step_footprint_peak_bytes": 5,
        "step_mlx_peak_bytes": 4,
        "step_watched_peak_bytes": 5,
        "footprint_peak_bytes": 6,
        "mlx_peak_memory_bytes": 5,
        "watched_peak_bytes": 6,
        "scenario_hash": H,
        "label": "MEASURED",
        "limits": {},
    }
    return {**data, **over}


def test_result_from_json_reads_the_fields_and_slices_off_the_warm_up_launches():
    # Bug caught: the child JSON's launches_per_step includes the warm-up steps; keeping them would
    # misreport a timed-step launch count.
    r = result_from_json(_json())
    assert r.condition == "df11"
    assert r.median == pytest.approx(1.05)
    assert r.launches_per_step == (57, 57)


@pytest.mark.parametrize("over", [{"exit_code": 2}, {"step_s": []}])
def test_result_from_json_refuses_a_failed_or_partial_child(over):
    # Red if: result_from_json stopped checking exit_code or an empty step_s.
    with pytest.raises(DFloatFormatError, match=r"exit_code|step_s"):
        result_from_json(_json(**over))


def test_result_from_json_names_a_missing_field():
    # Red if: a missing field raised a bare KeyError instead of DFloatFormatError naming it.
    data = _json()
    del data["scenario_hash"]
    with pytest.raises(DFloatFormatError, match="scenario_hash"):
        result_from_json(data)


def test_load_results_reads_complete_children_skips_incomplete_ones_and_refuses_a_hash_mix(
    tmp_path,
):
    # Red if: load_results kept exit_code != 0 children, globbed report.json, or skipped the hash check.
    (tmp_path / "round1-control.json").write_text(
        json.dumps(
            _json(mode="control", launches_per_step=[0, 0, 0, 0], launches_expected_per_step=0)
        )
    )
    (tmp_path / "round1-df11.json").write_text(json.dumps(_json(exit_code=70)))
    (tmp_path / "report.json").write_text("{}")
    results = load_results(tmp_path)
    assert [(r.condition, r.round) for r in results] == [("control", 1)]
    (tmp_path / "round2-df11.json").write_text(json.dumps(_json(round=2, scenario_hash="b" * 64)))
    with pytest.raises(DFloatFormatError, match="scenario"):
        load_results(tmp_path)
