"""The pure parts of the FLUX step bench: modes, interleaving, resume, launch counts, the report.

Expected values are worked by hand from the fixture step times below, never from the code.
No test here imports mflux or touches the GPU.
"""

import json
import sys

import pytest
from scripts._bench_common import write_json_atomic
from scripts.bench_flux_step import (
    MODES,
    BenchError,
    check_embeds_shapes,
    child_command,
    expected_launches,
    interleaved,
    mode_policy,
    pending_runs,
    read_results,
    report,
    result_is_complete,
    run_path,
)

# Two rounds of fixture step times (seconds). Medians: r1 df11 1.2, control 1.0, df11-depth2 1.1,
# control-depth2 1.0, control-noeval 0.9; r2 df11 1.5, control 1.1. Pooled df11 (6 reps) 1.35,
# pooled control 1.0. `verify_s` is the status validation timed outside the step window: pooled
# df11 (0.01, 0.02, 0.03, 0.02, 0.04, 0.03) has median 0.025; control-noeval carries none.
FIXTURES = [
    {"round": 1, "mode": "df11", "step_s": [1.2, 1.3, 1.1], "verify_s": [0.01, 0.02, 0.03]},
    {"round": 1, "mode": "control", "step_s": [1.0, 1.0, 1.0], "verify_s": [0.0, 0.0, 0.0]},
    {
        "round": 1,
        "mode": "df11-depth2",
        "step_s": [1.1, 1.05, 1.15],
        "verify_s": [0.02, 0.02, 0.02],
    },
    {"round": 1, "mode": "control-depth2", "step_s": [1.0, 1.0, 1.0], "verify_s": [0.0, 0.0, 0.0]},
    {"round": 1, "mode": "control-noeval", "step_s": [0.9, 0.9, 0.9]},
    {"round": 2, "mode": "df11", "step_s": [1.5, 1.4, 1.6], "verify_s": [0.02, 0.04, 0.03]},
    {"round": 2, "mode": "control", "step_s": [1.0, 1.2, 1.1], "verify_s": [0.0, 0.0, 0.0]},
]


# --- modes and policies --------------------------------------------------------------------------


def test_the_five_modes_run_in_the_paired_order():
    # Bug caught: a reordered or missing mode; the pairs must alternate df11/control so a drift in
    # machine state hits both sides of a pair.
    assert MODES == ("df11", "control", "df11-depth2", "control-depth2", "control-noeval")


@pytest.mark.parametrize(
    ("mode", "policy"),
    [
        ("df11", "per-block"),
        ("control", "per-block"),
        ("df11-depth2", "depth2"),
        ("control-depth2", "depth2"),
        ("control-noeval", "none"),
    ],
)
def test_each_mode_maps_to_its_eval_policy(mode, policy):
    # Bug caught: control-noeval running with per-block evals (no eval-policy cost measured), or a
    # depth2 mode falling back to per-block.
    assert mode_policy(mode) == policy


def test_an_unknown_mode_is_refused():
    with pytest.raises(BenchError, match="unknown mode"):
        mode_policy("df11-none")


# --- launches -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "steps", "want"),
    [
        ("df11", 5, 285),  # (19 + 38) * 5
        ("df11-depth2", 1, 57),
        ("control", 5, 0),
        ("control-depth2", 5, 0),
        ("control-noeval", 5, 0),
    ],
)
def test_expected_launches_is_one_per_block_per_step_for_df11_and_zero_for_control(
    mode, steps, want
):
    # Bug caught: counting only double blocks, forgetting the step multiplier, or expecting
    # launches from a control mode (which would make every control run fail parity).
    assert expected_launches(mode, n_double=19, n_single=38, steps=steps) == want


# --- interleaving and resume --------------------------------------------------------------------


def test_interleaved_runs_every_mode_once_per_round_in_mode_order():
    # Bug caught: grouping by mode (all df11 rounds first), which lets thermal drift land on one
    # side of every pair.
    assert interleaved(2) == [
        (1, "df11"),
        (1, "control"),
        (1, "df11-depth2"),
        (1, "control-depth2"),
        (1, "control-noeval"),
        (2, "df11"),
        (2, "control"),
        (2, "df11-depth2"),
        (2, "control-depth2"),
        (2, "control-noeval"),
    ]


def test_zero_rounds_is_empty():
    assert interleaved(0) == []


def test_run_path_names_round_and_mode(tmp_path):
    # Bug caught: a path that drops the round (later rounds overwrite the first).
    assert run_path(tmp_path, 3, "control-depth2") == tmp_path / "round3-control-depth2.json"


def test_pending_runs_skips_completed_results_only(tmp_path):
    # Bug caught: skipping on file existence alone (a failed or half-written child would count as
    # done and the report would be built from a hole), or never skipping (no resume).
    write_json_atomic(run_path(tmp_path, 1, "df11"), {"exit_code": 0, "step_s": [1.0]})
    write_json_atomic(run_path(tmp_path, 1, "control"), {"exit_code": 2, "error": "boom"})
    run_path(tmp_path, 1, "df11-depth2").write_text("{not json")
    assert pending_runs(tmp_path, 1) == [
        (1, "control"),
        (1, "df11-depth2"),
        (1, "control-depth2"),
        (1, "control-noeval"),
    ]


def test_a_result_without_step_times_is_not_complete(tmp_path):
    # Bug caught: treating exit_code 0 as complete without the data the report needs.
    path = tmp_path / "r.json"
    write_json_atomic(path, {"exit_code": 0})
    assert result_is_complete(path) is False


def test_a_missing_result_is_not_complete(tmp_path):
    assert result_is_complete(tmp_path / "absent.json") is False


# --- the child command --------------------------------------------------------------------------


def test_child_command_runs_the_module_for_one_mode_and_round_without_orchestrate(tmp_path):
    # Bug caught: forwarding --orchestrate (the child would orchestrate again, forever), running
    # the wrong interpreter, or dropping a setting so the child measures something else.
    cmd = child_command(
        mode="df11-depth2",
        round_no=2,
        out=tmp_path / "round2-df11-depth2.json",
        df11=tmp_path / "ckpt",
        embeds=tmp_path / "e.safetensors",
        steps=5,
        warmup=2,
        model="dev",
        size=768,
        seed=42,
        wall_budget=1200.0,
    )
    assert cmd[:3] == [sys.executable, "-m", "scripts.bench_flux_step"]
    assert "--orchestrate" not in cmd
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=True))
    assert flags == {
        "--mode": "df11-depth2",
        "--round": "2",
        "--out": str(tmp_path / "round2-df11-depth2.json"),
        "--df11": str(tmp_path / "ckpt"),
        "--embeds": str(tmp_path / "e.safetensors"),
        "--steps": "5",
        "--warmup": "2",
        "--model": "dev",
        "--size": "768",
        "--seed": "42",
        "--wall-budget": "1200.0",
    }


# --- embeddings shape check ---------------------------------------------------------------------


@pytest.mark.parametrize(("model", "seq"), [("schnell", 256), ("dev", 512)])
def test_embeds_of_the_models_token_length_pass(model, seq):
    check_embeds_shapes((1, seq, 4096), (1, 768), model)


@pytest.mark.parametrize(
    ("prompt_shape", "pooled_shape"),
    [
        ((1, 512, 4096), (1, 768)),  # dev-length embeddings for schnell: 2x the text tokens
        ((1, 256, 4096), (1, 1024)),
        ((256, 4096), (1, 768)),
    ],
)
def test_embeds_of_another_length_or_width_are_refused_for_schnell(prompt_shape, pooled_shape):
    # Bug caught: accepting any sequence length (a 512-token file would time a different step).
    with pytest.raises(BenchError, match="expected"):
        check_embeds_shapes(prompt_shape, pooled_shape, "schnell")


# --- the report ---------------------------------------------------------------------------------


def test_report_pairs_df11_with_control_per_round():
    # Bug caught: pairing a round's df11 with another round's control, or reporting a depth2 pair
    # for a round that has no depth2 runs. Pinned by hand: r1 1.2/1.0 - 1, r1 depth2 1.1/1.0 - 1,
    # r2 1.5/1.1 - 1.
    rounds = report(FIXTURES)["rounds"]
    assert rounds[1]["per-block"] == pytest.approx(0.2)
    assert rounds[1]["depth2"] == pytest.approx(0.1)
    assert rounds[2]["per-block"] == pytest.approx(1.5 / 1.1 - 1)
    assert "depth2" not in rounds[2]  # round 2 has no depth2 pair


def test_report_pools_every_timed_step_of_a_mode_across_rounds():
    # Bug caught: pooling per-round medians (df11 would be median(1.2, 1.5) = 1.35 too, so the
    # spread pins it: over 6 reps (1.6 - 1.1) / 1.35, over 2 medians (1.5 - 1.2) / 1.35).
    pooled = report(FIXTURES)["pooled"]
    assert pooled["df11"] == {
        "median": pytest.approx(1.35),
        "spread": pytest.approx(0.5 / 1.35),
        "n": 6,
        "verify_median_s": pytest.approx(0.025),
    }
    assert pooled["control"] == {
        "median": pytest.approx(1.0),
        "spread": pytest.approx(0.2),
        "n": 6,
        "verify_median_s": pytest.approx(0.0),
    }
    assert pooled["control-noeval"]["n"] == 3


def test_report_pools_the_status_validation_time_outside_the_step_window():
    # Bug caught: verify_s folded into step_s (df11's pooled median would read 1.37 instead of
    # 1.35), or the validation median taken per round (r1 0.02, r2 0.03) instead of over 6 reps.
    pooled = report(FIXTURES)["pooled"]
    assert pooled["df11"]["median"] == pytest.approx(1.35)
    assert pooled["df11"]["verify_median_s"] == pytest.approx(0.025)


def test_report_without_verify_times_has_a_none_validation_median():
    # Bug caught: a KeyError on a result that never recorded verify_s (control-noeval here).
    assert report(FIXTURES)["pooled"]["control-noeval"]["verify_median_s"] is None


def test_report_overhead_is_from_the_pooled_medians():
    # Bug caught: averaging the per-round overheads ((0.2 + 0.3636) / 2 = 0.28) instead of
    # 1.35 / 1.0 - 1.
    out = report(FIXTURES)
    assert out["overhead"]["per-block"] == pytest.approx(0.35)
    assert out["overhead"]["depth2"] == pytest.approx(0.1)


def test_report_eval_policy_cost_is_control_minus_noeval_pooled_medians():
    # Bug caught: the sign flipped (noeval - control), or using df11 instead of control.
    assert report(FIXTURES)["eval_policy_cost_s"] == pytest.approx(0.1)


def test_report_without_a_noeval_run_has_no_eval_policy_cost():
    # Bug caught: a KeyError or a 0.0 that reads as "evals are free".
    out = report([r for r in FIXTURES if r["mode"] != "control-noeval"])
    assert out["eval_policy_cost_s"] is None


def test_report_from_a_single_mode_has_no_overheads():
    out = report([FIXTURES[0]])
    assert out["overhead"] == {}
    assert out["rounds"] == {1: {}}


def test_read_results_loads_the_completed_files_of_every_round(tmp_path):
    # Bug caught: reading only round 1, or reading a failed child's file into the report.
    for r in FIXTURES:
        write_json_atomic(run_path(tmp_path, r["round"], r["mode"]), {"exit_code": 0, **r})
    write_json_atomic(run_path(tmp_path, 2, "df11-depth2"), {"exit_code": 2, "error": "boom"})
    loaded = read_results(tmp_path, 2)
    assert [(r["round"], r["mode"]) for r in loaded] == [(r["round"], r["mode"]) for r in FIXTURES]
    assert json.loads(run_path(tmp_path, 2, "df11").read_text())["step_s"] == [1.5, 1.4, 1.6]
