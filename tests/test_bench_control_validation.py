"""The pure parts of the reduced-depth control validation: modes, verdict, report, resume.

Expected values are worked by hand from the fixture step times below, never from the code.
No test here imports mflux or touches the GPU.
"""

import sys
from pathlib import Path

import pytest
from scripts._bench_common import write_json_atomic
from scripts.bench_control_validation import (
    MODES,
    BenchError,
    child_command,
    control_vs_bf16,
    df11_vs_bf16,
    expected_launches,
    interleaved,
    parse_args,
    pending_runs,
    read_results,
    report,
    resume_conflicts,
    run_key,
    within_spread,
)
from scripts.bench_flux_step import run_path

# Two complete rounds and a stopped third. Medians: r1 bf16 1.00, control 1.01, df11 1.30; r2 bf16
# 1.02, control 1.02, df11 1.32. Pooled over r1 + r2 (6 reps each):
#   bf16    0.98 1.00 1.00 1.02 1.02 1.04 -> median 1.01,  spread 0.06 / 1.01
#   control 1.00 1.00 1.01 1.02 1.03 1.04 -> median 1.015, spread 0.04 / 1.015
#   df11    1.28 1.30 1.30 1.32 1.32 1.34 -> median 1.31,  spread 0.06 / 1.31
# control_vs_bf16 = |1.01 - 1.015| / 1.015; within max(0.06/1.01, 0.04/1.015) -> True;
# df11_vs_bf16 = (1.31 - 1.01) / 1.01. Round 3 has bf16 only (5.0 s): pooled, it would move the
# bf16 median to 1.02.
FIXTURES = [
    {"round": 1, "mode": "bf16", "step_s": [1.00, 1.02, 0.98], "verify_s": [0.0, 0.0, 0.0]},
    {"round": 1, "mode": "control", "step_s": [1.01, 1.00, 1.03], "verify_s": [0.0, 0.0, 0.0]},
    {"round": 1, "mode": "df11", "step_s": [1.30, 1.28, 1.32], "verify_s": [0.01, 0.02, 0.03]},
    {"round": 2, "mode": "bf16", "step_s": [1.04, 1.00, 1.02], "verify_s": [0.0, 0.0, 0.0]},
    {"round": 2, "mode": "control", "step_s": [1.02, 1.04, 1.00], "verify_s": [0.0, 0.0, 0.0]},
    {"round": 2, "mode": "df11", "step_s": [1.34, 1.30, 1.32], "verify_s": [0.02, 0.04, 0.03]},
    {"round": 3, "mode": "bf16", "step_s": [5.0, 5.0, 5.0], "verify_s": [0.0, 0.0, 0.0]},
]
COMMON = {
    "model": "schnell",
    "size": 1024,
    "steps": 5,
    "warmup": 2,
    "seed": 42,
    "df11": Path("/ckpt"),
    "embeds": Path("/e.safetensors"),
    "embeds_meta": {"synthetic": "true"},
    "source": "abc",
    "mlx": "0.32.2",
    "cache_limit": 1_400_000_000,
}
KEY = run_key(**COMMON, double=4, single=8)


# --- modes and launches --------------------------------------------------------------------------


def test_the_three_modes_run_bf16_then_control_then_df11():
    # Bug caught: a missing or reordered mode; bf16 and control, the pair the verdict compares,
    # must run back to back so a drift in machine state lands on both.
    assert MODES == ("bf16", "control", "df11")


@pytest.mark.parametrize(
    ("mode", "steps", "want"),
    [("df11", 1, 12), ("df11", 5, 60), ("bf16", 5, 0), ("control", 5, 0)],
)
def test_df11_launches_once_per_block_of_the_reduced_depth_and_the_others_never(mode, steps, want):
    # Bug caught: the full-depth count (57 per step) expected at 4 + 8 blocks, or launches expected
    # from the resident bf16 mode (every bf16 run would fail its parity check).
    assert expected_launches(mode, n_double=4, n_single=8, steps=steps) == want


def test_an_unknown_mode_is_refused():
    # Bug caught: a full-depth mode name (df11-depth2) silently treated as a control.
    with pytest.raises(BenchError, match="unknown mode"):
        expected_launches("df11-depth2", n_double=4, n_single=8, steps=1)


def test_interleaved_runs_the_three_modes_once_per_round_in_order():
    # Bug caught: bench_flux_step's five full-depth modes interleaved instead of these three.
    assert interleaved(2) == [
        (1, "bf16"),
        (1, "control"),
        (1, "df11"),
        (2, "bf16"),
        (2, "control"),
        (2, "df11"),
    ]


# --- the verdict arithmetic ------------------------------------------------------------------------


@pytest.mark.parametrize(("t_bf16", "want"), [(1.2, 0.2), (0.8, 0.2)])
def test_control_vs_bf16_is_the_absolute_difference_over_the_control(t_bf16, want):
    # Bug caught: a signed difference (a faster bf16 would read -0.2 and always pass the verdict),
    # or dividing by bf16 (1.2 would give 0.2 / 1.2 = 0.1667).
    assert control_vs_bf16(t_bf16, 1.0) == pytest.approx(want)


def test_control_vs_bf16_refuses_a_non_positive_control():
    with pytest.raises(ValueError, match="positive"):
        control_vs_bf16(1.0, 0.0)


@pytest.mark.parametrize(
    ("deviation", "spread_bf16", "spread_control", "want"),
    [
        (0.05, 0.05, 0.01, True),  # on the bound: inside
        (0.0501, 0.05, 0.01, False),  # just above
        (0.03, 0.01, 0.05, True),  # the control's spread is the larger
        (0.03, 0.05, 0.01, True),  # the bf16 spread is the larger
    ],
)
def test_within_spread_compares_with_the_larger_of_the_two_spreads(
    deviation, spread_bf16, spread_control, want
):
    # Bug caught: `<` for `<=` (row 1), the smaller spread as the bound (rows 3 and 4 would fail),
    # or only one mode's spread used (row 3 or row 4 would fail).
    assert within_spread(deviation, spread_bf16, spread_control) is want


def test_df11_vs_bf16_is_the_df11_overhead_over_bf16():
    # Bug caught: the control as the denominator, or the ratio without the -1 (1.3 instead of 0.3).
    assert df11_vs_bf16(1.3, 1.0) == pytest.approx(0.3)
    assert df11_vs_bf16(0.9, 1.0) == pytest.approx(-0.1)


# --- the report ------------------------------------------------------------------------------------


def test_report_lists_each_rounds_medians():
    # Bug caught: a round's mode reported with another round's time, or the stopped round dropped
    # from the per-round list (it stays visible even though it is not pooled).
    rounds = report(FIXTURES)["rounds"]
    assert rounds[1] == {
        "T_bf16": pytest.approx(1.00),
        "T_control": pytest.approx(1.01),
        "T_df11": pytest.approx(1.30),
    }
    assert rounds[2] == {
        "T_bf16": pytest.approx(1.02),
        "T_control": pytest.approx(1.02),
        "T_df11": pytest.approx(1.32),
    }
    assert rounds[3] == {"T_bf16": pytest.approx(5.0)}


def test_report_lists_the_paired_control_bias_of_each_complete_round():
    # Bug caught: a consistent small bias hidden behind a passing pooled verdict, a signed value,
    # or the stopped round 3 (bf16 only) given a value. By hand: r1 |1.00 - 1.01| / 1.01,
    # r2 |1.02 - 1.02| / 1.02 = 0.
    per_round = report(FIXTURES)["control_vs_bf16_per_round"]
    assert per_round == {1: pytest.approx(0.01 / 1.01), 2: pytest.approx(0.0)}


def test_report_pools_every_timed_step_of_the_complete_rounds_only():
    # Bug caught: round 3's lone bf16 pooled (median 1.02, n 9), or per-round medians pooled
    # instead of the steps (bf16 spread would be (1.02 - 1.00) / 1.01).
    out = report(FIXTURES)
    assert out["complete_rounds"] == [1, 2]
    assert out["pooled"]["bf16"] == {
        "median": pytest.approx(1.01),
        "spread": pytest.approx(0.06 / 1.01),
        "n": 6,
        "verify_median_s": pytest.approx(0.0),
        "trace": None,
    }
    assert out["pooled"]["control"]["median"] == pytest.approx(1.015)
    assert out["pooled"]["control"]["spread"] == pytest.approx(0.04 / 1.015)
    assert out["pooled"]["df11"]["median"] == pytest.approx(1.31)
    assert out["pooled"]["df11"]["verify_median_s"] == pytest.approx(0.025)


def test_report_verdict_is_from_the_pooled_medians_and_spreads():
    # Bug caught: the verdict taken from one round (r1: |1.00 - 1.01| / 1.01), or the df11 anchor
    # divided by the control (0.29 / 1.015).
    out = report(FIXTURES)
    assert out["control_vs_bf16"] == pytest.approx(0.005 / 1.015)
    assert out["spread_bound"] == pytest.approx(0.06 / 1.01)
    assert out["within_spread"] is True
    assert out["df11_vs_bf16"] == pytest.approx(0.30 / 1.01)


def test_report_verdict_fails_when_the_control_is_off_by_more_than_the_spread():
    # Bug caught: a verdict that is always True. Constant steps have zero spread, so a control 20%
    # slower than bf16 lies outside it.
    results = [
        {"round": 1, "mode": "bf16", "step_s": [1.0, 1.0, 1.0]},
        {"round": 1, "mode": "control", "step_s": [1.2, 1.2, 1.2]},
        {"round": 1, "mode": "df11", "step_s": [1.5, 1.5, 1.5]},
    ]
    out = report(results)
    assert out["control_vs_bf16"] == pytest.approx(0.2 / 1.2)
    assert out["within_spread"] is False


def test_report_without_a_complete_round_has_no_verdict():
    # Bug caught: a verdict from a round missing its df11 (or a KeyError on the missing mode).
    out = report([r for r in FIXTURES if r["mode"] != "df11"])
    assert out["complete_rounds"] == []
    assert out["pooled"] == {}
    assert out["within_spread"] is None
    assert out["control_vs_bf16"] is None
    assert out["df11_vs_bf16"] is None


def test_report_of_a_stopped_orchestration_suppresses_the_verdict():
    # Bug caught: a stopped run printing a verdict as if the validation had finished.
    stopped = {"round": 3, "mode": "control", "exit_code": 2}
    out = report(FIXTURES, stopped=stopped)
    assert out["stopped"] == stopped
    assert out["within_spread"] is None
    assert out["control_vs_bf16"] is None
    assert out["df11_vs_bf16"] is None
    assert out["rounds"][1]["T_bf16"] == pytest.approx(1.00)  # the per-round medians stay


def test_report_refuses_a_result_without_timed_steps():
    with pytest.raises(BenchError, match="no timed steps"):
        report([{"round": 1, "mode": "bf16", "step_s": []}])


# --- resume ------------------------------------------------------------------------------------------


def test_the_key_carries_the_reduced_depth():
    # Bug caught: a --double 2 rerun resuming into files measured at --double 4.
    assert KEY["double"] == 4
    assert KEY["single"] == 8
    assert KEY["df11"] == str(Path("/ckpt").resolve())


@pytest.mark.parametrize("field", ["double", "single", "steps"])
def test_a_stored_key_of_another_depth_or_setting_is_a_conflict(tmp_path, field):
    changed = {**KEY, field: 99}
    write_json_atomic(
        run_path(tmp_path, 1, "control"), {"exit_code": 0, "step_s": [1.0], "key": changed}
    )
    assert resume_conflicts(tmp_path, 1, KEY) == [(1, "control", [field])]


def test_a_stored_result_without_a_key_is_a_conflict(tmp_path):
    write_json_atomic(run_path(tmp_path, 1, "bf16"), {"exit_code": 0, "step_s": [1.0]})
    assert resume_conflicts(tmp_path, 1, KEY) == [(1, "bf16", ["key"])]


def test_pending_runs_skips_only_complete_runs_of_the_three_modes(tmp_path):
    # Bug caught: resuming over bench_flux_step's mode names, so the bf16 file is never seen as
    # done (re-run forever) or df11-depth2 is expected here.
    write_json_atomic(run_path(tmp_path, 1, "bf16"), {"exit_code": 0, "step_s": [1.0], "key": KEY})
    write_json_atomic(run_path(tmp_path, 1, "control"), {"exit_code": 2, "key": KEY})
    assert pending_runs(tmp_path, 2) == [
        (1, "control"),
        (1, "df11"),
        (2, "bf16"),
        (2, "control"),
        (2, "df11"),
    ]


def test_read_results_reads_the_complete_files_of_the_three_modes(tmp_path):
    for r in FIXTURES:
        write_json_atomic(run_path(tmp_path, r["round"], r["mode"]), {"exit_code": 0, **r})
    write_json_atomic(run_path(tmp_path, 3, "control"), {"exit_code": 2, "error": "boom"})
    loaded = read_results(tmp_path, 3)
    assert [(r["round"], r["mode"]) for r in loaded] == [(r["round"], r["mode"]) for r in FIXTURES]


# --- the child command and the arguments -----------------------------------------------------------


def test_child_command_runs_this_module_for_one_mode_at_the_reduced_depth(tmp_path):
    # Bug caught: launching bench_flux_step (a full-depth run), dropping --double/--single (the
    # child would build its default depth), or passing no --mode (the child would orchestrate).
    cmd = child_command(
        mode="bf16",
        round_no=2,
        out=tmp_path / "round2-bf16.json",
        df11=tmp_path / "ckpt",
        embeds=tmp_path / "e.safetensors",
        double=3,
        single=5,
        steps=5,
        warmup=2,
        model="schnell",
        size=512,
        seed=7,
        wall_budget=900.0,
    )
    assert cmd[:3] == [sys.executable, "-m", "scripts.bench_control_validation"]
    assert dict(zip(cmd[3::2], cmd[4::2], strict=True)) == {
        "--mode": "bf16",
        "--round": "2",
        "--out": str(tmp_path / "round2-bf16.json"),
        "--df11": str(tmp_path / "ckpt"),
        "--embeds": str(tmp_path / "e.safetensors"),
        "--double": "3",
        "--single": "5",
        "--steps": "5",
        "--warmup": "2",
        "--model": "schnell",
        "--size": "512",
        "--seed": "7",
        "--wall-budget": "900.0",
    }


def _args(*extra):
    return ["--df11", "d", "--embeds", "e", "--out", "o", *extra]


def test_without_a_mode_the_run_orchestrates_and_paths_are_absolute(tmp_path, monkeypatch):
    # Bug caught: relative paths handed to children that run with the repository root as cwd
    # (they would read and write somewhere else, and their keys would differ from this one's).
    monkeypatch.chdir(tmp_path)
    args = parse_args(_args())
    assert args.mode is None
    assert (args.double, args.single, args.rounds) == (4, 8, 3)
    assert args.out == tmp_path / "o"
    assert args.df11 == tmp_path / "d"
    assert args.embeds == tmp_path / "e"


@pytest.mark.parametrize(
    "extra",
    [
        ("--double", "0"),
        ("--single", "0"),
        ("--double", "20"),  # FLUX.1 has 19 double blocks
        ("--single", "39"),  # and 38 single blocks
        ("--warmup", "0"),
        ("--steps", "0"),
        ("--mode", "df11-depth2"),
    ],
)
def test_out_of_range_depths_and_counts_are_usage_errors(extra):
    # Bug caught: a depth of 0 (the control needs one decoded block of each kind) or beyond the
    # model, or no warm-up step (the first timed step would carry the pipeline compile).
    with pytest.raises(SystemExit) as exc:
        parse_args(_args(*extra))
    assert exc.value.code == 2


def test_the_edge_depths_are_accepted():
    args = parse_args(_args("--double", "1", "--single", "38", "--mode", "df11", "--round", "1"))
    assert (args.double, args.single, args.mode, args.round) == (1, 38, "df11", 1)


# --- the providers ---------------------------------------------------------------------------------


def _counting_reference_decode(calls):
    from mlx_dfloat.decode import decode_group

    def decode(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    return decode


def test_make_provider_for_bf16_decodes_every_block_once_into_its_own_resident_dict():
    # Bug caught: bf16 decoding one block per kind (that is the control, and the validation would
    # compare the control with itself), decoding a block twice, or handing block 1 block 0's weights.
    import mlx.core as mx
    import numpy as np
    from scripts._flux_rig import ResidentProvider
    from scripts.bench_control_validation import make_provider
    from tests.test_bench_flux_step import _fake_rig

    ckpt, groups, shapes, source = _fake_rig(np.random.default_rng(6))
    calls = []
    provider = make_provider("bf16", ckpt, groups, shapes, decode=_counting_reference_decode(calls))
    assert calls == list(shapes)
    assert isinstance(provider, ResidentProvider)
    assert provider.launches == 0
    for block_name in shapes:
        w = provider.weights_for(block_name, shapes[block_name])
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert np.array_equal(np.array(w["attn.to_q"].view(mx.uint16)), want)


@pytest.mark.parametrize(("mode", "want"), [("control", 2), ("df11", 0)])
def test_make_provider_hands_control_and_df11_to_the_step_bench(mode, want):
    # Bug caught: the validation building its own control (two decodes) differently from the step
    # bench's, so the verdict would say nothing about the bench it validates.
    import numpy as np
    from scripts.bench_control_validation import make_provider
    from tests.test_bench_flux_step import _fake_rig

    ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(7))
    calls = []
    make_provider(mode, ckpt, groups, shapes, decode=_counting_reference_decode(calls))
    assert len(calls) == want


# --- orchestrator setup errors -----------------------------------------------------------------------


def test_an_orchestrator_setup_error_exits_2_and_launches_nothing(tmp_path, monkeypatch, capsys):
    # Bug caught: a mistyped --embeds escaping main as a traceback with exit 1, the bit-mismatch
    # kill signal of the exit-code contract, instead of the tool error 2.
    import scripts.bench_control_validation as cv
    from scripts.bench_control_validation import main

    calls = []
    monkeypatch.setattr(cv.subprocess, "run", lambda *a, **k: calls.append(a))
    code = main(
        [
            *("--df11", str(tmp_path / "ckpt"), "--out", str(tmp_path / "out")),
            *("--embeds", str(tmp_path / "missing.safetensors")),
        ]
    )
    assert code == 2
    assert calls == []
    assert "missing.safetensors" in capsys.readouterr().err
