"""The pure parts of the FLUX step bench: modes, interleaving, resume, launch counts, the report.

Expected values are worked by hand from the fixture step times below, never from the code.
No test here imports mflux or touches the GPU.
"""

import json
import sys
from pathlib import Path

import pytest
from scripts._bench_common import resume_key_diff, write_json_atomic
from scripts._flux_rig import FLUX_CACHE_LIMIT, DF11Provider, ReuseProvider, install_placeholders
from scripts.bench_flux_step import (
    MODES,
    BenchError,
    check_embeds_shapes,
    child_command,
    compressed_set_loaded,
    expected_launches,
    interleaved,
    limits_in_force,
    make_provider,
    mode_policy,
    parse_args,
    pending_runs,
    read_results,
    report,
    result_is_complete,
    resume_conflicts,
    run_key,
    run_path,
)

# Two rounds of fixture step times (seconds). Medians: r1 df11 1.2, control 1.0, df11-depth2 1.1,
# control-depth2 1.0, control-noeval 0.9; r2 df11 1.5, control 1.1. Pooled df11 (6 reps) 1.35,
# pooled control 1.0. `verify_s` is the status validation timed outside the step window: pooled
# df11 (0.01, 0.02, 0.03, 0.02, 0.04, 0.03) has median 0.025; control-noeval carries none. Round 3
# has df11 only (the orchestration stopped before its control): an unpaired round that must stay
# out of every pooled number.
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
    {"round": 3, "mode": "df11", "step_s": [9.0, 9.0, 9.0], "verify_s": [0.5, 0.5, 0.5]},
]
KEY_KWARGS = {
    "model": "schnell",
    "size": 1024,
    "steps": 5,
    "warmup": 2,
    "seed": 42,
    "df11": Path("/ckpt"),
    "embeds": Path("/e.safetensors"),
    "embeds_meta": {"synthetic": "true", "seed": "42"},
    "source": "abc",
    "mlx": "0.32.2",
    "cache_limit": FLUX_CACHE_LIMIT,
}
KEY = run_key(**KEY_KWARGS)
# The legacy orchestration's child settings (the scenario-free path).
CHILD_KWARGS = {
    "steps": 5,
    "warmup": 2,
    "model": "schnell",
    "size": 1024,
    "seed": 42,
    "wall_budget": 1200.0,
    "cache_limit": FLUX_CACHE_LIMIT,
    "trace": False,
}
REPO = Path(__file__).resolve().parents[1]
SCENARIO_FILE = REPO / "bench/scenarios/flux1-schnell-1024.toml"
# The committed schnell scenario's DF11 pin (bench/scenarios/flux1-schnell-1024.toml).
SCHNELL_DF11_REVISION = "51a428b928197e0531cb93d6e438941e2d0b247e"
BASE_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"  # the schnell scenario's pinned base
GIB = 1024**3
HOST_REC = 24 * GIB  # a 32 GB host's recommended working set, a round figure for the tests


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
        **{**CHILD_KWARGS, "model": "dev", "size": 768},
    )
    assert cmd[:3] == [sys.executable, "-m", "scripts.bench_flux_step"]
    assert "--orchestrate" not in cmd
    assert "--trace" not in cmd
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
        "--cache-limit": str(FLUX_CACHE_LIMIT),
    }


def test_child_command_forwards_trace_and_a_custom_cache_limit(tmp_path):
    # Bug caught: a traced or cache-limit orchestration whose children run untraced at the default
    # limit (the A/B would compare two identical runs).
    cmd = child_command(
        mode="df11",
        round_no=1,
        out=tmp_path / "r.json",
        df11=tmp_path / "ckpt",
        embeds=tmp_path / "e.safetensors",
        **{**CHILD_KWARGS, "cache_limit": 2_500_000_000, "trace": True},
    )
    assert cmd[-1] == "--trace"
    flags = dict(zip(cmd[3:-1:2], cmd[4:-1:2], strict=True))
    assert flags["--cache-limit"] == "2500000000"


def test_parse_args_defaults_to_the_rig_cache_limit_and_no_trace():
    args = parse_args(["--mode", "df11", "--out", "r.json", "--df11", "d", "--embeds", "e"])
    assert args.cache_limit == FLUX_CACHE_LIMIT
    assert args.trace is False
    traced = parse_args(
        [*("--mode", "df11", "--out", "r.json", "--df11", "d", "--embeds", "e"), "--trace"]
    )
    assert traced.trace is True


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
    assert rounds[3] == {}  # round 3 has no pair at all


def test_report_pools_every_timed_step_of_a_mode_across_rounds():
    # Bug caught: pooling per-round medians (df11 would be median(1.2, 1.5) = 1.35 too, so the
    # spread pins it: over 6 reps (1.6 - 1.1) / 1.35, over 2 medians (1.5 - 1.2) / 1.35).
    pooled = report(FIXTURES)["pooled"]
    assert pooled["df11"] == {
        "median": pytest.approx(1.35),
        "spread": pytest.approx(0.5 / 1.35),
        "n": 6,
        "verify_median_s": pytest.approx(0.025),
        "trace": None,
    }
    assert pooled["control"] == {
        "median": pytest.approx(1.0),
        "spread": pytest.approx(0.2),
        "n": 6,
        "verify_median_s": pytest.approx(0.0),
        "trace": None,
    }
    assert pooled["control-noeval"]["n"] == 3


def test_report_pools_the_status_validation_time_outside_the_step_window():
    # Bug caught: verify_s folded into step_s (df11's pooled median would read 1.38 instead of
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


def test_report_from_a_single_mode_has_no_overheads_and_nothing_pooled():
    # Bug caught: pooling an unpaired mode (its median would stand alone, as if measured).
    out = report([FIXTURES[0]])
    assert out["overhead"] == {}
    assert out["rounds"] == {1: {}}
    assert out["pooled"] == {}


def test_report_pools_only_rounds_where_both_modes_of_the_pair_completed():
    # Bug caught: round 3's unpaired df11 (9.0 s) pulled into the pool: the df11 median would jump
    # from 1.35 to 1.5 and the overhead from 0.35 to 0.5.
    out = report(FIXTURES)
    assert out["pooled"]["df11"]["n"] == 6
    assert out["pooled"]["df11"]["median"] == pytest.approx(1.35)
    assert out["pooled"]["df11"]["verify_median_s"] == pytest.approx(0.025)
    assert out["overhead"]["per-block"] == pytest.approx(0.35)
    assert out["paired_rounds"] == {
        "per-block": [1, 2],
        "depth2": [1],
        "prefetch": [],
        "prefetch-inline": [],
        "eval-policy": [1],
    }


def test_report_of_a_stopped_orchestration_suppresses_the_pooled_overhead():
    # Bug caught: a stopped run printing a pooled overhead as if the orchestration had finished.
    stopped = {"round": 3, "mode": "control", "exit_code": 2}
    out = report(FIXTURES, stopped=stopped)
    assert out["stopped"] == stopped
    assert out["overhead"] == {}
    assert out["eval_policy_cost_s"] is None
    assert out["rounds"][1]["per-block"] == pytest.approx(0.2)  # the per-round pairs stay


def test_report_of_a_finished_orchestration_records_no_stop():
    assert report(FIXTURES)["stopped"] is None


# --- the resume key -----------------------------------------------------------------------------


def test_run_key_resolves_the_checkpoint_path_and_carries_every_setting(tmp_path):
    # Bug caught: a relative --df11 in the key (the same checkpoint reached from another cwd would
    # read as a mismatch), or a setting left out (a 768 px rerun resuming into 1024 px files).
    key = run_key(
        model="dev",
        size=768,
        steps=3,
        warmup=1,
        seed=7,
        df11=tmp_path / "ckpt",
        embeds=tmp_path / "e.safetensors",
        embeds_meta={"synthetic": "false", "prompt": "a lighthouse"},
        source="deadbeef",
        mlx="0.32.2",
        cache_limit=2_500_000_000,
    )
    assert key == {
        "model": "dev",
        "size": 768,
        "steps": 3,
        "warmup": 1,
        "seed": 7,
        "df11": str((tmp_path / "ckpt").resolve()),
        "embeds": str(tmp_path / "e.safetensors"),
        "embeds_meta": {"synthetic": "false", "prompt": "a lighthouse"},
        "source": "deadbeef",
        "mlx": "0.32.2",
        "cache_limit": 2_500_000_000,
        "scenario_hash": None,
        "tier_gb": None,
    }


@pytest.mark.parametrize("field", sorted(KEY))
def test_a_stored_key_differing_in_one_field_is_a_conflict_naming_that_field(tmp_path, field):
    # Bug caught: a field missing from the comparison, so a run with other settings (or another
    # checkpoint, embeddings file, source or mlx) resumes into these files.
    changed = {**KEY, field: "other" if field != "embeds_meta" else {"synthetic": "false"}}
    write_json_atomic(
        run_path(tmp_path, 1, "df11"), {"exit_code": 0, "step_s": [1.0], "key": changed}
    )
    assert resume_conflicts(tmp_path, 1, KEY) == [(1, "df11", [field])]


def test_a_stored_result_without_a_key_is_a_conflict(tmp_path):
    # Bug caught: treating a keyless (older) file as matching.
    write_json_atomic(run_path(tmp_path, 1, "control"), {"exit_code": 0, "step_s": [1.0]})
    assert resume_conflicts(tmp_path, 1, KEY) == [(1, "control", ["key"])]


def test_a_failed_result_with_the_same_key_is_no_conflict_and_is_rerun(tmp_path):
    # Bug caught: a key check that skips incomplete files (they would be overwritten unchecked), or
    # a resume that skips a failed run because its key matches.
    write_json_atomic(run_path(tmp_path, 1, "df11"), {"exit_code": 2, "error": "boom", "key": KEY})
    write_json_atomic(
        run_path(tmp_path, 1, "control"), {"exit_code": 0, "step_s": [1.0], "key": KEY}
    )
    assert resume_conflicts(tmp_path, 1, KEY) == []
    assert pending_runs(tmp_path, 1)[:2] == [(1, "df11"), (1, "df11-depth2")]


def test_missing_files_are_no_conflict(tmp_path):
    assert resume_conflicts(tmp_path, 2, KEY) == []


# --- argument validation ------------------------------------------------------------------------


def _single_run_args(*extra):
    return ["--mode", "df11", "--out", "o.json", "--df11", "d", "--embeds", "e", *extra]


def test_a_warmup_below_one_is_a_usage_error():
    # Bug caught: allowing --warmup 0, so the first timed step carries the real-group pipeline
    # compile and the lazy scheduler construction.
    with pytest.raises(SystemExit) as exc:
        parse_args(_single_run_args("--warmup", "0"))
    assert exc.value.code == 2


def test_a_warmup_of_one_is_accepted():
    assert parse_args(_single_run_args("--warmup", "1")).warmup == 1


def test_read_results_loads_the_completed_files_of_every_round(tmp_path):
    # Bug caught: reading only round 1, or reading a failed child's file into the report.
    for r in FIXTURES:
        write_json_atomic(run_path(tmp_path, r["round"], r["mode"]), {"exit_code": 0, **r})
    write_json_atomic(run_path(tmp_path, 2, "df11-depth2"), {"exit_code": 2, "error": "boom"})
    loaded = read_results(tmp_path, 3)
    assert [(r["round"], r["mode"]) for r in loaded] == [(r["round"], r["mode"]) for r in FIXTURES]
    assert json.loads(run_path(tmp_path, 2, "df11").read_text())["step_s"] == [1.5, 1.4, 1.6]


# --- the shared step loop (fakes for the mflux transformer and scheduler) ------------------------


class _FakeScheduler:
    """Scales nothing; ``step`` returns ``latents - noise`` (or ``bad`` from step ``bad_from`` on)."""

    def __init__(self, bad_from=None, bad=float("nan")):
        self.bad_from = bad_from
        self.bad = bad

    def scale_model_input(self, latents, t):
        return latents

    def step(self, *, noise, timestep, latents):
        out = latents - noise
        if self.bad_from is not None and timestep >= self.bad_from:
            out = out * 0 + self.bad
        return out


class _FakeConfig:
    def __init__(self, scheduler):
        self.scheduler = scheduler


class _FakeProvider:
    def __init__(self):
        self.launches = 0


class _FakeTransformer:
    """Counts ``launches[t]`` decode launches on the provider at step ``t``; tracks verify calls.

    Steps before ``spike_until`` allocate and drop a 256 MiB array, a stand-in for the load spike.
    """

    def __init__(self, provider, launches, *, spike_until=0):
        self.provider = provider
        self.launches = launches
        self.spike_until = spike_until
        self.verified = 0

    def __call__(self, *, t, config, hidden_states, prompt_embeds, pooled_prompt_embeds):
        import mlx.core as mx

        self.provider.launches += self.launches[t]
        if t < self.spike_until:
            spike = mx.zeros((256 * 1024 * 1024,), dtype=mx.uint8)
            mx.eval(spike)
            del spike
        return hidden_states * 0.5

    def verify_step(self):
        self.verified += 1


class _FakeWatchdog:
    def __init__(self, peak_footprint=0, peak_watched=0):
        self.peak_footprint = peak_footprint
        self.peak_watched = peak_watched

    def reset_peak(self):
        self.peak_footprint = 0
        self.peak_watched = 0


def _time(
    transformer,
    provider,
    *,
    scheduler=None,
    warmup=2,
    steps=3,
    per_step=12,
    limits=True,
    loaded=True,
    watchdog=None,
    tracer=None,
):
    import mlx.core as mx
    from scripts.bench_flux_step import time_steps

    return time_steps(
        transformer,
        provider,
        _FakeConfig(scheduler or _FakeScheduler()),
        mx.ones((1, 4, 8), dtype=mx.float32),
        mx.zeros((1, 2, 4)),
        mx.zeros((1, 4)),
        warmup=warmup,
        steps=steps,
        per_step=per_step,
        compressed_loaded=loaded,
        limits_recorded=limits,
        watchdog=watchdog if watchdog is not None else _FakeWatchdog(),
        label="df11",
        tracer=tracer,
    )


def test_time_steps_times_only_the_steps_after_the_warmup_and_verifies_every_step():
    # Bug caught: the warm-up steps landing in step_s (their pipeline compile would inflate the
    # median), or verify_step skipped (a decode status error would go unseen).
    provider = _FakeProvider()
    transformer = _FakeTransformer(provider, [12] * 5)
    out = _time(transformer, provider)
    assert len(out["warmup_s"]) == 2
    assert len(out["step_s"]) == 3
    assert len(out["verify_s"]) == 3
    assert out["launches_per_step"] == [12, 12, 12, 12, 12]
    assert out["launches_total"] == 60
    assert out["launches_expected_per_step"] == 12
    assert transformer.verified == 5


def test_time_steps_refuses_a_timed_step_with_other_launches_than_the_mode():
    # Bug caught: only the warm-up total checked, so a timed step that skipped a block's decode
    # (11 launches instead of 12) would be timed as if it had decoded every block.
    provider = _FakeProvider()
    transformer = _FakeTransformer(provider, [12, 12, 12, 11, 12])
    with pytest.raises(BenchError, match=r"\[12, 11, 12\] launches; df11 expects 12"):
        _time(transformer, provider)


def test_time_steps_names_the_failed_parity_conditions_before_timing():
    # Bug caught: a failed condition ignored (the caps not recorded, the compressed set not loaded,
    # or a launch count off in the warm-up) and the steps timed anyway.
    from scripts.bench_flux_step import ParityError

    provider = _FakeProvider()
    transformer = _FakeTransformer(provider, [12, 13, 12, 12, 12])
    with pytest.raises(ParityError) as exc:
        _time(transformer, provider, limits=False, loaded=False)
    assert exc.value.failed == ["compressed_loaded", "limits_recorded", "launches_expected"]
    assert transformer.verified == 2  # only the warm-up ran


def test_time_steps_records_the_timed_steps_own_peaks_apart_from_the_lifetime_peaks():
    # Bug caught: peaks that span the process lifetime (the load and warm-up spike would hide what the
    # timed steps themselves reach), or a reset before the timed steps that loses the lifetime peak.
    provider = _FakeProvider()
    transformer = _FakeTransformer(provider, [0] * 5, spike_until=2)
    out = _time(transformer, provider, per_step=0, watchdog=_FakeWatchdog(peak_footprint=10**13))
    assert out["footprint_peak_bytes"] >= 10**13
    assert out["step_footprint_peak_bytes"] < 10**13
    assert out["mlx_peak_memory_bytes"] >= 256 * 1024**2
    assert out["step_mlx_peak_bytes"] < 256 * 1024**2


def test_time_steps_records_the_watched_peak_of_the_timed_steps_and_of_the_lifetime():
    # Bug caught: the watched peak (the number the watchdog enforces, the one the tier rows use)
    # left out of the result, read without the reset (the load spike lands in the step peak), or
    # the lifetime value lost to that reset.
    provider = _FakeProvider()
    transformer = _FakeTransformer(provider, [0] * 5)
    out = _time(transformer, provider, per_step=0, watchdog=_FakeWatchdog(peak_watched=10**13))
    assert out["watched_peak_bytes"] >= 10**13
    assert 0 < out["step_watched_peak_bytes"] < 10**13


# --- the parity condition helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("caps", "want"),
    [
        ((20, 22), True),
        ((0, 22), False),  # the wired cap failed to install
        ((20, 0), False),  # the memory cap failed to install
        ((0, 0), False),
        ((20,), False),  # one cap missing
    ],
)
def test_limits_in_force_needs_both_caps(caps, want):
    # Bug caught: checking caps[0] alone (a failed memory cap passes). The MLX cache limit is not
    # part of the condition: mlx 0.32.2 has no getter, so a read-back would compare the requested
    # value with itself.
    assert limits_in_force(caps) is want


def _fake_rig(rng, *, n_double=2, n_single=2):
    """A fake transformer's shapes, one real DF11 group per block, a ckpt-shaped view of their names."""
    from types import SimpleNamespace

    from tests._flux_fakes import FLUX_TABLE, FakeTransformer, Recorder, block_lists, df11_groups

    tf = FakeTransformer(Recorder(), n_double=n_double, n_single=n_single)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, source = df11_groups(shapes, rng)
    ckpt = SimpleNamespace(groups={n: SimpleNamespace(matrix_names=names[n]) for n in names})
    return ckpt, groups, shapes, source


def test_compressed_set_loaded_needs_every_block_resident_with_elements():
    # Bug caught: a constant True (the condition the run JSON reports could never fail), a resident
    # set missing a block the transformer has, or a group with no elements counted as loaded.
    from types import SimpleNamespace

    import numpy as np

    _ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(3), n_double=1, n_single=1)
    assert compressed_set_loaded(groups, shapes) is True
    missing = {k: v for k, v in groups.items() if k != "single_transformer_blocks.0"}
    assert compressed_set_loaded(missing, shapes) is False
    empty = {**groups, "transformer_blocks.0": SimpleNamespace(n_elements=0)}
    assert compressed_set_loaded(empty, shapes) is False


# --- the providers --------------------------------------------------------------------------------


def _counting_reference_decode(calls):
    from mlx_dfloat.decode import decode_group

    def decode(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    return decode


def test_make_provider_for_a_control_decodes_block_0_of_each_kind_once():
    # Bug caught: the control decoding every block (that is the resident BF16 model, not the
    # two-block control), decoding other blocks than the first of each kind, or launching in the step.
    import mlx.core as mx
    import numpy as np
    from tests._flux_fakes import FLUX_TABLE

    ckpt, groups, shapes, source = _fake_rig(np.random.default_rng(4))
    calls = []
    provider = make_provider(
        "control",
        ckpt,
        groups,
        shapes,
        decode=_counting_reference_decode(calls),
        name_map=FLUX_TABLE,
    )
    assert calls == ["transformer_blocks.0", "single_transformer_blocks.0"]
    assert isinstance(provider, ReuseProvider)
    # Every block of a kind gets block 0's decoded weights: the control's design.
    w = provider.weights_for("transformer_blocks.1", shapes["transformer_blocks.1"])
    want = source["transformer_blocks.0"]["transformer_blocks.0.attn.to_q.weight"]
    assert np.array_equal(np.array(w["attn.to_q"].view(mx.uint16)), want)
    assert provider.launches == 0


@pytest.mark.parametrize("mode", ["df11", "df11-depth2"])
def test_make_provider_for_df11_decodes_nothing_before_the_first_step(mode):
    # Bug caught: a df11 provider that decodes at construction (launches before the first step, which
    # the warm-up's launch count would then miss).
    import numpy as np
    from tests._flux_fakes import FLUX_TABLE

    ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(5))
    calls = []
    provider = make_provider(
        mode, ckpt, groups, shapes, decode=_counting_reference_decode(calls), name_map=FLUX_TABLE
    )
    assert calls == []
    assert isinstance(provider, DF11Provider)
    provider.weights_for("transformer_blocks.1", shapes["transformer_blocks.1"])
    assert calls == ["transformer_blocks.1"]


def test_time_steps_refuses_non_finite_latents_after_the_warmup():
    from scripts.bench_flux_step import ParityError

    provider = _FakeProvider()
    with pytest.raises(ParityError) as exc:
        _time(
            _FakeTransformer(provider, [0] * 5), provider, scheduler=_FakeScheduler(1), per_step=0
        )
    assert exc.value.failed == ["finite"]


def test_time_steps_refuses_non_finite_final_latents():
    # Bug caught: only the warm-up latents checked; an inf appearing in a timed step would be timed
    # and reported as a valid run.
    provider = _FakeProvider()
    with pytest.raises(BenchError, match="not finite"):
        _time(
            _FakeTransformer(provider, [0] * 5),
            provider,
            scheduler=_FakeScheduler(3, float("inf")),
            per_step=0,
        )


# --- run_one with an injected measurement and key ------------------------------------------------


@pytest.fixture
def restore_cache_limit():
    import mlx.core as mx

    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    yield
    mx.set_cache_limit(previous)


def _one_run_args(tmp_path):
    return parse_args(
        [
            *("--mode", "control", "--out", str(tmp_path / "r.json")),
            *("--df11", "d", "--embeds", "e", "--wall-budget", "60"),
        ]
    )


def test_run_one_writes_the_injected_measurement_under_the_injected_key(
    tmp_path, restore_cache_limit
):
    # Bug caught: run_one ignoring the injected key (the reduced-depth validation would write
    # bench_flux_step's key without its depth, so every resume would read as a conflict) or the
    # injected measurement (it would run the full-depth mode).
    from scripts.bench_flux_step import run_one

    def measure(args, watchdog, *, limits_recorded):
        return {
            "exit_code": 0,
            "policy": "per-block",
            "median_s": 1.0,
            "spread": 0.0,
            "verify_median_s": 0.0,
            "launches_expected_per_step": 0,
            "footprint_peak_bytes": 1,
            "limits_recorded": limits_recorded,
        }

    args = _one_run_args(tmp_path)
    code = run_one(args, measure=measure, key=lambda a: {"double": 4})
    written = json.loads((tmp_path / "r.json").read_text())
    assert code == 0
    assert written["key"] == {"double": 4}
    assert written["limits_recorded"] is True
    assert written["mode"] == "control"


def test_run_one_moves_a_stale_abort_artifact_aside_before_it_starts(tmp_path, restore_cache_limit):
    # Bug caught: a resumed orchestration finishing next to a previous child's abort.json, which then
    # reads as this run's outcome.
    from scripts.bench_flux_step import run_one

    (tmp_path / "abort.json").write_text('{"reason": "stale"}')

    def measure(args, watchdog, *, limits_recorded):
        return {
            "exit_code": 0,
            "policy": "per-block",
            "median_s": 1.0,
            "spread": 0.0,
            "verify_median_s": 0.0,
            "launches_expected_per_step": 0,
            "footprint_peak_bytes": 1,
        }

    assert run_one(_one_run_args(tmp_path), measure=measure, key=lambda a: {}) == 0
    assert (tmp_path / "abort.previous.json").read_text() == '{"reason": "stale"}'
    assert not (tmp_path / "abort.json").exists()


def test_run_one_records_a_failed_measurement_as_exit_2_with_its_key(tmp_path, restore_cache_limit):
    # Bug caught: an exception escaping without a JSON (the orchestrator could not tell a failed
    # run from a missing one) or the key left out of the failed record.
    from scripts.bench_flux_step import run_one

    def measure(args, watchdog, *, limits_recorded):
        raise RuntimeError("boom")

    code = run_one(_one_run_args(tmp_path), measure=measure, key=lambda a: {"double": 4})
    written = json.loads((tmp_path / "r.json").read_text())
    assert code == 2
    assert written["exit_code"] == 2
    assert "boom" in written["error"]
    assert written["key"] == {"double": 4}


def test_an_orchestrator_setup_error_exits_2_and_launches_nothing(tmp_path, monkeypatch, capsys):
    # Bug caught: a mistyped --embeds escaping main as a traceback with exit 1, the bit-mismatch
    # kill signal of the exit-code contract, instead of the tool error 2.
    import scripts.bench_flux_step as bfs

    calls = []
    monkeypatch.setattr(bfs.subprocess, "run", lambda *a, **k: calls.append(a))
    code = bfs.main(
        [
            *("--orchestrate", "--out-dir", str(tmp_path / "out"), "--df11", str(tmp_path / "c")),
            *("--embeds", str(tmp_path / "missing.safetensors")),
        ]
    )
    assert code == 2
    assert calls == []
    assert "missing.safetensors" in capsys.readouterr().err


def test_the_orchestrator_launches_children_from_the_repository_root_with_absolute_paths(
    tmp_path, monkeypatch
):
    # Children run `python -m scripts.bench_flux_step`, which only resolves with the repository root as cwd.
    # Bug caught: launching from the caller's cwd (an orchestration started elsewhere fails every child), or
    # handing the root-cwd children relative paths that then point inside the repository.
    from types import SimpleNamespace

    import scripts.bench_flux_step as bfs

    calls = []

    def run(cmd, **kwargs):
        if (
            "scripts.bench_flux_step" in cmd
        ):  # a child; provenance's git calls pass through the stub too
            calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=3, stdout="")  # the first child fails: stop there

    monkeypatch.setattr(bfs, "current_key", lambda args: {"model": "schnell"})
    monkeypatch.setattr(bfs.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)
    code = bfs.main(
        [
            *("--orchestrate", "--out-dir", "out", "--df11", "c", "--embeds", "e.safetensors"),
        ]
    )
    assert code == 2
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert kwargs["cwd"] == Path(__file__).resolve().parents[1]
    flags = dict(zip(cmd[3::2], cmd[4::2], strict=False))
    here = tmp_path.resolve()
    assert flags["--df11"] == str(here / "c")
    assert flags["--embeds"] == str(here / "e.safetensors")
    assert Path(flags["--out"]).parent == here / "out"


# --- trace ----------------------------------------------------------------------------------------


class _TracingTransformer(_FakeTransformer):
    """The fake, recording one block event per call into ``tracer`` the way the seam does."""

    def __init__(self, provider, launches, tracer):
        super().__init__(provider, launches)
        self.tracer = tracer

    def __call__(self, **kwargs):
        import time

        self.tracer.begin_step()
        t0 = time.perf_counter()
        out = super().__call__(**kwargs)
        t1 = time.perf_counter()
        self.tracer.record("b", decode_start=t0, decode_end=t0, encode_end=t1, eval_end=t1)
        self.tracer.end_step()
        return out


def test_time_steps_with_a_tracer_summarises_only_the_timed_steps():
    # Bug caught: warm-up steps' events in the trace (their compile time would pollute the
    # medians), or a summary built from another step's window.
    from scripts._flux_rig import Tracer

    provider = _FakeProvider()
    tracer = Tracer()
    transformer = _TracingTransformer(provider, [12] * 5, tracer)
    out = _time(transformer, provider, tracer=tracer)
    assert len(tracer.events) == 5
    assert [e["step"] for e in out["trace_events"]] == [2, 3, 4]
    assert [s["n_blocks"] for s in out["trace_steps"]] == [1, 1, 1]
    for summary, (start, end), event in zip(
        out["trace_steps"], tracer.steps[2:], out["trace_events"], strict=True
    ):
        assert summary["step_s"] == pytest.approx(end - start)
        assert summary["head_s"] == pytest.approx(event["t_decode_start"] - start)
        assert summary["head_s"] >= 0  # an off-by-one in the step filter goes negative here
    assert out["trace_medians"].keys() == out["trace_steps"][0].keys()


def test_time_steps_without_a_tracer_records_no_trace_fields():
    provider = _FakeProvider()
    out = _time(_FakeTransformer(provider, [12] * 5), provider)
    assert not {"trace_steps", "trace_events", "trace_medians"} & out.keys()


def test_trace_medians_takes_the_median_of_every_phase_across_steps():
    from scripts.bench_flux_step import trace_medians

    steps = [
        {"n_blocks": 2, "host_decode_s": 0.1, "eval_wait_s": 1.0},
        {"n_blocks": 2, "host_decode_s": 0.3, "eval_wait_s": 3.0},
        {"n_blocks": 2, "host_decode_s": 0.2, "eval_wait_s": 9.0},
    ]
    assert trace_medians(steps) == {
        "n_blocks": 2,
        "host_decode_s": pytest.approx(0.2),
        "eval_wait_s": pytest.approx(3.0),
    }


def test_report_pools_the_trace_phases_over_the_paired_rounds():
    # Bug caught: a trace pooled from one round only, or from rounds whose pair did not complete.
    traced = [
        {**FIXTURES[0], "trace_steps": [{"host_decode_s": 0.1}, {"host_decode_s": 0.2}]},
        FIXTURES[1],
        {**FIXTURES[5], "trace_steps": [{"host_decode_s": 0.4}]},
        FIXTURES[6],
        {**FIXTURES[7], "trace_steps": [{"host_decode_s": 9.0}]},  # unpaired round 3
    ]
    pooled = report(traced)["pooled"]
    assert pooled["df11"]["trace"] == {"host_decode_s": pytest.approx(0.2)}
    assert pooled["control"]["trace"] is None


@pytest.mark.parametrize("value", ["-1", "0", "18446744073709551616", "abc"])
def test_a_cache_limit_that_is_not_a_positive_size_is_a_usage_error(value):
    # Bug caught: a negative or oversized value reaching mx.set_cache_limit (a TypeError outside
    # run_one's guard, exit 1, which this project reserves for a real bit mismatch), or 0 silently
    # turning the buffer cache off.
    with pytest.raises(SystemExit) as exc:
        parse_args(
            [
                *("--mode", "df11", "--out", "r.json", "--df11", "d", "--embeds", "e"),
                "--cache-limit",
                value,
            ]
        )
    assert exc.value.code == 2


def test_run_one_sets_and_records_the_requested_cache_limit(tmp_path, restore_cache_limit):
    import mlx.core as mx
    from scripts.bench_flux_step import run_one

    seen = {}

    def measure(args, watchdog, *, limits_recorded):
        seen["limit"] = int(mx.set_cache_limit(0))
        mx.set_cache_limit(seen["limit"])
        return {
            "exit_code": 0,
            "policy": "per-block",
            "median_s": 1.0,
            "spread": 0.0,
            "verify_median_s": 0.0,
            "launches_expected_per_step": 0,
            "footprint_peak_bytes": 1,
            "limits_recorded": limits_recorded,
        }

    args = _one_run_args(tmp_path)
    code = run_one(args, measure=measure, key=lambda a: {}, cache_limit=2_000_000_000)
    written = json.loads((tmp_path / "r.json").read_text())
    assert code == 0
    assert seen["limit"] == 2_000_000_000
    assert written["cache_limit_bytes"] == 2_000_000_000
    assert written["limits_recorded"] is True


# --- prefetch modes -------------------------------------------------------------------------------


def test_prefetch_modes_are_df11_modes_under_the_per_block_policy():
    from scripts.bench_flux_step import EXTRA_MODES, PAIRS

    for mode in EXTRA_MODES:
        assert mode_policy(mode) == "per-block"
        assert expected_launches(mode, n_double=19, n_single=38, steps=2) == 114
    assert ("prefetch", "df11-prefetch", "control") in PAIRS
    assert ("prefetch-inline", "df11-prefetch-inline", "control") in PAIRS
    assert not set(EXTRA_MODES) & set(MODES)  # the verdict recipe stays the five modes


def test_make_provider_for_prefetch_modes_wraps_the_decoder_with_the_right_stream():
    # Bug caught: the second-stream mode decoding on the default stream (the A/B would compare two
    # identical runs), or the inline mode given a new stream.
    import mlx.core as mx
    import numpy as np
    from scripts._flux_rig import PrefetchProvider
    from tests._flux_fakes import FLUX_TABLE

    from mlx_dfloat.decode import decode_group

    ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(5), n_double=1, n_single=1)
    streams = []

    def recording_decode(group):
        streams.append(mx.default_stream(mx.gpu))  # the stream a kernel launched here would use
        return decode_group(group, backend="reference")

    two = make_provider(
        "df11-prefetch", ckpt, groups, shapes, decode=recording_decode, name_map=FLUX_TABLE
    )
    one = make_provider(
        "df11-prefetch-inline", ckpt, groups, shapes, decode=recording_decode, name_map=FLUX_TABLE
    )
    assert isinstance(two, PrefetchProvider)
    assert isinstance(one, PrefetchProvider)
    assert two.stream is not None
    assert two.stream != mx.default_stream(mx.gpu)
    assert one.stream is None
    assert streams == []  # nothing decoded before the first step
    two.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    assert streams == [two.stream, two.stream]  # the cold decode and the look-ahead
    one.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    assert streams[2:] == [mx.default_stream(mx.gpu)] * 2


def test_orchestrate_runs_only_the_requested_modes(tmp_path, monkeypatch):
    # Bug caught: --modes ignored (the prefetch experiment would run the five-mode recipe).
    from types import SimpleNamespace

    import scripts.bench_flux_step as bfs

    launched = []
    commands = []

    def run(cmd, **kwargs):
        if "scripts.bench_flux_step" in cmd:
            launched.append(cmd[cmd.index("--mode") + 1])
            commands.append(cmd)
            out = Path(cmd[cmd.index("--out") + 1])
            write_json_atomic(
                out, {"exit_code": 0, "step_s": [1.0], "mode": launched[-1], "round": 1}
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(bfs, "current_key", lambda args: {"model": "schnell"})
    monkeypatch.setattr(bfs.subprocess, "run", run)
    code = bfs.main(
        [
            *("--orchestrate", "--rounds", "1", "--out-dir", str(tmp_path / "out")),
            *("--df11", "c", "--embeds", "e.safetensors"),
            *("--modes", "df11-prefetch", "control"),
            *("--trace", "--cache-limit", "2500000000"),
        ]
    )
    assert code == 0
    assert launched == ["df11-prefetch", "control"]
    for cmd in commands:  # the children run what the orchestration asked for
        assert cmd[-1] == "--trace"
        assert cmd[cmd.index("--cache-limit") + 1] == "2500000000"
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["overhead"] == {"prefetch": pytest.approx(0.0)}


def test_modes_without_orchestrate_is_a_usage_error():
    # Bug caught: `--modes` silently ignored on a single-mode run.
    with pytest.raises(SystemExit):
        parse_args(
            [
                *("--mode", "df11", "--out", "r.json", "--df11", "d", "--embeds", "e"),
                "--modes",
                "df11",
            ]
        )


def test_main_runs_one_mode_at_the_requested_cache_limit(tmp_path, monkeypatch):
    # Bug caught: main dropping the kwarg, so a --cache-limit child runs at the rig's default
    # while its key and limits still claim the requested one.
    import scripts.bench_flux_step as bfs

    seen = {}

    def run_one(args, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(bfs, "run_one", run_one)
    code = bfs.main(
        [
            *("--mode", "df11", "--out", str(tmp_path / "r.json"), "--df11", "d", "--embeds", "e"),
            *("--cache-limit", "2000000000"),
        ]
    )
    assert code == 0
    assert seen["cache_limit"] == 2_000_000_000


def test_current_key_carries_the_cache_limit(tmp_path, monkeypatch):
    import scripts.bench_flux_step as bfs

    monkeypatch.setattr(bfs, "embeds_metadata", lambda path: {"synthetic": "true"})
    monkeypatch.setattr(bfs, "source_hash", lambda: "abc")
    args = parse_args(
        [
            *("--mode", "df11", "--out", "r.json", "--df11", "d", "--embeds", "e"),
            *("--cache-limit", "2500000000"),
        ]
    )
    assert bfs.current_key(args)["cache_limit"] == 2_500_000_000


def test_report_pools_a_shared_control_over_each_pairs_own_rounds():
    # Bug caught: control pooled once, over the rounds of whichever pair came last, so the per-block
    # overhead compared df11 over rounds 1-2 with a control over round 1 only (+40 % here instead of
    # -6.7 %).
    results = [
        {"round": 1, "mode": "df11", "step_s": [1.2]},
        {"round": 1, "mode": "control", "step_s": [1.0]},
        {"round": 1, "mode": "df11-prefetch", "step_s": [1.1]},
        {"round": 2, "mode": "df11", "step_s": [1.6]},
        {"round": 2, "mode": "control", "step_s": [2.0]},
    ]
    out = report(results)
    assert out["paired_rounds"]["per-block"] == [1, 2]
    assert out["paired_rounds"]["prefetch"] == [1]
    assert out["overhead"]["per-block"] == pytest.approx((1.4 - 1.5) / 1.5)
    assert out["overhead"]["prefetch"] == pytest.approx(0.1)
    assert out["pooled"]["control"]["n"] == 2  # the per-block pair's rounds
    assert out["pooled"]["df11-prefetch"]["n"] == 1


# --- scenario runs, the tier flag and the limits record -----------------------------------------


def _scenario_run_args(tmp_path, *extra, mode="df11", df11=None):
    df11 = df11 if df11 is not None else tmp_path / SCHNELL_DF11_REVISION
    return [
        *("--scenario", str(SCENARIO_FILE), "--mode", mode, "--out", str(tmp_path / "r.json")),
        *("--df11", str(df11), "--embeds", "e", *extra),
    ]


def _error_line(capsys):
    """argparse's own error line (the usage block above it names every flag, so it proves nothing)."""
    return capsys.readouterr().err.strip().splitlines()[-1]


def test_q8_is_scenario_only_with_no_launches_and_the_none_policy():
    # Bug caught: q8 appended to MODES (the legacy five-mode recipe would grow a sixth child), q8
    # counted as a DF11 mode (its steps would be refused for making no launches), or given an
    # eval policy (the quantized transformer has no seam to evaluate at).
    from scripts.bench_flux_step import SCENARIO_MODES, is_df11

    assert "q8" not in MODES
    assert "q8" in SCENARIO_MODES
    assert mode_policy("q8") == "none"
    assert is_df11("q8") is False
    assert expected_launches("q8", n_double=19, n_single=38, steps=5) == 0


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (["--scenario", "s.toml", "--mode", "df11", "--round", "1", "--trace"], []),
        (["--scenario", "s.toml", "--steps", "9"], ["--steps"]),
        (["--scenario", "s.toml", "--steps=9"], ["--steps"]),
        (
            ["--model", "dev", "--size", "512", "--warmup", "1", "--seed", "7"],
            ["--model", "--size", "--warmup", "--seed"],
        ),
        (["--cache-limit", "5", "--wall-budget", "60"], ["--cache-limit", "--wall-budget"]),
        (["--steps", "1", "--steps", "2"], ["--steps"]),  # named once
        (["--out", "--steps.json"], []),  # begins with a flag's text but is not that flag
    ],
)
def test_conflicting_flags_names_every_scenario_fixed_flag_given(argv, want):
    # Bug caught: a fixed flag missing from SCENARIO_FIXED_FLAGS (the scenario would silently win
    # over the command line, or the other way round), or the `--flag=value` form slipping through.
    from scripts.bench_flux_step import conflicting_flags

    assert conflicting_flags(argv) == want


def test_parse_args_refuses_an_abbreviated_fixed_flag_with_a_scenario(tmp_path, capsys):
    # Bug caught: argparse's prefix matching left on, so `--step 9` sets --steps unnoticed by the
    # exact-name conflict check and the scenario run's recipe is changed or silently overridden.
    import re

    with pytest.raises(SystemExit) as exc:
        parse_args(_scenario_run_args(tmp_path, "--step", "9"))
    assert exc.value.code == 2
    assert re.search(r"--step(?!s)", _error_line(capsys))


def test_parse_args_scenario_mode_refuses_fixed_flags(tmp_path, capsys):
    # Bug caught: a --steps next to --scenario accepted, so the child's JSON claims the scenario's
    # hash while it ran another step count (or the flag is dropped without a word).
    with pytest.raises(SystemExit) as exc:
        parse_args(_scenario_run_args(tmp_path, "--steps", "9"))
    assert exc.value.code == 2
    line = _error_line(capsys)
    assert "--steps" in line
    assert "--scenario" in line


def test_parse_args_scenario_mode_takes_the_settings_from_the_file(tmp_path):
    # The values are the committed schnell scenario's (bench/scenarios/flux1-schnell-1024.toml).
    # Bug caught: a scenario run left on the parser defaults (1.4 GB cache limit instead of the
    # file's 2.5 GB, 3600 s kept by accident), or no hash carried to the key and the JSON.
    from mlx_dfloat.bench.scenario import load_scenario, scenario_hash

    args = parse_args(_scenario_run_args(tmp_path))
    assert (args.model, args.size, args.steps, args.warmup, args.seed) == (
        "schnell",
        1024,
        5,
        2,
        42,
    )
    assert args.cache_limit == 2_500_000_000
    assert args.wall_budget == 3600.0
    assert args.scenario_hash == scenario_hash(load_scenario(SCENARIO_FILE))
    assert args.scenario_spec.df11_revision == SCHNELL_DF11_REVISION


def test_parse_args_refuses_q8_without_a_scenario(capsys):
    # Bug caught: a scenario-free q8 run, which has no pinned base checkpoint to quantize.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--mode", "q8", "--out", "r.json", "--df11", "d", "--embeds", "e"])
    assert exc.value.code == 2
    line = _error_line(capsys)
    assert "q8" in line
    assert "--scenario" in line


def test_parse_args_accepts_q8_with_a_scenario(tmp_path):
    # Bug caught: q8 missing from --mode's choices, so the orchestrator's q8 child cannot start.
    assert parse_args(_scenario_run_args(tmp_path, mode="q8")).mode == "q8"


@pytest.mark.parametrize(
    "extra",
    [
        ("--scenario", str(SCENARIO_FILE)),
        ("--tier", "24"),
    ],
)
def test_parse_args_refuses_a_scenario_or_tier_on_the_legacy_orchestration(extra, capsys):
    # Bug caught: the scenario-free orchestration accepting a scenario or a tier it never forwards
    # to its children (their JSONs would claim neither).
    with pytest.raises(SystemExit):
        parse_args(["--orchestrate", "--out-dir", "o", "--df11", "d", "--embeds", "e", *extra])
    line = _error_line(capsys)
    assert "--orchestrate" in line
    assert extra[0] in line


def test_parse_args_refuses_an_unreadable_scenario(tmp_path, capsys):
    # Bug caught: a DFloatFormatError escaping parse_args as a traceback (exit 1, the bit-mismatch
    # code) instead of a usage error naming the file.
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n')
    with pytest.raises(SystemExit) as exc:
        parse_args(
            ["--scenario", str(bad), "--mode", "df11", "--out", "r", "--df11", "d", "--embeds", "e"]
        )
    assert exc.value.code == 2
    line = _error_line(capsys)
    assert "bad.toml" in line
    assert "missing" in line


def test_parse_args_refuses_a_tier_below_one(capsys):
    # Bug caught: --tier 0 reaching tier_limits (a zero-byte budget; the watchdog aborts at once).
    with pytest.raises(SystemExit):
        parse_args(["--mode", "df11", "--out", "r", "--df11", "d", "--embeds", "e", "--tier", "0"])
    assert "--tier must be >= 1" in _error_line(capsys)


def test_legacy_orchestrate_still_runs_the_five_modes():
    # Bug caught: q8 (or any scenario-only mode) reaching the legacy orchestration's default list or
    # its --modes choices.
    args = parse_args(["--orchestrate", "--out-dir", "o", "--df11", "d", "--embeds", "e"])
    assert args.modes == list(MODES)
    with pytest.raises(SystemExit):
        parse_args(
            ["--orchestrate", "--out-dir", "o", "--df11", "d", "--embeds", "e", "--modes", "q8"]
        )


def test_settings_from_scenario_maps_every_keyed_field():
    # Bug caught: a keyed setting not taken from the file (the run would use the parser default
    # while its key claims the scenario), or a field mapped to the wrong option.
    from scripts.bench_flux_step import settings_from_scenario

    from mlx_dfloat.bench.scenario import Scenario

    scenario = Scenario(
        name="t",
        model="dev",
        df11_repo="o/d",
        df11_revision="a" * 40,
        base_repo="o/b",
        base_revision="b" * 40,
        prompt="p",
        seed=7,
        steps=3,
        warmup=1,
        size=512,
        rounds=2,
        cache_limit_bytes=2_000_000_001,
        conditions=("df11", "control"),
        wall_budget_s=900.0,
    )
    assert settings_from_scenario(scenario) == {
        "model": "dev",
        "size": 512,
        "steps": 3,
        "warmup": 1,
        "seed": 7,
        "cache_limit": 2_000_000_001,
        "wall_budget": 900.0,
    }


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("/Users/x/hf/snap", "~/hf/snap"),
        ("/Users/x", "~"),
        ("/Users/xy/hf", "/Users/xy/hf"),  # another user whose name extends this one
        ("/opt/Users/x/hf", "/opt/Users/x/hf"),  # the home path not at the start
    ],
)
def test_redact_home_replaces_only_a_leading_home_directory(text, want):
    # Bug caught: a bare prefix match (/Users/xy redacted as ~y) or a replace anywhere in the text.
    from scripts.bench_flux_step import redact_home

    assert redact_home(text, home="/Users/x") == want


def test_run_key_includes_the_scenario_hash_and_the_tier_and_redacts_home():
    # Bug caught: a result file that carries the user's home path (committed results must not),
    # or a key without the scenario hash or the tier (a tier-24 child would resume into the
    # 32 GB rows' files, or a changed recipe into the old one's).
    key = run_key(
        **{
            **KEY_KWARGS,
            "df11": Path.home() / "hf" / "snap",
            "embeds": Path.home() / "runs" / "e.safetensors",
        },
        scenario_hash="f" * 64,
        tier_gb=24,
    )
    assert key["df11"] == "~/hf/snap"
    assert key["embeds"] == "~/runs/e.safetensors"
    assert key["scenario_hash"] == "f" * 64
    assert key["tier_gb"] == 24


def test_check_df11_pin():
    # Bug caught: a scenario child timing a checkpoint other than the pinned DF11 revision.
    from scripts.bench_flux_step import check_df11_pin

    check_df11_pin(Path("/hub/snapshots") / SCHNELL_DF11_REVISION, SCHNELL_DF11_REVISION)
    with pytest.raises(BenchError, match=SCHNELL_DF11_REVISION):
        check_df11_pin(Path("/hub/snapshots/main"), SCHNELL_DF11_REVISION)


def test_check_df11_pin_names_the_directory_without_the_home_path():
    # Bug caught: the refusal (written into the failed child's JSON as `error`) carrying the
    # absolute snapshot path, so a committed result names the user.
    from scripts.bench_flux_step import check_df11_pin

    with pytest.raises(BenchError) as exc:
        check_df11_pin(Path.home() / "hub" / "main", SCHNELL_DF11_REVISION)
    assert "~/hub/main" in str(exc.value)
    assert str(Path.home()) not in str(exc.value)


def test_embeds_record_and_the_key_redact_the_home_path_in_the_metadata():
    # encode_prompt records the resolved snapshot root (under ~/.cache) in the metadata.
    # Bug caught: that root, or the embeddings path, written verbatim into the key or the run
    # JSON's `embeds` record, so every committed child JSON names the user.
    from scripts.bench_flux_step import embeds_record

    meta = {"root": str(Path.home() / ".cache" / "hf" / "snap"), "prompt": "a lighthouse"}
    record = embeds_record(Path.home() / "runs" / "e.safetensors", meta)
    assert record == {
        "path": "~/runs/e.safetensors",
        "metadata": {"root": "~/.cache/hf/snap", "prompt": "a lighthouse"},
    }
    key = run_key(**{**KEY_KWARGS, "embeds_meta": meta})
    assert key["embeds_meta"] == {"root": "~/.cache/hf/snap", "prompt": "a lighthouse"}


def test_limits_for_process_gives_the_host_record_without_a_tier_and_capped_with_one():
    # Worked by hand: 32 GiB host, recommended 24 GiB. Host tier 32: MEASURED, reserve 2 GiB (above
    # 24 GB), ceiling 22 GiB. Tier 24: recommended 24 * 2/3 = 16 GiB, reserve 1.5 GiB, ceiling
    # 14.5 GiB, CAPPED. Bug caught: a missing record without --tier, or the host tier labelled CAPPED.
    from scripts.bench_flux_step import limits_for_process

    from mlx_dfloat.errors import DFloatUnsupportedError

    host = limits_for_process(None, host_ram_bytes=32 * GIB, host_recommended_bytes=HOST_REC)
    assert (host.tier_gb, host.is_host, host.label) == (32, True, "MEASURED")
    assert host.ceiling_bytes == 22 * GIB
    capped = limits_for_process(24, host_ram_bytes=32 * GIB, host_recommended_bytes=HOST_REC)
    assert (capped.tier_gb, capped.is_host, capped.label) == (24, False, "CAPPED")
    assert capped.ceiling_bytes == int(14.5 * GIB)
    with pytest.raises(DFloatUnsupportedError):
        limits_for_process(48, host_ram_bytes=32 * GIB, host_recommended_bytes=HOST_REC)


def _measured(seen):
    def measure(args, watchdog, *, limits_recorded):
        seen["limits_recorded"] = limits_recorded
        return {
            "exit_code": 0,
            "policy": "per-block",
            "median_s": 1.0,
            "spread": 0.0,
            "verify_median_s": 0.0,
            "launches_expected_per_step": 0,
            "footprint_peak_bytes": 1,
        }

    return measure


@pytest.fixture
def fake_limits(monkeypatch):
    """Record the cap installers instead of running them; the fake apply sets the tier's cache limit
    the way the real one does, so a later set_cache_limit is what the effective value shows."""
    import mlx.core as mx
    import scripts.bench_flux_step as bfs

    calls = []

    def install():
        calls.append("install_memory_caps")
        return (20, 22)

    def apply(limits):
        calls.append(("apply_limits", limits.tier_gb))
        mx.set_cache_limit(limits.cache_limit_bytes)
        return {}

    monkeypatch.setattr(bfs, "install_memory_caps", install)
    monkeypatch.setattr(bfs, "apply_limits", apply)
    monkeypatch.setattr(bfs, "default_ceiling", lambda: 30 * GIB + 7)
    monkeypatch.setattr(
        bfs,
        "host_memory",
        lambda: {"host_ram_bytes": 32 * GIB, "host_recommended_bytes": HOST_REC},
    )
    return calls


def test_run_one_records_the_limits_effective_values_and_label(
    tmp_path, restore_cache_limit, fake_limits
):
    # Bug caught (host): no limits record without --tier, the host tier labelled CAPPED, or the
    # tier defaults installed on the host. Bug caught (tier 24): the host caps installed over the
    # tier's limits, the host ceiling kept (the run would never abort at the tier's budget), or the
    # scenario's cache limit set before apply_limits (the tier's 15.2 GiB would stay in force).
    from scripts.bench_flux_step import limits_for_process, run_one

    args = parse_args(_scenario_run_args(tmp_path))
    seen = {}
    assert run_one(args, measure=_measured(seen), key=lambda a: {}, cache_limit=2_500_000_000) == 0
    host = json.loads((tmp_path / "r.json").read_text())
    assert fake_limits == ["install_memory_caps"]
    assert host["limits"]["applied"] == "host-caps"
    assert host["label"] == "MEASURED"
    assert host["tier_gb"] == 32
    assert host["watchdog_ceiling_bytes"] == 30 * GIB + 7
    assert host["memory_caps_gb"] == [20, 22]
    assert host["limits"]["effective"]["cache_limit_bytes"] == 2_500_000_000
    assert host["limits"]["tier"]["tier_gb"] == 32
    assert host["scenario_hash"] == args.scenario_hash
    assert seen["limits_recorded"] is True

    fake_limits.clear()
    tier = limits_for_process(24, host_ram_bytes=32 * GIB, host_recommended_bytes=HOST_REC)
    code = run_one(
        args, measure=_measured(seen), key=lambda a: {}, cache_limit=2_500_000_000, limits=tier
    )
    capped = json.loads((tmp_path / "r.json").read_text())
    assert code == 0
    assert fake_limits == [("apply_limits", 24)]
    assert capped["limits"]["applied"] == "tier-defaults"
    assert capped["label"] == "CAPPED"
    assert capped["tier_gb"] == 24
    assert capped["watchdog_ceiling_bytes"] == int(14.5 * GIB)
    assert capped["memory_caps_gb"] == [0, 0]
    assert capped["provenance"]["memory_caps_gb"] == [0, 0]
    assert capped["limits"]["effective"]["cache_limit_bytes"] == 2_500_000_000
    assert seen["limits_recorded"] is True


def test_run_one_refuses_a_df11_dir_off_the_scenario_pin(
    tmp_path, restore_cache_limit, fake_limits
):
    # Bug caught: the pin never checked by the child, so a scenario JSON times another checkpoint.
    from scripts.bench_flux_step import run_one

    args = parse_args(_scenario_run_args(tmp_path, df11=tmp_path / "main"))
    seen = {}
    assert run_one(args, measure=_measured(seen), key=lambda a: {}) == 2
    written = json.loads((tmp_path / "r.json").read_text())
    assert SCHNELL_DF11_REVISION in written["error"]
    assert seen == {}  # nothing measured


class _Q8Transformer:
    """mflux's plain transformer as the q8 mode sees it: callable, no seam (no attach, no verify_step)."""

    def __call__(self, *, t, config, hidden_states, prompt_embeds, pooled_prompt_embeds):
        return hidden_states * 0.5


def test_run_mode_q8_times_the_quantized_transformer_from_the_pinned_base(tmp_path, monkeypatch):
    # Bug caught: the q8 child opening the DF11 checkpoint or loading its resident set (16 GB it never
    # uses, and a footprint that is not q8's), resolving the base off the scenario's pin or with more
    # than the transformer's files, building at reduced depth, attaching a seam, or a record without
    # the q8 settings the report cites (bits, group size, eval policy, cache limit).
    import mlx.core as mx
    import scripts.bench_flux_step as bfs

    snap = Path.home() / "hub" / BASE_REVISION
    seen = {}

    def pinned(repo, revision, *, allow_patterns):
        seen["pinned"] = (repo, revision, list(allow_patterns))
        return snap

    def build(model, root, *, n_double, n_single):
        seen["build"] = (model, root, n_double, n_single)
        return _Q8Transformer()

    def refuse(*a, **k):
        raise AssertionError("the q8 mode touched the DF11 checkpoint")

    inputs = (
        _FakeConfig(_FakeScheduler()),
        mx.ones((1, 4, 8), dtype=mx.float32),
        mx.zeros((1, 2, 4)),
        mx.zeros((1, 4)),
        {"latent_dtype": "mlx.core.float32"},
    )
    monkeypatch.setattr(bfs, "pinned_snapshot", pinned)
    monkeypatch.setattr(bfs, "build_q8_transformer", build)
    monkeypatch.setattr(bfs, "open_checkpoint", refuse)
    monkeypatch.setattr(bfs, "load_resident_set", refuse)
    monkeypatch.setattr(bfs, "build_transformer", refuse)
    monkeypatch.setattr(bfs, "step_inputs", lambda args: inputs)
    args = parse_args(_scenario_run_args(tmp_path, mode="q8"))
    out = bfs.run_mode(args, _FakeWatchdog(), limits_recorded=True)
    assert seen["pinned"] == (
        "black-forest-labs/FLUX.1-schnell",
        BASE_REVISION,
        ["transformer/*"],
    )
    assert seen["build"] == ("schnell", snap, 19, 38)
    assert out["mode"] == "q8"
    assert out["policy"] == "none"
    assert out["compressed_set"] is None
    assert out["q8"] == {
        "base_root": f"~/hub/{BASE_REVISION}",
        "bits": 8,
        "group_size": 64,
        "eval_policy": "none",
        "cache_limit_bytes": 2500000000,
    }
    assert "load_q8_s" in out["timings_s"]
    assert "load_resident_s" not in out["timings_s"]
    assert out["launches_per_step"] == [0] * 7  # 2 warm-up + 5 timed, none of them launching
    assert out["verify_s"] == [0.0] * 5
    assert out["latent_dtype"] == "mlx.core.float32"


def test_run_mode_q8_refuses_trace_before_building(tmp_path, monkeypatch):
    # Bug caught: --trace accepted for q8, so the run either crashes in the report (no seam recorded
    # a step) or writes "traced" for a run nothing traced; refused before the base is resolved.
    import scripts.bench_flux_step as bfs

    def refuse(*a, **k):
        raise AssertionError("resolved the base for a run that must be refused")

    monkeypatch.setattr(bfs, "pinned_snapshot", refuse)
    args = parse_args(_scenario_run_args(tmp_path, "--trace", mode="q8"))
    with pytest.raises(BenchError, match="trace"):
        bfs.run_mode(args, _FakeWatchdog(), limits_recorded=True)


def test_make_provider_refuses_the_q8_mode():
    # Bug caught: q8 falling through to the control branch (not a DF11 mode), which would decode two
    # DF11 blocks and hand a ReuseProvider to a run that must never touch the checkpoint.
    import numpy as np
    from tests._flux_fakes import FLUX_TABLE

    ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(6))
    calls = []
    with pytest.raises(BenchError, match="q8"):
        make_provider(
            "q8",
            ckpt,
            groups,
            shapes,
            decode=_counting_reference_decode(calls),
            name_map=FLUX_TABLE,
        )
    assert calls == []


def test_denoise_step_on_a_transformer_without_verify_step_reports_no_verify_time():
    # Bug caught: `verify_step()` called unconditionally (mflux's plain q8 transformer has none, so
    # every q8 step raises AttributeError), or a verify time other than 0.0 recorded for it.
    import mlx.core as mx
    from scripts.bench_flux_step import denoise_step

    latents, took, verify = denoise_step(
        _Q8Transformer(),
        _FakeConfig(_FakeScheduler()),
        mx.ones((1, 4, 8), dtype=mx.float32),
        mx.zeros((1, 2, 4)),
        mx.zeros((1, 4)),
        0,
    )
    assert verify == 0.0
    assert took > 0.0
    assert latents.tolist() == [[[0.5] * 8] * 4]  # 1 - 0.5, the fake scheduler's step


def test_time_steps_passes_a_run_with_no_compressed_set():
    # Bug caught: compressed_loaded=None (the q8 mode) failing the parity check before any step is
    # timed.
    provider = _FakeProvider()
    out = _time(_FakeTransformer(provider, [0] * 5), provider, per_step=0, loaded=None)
    assert len(out["step_s"]) == 3


def test_main_hands_run_one_the_tier_limits_and_refuses_a_tier_above_the_host(
    tmp_path, monkeypatch, capsys
):
    # Bug caught: --tier parsed but never turned into limits (the child runs under the host caps
    # while its key says tier 24), or a tier above the host escaping as a traceback (exit 1).
    import scripts.bench_flux_step as bfs

    seen = {}

    def run_one(args, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(bfs, "run_one", run_one)
    monkeypatch.setattr(
        bfs,
        "host_memory",
        lambda: {"host_ram_bytes": 32 * GIB, "host_recommended_bytes": HOST_REC},
    )
    assert bfs.main(_scenario_run_args(tmp_path, "--tier", "24")) == 0
    assert seen["limits"].tier_gb == 24
    assert seen["limits"].label == "CAPPED"
    seen.clear()
    assert bfs.main(_scenario_run_args(tmp_path)) == 0
    assert seen["limits"] is None  # run_one records the host tier itself
    assert bfs.main(_scenario_run_args(tmp_path, "--tier", "48")) == 2
    assert "48" in capsys.readouterr().err


def test_current_key_carries_the_scenario_hash_and_the_tier(tmp_path, monkeypatch):
    # Bug caught: a child key without the tier or the hash, so the orchestrator's key (which has
    # them) reads every finished child as a conflict, or a tier-24 child as a 32 GB one.
    import scripts.bench_flux_step as bfs

    monkeypatch.setattr(bfs, "embeds_metadata", lambda path: {"synthetic": "true"})
    monkeypatch.setattr(bfs, "source_hash", lambda: "abc")
    args = parse_args(_scenario_run_args(tmp_path, "--tier", "24"))
    key = bfs.current_key(args)
    assert key["tier_gb"] == 24
    assert key["scenario_hash"] == args.scenario_hash
    assert key["cache_limit"] == 2_500_000_000


def _host(monkeypatch, bfs, ram_gb):
    monkeypatch.setattr(bfs, "embeds_metadata", lambda path: {"synthetic": "true"})
    monkeypatch.setattr(bfs, "source_hash", lambda: "abc")
    monkeypatch.setattr(
        bfs,
        "host_memory",
        lambda: {"host_ram_bytes": ram_gb * GIB, "host_recommended_bytes": ram_gb * GIB * 3 // 4},
    )


def test_current_key_keys_the_resolved_tier_so_the_host_tier_matches_no_tier(tmp_path, monkeypatch):
    # Bug caught: the key carrying the requested --tier (None without it), so `--tier 32` on a
    # 32 GB host (the same limits) reads as a conflict with a no-tier run of the same recipe.
    import scripts.bench_flux_step as bfs

    _host(monkeypatch, bfs, 32)
    plain = bfs.current_key(parse_args(_scenario_run_args(tmp_path)))
    tiered = bfs.current_key(parse_args(_scenario_run_args(tmp_path, "--tier", "32")))
    assert plain["tier_gb"] == 32
    assert plain == tiered


def test_current_key_of_another_host_tier_differs(tmp_path, monkeypatch):
    # Bug caught: a no-tier key with nothing host-specific in it (paths are redacted), so a
    # results directory measured on this 32 GB Mac resumes as complete on a 64 GB one.
    import scripts.bench_flux_step as bfs

    _host(monkeypatch, bfs, 32)
    here = bfs.current_key(parse_args(_scenario_run_args(tmp_path)))
    _host(monkeypatch, bfs, 64)
    there = bfs.current_key(parse_args(_scenario_run_args(tmp_path)))
    assert there["tier_gb"] == 64
    assert resume_key_diff(here, there) == ["tier_gb"]
