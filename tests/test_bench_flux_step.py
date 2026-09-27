"""The pure parts of the FLUX step bench: modes, interleaving, resume, launch counts, the report.

Expected values are worked by hand from the fixture step times below, never from the code.
No test here imports mflux or touches the GPU.
"""

import json
import sys
from pathlib import Path

import pytest
from scripts._bench_common import write_json_atomic
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
KEY = run_key(
    model="schnell",
    size=1024,
    steps=5,
    warmup=2,
    seed=42,
    df11=Path("/ckpt"),
    embeds=Path("/e.safetensors"),
    embeds_meta={"synthetic": "true", "seed": "42"},
    source="abc",
    mlx="0.32.2",
)


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
    }
    assert pooled["control"] == {
        "median": pytest.approx(1.0),
        "spread": pytest.approx(0.2),
        "n": 6,
        "verify_median_s": pytest.approx(0.0),
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
    assert out["paired_rounds"] == {"per-block": [1, 2], "depth2": [1], "eval-policy": [1]}


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
    def __init__(self, peak_footprint=0):
        self.peak_footprint = peak_footprint

    def reset_peak(self):
        self.peak_footprint = 0


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


# --- the parity condition helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("caps", "cache_limit", "want"),
    [
        ((20, 22), FLUX_CACHE_LIMIT, True),
        ((0, 22), FLUX_CACHE_LIMIT, False),  # the wired cap failed to install
        ((20, 0), FLUX_CACHE_LIMIT, False),  # the memory cap failed to install
        ((0, 0), FLUX_CACHE_LIMIT, False),
        ((20, 22), FLUX_CACHE_LIMIT + 1, False),  # another cache limit is in force
    ],
)
def test_limits_in_force_needs_both_caps_and_the_cache_limit(caps, cache_limit, want):
    # Bug caught: checking caps[0] alone (a failed memory cap passes), or ignoring the cache limit.
    assert limits_in_force(caps, cache_limit) is want


def _fake_rig(rng, *, n_double=2, n_single=2):
    """A fake transformer's shapes, one real DF11 group per block, a ckpt-shaped view of their names."""
    from types import SimpleNamespace

    from tests.test_flux_rig import FakeTransformer, Recorder, _df11_groups

    shapes = install_placeholders(FakeTransformer(Recorder(), n_double=n_double, n_single=n_single))
    groups, names, source = _df11_groups(shapes, rng)
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

    ckpt, groups, shapes, source = _fake_rig(np.random.default_rng(4))
    calls = []
    provider = make_provider(
        "control", ckpt, groups, shapes, decode=_counting_reference_decode(calls)
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

    ckpt, groups, shapes, _source = _fake_rig(np.random.default_rng(5))
    calls = []
    provider = make_provider(mode, ckpt, groups, shapes, decode=_counting_reference_decode(calls))
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
            "limits": limits_recorded,
        }

    args = _one_run_args(tmp_path)
    code = run_one(args, measure=measure, key=lambda a: {"double": 4})
    written = json.loads((tmp_path / "r.json").read_text())
    assert code == 0
    assert written["key"] == {"double": 4}
    assert written["limits"] is True
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
