import json

import pytest
from scripts.flux_rig_smoke import config_kwargs, parse_args, verdict


def test_finite_outputs_exit_0():
    assert verdict(finite=True) == 0


def test_a_non_finite_output_is_a_tool_error_not_the_mismatch_code():
    # The scripts reserve exit 1 for a real bit mismatch. Bug caught: a NaN/Inf smoke exiting 1 and reading as a
    # decode mismatch.
    assert verdict(finite=False) == 2


@pytest.mark.parametrize("model", ["schnell", "dev"])
def test_the_smoke_uses_the_step_bench_guidance(model):
    # mflux's CLI default guidance is 3.5 (inert for schnell, which has no guidance embedder). Bug caught: the smoke
    # passing 0.0 for schnell while the step bench passes 3.5, so the two configs silently differ.
    args = parse_args(
        ["--df11", "x", "--synthetic", "--model", model, "--steps", "2", "--size", "256"]
    )
    assert config_kwargs(args) == {
        "num_inference_steps": 4,
        "height": 256,
        "width": 256,
        "guidance": 3.5,
    }


@pytest.mark.parametrize(
    "extra",
    [("--double", "0"), ("--single", "0"), ("--double", "20"), ("--single", "39")],
)
def test_depths_outside_the_model_are_usage_errors(extra):
    # Bug caught: a depth of 0 (nothing to run) or beyond FLUX.1's 19 + 38 blocks accepted and failing
    # deep inside mflux's constructor instead of at the command line.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--df11", "x", "--synthetic", *extra])
    assert exc.value.code == 2


def test_the_edge_depths_are_accepted():
    args = parse_args(["--df11", "x", "--synthetic", "--double", "19", "--single", "38"])
    assert (args.double, args.single) == (19, 38)


@pytest.fixture
def restore_cache_limit():
    import mlx.core as mx

    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    yield
    mx.set_cache_limit(previous)


def test_main_moves_a_stale_abort_artifact_aside_before_the_watchdog_starts(
    tmp_path, monkeypatch, restore_cache_limit
):
    # Bug caught: a smoke that finishes next to a previous run's abort.json, which then reads as this
    # run's outcome.
    import scripts.flux_rig_smoke as fs

    monkeypatch.setattr(
        fs,
        "smoke",
        lambda args, watchdog: {
            "exit_code": 0,
            "df11": {"launches": 2, "step_s": [0.1]},
            "control": {"step_s": [0.1]},
            "footprint_peak_bytes": 1,
            "outputs_bit_identical": True,
        },
    )
    out = tmp_path / "o"
    out.mkdir()
    (out / "abort.json").write_text('{"reason": "stale"}')
    argv = ["--df11", "x", "--synthetic", "--out", str(out), "--wall-budget", "60"]
    assert fs.main(argv) == 0
    assert (out / "abort.previous.json").read_text() == '{"reason": "stale"}'
    assert not (out / "abort.json").exists()
    assert json.loads((out / "smoke.json").read_text())["exit_code"] == 0
