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
