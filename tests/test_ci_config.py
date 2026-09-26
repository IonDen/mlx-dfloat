import shlex
from pathlib import Path

import yaml

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _test_job_steps() -> list[dict]:
    config = yaml.safe_load(CI.read_text())
    return config["jobs"]["test"]["steps"]


def _step_named(name: str) -> dict:
    steps = [step for step in _test_job_steps() if step.get("name") == name]
    assert len(steps) == 1, f"expected exactly one step named {name!r}, found {len(steps)}"
    return steps[0]


def test_ci_measures_the_package_and_the_parity_scripts():
    # Bug caught: a --cov=<pkg> flag replaces coverage's configured `source`, so dropping
    # --cov=scripts from the CI step silently stops measuring the scripts that carry the
    # 0 / 1 / 2 exit-code contract, while the job stays green. Also catches the coverage
    # gate being dropped outright, or the Metal suite being folded back into this step
    # before a runner has ever reported metal_ready=True.
    step = _step_named("Test (required, without the Metal suite)")
    args = shlex.split(step["run"])
    assert "--cov=mlx_dfloat" in args
    assert "--cov=scripts" in args
    assert "--cov-fail-under=85" in args
    assert "-m" in args
    assert args[args.index("-m") + 1] == "not metal"
    assert "continue-on-error" not in step


def test_metal_lanes_are_informational_only():
    # Bug caught: a runner limitation (no Metal device, an unsigned binary, whatever) turning
    # into a red *required* gate because continue-on-error was dropped from one of these steps.
    probe = _step_named("Metal capability probe (informational)")
    metal_suite = _step_named("Metal suite (informational until the probe reports ready)")
    assert probe["continue-on-error"] is True
    assert metal_suite["continue-on-error"] is True
