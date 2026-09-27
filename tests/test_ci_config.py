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


def test_ci_runs_the_whole_suite_measuring_the_package_and_the_scripts():
    # Bug caught: a --cov=<pkg> flag replaces coverage's configured `source`, so dropping
    # --cov=scripts from the CI step silently stops measuring the scripts that carry the
    # 0 / 1 / 2 exit-code contract, while the job stays green. Also catches the coverage gate
    # being dropped, or a `-m` filter creeping back in and taking the Metal tests out of the
    # required step (the runner reports a Metal device and the kernel suite passed there).
    step = _step_named("Test (required)")
    args = shlex.split(step["run"])
    assert "--cov=mlx_dfloat" in args
    assert "--cov=scripts" in args
    assert "--cov-fail-under=85" in args
    assert "-m" not in args
    assert "continue-on-error" not in step


def test_the_metal_probe_stays_informational_and_no_separate_metal_suite_exists():
    # Bug caught: the device probe turned into a red required gate (continue-on-error dropped), or
    # the Metal tests split back out into an informational step that can go red unnoticed.
    probe = _step_named("Metal capability probe (informational)")
    assert probe["continue-on-error"] is True
    names = [step.get("name", "") for step in _test_job_steps()]
    assert not any("Metal suite" in name for name in names)
