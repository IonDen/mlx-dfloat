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


def _integration_job() -> dict:
    return yaml.safe_load(CI.read_text())["jobs"]["integration"]


def test_the_mflux_lane_installs_the_locked_extra_and_proves_it_ran_tests():
    # Bug caught: the lane installing mflux outside the lock (torch and transformers float to
    # whatever resolves that day), a sanity step reading `mflux.__version__` (mflux has none, so the
    # job dies before a single test), dropping `-m mflux` / `--run-network`, or a lane that goes
    # green having collected zero mflux tests, or a `uv run` step that could drop the extra.
    job = _integration_job()
    assert job["strategy"]["matrix"]["mflux"] == ["0.20.0"]
    runs = [step.get("run", "") for step in job["steps"]]
    assert "uv sync --locked --group dev --extra mflux" in runs
    assert not any("uv pip install" in run or "--no-sync" in run for run in runs)
    assert not any("__version__" in run for run in runs)
    assert any("importlib.metadata" in run and "version('mflux')" in run for run in runs)
    # Every `uv run` names the extra and the group, so no step depends on uv's sync mode keeping
    # what `uv sync --extra mflux` installed.
    uv_runs = [run for run in runs if "uv run" in run]
    assert len(uv_runs) == 3
    assert all("uv run --extra mflux --group dev " in run for run in uv_runs)
    pytest_runs = [
        shlex.split(run)
        for run in runs
        if run.startswith("uv run --extra mflux --group dev pytest")
    ]
    tests = [args for args in pytest_runs if "--co" not in args]
    assert len(tests) == 1
    assert {"-m", "mflux", "--run-network", "-rs"} <= set(tests[0])
    # model.py and init.py are exercised only by this lane, so its own coverage gate measures
    # them (the required job's gate omits both; see pyproject.toml).
    assert "--cov=mlx_dfloat.mflux.flux1.model" in tests[0]
    assert "--cov=mlx_dfloat.mflux.flux1.init" in tests[0]
    assert "--cov-fail-under=80" in tests[0]
    guard = [run for run in runs if "--co" in run and "grep -c" in run]
    assert len(guard) == 1
    assert "-m mflux" in guard[0]
    assert "-ge 2" in guard[0]
