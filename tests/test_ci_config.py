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


def test_ci_runs_the_gpu_decoder_selftest_as_a_required_step():
    # Bug caught: the self-check dropped from CI, or marked continue-on-error, so a runner whose GPU
    # decodes wrong bits would still go green.
    step = _step_named("GPU decoder self-check")
    assert step["run"].strip() == "uv run mlx-dfloat selftest"
    assert "continue-on-error" not in step
    assert "if" not in step


def test_ci_checks_that_the_built_wheel_ships_the_canary_data_as_a_required_step():
    # Bug caught: a packaging change (an exclude, a moved directory, a data file outside the package) that drops
    # the canary files from the wheel: every install would then refuse its GPU decoder or fail on first
    # decode, while the checkout-based tests all pass. Also catches the step marked continue-on-error or
    # made conditional, and a listing check that forgets one of the three files.
    steps = _test_job_steps()
    step = _step_named("Wheel ships the canary data")
    assert "continue-on-error" not in step
    assert "if" not in step
    run = step["run"]
    assert "uv build --wheel --out-dir dist" in run
    assert "zipfile" in run or "unzip" in run
    for name in ("qwen3_4b_layer0_4blocks.npz", "qwen3_4b_layer0_4blocks.json", "long_codes.npz"):
        assert name in run
    names = [s.get("name") for s in steps]
    assert names.index("Wheel ships the canary data") > names.index("GPU decoder self-check")


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
    # sanity + lane + six per-model and ten per-module coverage gates + the collection check
    assert len(uv_runs) == 19
    assert all("uv run --extra mflux --group dev " in run for run in uv_runs)
    pytest_runs = [
        shlex.split(run)
        for run in runs
        if run.startswith("uv run --extra mflux --group dev pytest")
    ]
    tests = [args for args in pytest_runs if "--co" not in args]
    assert len(tests) == 1
    assert {"-m", "mflux", "--run-network", "-rs"} <= set(tests[0])
    # model.py has no offline test at all (every test is @pytest.mark.mflux), so only this lane
    # measures it (the required job's gate omits it; see pyproject.toml). The same holds for the
    # init.py modules, which drive mflux's loaders; each is gated on its own after this step.
    assert "--cov=mlx_dfloat.mflux.flux1.model" in tests[0]
    assert "--cov=mlx_dfloat.mflux.flux1.init" in tests[0]
    assert "--cov-config=.coveragerc-integration" in tests[0]
    # No combined threshold: one module's lines would dilute another's; the per-file steps after
    # this one are the lane's gates.
    assert not any(arg.startswith("--cov-fail-under") for arg in tests[0])
    guard = [run for run in runs if "--co" in run and "grep -c" in run]
    assert len(guard) == 1
    assert "-m mflux" in guard[0]
    assert "-ge 2" in guard[0]


def test_the_mflux_lane_gates_the_model_module_on_its_own():
    # Bug caught: model.py's lines diluted by init.py's in the lane's combined 80 % gate, so the
    # module with no offline test at all could drop far below 80 % while the job stays green; or
    # the per-file report running before the lane's pytest has written the data.
    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/flux1/model.py' --fail-under=80"
    assert report in runs
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    assert runs.index(report) == lane + 1


def test_the_mflux_lane_measures_and_gates_the_zimage_model_module_on_its_own():
    # Bug caught: zimage/model.py (every test @pytest.mark.mflux) omitted from the required job's gate but measured
    # by no job either, so its coverage could fall to zero with every check green; or its gate diluted by
    # flux1/model.py's lines, or run before the lane's pytest wrote the data.
    import tomllib

    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    assert "--cov=mlx_dfloat.mflux.zimage.model" in shlex.split(runs[lane])
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/zimage/model.py' --fail-under=80"
    assert report in runs
    assert lane < runs.index(report) < lane + 3
    pyproject = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())
    assert "src/mlx_dfloat/mflux/zimage/model.py" in pyproject["tool"]["coverage"]["run"]["omit"]


def test_the_mflux_lane_measures_and_gates_the_flux2_model_module_on_its_own():
    # Bug caught: flux2/model.py (every test @pytest.mark.mflux) omitted from the required job's gate but measured
    # by no job either, so its coverage could fall to zero with every check green; or its gate diluted by the other
    # model modules' lines, or run before the lane's pytest wrote the data.
    import tomllib

    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    assert "--cov=mlx_dfloat.mflux.flux2.model" in shlex.split(runs[lane])
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/flux2/model.py' --fail-under=80"
    assert report in runs
    assert lane < runs.index(report) < lane + 4
    pyproject = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())
    assert "src/mlx_dfloat/mflux/flux2/model.py" in pyproject["tool"]["coverage"]["run"]["omit"]


def test_the_mflux_lane_measures_and_gates_the_qwen21_model_module_on_its_own():
    # Bug caught: qwen21/model.py (every test @pytest.mark.mflux) omitted from the required job's gate but measured
    # by no job either, so its coverage could fall to zero with every check green; or its gate diluted by the other
    # model modules' lines, or run before the lane's pytest wrote the data; or Qwen's init.py and transformer.py not
    # reported in the lane beside its model (the lane's mflux-only tests cover lines the main job's report cannot).
    import tomllib

    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    args = shlex.split(runs[lane])
    assert "--cov=mlx_dfloat.mflux.qwen21.model" in args
    assert "--cov=mlx_dfloat.mflux.qwen21.init" in args
    assert "--cov=mlx_dfloat.mflux.qwen21.transformer" in args
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/qwen21/model.py' --fail-under=80"
    assert report in runs
    assert lane < runs.index(report) < lane + 5
    pyproject = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())
    assert "src/mlx_dfloat/mflux/qwen21/model.py" in pyproject["tool"]["coverage"]["run"]["omit"]


def test_the_mflux_lane_measures_and_gates_the_ernie_model_module_on_its_own():
    # Bug caught: ernie/model.py (every test @pytest.mark.mflux) omitted from the required job's gate but measured by
    # no job either, so its coverage could fall to zero with every check green; or its gate diluted by the other model
    # modules' lines, or run before the lane's pytest wrote the data; or ERNIE's init.py and transformer.py not
    # measured in the lane beside its model (each is gated on its own, see the test below).
    import tomllib

    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    args = shlex.split(runs[lane])
    assert "--cov=mlx_dfloat.mflux.ernie.model" in args
    assert "--cov=mlx_dfloat.mflux.ernie.init" in args
    assert "--cov=mlx_dfloat.mflux.ernie.transformer" in args
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/ernie/model.py' --fail-under=80"
    assert report in runs
    assert lane < runs.index(report) < lane + 6
    pyproject = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())
    assert "src/mlx_dfloat/mflux/ernie/model.py" in pyproject["tool"]["coverage"]["run"]["omit"]


def test_the_mflux_lane_measures_and_gates_the_krea2_model_module_on_its_own():
    # Bug caught: krea2/model.py (every test @pytest.mark.mflux) omitted from the required job's gate but measured by
    # no job either, so its coverage could fall to zero with every check green; or its gate diluted by the other model
    # modules' lines, or run before the lane's pytest wrote the data; or Krea 2's init.py and transformer.py not
    # measured in the lane beside its model (each is gated on its own, see the test below).
    import tomllib

    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    args = shlex.split(runs[lane])
    assert "--cov=mlx_dfloat.mflux.krea2.model" in args
    assert "--cov=mlx_dfloat.mflux.krea2.init" in args
    assert "--cov=mlx_dfloat.mflux.krea2.transformer" in args
    report = "uv run --extra mflux --group dev coverage report --include='*/mflux/krea2/model.py' --fail-under=80"
    assert report in runs
    assert lane < runs.index(report) < lane + 7  # the sixth per-model gate after the lane
    pyproject = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())
    assert "src/mlx_dfloat/mflux/krea2/model.py" in pyproject["tool"]["coverage"]["run"]["omit"]


def _config() -> dict:
    return yaml.safe_load(CI.read_text())


def _live_test_step(job: dict) -> dict:
    live = [
        step
        for step in job["steps"]
        if str(step.get("run", "")).startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(step["run"])
    ]
    assert len(live) == 1
    return live[0]


def test_only_the_live_test_step_of_the_mflux_lane_gets_the_hub_token():
    # Bug caught: the HF_TOKEN env dropped from the lane's pytest step, or renamed to something
    # huggingface_hub does not read, or blanked by an override: the three live tokenizer tests then skip
    # on every CI run (`get_token() is None` on the runner) and the lane stays green while the gated
    # base-repo path is never exercised there. Also caught: the network marker gated off again, which
    # makes the token useless; and the token widened to the job, the workflow, another step or the
    # default lane, where the checkout and setup-uv actions and the collection and coverage steps
    # would read it for no reason. Only the value's presence can be checked here: an unset secret
    # expands to "" and the tests skip, so this test proves the wiring, not that the secret exists.
    config = _config()
    job = config["jobs"]["integration"]
    live = _live_test_step(job)
    assert live.get("env", {}).get("HF_TOKEN") == "${{ secrets.HF_TOKEN }}"
    assert "--run-network" in shlex.split(live["run"])
    others = [step for step in job["steps"] if step is not live]
    assert not any("HF_TOKEN" in (step.get("env") or {}) for step in others)
    assert "HF_TOKEN" not in (job.get("env") or {})
    assert "HF_TOKEN" not in (config.get("env") or {})
    assert "HF_TOKEN" not in yaml.safe_dump(config["jobs"]["test"])
    assert "HF_TOKEN" not in yaml.safe_dump(config["jobs"]["build"])


def test_the_workflow_token_is_read_only():
    # Bug caught: the `permissions` block dropped, so the workflow's GITHUB_TOKEN falls back to the
    # repository default (read/write on older defaults), in a workflow whose mflux lane now carries a
    # Hub secret; or a write scope added without a step that needs it.
    assert _config()["permissions"] == {"contents": "read"}


# The family modules that drive mflux (its loaders, its transformer classes): their mflux-free tests leave them at
# 40-60 % in the required job, so that job's gate omits them and the mflux lane gates each one at 80 %, as it does
# model.py. flux1/transformer.py and zimage/transformer.py stay in the required gate: their offline tests (the seam
# fakes) cover them past 90 % without mflux, while the lane alone reaches only 57 % of flux1/transformer.py.
MFLUX_ONLY = [
    "flux1/init",
    "zimage/init",
    "flux2/init",
    "flux2/transformer",
    "qwen21/init",
    "qwen21/transformer",
    "ernie/init",
    "ernie/transformer",
    "krea2/init",
    "krea2/transformer",
]


def test_the_mflux_only_modules_are_omitted_from_the_required_gate_and_gated_in_the_lane():
    # Bug caught: a module omitted from the required job's 85 % gate but measured or gated by no job (its coverage
    # could fall to zero with every check green), a gate diluted by another module's lines, a gate run before the
    # lane's pytest wrote the data, or a module that runs without mflux (names.py, memory.py, the two seam-covered
    # transformer.py files) dropped from the required gate.
    import tomllib

    omit = tomllib.loads((CI.parents[2] / "pyproject.toml").read_text())["tool"]["coverage"]["run"][
        "omit"
    ]
    runs = [step.get("run", "") for step in _integration_job()["steps"]]
    lane = next(
        i
        for i, run in enumerate(runs)
        if run.startswith("uv run --extra mflux --group dev pytest")
        and "--co" not in shlex.split(run)
    )
    args = shlex.split(runs[lane])
    collect = next(i for i, run in enumerate(runs) if "--co" in run and "grep -c" in run)
    for module in MFLUX_ONLY:
        family, name = module.split("/")
        assert f"src/mlx_dfloat/mflux/{module}.py" in omit, module
        assert f"--cov=mlx_dfloat.mflux.{family}.{name}" in args, module
        report = f"uv run --extra mflux --group dev coverage report --include='*/mflux/{module}.py' --fail-under=80"
        assert report in runs, module
        assert lane < runs.index(report) < collect, module
    for kept in (
        "flux1/transformer",
        "zimage/transformer",
        "krea2/names",
        "krea2/memory",
        "ernie/memory",
    ):
        assert f"src/mlx_dfloat/mflux/{kept}.py" not in omit, kept
