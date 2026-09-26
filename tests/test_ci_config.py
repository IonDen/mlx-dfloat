from pathlib import Path

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _pytest_steps():
    return [
        line.split("run:", 1)[1].split()
        for line in CI.read_text().splitlines()
        if "run:" in line and "pytest" in line
    ]


def test_ci_measures_the_package_and_the_parity_scripts():
    # Bug caught: a --cov=<pkg> flag replaces coverage's configured `source`, so dropping
    # --cov=scripts from the CI step silently stops measuring the scripts that carry the
    # 0 / 1 / 2 exit-code contract, while the job stays green.
    steps = _pytest_steps()
    assert len(steps) == 1, steps
    args = steps[0]
    assert "--cov=mlx_dfloat" in args
    assert "--cov=scripts" in args
    assert "--cov-fail-under=85" in args
