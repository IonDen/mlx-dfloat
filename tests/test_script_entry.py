import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ["scripts/verify_checkpoint.py", "scripts/verify_remote_group.py"]


def _run(args, env=None):
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_running_the_script_file_directly_starts(script):
    # Bug caught: `python scripts/x.py` puts scripts/ first on sys.path, so `from scripts._watchdog
    # import ...` raises ModuleNotFoundError: exit 1 (the mismatch kill signal) and no summary.
    result = _run([script, "--help"])
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("error", ["ImportError", "RuntimeError", "OSError"])
def test_a_project_import_failure_exits_2_not_1(script, error, tmp_path):
    # A broken environment is a tool error (exit 2), never the bit-mismatch exit code 1. Importing
    # mlx.core on a host without Metal raises RuntimeError or OSError, not only ImportError.
    shadow = tmp_path / "mlx_dfloat"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(f'raise {error}("simulated broken install")\n')
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = _run([script, "--help"], env=env)
    assert result.returncode == 2, result.stderr[-800:]
    assert "simulated broken install" in result.stderr
