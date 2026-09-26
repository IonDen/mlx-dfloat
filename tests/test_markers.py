import subprocess
import sys
from pathlib import Path

import conftest  # pytest prepend mode adds tests/ to sys.path
from conftest import GATED_MARKERS, _hard_exit_code, _markers_to_skip

REPO = Path(__file__).resolve().parents[1]


def test_all_markers_skipped_when_no_flags():
    skipped = dict(_markers_to_skip(enabled_flags=set()))
    assert set(skipped) == {marker for marker, _flag, _description in GATED_MARKERS}


def test_enabled_flag_unskips_its_marker():
    skipped = dict(_markers_to_skip(enabled_flags={"--run-slow"}))
    assert "slow" not in skipped
    assert "network" in skipped


def test_no_hard_exit_code_until_a_session_finishes():
    # Bug caught: starting the recorded code at 0 makes the atexit hard exit report success for a
    # run that never had a session (an unknown flag, a --strict-config error).
    assert conftest._FINAL_EXIT_CODE is None
    assert _hard_exit_code(None) is None


def test_a_recorded_session_code_is_used_as_is():
    assert _hard_exit_code(0) == 0
    assert _hard_exit_code(1) == 1


def test_a_pytest_usage_error_still_exits_non_zero():
    # The end-to-end form of the bug above: a typo'd CI command must not go green.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-such-flag", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 4, result.stderr[-500:]
