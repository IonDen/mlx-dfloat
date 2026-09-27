import importlib.util
import subprocess
import sys
from pathlib import Path

import conftest  # pytest prepend mode adds tests/ to sys.path
import pytest
from conftest import (
    GATED_MARKERS,
    _hard_exit_code,
    _markers_to_skip,
    _metal_marker_action,
    _mflux_marker_action,
)

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


def test_metal_marker_runs_on_apple_silicon():
    # Bug caught: skipping the Metal suite on the one platform that must run it.
    assert _metal_marker_action("Darwin", "arm64") == "run"


def test_metal_marker_skips_elsewhere():
    # Bug caught: a Linux or Intel job trying to dispatch a Metal kernel.
    assert _metal_marker_action("Linux", "x86_64") == "skip"
    assert _metal_marker_action("Darwin", "x86_64") == "skip"


def test_metal_marker_collection_succeeds():
    # Bug caught: a typo in the collection hook (e.g. `_metal_marker_action_TYPO`) that crashes
    # `pytest_collection_modifyitems` outright. Confirmed empirically: that specific crash exits
    # 3 (INTERNALERROR), traceback pointing at the bad name. `tests/test_decode_parity.py` is
    # marked `metal`, so a working hook collects at least that file and pytest exits 0; a hook
    # that deselected everything would exit 5 ("no tests collected") and fail here too.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "metal",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "test_decode_parity.py" in result.stdout


def test_mflux_marker_runs_only_when_mflux_is_importable():
    # Bug caught: mflux tests silently skipped on a machine that has mflux (a green run that tested
    # nothing), or collected where mflux is absent (an ImportError instead of a skip).
    assert _mflux_marker_action(available=True) == "run"
    assert _mflux_marker_action(available=False) == "skip"


def test_mflux_marker_tests_skip_with_the_install_hint_where_mflux_is_missing():
    # Bug caught: the collection hook not skipping `mflux` tests here (they would fail at their
    # `from mflux ...` import, exit 1) or crashing (exit 3), or a skip reason that no longer tells
    # the reader which extra to install. The mirror image (the tests run where mflux is present)
    # is the CI lane's own collected-tests guard.
    if importlib.util.find_spec("mflux") is not None:
        pytest.skip("mflux is installed here: the mflux lane runs these tests for real")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "mflux", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "mflux: install the mlx-dfloat[mflux] extra" in result.stdout
    summary = result.stdout.strip().splitlines()[-1]
    assert "skipped" in summary
    assert "passed" not in summary
    assert "failed" not in summary
