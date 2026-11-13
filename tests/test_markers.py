import importlib.machinery
import importlib.metadata
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
    _mflux_is_installed,
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


def _version_lookup(installed: dict[str, str]):
    """A distribution-metadata lookup over a fixed table; a missing name raises like ``importlib.metadata.version``."""

    def version(name: str) -> str:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    return version


def _namespace_spec(tmp_path):
    """The spec a leftover ``mflux/`` directory (no ``__init__.py``) really gets: a namespace package, no origin."""
    (tmp_path / "mflux").mkdir()
    spec = importlib.machinery.PathFinder.find_spec("mflux", [str(tmp_path)])
    assert spec is not None
    assert spec.origin is None
    return spec


def _package_spec(tmp_path):
    """The spec of an importable ``mflux`` package (an ``__init__.py``, so a real origin)."""
    (tmp_path / "mflux").mkdir()
    (tmp_path / "mflux" / "__init__.py").write_text("")
    spec = importlib.machinery.PathFinder.find_spec("mflux", [str(tmp_path)])
    assert spec is not None
    assert spec.origin is not None
    return spec


def test_a_leftover_mflux_directory_without_metadata_is_not_an_install(tmp_path):
    # Bug caught: `find_spec("mflux") is not None` as the gate; a namespace-only leftover directory then counts as an
    # install and every `mflux` test fails at its first `from mflux...` import instead of skipping.
    assert _mflux_is_installed(_namespace_spec(tmp_path), _version_lookup({})) is False


def test_a_leftover_mflux_directory_is_not_an_install_even_when_metadata_remains(tmp_path):
    # Bug caught: the two conditions joined with `or` (metadata alone suffices), or the origin check dropped; a stale
    # dist-info next to an emptied package directory then counts as an install.
    assert (
        _mflux_is_installed(_namespace_spec(tmp_path), _version_lookup({"mflux": "0.24.0"}))
        is False
    )


def test_an_importable_mflux_without_distribution_metadata_is_not_an_install(tmp_path):
    # Bug caught: the metadata check dropped or its PackageNotFoundError not caught (the hook would crash collection
    # with exit 3 instead of skipping); a bare source tree on sys.path then counts as the installed extra.
    assert _mflux_is_installed(_package_spec(tmp_path), _version_lookup({})) is False


def test_no_mflux_spec_is_not_an_install():
    # Bug caught: a `None` spec dereferenced (`spec.origin` on None crashes collection) or treated as installed.
    assert _mflux_is_installed(None, _version_lookup({"mflux": "0.24.0"})) is False


def test_a_real_mflux_install_counts_as_available(tmp_path):
    # Bug caught: a hardened gate that is too strict (for example `spec.origin is None`, or the metadata lookup made to
    # always fail); the mflux lane then skips every family test and reports a green run that tested nothing.
    assert (
        _mflux_is_installed(_package_spec(tmp_path), _version_lookup({"mflux": "0.24.0"})) is True
    )


def test_the_installed_mflux_in_this_interpreter_counts_as_available():
    # Bug caught: the same over-strict gate, seen on the real install (real `find_spec`, real metadata) rather than a
    # built spec; skipped where mflux is not installed.
    try:
        importlib.metadata.version("mflux")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("mflux is not installed in this interpreter")
    assert (
        _mflux_is_installed(importlib.util.find_spec("mflux"), importlib.metadata.version) is True
    )


def test_mflux_marker_tests_skip_with_the_install_hint_where_mflux_is_missing():
    # Bug caught: the collection hook not skipping `mflux` tests here (they would fail at their
    # `from mflux ...` import, exit 1) or crashing (exit 3), or a skip reason that no longer tells
    # the reader which extra to install. The mirror image (the tests run where mflux is present)
    # is the CI lane's own collected-tests guard.
    if _mflux_is_installed(importlib.util.find_spec("mflux"), importlib.metadata.version):
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
