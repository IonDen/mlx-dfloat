"""The once-per-process warning on an mflux newer than the newest tested release."""

import builtins
import importlib
import tomllib
import warnings
from pathlib import Path

import pytest

import mlx_dfloat.mflux as adapters
from mlx_dfloat.errors import DFloatDependencyError, DFloatUntestedMfluxWarning
from mlx_dfloat.mflux import _versions
from mlx_dfloat.mflux._versions import NEWEST_TESTED_MFLUX, is_newer_than_tested, release_tuple


def _bump(text: str, field: int, by: int) -> str:
    """``text`` with release field ``field`` (0 major, 1 minor, 2 patch) moved by ``by``, later fields zeroed."""
    parts = [int(p) for p in text.split(".")]
    parts[field] += by
    return ".".join(str(p) if i <= field else "0" for i, p in enumerate(parts))


NEWER_MINOR = _bump(NEWEST_TESTED_MFLUX, 1, 1)
NEWER_PATCH = _bump(NEWEST_TESTED_MFLUX, 2, 1)
OLDER_MINOR = _bump(NEWEST_TESTED_MFLUX, 1, -1)


@pytest.mark.parametrize("text", ["0.24.0", "0.24", "0.24.0rc1", "0.24.0.dev3+gabc", " 0.24.0\n"])
def test_release_tuple_reads_the_release_part(text):
    # Bug caught: a pre-release or local suffix breaks the parse, or a two-field version is not padded (so
    # (0, 24) < (0, 24, 0) and the tested release reads as older).
    assert release_tuple(text) == (0, 24, 0)


def test_release_tuple_is_none_without_a_release_number():
    # Bug caught: a branch name parsed as a version (or raising) instead of being treated as unknown.
    assert release_tuple("main") is None


@pytest.mark.parametrize(
    ("installed", "newer"),
    [
        (NEWER_MINOR, True),
        (NEWER_PATCH, True),
        (NEWEST_TESTED_MFLUX, False),
        (OLDER_MINOR, False),
        (None, False),
        ("garbage", False),
    ],
)
def test_is_newer_than_tested(installed, newer):
    # Bug caught: >= instead of > (the tested release warns), a string comparison ("0.100" < "0.24"), or a patch
    # release slipping through silently.
    assert is_newer_than_tested(installed) is newer


def test_a_two_digit_minor_counts_as_newer():
    # Bug caught: comparing version strings instead of numbers ("0.100.0" < "0.24.0" as text).
    assert is_newer_than_tested("0.100.0", "0.24.0") is True


@pytest.fixture
def fresh(monkeypatch):
    """A process that has not warned yet, with ``installed`` as the installed mflux."""

    def install(installed):
        monkeypatch.setattr(_versions, "_warned", False)
        monkeypatch.setattr(_versions, "installed_mflux_version", lambda: installed)

    return install


def _untested(caught):
    return [w for w in caught if issubclass(w.category, DFloatUntestedMfluxWarning)]


def test_warns_once_per_process_naming_both_versions(fresh):
    # Bug caught: a newer mflux passes silently, the warning repeats on every adapter call, or the message omits a
    # version the user needs for the report.
    fresh(NEWER_MINOR)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _versions.warn_if_untested_mflux()
        _versions.warn_if_untested_mflux()
    got = _untested(caught)
    assert len(got) == 1
    text = str(got[0].message)
    assert NEWER_MINOR in text
    assert NEWEST_TESTED_MFLUX in text
    assert "https://github.com/IonDen/mlx-dfloat/issues" in text


def test_the_warning_is_a_user_warning_from_the_package():
    # Bug caught: a category that a caller's "ignore::UserWarning" or the package-rooted filter cannot reach.
    assert issubclass(DFloatUntestedMfluxWarning, UserWarning)


def test_an_error_filter_does_not_silence_later_calls(fresh):
    # Bug caught: the once-flag set before warn(), so a caller's "error" filter raising on the first call hides the
    # warning from every later call.
    fresh(NEWER_MINOR)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DFloatUntestedMfluxWarning):
            _versions.warn_if_untested_mflux()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _versions.warn_if_untested_mflux()
    assert len(_untested(caught)) == 1


@pytest.mark.parametrize("installed", [NEWEST_TESTED_MFLUX, None])
def test_no_warning_on_the_tested_release_or_without_mflux(fresh, installed):
    # Bug caught: the tested release (or an install without mflux) warning on every run.
    fresh(installed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _versions.warn_if_untested_mflux()
    assert _untested(caught) == []


def test_installed_mflux_version_is_none_without_the_distribution(monkeypatch):
    # Bug caught: PackageNotFoundError escaping on an install without the [mflux] extra.
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_versions, "version", missing)
    assert _versions.installed_mflux_version() is None


def test_installed_mflux_version_reads_the_distribution_metadata(monkeypatch):
    # Bug caught: the version read from another distribution than mflux.
    monkeypatch.setattr(_versions, "version", lambda name: {"mflux": "9.8.7"}[name])
    assert _versions.installed_mflux_version() == "9.8.7"


def test_require_mflux_issues_the_check_when_mflux_is_present(monkeypatch):
    # Bug caught: the warning never wired into the adapters' entry check.
    calls = []
    monkeypatch.setattr(_versions, "installed_mflux_version", lambda: NEWEST_TESTED_MFLUX)
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(_versions, "warn_if_untested_mflux", lambda: calls.append("checked"))
    adapters.require_mflux()
    assert calls == ["checked"]


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _configured_filters() -> list[str]:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]["filterwarnings"]


def _category_field(entry: str) -> str:
    """The category part of a pytest ``action:message:category:module:lineno`` filter."""
    fields = entry.split(":")
    return fields[2] if len(fields) > 2 else ""


def _apply(entries: list[str]) -> None:
    """Install ``entries`` as pytest does: later entries take precedence, categories resolved by name."""
    for entry in entries:
        action, message, category, module, *_ = [*entry.split(":"), "", "", "", ""]
        if category and "." in category:
            module_name, _dot, cls = category.rpartition(".")
            resolved = getattr(importlib.import_module(module_name), cls)
        else:
            resolved = getattr(builtins, category) if category else Warning
        warnings.filterwarnings(action, message, resolved, module)


def test_the_pytest_warning_filters_name_only_builtin_categories():
    # Bug caught: a filter naming a package class (`ignore::mlx_dfloat.errors.DFloatUntestedMfluxWarning`) makes pytest
    # import mlx_dfloat while it reads its configuration, before coverage starts, so errors.py and __init__.py report
    # 0 % in the coverage gate.
    named = [e for e in _configured_filters() if "." in _category_field(e)]
    assert named == []


def test_the_configured_filters_ignore_the_untested_mflux_warning_only(fresh):
    # Bug caught: the message filter no longer matching the warning's text (every test on a newer mflux then fails
    # under "error"), or a filter so broad (`ignore::UserWarning`) that the suite stops being warning-strict.
    fresh(NEWER_MINOR)
    with warnings.catch_warnings(record=True) as caught:
        _apply(_configured_filters())
        _versions.warn_if_untested_mflux()
        with pytest.raises(UserWarning, match="some other warning"):
            warnings.warn("some other warning", UserWarning, stacklevel=1)
    assert _untested(caught) == []


def test_the_warning_points_at_the_caller_of_require_mflux(fresh, monkeypatch):
    # Bug caught: a stacklevel one short (the warning names mlx_dfloat/mflux/__init__.py, a file the user never wrote)
    # or one too far (a frame in pytest).
    fresh(NEWER_MINOR)
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapters.require_mflux()
    got = _untested(caught)
    assert len(got) == 1
    assert got[0].filename == __file__


@pytest.mark.parametrize("installed", ["0.23.0", "0.23.9", "0.22.1"])
def test_require_mflux_refuses_an_mflux_below_the_floor_naming_both_versions(
    fresh, monkeypatch, installed
):
    # Bug caught: an mflux older than 0.24 passing the entry check, so three families later die with a bare
    # ModuleNotFoundError instead of a message naming the version to install. Floor from pyproject: mflux>=0.24.
    fresh(installed)
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(DFloatDependencyError) as raised:
        adapters.require_mflux()
    assert installed in str(raised.value)
    assert "0.24.0" in str(raised.value)


@pytest.mark.parametrize("installed", ["0.24.0", "0.24.1", "1.0.0"])
def test_require_mflux_accepts_the_floor_and_newer(fresh, monkeypatch, installed):
    # Bug caught: > instead of >= (the floor release itself refused), or a string comparison refusing 1.0.0.
    fresh(installed)
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DFloatUntestedMfluxWarning)
        adapters.require_mflux()


def test_a_namespace_leftover_without_a_version_is_not_refused_as_below_the_floor(
    fresh, monkeypatch
):
    # Bug caught: a missing version (a leftover mflux/ directory with no distribution) read as an old release, so
    # require_mflux says "mflux None is installed" instead of behaving as before the floor existed (no version, no
    # version check; tests/test_mflux_guard.py covers the find_spec side).
    fresh(None)
    monkeypatch.setattr(adapters.importlib.util, "find_spec", lambda name: object())
    adapters.require_mflux()
