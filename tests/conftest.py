"""Pytest gates and MLX memory-safety guard."""

import atexit
import importlib.metadata
import importlib.util
import os
import platform
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from typing import Any, Literal

import pytest

from mlx_dfloat._memory_caps import install_memory_caps

# pytester runs the autouse fixture's teardown check end to end in a child pytest (tests/test_markers.py).
pytest_plugins = ["pytester"]

GATED_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("slow", "--run-slow", "real-weights decode"),
    ("network", "--run-network", "real network I/O"),
)

# Install caps at import, before collection imports any MLX-heavy module.
INSTALLED_CAPS_GB = install_memory_caps()

# None until pytest_sessionfinish runs. A usage or config error (unknown flag, --strict-config)
# never starts a session, so the hard exit below must not replace pytest's own exit code.
_FINAL_EXIT_CODE: int | None = None


def _markers_to_skip(enabled_flags: set[str]) -> list[tuple[str, str]]:
    return [
        (marker, f"requires {flag} ({description})")
        for marker, flag, description in GATED_MARKERS
        if flag not in enabled_flags
    ]


def _metal_marker_action(system: str, machine: str) -> Literal["run", "skip"]:
    """`metal` tests run (and may fail) on Apple Silicon; elsewhere they skip with a reason."""
    return "run" if (system == "Darwin" and machine == "arm64") else "skip"


def _mflux_marker_action(available: bool) -> Literal["run", "skip"]:
    """`mflux` tests run where mflux is importable and skip elsewhere (the extra is optional)."""
    return "run" if available else "skip"


def _mflux_is_installed(spec: ModuleSpec | None, version: Callable[[str], str]) -> bool:
    """True only for a real mflux install: an importable module with an origin and its distribution metadata.

    A leftover ``mflux/`` directory with no ``__init__.py`` and no dist-info is a namespace package: ``find_spec``
    still returns a spec (with ``origin=None``), but nothing in it imports.
    """
    if spec is None or spec.origin is None:
        return False
    try:
        version("mflux")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _hard_exit_code(recorded: int | None) -> int | None:
    """Code for the atexit hard exit, or None to let the interpreter exit normally.

    None means no session finished, so no Metal work ran and there is no destructor to skip.
    """
    return recorded


class _ExitRecorder:
    """Stands in for the process exit of an armed model call: records each code instead of ending pytest."""

    def __init__(self) -> None:
        self.codes: list[int] = []

    def __call__(self, code: int) -> None:
        self.codes.append(code)

    def take(self) -> list[int]:
        """Hand the recorded codes to a test that expected an escalation; the teardown then sees none."""
        taken, self.codes = self.codes, []
        return taken


def _host_ceiling_32gib() -> int:
    # A 32 GiB Mac's ceiling (32 - 4 GiB) as a literal, so no expected number depends on this runner's RAM.
    return 30_064_771_072


def _api_watchdog_patches() -> list[tuple[str, object]]:
    """The ``mlx_dfloat.mflux._pipeline`` attributes every test replaces, with fresh fakes.

    The counter reader becomes a ``FakeCounters`` at zero, the host's default ceiling the 32 GiB literal, and the
    process exit a recorder, so a decorated model call is armed in every test, never trips on the runner's real
    memory, and can never end pytest.
    """
    from tests._fake_counters import FakeCounters

    return [
        ("_counters", FakeCounters()),
        ("_default_ceiling", _host_ceiling_32gib),
        ("_exit_process", _ExitRecorder()),
    ]


def _untaken_exits(codes: list[int]) -> str | None:
    """The teardown's failure message for exit codes left in the recorder, or None when there are none."""
    if not codes:
        return None
    return (
        f"an armed model call escalated to process exit {codes} and the test did not take the code"
    )


@dataclass(frozen=True, slots=True)
class ApiWatchdogFakes:
    """What ``_api_watchdog_fakes`` yields: the fake allocator the watchdog reads and the exit recorder."""

    counters: Any
    exits: _ExitRecorder


@pytest.fixture(autouse=True)
def _api_watchdog_fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[ApiWatchdogFakes]:
    from mlx_dfloat import _watchdog
    from mlx_dfloat.mflux import _pipeline

    patches = dict(_api_watchdog_patches())
    for name, value in patches.items():
        monkeypatch.setattr(_pipeline, name, value)
    exits = patches["_exit_process"]
    assert isinstance(exits, _ExitRecorder)
    # A watchdog built without ``exit_process`` falls back to the module's ``_exit``: record it the same way.
    monkeypatch.setattr(_watchdog, "_exit", exits)
    yield ApiWatchdogFakes(counters=patches["_counters"], exits=exits)
    message = _untaken_exits(exits.codes)
    if message is not None:
        pytest.fail(message)


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, flag, description in GATED_MARKERS:
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run `{marker}` tests ({description}); skipped by default",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = {flag for _marker, flag, _description in GATED_MARKERS if config.getoption(flag)}
    for marker, reason in _markers_to_skip(enabled):
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)

    if _metal_marker_action(platform.system(), platform.machine()) == "skip":
        metal_skip = pytest.mark.skip(reason="metal: requires Apple Silicon")
        for item in items:
            if "metal" in item.keywords:
                item.add_marker(metal_skip)

    mflux_available = _mflux_is_installed(
        importlib.util.find_spec("mflux"), importlib.metadata.version
    )
    if _mflux_marker_action(available=mflux_available) == "skip":
        mflux_skip = pytest.mark.skip(reason="mflux: install the mlx-dfloat[mflux] extra")
        for item in items:
            if "mflux" in item.keywords:
                item.add_marker(mflux_skip)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Record pytest's final exit code (already reflects test failures and the
    # --cov-fail-under gate) for the atexit hard-exit below.
    global _FINAL_EXIT_CODE
    _FINAL_EXIT_CODE = int(session.exitstatus)


@atexit.register
def _hard_exit_past_metal_teardown() -> None:  # pragma: no cover - runs at interpreter shutdown
    """Skip MLX's Metal backend C++ destructor, which can segfault at interpreter shutdown.

    Runs after pytest has printed its summary and recorded the final exit code, so output
    is preserved and a real failure still exits non-zero.
    """
    code = _hard_exit_code(_FINAL_EXIT_CODE)
    if code is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
