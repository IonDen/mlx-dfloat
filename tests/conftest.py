"""Pytest gates and MLX memory-safety guard."""

import atexit
import os
import platform
import sys
from typing import Literal

import pytest

from mlx_dfloat._memory_caps import install_memory_caps

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


def _hard_exit_code(recorded: int | None) -> int | None:
    """Code for the atexit hard exit, or None to let the interpreter exit normally.

    None means no session finished, so no Metal work ran and there is no destructor to skip.
    """
    return recorded


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
