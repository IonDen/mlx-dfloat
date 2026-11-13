"""The newest mflux release these adapters were tested on, and how a newer one is recognised.

Imports no mflux: the installed version comes from package metadata, so this module loads on an install without the
[mflux] extra.
"""

import re
import warnings
from importlib.metadata import PackageNotFoundError, version

from mlx_dfloat.errors import DFloatUntestedMfluxWarning

# The newest mflux release the test suite ran against. The [mflux] extra has a floor and no upper bound: a newer mflux
# installs and runs, and the adapters warn once per process that it was not tested.
NEWEST_TESTED_MFLUX = "0.24.0"

# The oldest mflux the adapters run on: the [mflux] extra's floor in pyproject.toml. Older releases lack modules three
# families import, so ``require_mflux`` refuses them by name instead of a bare ModuleNotFoundError later.
MIN_MFLUX = "0.24.0"

_LEADING_RELEASE = re.compile(r"\d+(?:\.\d+)*")

_warned = False


def release_tuple(text: str) -> tuple[int, ...] | None:
    """The numeric release part of a version string, padded to three fields.

    "0.24.0", "0.24", "0.24.0rc1" and "0.24.0.dev3+gabc" all give (0, 24, 0). None when the string does not start
    with a number.
    """
    match = _LEADING_RELEASE.match(text.strip())
    if match is None:
        return None
    parts = tuple(int(part) for part in match.group(0).split("."))
    return parts + (0,) * (3 - len(parts))


def is_newer_than_tested(installed: str | None, newest: str = NEWEST_TESTED_MFLUX) -> bool:
    """True when ``installed`` is a later release than ``newest``; False for None or a string with no release."""
    if installed is None:
        return False
    mine = release_tuple(installed)
    tested = release_tuple(newest)
    if mine is None or tested is None:
        return False
    return mine > tested


def is_below_floor(installed: str, floor: str = MIN_MFLUX) -> bool:
    """True when ``installed`` is an earlier release than ``floor``; False for a string with no release number."""
    mine = release_tuple(installed)
    oldest = release_tuple(floor)
    if mine is None or oldest is None:
        return False
    return mine < oldest


def installed_mflux_version() -> str | None:
    """The installed mflux distribution's version, or None when mflux is not installed."""
    try:
        return version("mflux")
    except PackageNotFoundError:
        return None


def warn_if_untested_mflux() -> None:
    """Warn once per process when the installed mflux is newer than ``NEWEST_TESTED_MFLUX``.

    Called by ``require_mflux``; the warning points at its caller. Never refuses: generation still runs.
    """
    global _warned
    if _warned:
        return
    installed = installed_mflux_version()
    if not is_newer_than_tested(installed):
        return
    warnings.warn(
        f"mflux {installed} is newer than {NEWEST_TESTED_MFLUX}, the newest mflux this version of mlx-dfloat was "
        "tested with. Image generation still runs. If it fails or the images look wrong, please report it with both "
        "versions at https://github.com/IonDen/mlx-dfloat/issues",
        DFloatUntestedMfluxWarning,
        stacklevel=3,
    )
    # After the call: a caller's "error" filter makes warn() raise, and that must not silence later calls.
    _warned = True
