"""Compatibility shim: the watchdog lives in the package."""

from mlx_dfloat._watchdog import *  # noqa: F403
from mlx_dfloat._watchdog import Watchdog, default_ceiling, phys_footprint  # noqa: F401
