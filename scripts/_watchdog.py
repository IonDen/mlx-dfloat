"""Compatibility shim: the watchdog lives in the package."""

from mlx_dfloat._watchdog import *  # noqa: F403
from mlx_dfloat._watchdog import (  # noqa: F401
    Watchdog,
    default_ceiling,
    phys_footprint,
    watched_memory,
)
