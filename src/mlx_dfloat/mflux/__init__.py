"""mflux adapters. Everything here imports mflux lazily; ``import mlx_dfloat`` never needs it."""

import importlib.util

from mlx_dfloat.errors import DFloatDependencyError


def require_mflux() -> None:
    """Raise a package-rooted error when the optional ``mflux`` extra is missing.

    Raises:
        DFloatDependencyError: ``mflux`` cannot be imported.
    """
    if importlib.util.find_spec("mflux") is None:
        raise DFloatDependencyError(
            "the mflux adapters need the optional extra: install mlx-dfloat[mflux]"
        )
