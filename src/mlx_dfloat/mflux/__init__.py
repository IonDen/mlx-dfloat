"""mflux adapters. Everything here imports mflux lazily; ``import mlx_dfloat`` never needs it."""

import importlib.util
from typing import Any

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


def __getattr__(name: str) -> Any:
    """``DFloatModel``, ``DFloatFlux1``, ``DFloatZImage`` and ``DFloatFlux2Klein`` are imported on first use.

    So ``import mlx_dfloat.mflux`` needs neither mflux nor MLX.
    """
    if name == "DFloatModel":
        from mlx_dfloat.mflux.families import DFloatModel

        return DFloatModel
    if name == "DFloatFlux1":
        from mlx_dfloat.mflux.flux1.model import DFloatFlux1

        return DFloatFlux1
    if name == "DFloatZImage":
        from mlx_dfloat.mflux.zimage.model import DFloatZImage

        return DFloatZImage
    if name == "DFloatFlux2Klein":
        from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

        return DFloatFlux2Klein
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The model classes stay out of __all__: a star-import must not need mflux; `mlx_dfloat.mflux.DFloatFlux1`,
# `mlx_dfloat.mflux.DFloatZImage` and `mlx_dfloat.mflux.DFloatFlux2Klein` still work.
__all__ = ["DFloatModel", "require_mflux"]
