"""Run DFloat11 losslessly compressed BF16 models on Apple Silicon with MLX."""

from mlx_dfloat._version import __version__
from mlx_dfloat.errors import (
    DFloatBackendError,
    DFloatDependencyError,
    DFloatError,
    DFloatFormatError,
    DFloatResourceError,
    DFloatUnsupportedError,
)

__all__ = [
    "DFloatBackendError",
    "DFloatDependencyError",
    "DFloatError",
    "DFloatFormatError",
    "DFloatResourceError",
    "DFloatUnsupportedError",
    "__version__",
]
