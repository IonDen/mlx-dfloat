"""Run DFloat11 losslessly compressed BF16 models on Apple Silicon with MLX."""

from mlx_dfloat._version import __version__
from mlx_dfloat.errors import (
    DFloatAccessError,
    DFloatBackendError,
    DFloatDependencyError,
    DFloatError,
    DFloatFormatError,
    DFloatIntegrationError,
    DFloatResourceError,
    DFloatUnsupportedError,
)

__all__ = [
    "DFloatAccessError",
    "DFloatBackendError",
    "DFloatDependencyError",
    "DFloatError",
    "DFloatFormatError",
    "DFloatIntegrationError",
    "DFloatResourceError",
    "DFloatUnsupportedError",
    "__version__",
]
