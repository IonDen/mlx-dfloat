"""Package-rooted exceptions for mlx-dfloat."""


class DFloatError(Exception):
    """Base class for every error raised by mlx-dfloat."""


class DFloatFormatError(DFloatError, ValueError):
    """A DFloat11 checkpoint is malformed or uses a layout this version cannot read."""


class DFloatResourceError(DFloatError, MemoryError):
    """A decode would need more memory than the configured budget allows."""


class DFloatBackendError(DFloatError):
    """A decode backend is unavailable here or its kernel cannot run."""


class DFloatIntegrationError(DFloatError, RuntimeError):
    """A block-seam or name-map invariant failed: a name outside the map, a missing or extra layer, a shape mismatch."""


class DFloatUnsupportedError(DFloatError, ValueError):
    """An option this integration does not support on the DFloat11 path (quantisation, LoRA, img2img, ...)."""


class DFloatDependencyError(DFloatError, ImportError):
    """An optional dependency this feature needs is not installed."""


__all__ = [
    "DFloatBackendError",
    "DFloatDependencyError",
    "DFloatError",
    "DFloatFormatError",
    "DFloatIntegrationError",
    "DFloatResourceError",
    "DFloatUnsupportedError",
]
