"""Package-rooted exceptions for mlx-dfloat."""


class DFloatError(Exception):
    """Base class for every error raised by mlx-dfloat."""


class DFloatFormatError(DFloatError, ValueError):
    """A DFloat11 checkpoint is malformed or uses a layout this version cannot read."""


class DFloatResourceError(DFloatError, MemoryError):
    """A decode would need more memory than the configured budget allows."""


__all__ = ["DFloatError", "DFloatFormatError", "DFloatResourceError"]
