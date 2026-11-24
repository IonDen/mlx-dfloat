"""Package-rooted exceptions for mlx-dfloat."""

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DFloatError(Exception):
    """Base class for every error raised by mlx-dfloat."""


class DFloatFormatError(DFloatError, ValueError):
    """A DFloat11 checkpoint is malformed or uses a layout this version cannot read."""


class DFloatResourceError(DFloatError, MemoryError):
    """A decode would need more memory than the configured budget allows."""


class DFloatWatchdogError(DFloatResourceError):
    """A watched call stopped at a checkpoint because the watchdog tripped.

    Attributes:
        reason: ``"memory"``, ``"wall"`` or ``"sample_error"``.
        phase: The phase the trip happened in, or None when no phase was running.
        site: The checkpoint where the call stopped.
        artifact: The abort record written to disk, or None when none was written.
        record: The abort record's contents.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        phase: str | None,
        site: str,
        artifact: Path | None,
        record: Mapping[str, Any],
    ) -> None:
        """Keep the trip's fields next to the message."""
        super().__init__(message)
        self.reason = reason
        self.phase = phase
        self.site = site
        self.artifact = artifact
        self.record: dict[str, Any] = dict(record)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the keyword-only fields, which the default exception pickling drops."""
        rebuild = functools.partial(
            type(self),
            reason=self.reason,
            phase=self.phase,
            site=self.site,
            artifact=self.artifact,
            record=self.record,
        )
        return (rebuild, self.args)


class DFloatBackendError(DFloatError):
    """A decode backend is unavailable here or its kernel cannot run."""


class DFloatIntegrationError(DFloatError, RuntimeError):
    """A block-seam or name-map invariant failed: a name outside the map, a missing or extra layer, a shape mismatch."""


class DFloatUnsupportedError(DFloatError, ValueError):
    """An option this integration does not support on the DFloat11 path (quantisation, LoRA, img2img, ...)."""


class DFloatDependencyError(DFloatError, ImportError):
    """An optional dependency this feature needs is not installed."""


class DFloatAccessError(DFloatError, PermissionError):
    """A model repository on the Hub that this account may not read (a gated licence not accepted, or no token)."""


class DFloatUntestedMfluxWarning(UserWarning):
    """The installed mflux is newer than the newest release mlx-dfloat was tested with."""


__all__ = [
    "DFloatAccessError",
    "DFloatBackendError",
    "DFloatDependencyError",
    "DFloatError",
    "DFloatFormatError",
    "DFloatIntegrationError",
    "DFloatResourceError",
    "DFloatUnsupportedError",
    "DFloatUntestedMfluxWarning",
    "DFloatWatchdogError",
]
