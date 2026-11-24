"""The watchdog settings a model call runs under.

Standard library only, so ``import mlx_dfloat.mflux`` still loads neither MLX nor mflux.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from mlx_dfloat.errors import DFloatUnsupportedError


@dataclass(frozen=True, slots=True, kw_only=True)
class WatchdogConfig:
    """How the watchdog guards a model call.

    Memory values are in bytes. A checkpoint is a place where the call can stop cleanly, such as between
    transformer blocks.

    Attributes:
        ceiling: The memory ceiling in bytes. None means the Mac's RAM minus 4 GiB, read at call time. A ceiling
            can only be lower than that: a higher one is refused when the call starts.
        stop_margin: Bytes below the ceiling at which the call stops at its next checkpoint. It must be above 0
            and below the ceiling.
        grace: Seconds the call may take to reach a checkpoint after a wall-time trip, or after a trip when the
            latest memory reading failed (counted from the first failed sample when the trip was on memory). A
            memory trip with live readings has no grace: the process ends only at the ceiling.
        wall_budget: Seconds the whole call may take, or None for no wall limit.
        artifact_dir: Folder for the abort record, a str or a path (stored as a ``Path``). None means
            ``Path(tempfile.gettempdir()) / "mlx-dfloat"``. A record file is written when a trip happens, unless
            the write fails or the default folder is refused (a symlink, not a folder, or owned by another user);
            the error then says "record not written". The folder is created only when a trip happens.
    """

    ceiling: int | None = None
    stop_margin: int = 1_073_741_824
    grace: float = 10.0
    wall_budget: float | None = None
    artifact_dir: Path | None = None

    def __post_init__(self) -> None:
        """Refuse a value the watchdog cannot work with.

        Raises:
            DFloatUnsupportedError: a ceiling of 0 or less, a stop margin of 0 or less, a stop margin not below an
                explicit ceiling, a grace of 0 or less, a wall budget of 0 or less, or an artifact folder that is
                neither a str nor a path.
        """
        folder: object = self.artifact_dir
        if folder is not None and not isinstance(folder, Path):
            text = os.fspath(folder) if isinstance(folder, str | os.PathLike) else None
            if not isinstance(text, str):
                raise DFloatUnsupportedError(
                    f"WatchdogConfig.artifact_dir must be a str or a path, got {type(folder).__name__}"
                )
            object.__setattr__(self, "artifact_dir", Path(text))
        if self.ceiling is not None and self.ceiling <= 0:
            raise DFloatUnsupportedError(
                f"WatchdogConfig.ceiling must be above 0 bytes, got {self.ceiling}"
            )
        if self.stop_margin <= 0:
            raise DFloatUnsupportedError(
                f"WatchdogConfig.stop_margin must be above 0 bytes, got {self.stop_margin}"
            )
        if self.ceiling is not None and self.stop_margin >= self.ceiling:
            raise DFloatUnsupportedError(
                f"WatchdogConfig.stop_margin ({self.stop_margin}) must be below the ceiling "
                f"({self.ceiling})"
            )
        if self.grace <= 0:
            raise DFloatUnsupportedError(
                f"WatchdogConfig.grace must be above 0 seconds, got {self.grace}"
            )
        if self.wall_budget is not None and self.wall_budget <= 0:
            raise DFloatUnsupportedError(
                f"WatchdogConfig.wall_budget must be above 0 seconds, got {self.wall_budget}"
            )


DEFAULT_WATCHDOG = WatchdogConfig()


def check_watchdog_arg(value: object) -> WatchdogConfig | None:
    """Return a model's ``watchdog`` argument unchanged when it is None or a ``WatchdogConfig``.

    Raises:
        DFloatUnsupportedError: any other value, such as ``False``, ``True`` or a dict.
    """
    if value is None or isinstance(value, WatchdogConfig):
        return value
    raise DFloatUnsupportedError(
        f"watchdog must be a WatchdogConfig, or watchdog=None to turn the watchdog off; "
        f"got {type(value).__name__} {value!r}"
    )
