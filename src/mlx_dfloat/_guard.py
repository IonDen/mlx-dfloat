"""The call guard: which watchdog watches the running call, and the checkpoints that stop it.

A watched call runs inside ``scope(watchdog)``, which makes the watchdog the active one for that
call's context and resets it on the way out, also after a raise. ``checkpoint(site)`` is called at
the boundaries where the main thread runs Python (a transformer block, a prompt's encode, a group
of the set load). It does nothing unless the active watchdog has tripped; then it stops the
watchdog's grace clock and raises ``DFloatWatchdogError`` naming the site.

The watchdog is typed by a ``Protocol``, so this module does not import the watchdog's.
"""

import contextlib
import contextvars
import typing

from mlx_dfloat.errors import DFloatWatchdogError

_GIB = 1024**3


class _Flag(typing.Protocol):
    def is_set(self) -> bool: ...


class GuardedWatchdog(typing.Protocol):
    """What the guard reads and calls on a raise-mode watchdog."""

    @property
    def tripped(self) -> _Flag:
        """Set once the stop line was crossed."""
        ...

    @property
    def trip(self) -> dict[str, typing.Any] | None:
        """The trip's record, or None before a trip."""
        ...

    @property
    def artifact(self) -> typing.Any:
        """The path the record was written to, or None."""
        ...

    @property
    def ceiling(self) -> int:
        """The ceiling in bytes."""
        ...

    def consume(self) -> None:
        """Stop the grace clock."""
        ...

    def set_ceiling(self, ceiling: int) -> int:
        """Replace the ceiling; return the previous one."""
        ...


_ACTIVE: contextvars.ContextVar[GuardedWatchdog | None] = contextvars.ContextVar(
    "mlx_dfloat_watchdog", default=None
)


def active() -> GuardedWatchdog | None:
    """The watchdog watching the running call, or None."""
    return _ACTIVE.get()


@contextlib.contextmanager
def scope(watchdog: GuardedWatchdog) -> typing.Iterator[None]:
    """Make ``watchdog`` the active one for the block; the previous one is back afterward."""
    token = _ACTIVE.set(watchdog)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def tightened(ceiling: int | None) -> typing.Iterator[None]:
    """Lower the active watchdog's ceiling to ``ceiling`` for the block, then restore it.

    Nothing changes when no watchdog is active, when ``ceiling`` is None, or when it is not
    lower than the active one: the lower ceiling wins.
    """
    watchdog = _ACTIVE.get()
    if watchdog is None or ceiling is None or ceiling >= watchdog.ceiling:
        yield
        return
    previous = watchdog.set_ceiling(ceiling)
    try:
        yield
    finally:
        watchdog.set_ceiling(previous)


def checkpoint(site: str) -> None:
    """Raise if the active watchdog has tripped; a no-op otherwise.

    The ``encoder_loaded`` checkpoint runs before the text encoder's weights land: the encoder
    loads lazily and its weights arrive in the first prompt's evaluation. So loading the encoder
    and encoding the first prompt form one stretch with no checkpoint between them.

    Raises:
        DFloatWatchdogError: The active watchdog tripped. Its grace clock is stopped first, so
            the cleanup that follows the error cannot be ended by the clock.
    """
    watchdog = _ACTIVE.get()
    if watchdog is None or not watchdog.tripped.is_set():
        return
    watchdog.consume()
    raise error_for(watchdog, site)


def _gib(value: object) -> str:
    return f"{value / _GIB:.2f} GiB" if isinstance(value, int | float) else "unread"


def _seconds(value: object) -> str:
    return f"{value:.1f} s" if isinstance(value, int | float) else "unread"


def error_for(watchdog: GuardedWatchdog, site: str) -> DFloatWatchdogError:
    """The error a checkpoint raises for ``watchdog``'s trip, stopped at ``site``."""
    record = dict(watchdog.trip or {})
    reason = str(record.get("reason"))
    phase = (record.get("context") or {}).get("phase")
    artifact = watchdog.artifact
    where = f"record: {artifact}" if artifact is not None else "record not written"
    if reason == "wall":
        measured = (
            f"ran {_seconds(record.get('elapsed'))} against the wall budget of "
            f"{_seconds(record.get('budget'))}"
        )
    else:
        measured = (
            f"watched {_gib(record.get('verdict_memory'))} against the stop line "
            f"{_gib(record.get('stop_line'))}"
        )
    message = f"the memory watchdog stopped the call at {site} ({reason}, phase {phase}): {measured}; {where}"
    return DFloatWatchdogError(
        message,
        reason=reason,
        phase=None if phase is None else str(phase),
        site=site,
        artifact=artifact,
        record=record,
    )
