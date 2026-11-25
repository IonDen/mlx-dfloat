"""The drain an armed call runs after it frees memory, so the process footprint has fallen before the next sample.

A buffer MLX releases leaves the process footprint only after the next GPU submission (about 20 ms with the GPU busy,
80-700 ms idle). An armed call's watchdog samples that footprint, so freed pages that linger can trip it at the next
checkpoint. The drain waits for the GPU, empties MLX's buffer pool, submits one tiny evaluation, and polls the footprint
until it has fallen at least ``FALL_BYTES`` below the first reading and then stopped falling, for at most ``budget_s``
seconds (before that fall, a flat reading is the late release on an idle GPU and a smaller move is page-level noise, so
neither ends the poll); then it waits and empties the pool once more, since the tiny evaluation's own buffers reach the
pool when its command completes.
"""

import math
import time
from collections.abc import Callable

import mlx.core as mx

POLL_S = 0.02
# The smallest fall below the first reading that counts as the release. An idle footprint moves by a few pages
# (dips of 352 KiB and rises of 32 KiB were seen), far less than a released buffer.
FALL_BYTES = 1 << 20
# Looked up at call time, so a test can record the waits.
_sleep: Callable[[float], object] = time.sleep


def _kick() -> None:
    """One tiny GPU evaluation: the driver frees released buffers after the next command submission."""
    mx.eval(mx.zeros((1,)) + 0)


def drain(
    read_footprint: Callable[[], int],
    *,
    budget_s: float = 0.25,
    sleep: Callable[[float], object] | None = None,
) -> None:
    """Wait for the GPU, empty the pool, kick, poll the footprint until its fall ends (``budget_s`` at most), then again.

    ``read_footprint`` returns the process footprint in bytes. The poll waits ``POLL_S`` seconds between readings.
    Until a reading is at least ``FALL_BYTES`` below the first one, the poll goes on; after that, it ends at the first
    reading that is not below the one before. It ends after ``budget_s`` seconds of waits in any case. ``sleep``
    replaces the module's sleep (``time.sleep``).
    """
    wait = _sleep if sleep is None else sleep
    mx.synchronize()
    mx.clear_cache()
    _kick()
    first = last = read_footprint()
    fell = False
    for _ in range(math.floor(budget_s / POLL_S + 1e-9)):
        wait(POLL_S)
        now = read_footprint()
        if fell and now >= last:
            break
        if first - now >= FALL_BYTES:
            fell = True
        last = now
    mx.synchronize()
    mx.clear_cache()
