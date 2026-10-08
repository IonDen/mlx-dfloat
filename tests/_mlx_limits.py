"""MLX's memory limits as a Python-API user starts with them, for the per-call caps tests."""

from collections.abc import Iterator
from contextlib import contextmanager

import mlx.core as mx

from mlx_dfloat._memory_caps import compute_safe_caps_gb
from mlx_dfloat.bench.capped import current_limits

GIB = 1024**3

__all__ = ["command_caps", "current_limits", "mlx_without_wired_cap"]


def command_caps() -> tuple[int, int]:
    """The (wired, memory) caps in bytes the ``mlx-dfloat`` command installs here (``install_memory_caps``)."""
    wired_gb, memory_gb = compute_safe_caps_gb()
    return (wired_gb * GIB, memory_gb * GIB)


@contextmanager
def mlx_without_wired_cap() -> Iterator[dict[str, int]]:
    """MLX's default wired limit (0, no cap) for the block, as a Python-API user's process starts.

    The suite installs the caps at import (``tests/conftest.py``); this lifts the wired one and puts every limit back
    afterwards. Yields the limits in force on entry, the wired one then 0.
    """
    before = current_limits()
    mx.set_wired_limit(0)
    try:
        yield {**before, "wired": 0}
    finally:
        mx.set_memory_limit(before["memory"])
        mx.set_wired_limit(before["wired"])
