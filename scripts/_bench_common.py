"""The pure, unit-tested arithmetic behind the bench and parity scripts.

Nothing here touches MLX, Metal or the filesystem, so the numbers these scripts act on can be
checked without a GPU.
"""


def per_dispatch_guard(bytes_out: int, measured_bps: float, limit_s: float = 0.25) -> None:
    """Refuse a single kernel dispatch projected to run longer than ``limit_s``.

    One long dispatch can trip the macOS GPU watchdog and stall the display, so a whole group is
    decoded in one dispatch only while its output, at the measured throughput, fits the limit.

    Args:
        bytes_out: Bytes the dispatch writes (2 per BF16 element).
        measured_bps: Measured decode throughput, in bytes per second.
        limit_s: Longest acceptable projected dispatch, in seconds.

    Raises:
        RuntimeError: The projected time ``bytes_out / measured_bps`` exceeds ``limit_s``.
    """
    projected = bytes_out / measured_bps
    if projected > limit_s:
        raise RuntimeError(
            f"one dispatch of {bytes_out} bytes is projected at {projected:.3g} s, over the "
            f"{limit_s:.3g} s per-dispatch limit"
        )
