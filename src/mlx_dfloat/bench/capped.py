"""Capped mode: run under a smaller Mac's MLX limits on this one.

A CAPPED tier installs the wired and memory caps ``mlx-dfloat`` itself installs on a Mac of that
size (``install_memory_caps`` on a device whose recommended working set is the tier's; see
``_memory_caps.caps_for_recommended_bytes``), so the run sees the limits a user of that Mac runs
under. The wired cap matters: without it MLX allocated noticeably more for the same FLUX.2 Klein
VAE decode (``mx.get_peak_memory``), with bit-identical output, whatever the memory limit (mlx
0.32.2). When that Mac would get no caps (a working set under 1 GiB), the tier keeps MLX's own
defaults: memory limit ``min(1.5 x recommended working set, 0.95 x RAM)`` and no wired limit. MLX
starts reclaiming its buffer cache at ``min(memory_limit, 0.95 x recommended)`` (verified
2026-09-29 on mlx 0.32.2), so that is the cache limit installed as the stand-in for the real
reclaim point; the worker then installs its own, smaller cache limit and records the effective
value. Two numbers sit below the tier's recommended working set. The fit budget is what
``generate``'s fit check uses on a real Mac of that size (``integrate.memory.budget_for``: the
working set minus 2 GiB). The watchdog ceiling is the working set minus ``reserve_for(tier)``
(1.5 GiB up to 24 GB, 2 GiB above); crossing it aborts the run, because ``set_memory_limit`` only
throttles and never fails an allocation. The recommended working set comes from device data when a datapoint exists, else from
the 2/3 (16 and 24 GB) and 3/4 (above 24 GB) ratios, and for the host tier from the host's own
value; the host tier's record keeps MLX's defaults (wired 0) because a host-tier run installs the
host's own caps itself (``is_host``), so MEASURED rows run under the limits a plain ``generate``
uses.
"""

import dataclasses

import mlx.core as mx

from mlx_dfloat._memory_caps import caps_for_recommended_bytes, remember_wired_limit
from mlx_dfloat.errors import DFloatUnsupportedError
from mlx_dfloat.integrate.memory import budget_for

GIB = 1024**3


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TierLimits:
    """The limits capped mode installs for one tier, and the label its rows get.

    ``budget_bytes`` is the tier's recommended working set (the figure both reserves come off, kept
    under this name for the committed result files); ``reserve_bytes`` and ``ceiling_bytes`` are the
    watchdog's reserve and abort line; ``fit_budget_bytes`` is the fit check's budget, the one a real
    Mac of that size applies.
    """

    tier_gb: int
    ram_bytes: int
    recommended_bytes: int
    memory_limit_bytes: int
    cache_limit_bytes: int
    wired_limit_bytes: int
    reserve_bytes: int
    budget_bytes: int
    ceiling_bytes: int
    fit_budget_bytes: int
    budget_source: str
    is_host: bool
    label: str

    def as_dict(self) -> dict[str, int | str | bool]:
        """Every field, for a result file."""
        return dataclasses.asdict(self)


def reserve_for(tier_gb: int) -> int:
    """The watchdog reserve below a tier's recommended working set: 1.5 GiB up to 24 GB, 2 GiB above."""
    return int(1.5 * GIB) if tier_gb <= 24 else 2 * GIB


def recommended_ratio(tier_gb: int) -> float:
    """The share of RAM assumed as the recommended working set until device data exists: 2/3 up to 24 GB, 0.75 above 24."""
    return 2 / 3 if tier_gb <= 24 else 0.75


def host_tier_gb(host_ram_bytes: int) -> int:
    """The host's own tier: its RAM rounded to whole GB."""
    return round(host_ram_bytes / GIB)


def tier_limits(
    tier_gb: int,
    *,
    host_ram_bytes: int,
    host_recommended_bytes: int,
    device_recommended_bytes: int | None = None,
) -> TierLimits:
    """The limits for ``tier_gb`` on a host with ``host_ram_bytes`` and ``host_recommended_bytes``.

    Raises:
        DFloatUnsupportedError: The tier is larger than the host (it cannot be emulated here).
    """
    host_tier = host_tier_gb(host_ram_bytes)
    if tier_gb > host_tier:
        raise DFloatUnsupportedError(
            f"a {tier_gb} GB tier cannot be emulated on a {host_tier} GB host"
        )
    ram = tier_gb * GIB
    is_host = tier_gb == host_tier
    if is_host:
        recommended, source = host_recommended_bytes, "device"
    elif device_recommended_bytes is not None:
        recommended, source = device_recommended_bytes, "device"
    else:
        # Integer arithmetic (the float ratio above is for display): 2/3 up to 24 GB, 3/4 above.
        recommended, source = (ram * 2 // 3 if tier_gb <= 24 else ram * 3 // 4), "ratio"
    wired, limit = (0, 0) if is_host else caps_for_recommended_bytes(recommended)
    if limit == 0:
        # The host tier, or a Mac on which install_memory_caps installs nothing: MLX's own defaults.
        limit = min(int(1.5 * recommended), int(0.95 * ram))
    reserve = reserve_for(tier_gb)
    return TierLimits(
        tier_gb=tier_gb,
        ram_bytes=ram,
        recommended_bytes=recommended,
        memory_limit_bytes=limit,
        cache_limit_bytes=min(limit, int(0.95 * recommended)),
        wired_limit_bytes=wired,
        reserve_bytes=reserve,
        budget_bytes=recommended,
        ceiling_bytes=recommended - reserve,
        fit_budget_bytes=budget_for(recommended),
        budget_source=source,
        is_host=is_host,
        label="MEASURED" if is_host else "CAPPED",
    )


def apply(limits: TierLimits) -> dict[str, int]:
    """Install the tier's memory, cache and wired limits; returns the previous values.

    The wired limit is recorded (``_memory_caps.remember_wired_limit``), so a model's per-call caps see it without
    probing it again.
    """
    previous = {
        "memory": int(mx.set_memory_limit(limits.memory_limit_bytes)),
        "cache": int(mx.set_cache_limit(limits.cache_limit_bytes)),
        "wired": int(mx.set_wired_limit(limits.wired_limit_bytes)),
    }
    remember_wired_limit(mx, limits.wired_limit_bytes)
    return previous


def current_limits() -> dict[str, int]:
    """The three MLX limits in force (MLX 0.32.2 has no getters; the setters return the previous value).

    Each limit is set to 0 and restored to the value that returned. Verified on mlx 0.32.2: 0 is accepted
    by all three setters, and ``set_cache_limit(0)`` does not release cached buffers on the spot (the
    pool size was unchanged right after), so the read has no lasting side effect; call it at worker
    start, not while a run is allocating.
    """
    out: dict[str, int] = {}
    for name, setter in (
        ("memory", mx.set_memory_limit),
        ("cache", mx.set_cache_limit),
        ("wired", mx.set_wired_limit),
    ):
        previous = int(setter(0))
        setter(previous)
        out[name] = previous
    return out


def limits_record(
    limits: TierLimits,
    *,
    effective_memory_limit: int,
    effective_cache_limit: int,
    effective_wired_limit: int,
    applied: str,
) -> dict[str, object]:
    """What a result file stores under ``limits``: the tier's numbers, the values in force, and which path installed them."""
    return {
        "tier": limits.as_dict(),
        "effective": {
            "memory_limit_bytes": effective_memory_limit,
            "cache_limit_bytes": effective_cache_limit,
            "wired_limit_bytes": effective_wired_limit,
        },
        "applied": applied,
    }


__all__ = [
    "GIB",
    "TierLimits",
    "apply",
    "current_limits",
    "host_tier_gb",
    "limits_record",
    "recommended_ratio",
    "reserve_for",
    "tier_limits",
]
