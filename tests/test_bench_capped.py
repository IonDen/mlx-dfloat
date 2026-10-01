"""Capped mode: a tier's MLX limits and watchdog ceiling, worked by hand from the north star §4.3."""

import pytest

import mlx_dfloat.bench.capped as capped
from mlx_dfloat.bench.capped import (
    GIB,
    TierLimits,
    apply,
    current_limits,
    host_tier_gb,
    limits_record,
    recommended_ratio,
    tier_limits,
)
from mlx_dfloat.errors import DFloatUnsupportedError

HOST_RAM = 32 * GIB
HOST_REC = 26_800_000_000  # what this M1 Max reports (24.96 GiB)


def test_the_host_tier_uses_the_devices_own_recommended_set_and_is_measured():
    # Bug caught: the host row computed from the 0.75 ratio (24 GiB) instead of the device's
    # 24.96 GiB, or labelled CAPPED.
    lim = tier_limits(32, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == HOST_REC
    assert lim.budget_source == "device"
    assert lim.label == "MEASURED"
    assert lim.is_host is True
    assert lim.reserve_bytes == 2 * GIB
    assert lim.ceiling_bytes == HOST_REC - 2 * GIB
    assert lim.memory_limit_bytes == min(int(1.5 * HOST_REC), int(0.95 * HOST_RAM))
    assert lim.cache_limit_bytes == min(lim.memory_limit_bytes, int(0.95 * HOST_REC))
    assert lim.wired_limit_bytes == 0


def test_a_24_gb_tier_is_capped_with_the_two_thirds_ratio_and_a_1_5_gib_reserve():
    # By hand: rec = 24 GiB x 2/3 = 16 GiB; memory limit = min(24 GiB, 22.8 GiB) = 22.8 GiB;
    # cache limit = min(22.8, 0.95 x 16 = 15.2) = 15.2 GiB; ceiling = 16 - 1.5 = 14.5 GiB.
    lim = tier_limits(24, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == 16 * GIB
    assert lim.memory_limit_bytes == int(0.95 * 24 * GIB)
    assert lim.cache_limit_bytes == int(0.95 * 16 * GIB)
    assert lim.ceiling_bytes == int(14.5 * GIB)
    assert lim.budget_source == "ratio"
    assert lim.label == "CAPPED"
    assert lim.is_host is False


def test_a_16_gb_tier():
    # By hand: rec = 16 GiB x 2 // 3; memory limit = min(1.5 x rec, 15.2 GiB) = 15.2 GiB; ceiling = rec - 1.5 GiB.
    lim = tier_limits(16, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == 16 * GIB * 2 // 3
    assert lim.memory_limit_bytes == int(0.95 * 16 * GIB)
    assert lim.ceiling_bytes == lim.recommended_bytes - int(1.5 * GIB)


def test_a_48_gb_tier_on_a_64_gb_host_uses_the_three_quarters_ratio():
    # The 0.75 branch: rec = 48 x 3 // 4 = 36 GiB; reserve 2 GiB; CAPPED.
    lim = tier_limits(48, host_ram_bytes=64 * GIB, host_recommended_bytes=int(0.78 * 64 * GIB))
    assert lim.recommended_bytes == 36 * GIB
    assert lim.reserve_bytes == 2 * GIB
    assert lim.label == "CAPPED"


def test_device_data_overrides_the_ratio_and_is_labelled_device():
    # Bug caught: a device datapoint (from a real 16 GB Mac) ignored in favour of the ratio.
    lim = tier_limits(
        16,
        host_ram_bytes=HOST_RAM,
        host_recommended_bytes=HOST_REC,
        device_recommended_bytes=11_453_000_000,
    )
    assert lim.recommended_bytes == 11_453_000_000
    assert lim.budget_source == "device"


def test_a_tier_above_the_host_is_refused():
    # Review Focus 2: a 48 GB "cap" on a 32 GB Mac would set limits above RAM and look like a pass.
    with pytest.raises(DFloatUnsupportedError, match="48"):
        tier_limits(48, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)


def test_host_tier_rounds_the_ram_to_whole_gb():
    assert host_tier_gb(34_359_738_368) == 32
    assert host_tier_gb(17_179_869_184) == 16


def test_as_dict_carries_every_field_for_the_result_file():
    lim = tier_limits(24, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert set(lim.as_dict()) == {f.name for f in TierLimits.__dataclass_fields__.values()}


def test_apply_installs_the_three_limits_and_returns_the_previous_ones(monkeypatch):
    # Bug caught: apply() setting the memory limit but not the cache limit (MLX's default cache
    # limit equals the host's memory limit, so the pool could hold the whole overrun), or
    # returning the new values instead of the previous ones.
    calls: dict[str, int] = {}

    monkeypatch.setattr(
        capped.mx, "set_memory_limit", lambda b: (calls.__setitem__("memory", b), 11)[1]
    )
    monkeypatch.setattr(
        capped.mx, "set_cache_limit", lambda b: (calls.__setitem__("cache", b), 22)[1]
    )
    monkeypatch.setattr(
        capped.mx, "set_wired_limit", lambda b: (calls.__setitem__("wired", b), 33)[1]
    )
    lim = tier_limits(24, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    previous = apply(lim)
    assert calls == {"memory": lim.memory_limit_bytes, "cache": lim.cache_limit_bytes, "wired": 0}
    assert previous == {"memory": 11, "cache": 22, "wired": 33}


def test_recommended_ratio_is_two_thirds_up_to_24_and_three_quarters_above():
    # Bug caught: the boundary moved (< 24), so the 24 GB tier got 0.75 instead of 2/3; or the
    # displayed ratio and the integer arithmetic tier_limits uses disagree.
    assert recommended_ratio(24) == 2 / 3
    assert recommended_ratio(32) == 0.75
    big_host = {"host_ram_bytes": 64 * GIB, "host_recommended_bytes": 48 * GIB}
    assert tier_limits(24, **big_host).recommended_bytes == 16 * GIB  # 2/3 of 24 GiB
    assert tier_limits(32, **big_host).recommended_bytes == 24 * GIB  # 3/4 of 32 GiB
    for tier in (16, 24, 32, 48):
        lim = tier_limits(tier, **big_host)
        assert lim.budget_source == "ratio"
        assert lim.recommended_bytes == int(tier * GIB * recommended_ratio(tier))


def test_limits_record_carries_the_tier_the_effective_values_and_the_path():
    # Bug caught: the effective block dropped, or "applied" not passed through, so a result file could
    # not say which limits were really in force.
    lim = tier_limits(24, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    rec = limits_record(
        lim,
        effective_memory_limit=1,
        effective_cache_limit=2,
        effective_wired_limit=3,
        applied="tier-defaults",
    )
    assert rec == {
        "tier": lim.as_dict(),
        "effective": {"memory_limit_bytes": 1, "cache_limit_bytes": 2, "wired_limit_bytes": 3},
        "applied": "tier-defaults",
    }


def test_current_limits_reads_each_limit_with_a_swap_and_restore(monkeypatch):
    # Bug caught: the limit read but not restored (left at 0), or the restore skipped for one limit.
    calls: dict[str, list[int]] = {"memory": [], "cache": [], "wired": []}

    def setter(name, value):
        def _set(b):
            calls[name].append(b)
            return value

        return _set

    monkeypatch.setattr(capped.mx, "set_memory_limit", setter("memory", 111))
    monkeypatch.setattr(capped.mx, "set_cache_limit", setter("cache", 222))
    monkeypatch.setattr(capped.mx, "set_wired_limit", setter("wired", 333))
    assert current_limits() == {"memory": 111, "cache": 222, "wired": 333}
    assert calls == {"memory": [0, 111], "cache": [0, 222], "wired": [0, 333]}
