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
    # By hand: 26_800_000_000 - 2_147_483_648 = 24_652_516_352, both for the ceiling and for the fit
    # budget (the host's own budget_bytes()).
    assert lim.ceiling_bytes == 24_652_516_352
    assert lim.fit_budget_bytes == 24_652_516_352
    # MLX's defaults by hand: memory min(int(1.5 x 26.8e9) = 40_200_000_000,
    # int(0.95 x 34_359_738_368) = 32_641_751_449); cache min(that, int(0.95 x 26.8e9) = 25_460_000_000).
    assert lim.memory_limit_bytes == 32_641_751_449
    assert lim.cache_limit_bytes == 25_460_000_000
    assert lim.wired_limit_bytes == 0


def test_a_24_gb_tier_gets_the_caps_mlx_dfloat_installs_on_a_24_gb_mac():
    # Bug caught: a CAPPED tier left at MLX's own defaults (wired 0, memory 1.5 x rec), which a
    # user's generate never runs under: it installs wired/memory caps from the working set, and
    # the wired cap alone moved a Klein VAE peak from 9.49 to 6.89 GiB.
    # By hand: rec = 24 GiB x 2/3 = 16 GiB = 17_179_869_184 B -> 16 GiB floor -> wired
    # min(20, 16 - 2) = 14 GiB, memory min(22, max(15, 16)) = 16 GiB; cache = min(16 GiB,
    # int(0.95 x 17_179_869_184) = 16_320_875_724) = 16_320_875_724; ceiling = 16 - 1.5 = 14.5 GiB.
    lim = tier_limits(24, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == 17_179_869_184
    assert lim.wired_limit_bytes == 15_032_385_536
    assert lim.memory_limit_bytes == 17_179_869_184
    assert lim.cache_limit_bytes == 16_320_875_724
    assert lim.reserve_bytes == 1_610_612_736
    assert lim.budget_bytes == 17_179_869_184
    assert lim.ceiling_bytes == 15_569_256_448
    # The fit budget is a real 24 GB Mac's: 17_179_869_184 - 2 GiB, not the watchdog's 1.5 GiB reserve.
    assert lim.fit_budget_bytes == 15_032_385_536
    assert lim.budget_source == "ratio"
    assert lim.label == "CAPPED"
    assert lim.is_host is False


def test_a_16_gb_tier_gets_the_caps_mlx_dfloat_installs_on_a_16_gb_mac():
    # By hand: rec = 16 GiB x 2 // 3 = 11_453_246_122 B (10.67 GiB) -> 10 GiB floor -> wired
    # min(20, 10 - 2) = 8 GiB, memory min(22, max(9, 10)) = 10 GiB; cache = min(10_737_418_240,
    # int(0.95 x 11_453_246_122) = 10_880_583_815) = 10_737_418_240 (the memory cap binds);
    # ceiling = 11_453_246_122 - 1_610_612_736 = 9_842_633_386.
    lim = tier_limits(16, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == 11_453_246_122
    assert lim.wired_limit_bytes == 8_589_934_592
    assert lim.memory_limit_bytes == 10_737_418_240
    assert lim.cache_limit_bytes == 10_737_418_240
    assert lim.reserve_bytes == 1_610_612_736
    assert lim.budget_bytes == 11_453_246_122
    assert lim.ceiling_bytes == 9_842_633_386
    # The fit budget a real 16 GB Mac's generate applies: 11_453_246_122 - 2_147_483_648.
    assert lim.fit_budget_bytes == 9_305_762_474


def test_a_tiers_fit_budget_is_what_budget_bytes_gives_on_a_mac_of_that_size(monkeypatch):
    # Bug caught: the tier's fit budget drifting from the rule generate applies on a real Mac (a
    # reserve changed in one place only), so a CAPPED row passes a fit check that Mac would refuse.
    import mlx_dfloat.integrate.memory as memory

    for tier in (16, 24):
        lim = tier_limits(tier, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
        monkeypatch.setattr(
            memory.mx,
            "device_info",
            lambda rec=lim.recommended_bytes: {"max_recommended_working_set_size": rec},
        )
        assert lim.fit_budget_bytes == memory.budget_bytes()


def test_a_tier_whose_mac_gets_no_caps_keeps_mlx_defaults():
    # Bug caught: a working set under 1 GiB (install_memory_caps installs nothing there) turned
    # into a 0-byte memory limit instead of the MLX defaults that Mac would really run under.
    # By hand: rec = 1 GiB x 2 // 3 = 715_827_882 -> 0 GiB floor -> no caps; memory =
    # min(int(1.5 x 715_827_882) = 1_073_741_823, int(0.95 x 1 GiB) = 1_020_054_732);
    # cache = min(1_020_054_732, int(0.95 x 715_827_882) = 680_036_487).
    lim = tier_limits(1, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    assert lim.recommended_bytes == 715_827_882
    assert lim.wired_limit_bytes == 0
    assert lim.memory_limit_bytes == 1_020_054_732
    assert lim.cache_limit_bytes == 680_036_487


def test_a_48_gb_tier_on_a_64_gb_host_uses_the_three_quarters_ratio():
    # The 0.75 branch: rec = 48 x 3 // 4 = 36 GiB; reserve 2 GiB; CAPPED.
    # Bug caught: the caps above 24 GB not clamped to mlx-dfloat's 20 GiB wired / 22 GiB memory
    # ceiling (wired 34 GiB, memory 36 GiB), or the cache left at 0.95 x rec above the memory cap.
    # By hand: wired min(20, 36 - 2) = 20 GiB; memory min(22, max(20 + 1, 36)) = 22 GiB; cache
    # min(22 GiB, int(0.95 x 38_654_705_664) = 36_721_970_380) = 22 GiB.
    lim = tier_limits(48, host_ram_bytes=64 * GIB, host_recommended_bytes=int(0.78 * 64 * GIB))
    assert lim.recommended_bytes == 38_654_705_664
    assert lim.reserve_bytes == 2 * GIB
    assert lim.label == "CAPPED"
    assert lim.wired_limit_bytes == 21_474_836_480
    assert lim.memory_limit_bytes == 23_622_320_128
    assert lim.cache_limit_bytes == 23_622_320_128


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
    # The caps follow the device's own working set: 11_453_000_000 B -> 10 GiB -> (8, 10) GiB.
    assert lim.wired_limit_bytes == 8_589_934_592
    assert lim.memory_limit_bytes == 10_737_418_240


def test_a_tier_above_the_host_is_refused():
    # A 48 GB "cap" on a 32 GB Mac would set limits above RAM and look like a pass.
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
    from mlx_dfloat import _memory_caps

    known = _memory_caps.known_wired_limit(capped.mx)
    lim = tier_limits(16, host_ram_bytes=HOST_RAM, host_recommended_bytes=HOST_REC)
    try:
        previous = apply(lim)
        # The per-call caps read the wired limit from this record instead of probing it (P2): a tier's 8 GiB.
        assert _memory_caps.known_wired_limit(capped.mx) == 8_589_934_592
    finally:
        if known is not None:
            _memory_caps.remember_wired_limit(capped.mx, known)
    # The 16 GB tier's caps, worked by hand in test_a_16_gb_tier_gets_the_caps_...: wired 8 GiB,
    # memory 10 GiB, cache 10 GiB.
    assert calls == {
        "memory": 10_737_418_240,
        "cache": 10_737_418_240,
        "wired": 8_589_934_592,
    }
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
        applied="tier-caps",
    )
    assert rec == {
        "tier": lim.as_dict(),
        "effective": {"memory_limit_bytes": 1, "cache_limit_bytes": 2, "wired_limit_bytes": 3},
        "applied": "tier-caps",
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
