from mlx_dfloat import _memory_caps
from mlx_dfloat._memory_caps import device_string


def test_device_string_formats_name_and_memory(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx,
        "device_info",
        lambda: {"device_name": "Apple M1 Max", "memory_size": 34359738368},
    )
    assert device_string() == "Apple M1 Max, 32 GB"


def test_device_string_none_when_unreported(monkeypatch):
    monkeypatch.setattr(_memory_caps.mx, "device_info", dict)
    assert device_string() is None


def test_device_string_none_when_device_info_raises(monkeypatch):
    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(_memory_caps.mx, "device_info", _boom)
    assert device_string() is None


def test_device_string_name_only_when_memory_size_non_positive(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx,
        "device_info",
        lambda: {"device_name": "Apple M1 Max", "memory_size": 0},
    )
    assert device_string() == "Apple M1 Max"


def test_clamp_uses_desired_on_large_device():
    # 25 GB recommended -> desired (20, 22) fits unchanged
    assert _memory_caps._clamp_caps_gb(25) == (20, 22)


def test_clamp_shrinks_on_small_device():
    # 10 GB recommended -> wired = min(20, 10-2) = 8; memory = min(22, max(9, 10)) = 10
    assert _memory_caps._clamp_caps_gb(10) == (8, 10)


def test_zero_recommended_is_noop_signal():
    assert _memory_caps._clamp_caps_gb(0) == (0, 0)


def test_compute_safe_caps_handles_device_info_failure(monkeypatch):
    def _boom():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(_memory_caps.mx, "device_info", _boom)
    assert _memory_caps.compute_safe_caps_gb() == (0, 0)


def test_install_memory_caps_noop_when_no_working_set(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 0}
    )
    assert _memory_caps.install_memory_caps() == (0, 0)


def _boom(_limit):
    raise RuntimeError("metal unavailable")


def _healthy_device(monkeypatch):
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )


def test_install_memory_caps_swallows_set_limit_failure(monkeypatch):
    # A Metal-less device (e.g. CI) returns (0, 0) and never crashes.
    _healthy_device(monkeypatch)
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", _boom)
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", _boom)
    assert _memory_caps.install_memory_caps() == (0, 0)


def test_a_failed_wired_cap_still_installs_the_memory_cap(monkeypatch):
    # Bug caught: one try block around both calls skips set_memory_limit when set_wired_limit
    # raises, and reports (0, 0) although the memory cap could have been applied.
    _healthy_device(monkeypatch)
    seen: dict[str, int] = {}
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", _boom)
    monkeypatch.setattr(
        _memory_caps.mx, "set_memory_limit", lambda b: seen.__setitem__("memory", b)
    )
    assert _memory_caps.install_memory_caps() == (0, 22)
    assert seen == {"memory": 22 * 1024**3}


def test_a_failed_memory_cap_reports_only_the_wired_cap(monkeypatch):
    _healthy_device(monkeypatch)
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda _b: None)
    monkeypatch.setattr(_memory_caps.mx, "set_memory_limit", _boom)
    assert _memory_caps.install_memory_caps() == (20, 0)


def test_install_memory_caps_pushes_strict_byte_caps_on_healthy_device(monkeypatch):
    # Bug this catches: dropping the `* 1024**3` installs 20 *bytes* instead of 20 GB,
    # silently disabling the kernel-panic guard while still returning (20, 22).
    max_bytes = 25 * 1024**3
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": max_bytes}
    )
    seen: dict[str, int] = {}
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: seen.__setitem__("wired", b))
    monkeypatch.setattr(
        _memory_caps.mx, "set_memory_limit", lambda b: seen.__setitem__("memory", b)
    )

    assert _memory_caps.install_memory_caps() == (20, 22)
    assert seen["wired"] == 20 * 1024**3
    assert seen["memory"] == 22 * 1024**3
    assert seen["wired"] < max_bytes
    assert seen["memory"] < max_bytes


def test_caps_for_recommended_bytes_floors_to_whole_gib_like_install_memory_caps():
    # Bug caught: the helper rounding the working set up (or to nearest) instead of flooring as
    # compute_safe_caps_gb does, so a CAPPED tier would run under looser caps than that Mac gets.
    # By hand: 11_453_246_122 B = 10.67 GiB -> 10 -> wired min(20, 10 - 2) = 8 GiB, memory
    # min(22, max(9, 10)) = 10 GiB. 17_179_869_183 B is one byte under 16 GiB -> 15 -> (13, 15).
    assert _memory_caps.caps_for_recommended_bytes(11_453_246_122) == (
        8_589_934_592,
        10_737_418_240,
    )
    assert _memory_caps.caps_for_recommended_bytes(17_179_869_184) == (
        15_032_385_536,
        17_179_869_184,
    )
    assert _memory_caps.caps_for_recommended_bytes(17_179_869_183) == (
        13_958_643_712,
        16_106_127_360,
    )


def test_caps_for_recommended_bytes_below_one_gib_is_the_no_cap_signal():
    # Bug caught: a sub-GiB working set yielding a 1 GiB wired cap, where install_memory_caps on
    # that device installs nothing (it returns (0, 0) for a 0 GiB working set).
    assert _memory_caps.caps_for_recommended_bytes(1_073_741_823) == (0, 0)
    assert _memory_caps.caps_for_recommended_bytes(0) == (0, 0)


def test_install_memory_caps_on_a_16_gb_mac_matches_the_helper(monkeypatch):
    # Bug caught: install_memory_caps and caps_for_recommended_bytes drifting apart, so the CAPPED
    # emulation stops matching what generate installs on a real 16 GB Mac.
    monkeypatch.setattr(
        _memory_caps.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 11_453_246_122},
    )
    seen: dict[str, int] = {}
    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", lambda b: seen.__setitem__("wired", b))
    monkeypatch.setattr(
        _memory_caps.mx, "set_memory_limit", lambda b: seen.__setitem__("memory", b)
    )
    assert _memory_caps.install_memory_caps() == (8, 10)
    assert seen == {"wired": 8_589_934_592, "memory": 10_737_418_240}
