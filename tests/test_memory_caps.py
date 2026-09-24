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


def test_install_memory_caps_swallows_set_limit_failure(monkeypatch):
    # A Metal-less device (e.g. CI) returns (0, 0) and never crashes.
    monkeypatch.setattr(
        _memory_caps.mx, "device_info", lambda: {"max_recommended_working_set_size": 25 * 1024**3}
    )

    def _boom(_limit):
        raise RuntimeError("metal unavailable")

    monkeypatch.setattr(_memory_caps.mx, "set_wired_limit", _boom)
    assert _memory_caps.install_memory_caps() == (0, 0)


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
