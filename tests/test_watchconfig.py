import dataclasses
from pathlib import Path

import pytest

from mlx_dfloat.errors import DFloatUnsupportedError
from mlx_dfloat.mflux._watchconfig import DEFAULT_WATCHDOG, WatchdogConfig, check_watchdog_arg


def test_defaults():
    # Bug caught: a default changed from the documented one (a 1 GiB stop margin, a 10 s grace, no ceiling, no
    # wall budget, no artifact folder), e.g. stop_margin written as 1_000_000_000 or grace as 1.0.
    config = WatchdogConfig()
    assert config.ceiling is None
    assert config.stop_margin == 1_073_741_824
    assert config.grace == 10.0
    assert config.wall_budget is None
    assert config.artifact_dir is None


def test_the_default_watchdog_is_the_default_config():
    # Bug caught: DEFAULT_WATCHDOG built with non-default arguments, so the constructor default differs from
    # WatchdogConfig().
    assert WatchdogConfig() == DEFAULT_WATCHDOG


def test_the_config_has_exactly_the_documented_fields():
    # Bug caught: a field added, dropped or renamed.
    names = [f.name for f in dataclasses.fields(WatchdogConfig)]
    assert names == ["ceiling", "stop_margin", "grace", "wall_budget", "artifact_dir"]


def test_the_config_is_frozen():
    # Bug caught: a mutable config, so one model reassigning a field changes the shared DEFAULT_WATCHDOG for
    # every other model.
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_WATCHDOG.grace = 1.0  # type: ignore[misc]


def test_the_config_is_keyword_only():
    # Bug caught: kw_only dropped, so WatchdogConfig(5) silently sets the ceiling to 5 bytes.
    with pytest.raises(TypeError):
        WatchdogConfig(5)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"ceiling": 0}, "ceiling"),
        ({"stop_margin": -1}, "stop_margin"),
        ({"stop_margin": 0}, "stop_margin"),
        ({"ceiling": 10, "stop_margin": 10}, "stop_margin"),
        ({"grace": 0}, "grace"),
        ({"wall_budget": 0}, "wall_budget"),
    ],
)
def test_each_invalid_value_is_refused_by_name(kwargs, field):
    # Bug caught: a missing check (the value is accepted), a check off by one (`<` for `<=`), or an error that
    # names a different field than the one at fault.
    with pytest.raises(DFloatUnsupportedError, match=rf"^WatchdogConfig\.{field} "):
        WatchdogConfig(**kwargs)


def test_a_margin_one_below_the_ceiling_passes():
    # Bug caught: `stop_margin > ceiling` flipped to `>=` one step further (`stop_margin + 1 >= ceiling`), refusing
    # the boundary that leaves one byte of room.
    config = WatchdogConfig(ceiling=10, stop_margin=9)
    assert config.ceiling == 10
    assert config.stop_margin == 9


def test_the_smallest_valid_values_pass():
    # Bug caught: `stop_margin <= 0` written as `<= 1`, or `ceiling <= 0` as `<= 2`, refusing the lowest valid value.
    config = WatchdogConfig(ceiling=2, stop_margin=1, grace=0.001, wall_budget=0.001)
    assert config.ceiling == 2
    assert config.stop_margin == 1


def test_check_watchdog_arg_passes_none_and_a_config_through_unchanged():
    # Bug caught: the check replaces None with the default config, or copies the caller's config, so a model
    # built with watchdog=None is armed anyway or the identity of the given config is lost.
    config = WatchdogConfig(grace=2.0)
    assert check_watchdog_arg(None) is None
    assert check_watchdog_arg(config) is config


@pytest.mark.parametrize("value", [False, True, {"grace": 1.0}, 0, "off"])
def test_check_watchdog_arg_refuses_anything_else_naming_none(value):
    # Bug caught: a falsy check (`if not value: return None`) lets watchdog=False through as "no watchdog", or a
    # truthy one lets True or a dict through; the message must point at watchdog=None as the way to turn it off.
    with pytest.raises(DFloatUnsupportedError, match=r"watchdog=None"):
        check_watchdog_arg(value)


class _Folder:
    """A path-like that is not a ``Path``."""

    def __init__(self, path):
        self._path = path

    def __fspath__(self):
        return self._path


@pytest.mark.parametrize("folder", ["/x/records", _Folder("/x/records")])
def test_a_str_or_path_like_folder_becomes_a_path(folder):
    # Bug caught: a str folder stored as given, so the watchdog's `out_dir / name` join raises TypeError at the first
    # trip and the call runs on unwatched.
    config = WatchdogConfig(artifact_dir=folder)
    assert type(config.artifact_dir) is not str
    assert config.artifact_dir == Path("/x/records")


@pytest.mark.parametrize("folder", [5, b"/x/records", _Folder(b"/x/records")])
def test_a_folder_that_is_not_a_str_or_path_is_refused(folder):
    # Bug caught: any value accepted as the folder (an int, bytes), failing later inside the sampler instead of at
    # construction; or bytes from __fspath__ turned into a Path of the repr.
    with pytest.raises(DFloatUnsupportedError, match=r"^WatchdogConfig\.artifact_dir "):
        WatchdogConfig(artifact_dir=folder)
