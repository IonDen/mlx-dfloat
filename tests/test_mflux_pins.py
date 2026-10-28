"""The mflux pin: the ``[mflux]`` extra and the ``bench`` group name the same checked range and cap Pillow with it."""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
MFLUX_PIN = "mflux>=0.23,<0.24; sys_platform == 'darwin'"
PILLOW_CAP = "pillow<13"


def _requirement_lists() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return {
        "the [mflux] extra": data["project"]["optional-dependencies"]["mflux"],
        "the bench group": data["dependency-groups"]["bench"],
    }


@pytest.mark.parametrize("where", ["the [mflux] extra", "the bench group"])
def test_each_mflux_requirement_list_pins_the_checked_range(where):
    # Bug caught: one list left on the old range (the lock then resolves an mflux the adapters were not
    # checked against), or the range widened past the release the seams were read against.
    requirements = _requirement_lists()[where]
    assert [r for r in requirements if r.startswith("mflux")] == [MFLUX_PIN]


@pytest.mark.parametrize("where", ["the [mflux] extra", "the bench group"])
def test_each_mflux_requirement_list_caps_pillow_below_13(where):
    # Bug caught: the cap dropped from one list; mflux 0.23.0 imports a private Pillow module that
    # Pillow 13 removed, so a fresh install fails at `import mflux`.
    requirements = _requirement_lists()[where]
    assert [r for r in requirements if r.lower().startswith("pillow")] == [PILLOW_CAP]
