"""The mflux pin: the ``[mflux]`` extra and the ``bench`` group name the same floors, with no ceilings."""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
MFLUX_FLOOR = "mflux>=0.24; sys_platform == 'darwin'"
PILLOW_FLOOR = "pillow>=13"
WHERE = ["the [mflux] extra", "the bench group"]


def _requirement_lists() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return {
        "the [mflux] extra": data["project"]["optional-dependencies"]["mflux"],
        "the bench group": data["dependency-groups"]["bench"],
    }


@pytest.mark.parametrize("where", WHERE)
def test_each_mflux_requirement_list_names_the_floor(where):
    # Bug caught: one list left on `>=0.23,<0.24`, so the lock resolves an mflux the adapters were not
    # tested against.
    requirements = _requirement_lists()[where]
    assert [r for r in requirements if r.startswith("mflux")] == [MFLUX_FLOOR]


@pytest.mark.parametrize("where", WHERE)
def test_each_list_floors_pillow_at_13(where):
    # Bug caught: `pillow<13` left in one list, which makes the extra unresolvable beside mflux 0.24
    # (it needs `pillow>=13.0.0`).
    requirements = _requirement_lists()[where]
    assert [r for r in requirements if r.lower().startswith("pillow")] == [PILLOW_FLOOR]


@pytest.mark.parametrize("where", WHERE)
def test_no_mflux_or_pillow_requirement_has_an_upper_bound(where):
    # Bug caught: a ceiling re-added by hand, against the floor-only policy.
    requirements = _requirement_lists()[where]
    pinned = [r for r in requirements if r.lower().startswith(("mflux", "pillow"))]
    assert pinned, f"no mflux or pillow requirement in {where}"
    assert [r for r in pinned if "<" in r.split(";")[0]] == []


def test_the_runtime_floor_is_the_extras_floor():
    # Bug caught: the [mflux] floor raised in pyproject while require_mflux() still accepts the older release (or the
    # reverse), so an install the extra would refuse passes the entry check.
    from mlx_dfloat.mflux._versions import MIN_MFLUX, release_tuple

    floor = MFLUX_FLOOR.split(";")[0].removeprefix("mflux>=")
    assert release_tuple(MIN_MFLUX) == release_tuple(floor)
    assert MIN_MFLUX == "0.24.0"
