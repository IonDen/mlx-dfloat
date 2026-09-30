"""Scenario files: every field validated, unknown keys refused, the hash stable and content-bound."""

import dataclasses
from pathlib import Path

import pytest

from mlx_dfloat.bench.scenario import (
    CONDITIONS,
    MODELS,
    Scenario,
    load_scenario,
    scenario_from_mapping,
    scenario_hash,
)
from mlx_dfloat.errors import DFloatFormatError

REPO = Path(__file__).resolve().parents[1]
SHA = "51a428b928197e0531cb93d6e438941e2d0b247e"
GOOD = {
    "name": "flux1-schnell-1024",
    "model": "schnell",
    "df11_repo": "DFloat11/FLUX.1-schnell-DF11",
    "df11_revision": SHA,
    "base_repo": "black-forest-labs/FLUX.1-schnell",
    "base_revision": "741f7c3ce8b383c54771c7003378a50191e9efe9",
    "prompt": "a lighthouse",
    "seed": 42,
    "steps": 5,
    "warmup": 2,
    "size": 1024,
    "rounds": 3,
    "cache_limit_bytes": 2_500_000_000,
    "conditions": ["df11", "control", "q8"],
    "wall_budget_s": 3600,
}


def test_a_complete_mapping_builds_a_frozen_scenario_with_tuples():
    # Bug caught: lists kept as lists (unhashable, mutable).
    s = scenario_from_mapping(GOOD)
    assert s.conditions == ("df11", "control", "q8")
    assert s.wall_budget_s == 3600.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("warmup_steps", 2, "warmup_steps"),  # unknown key (Review Focus 1)
        ("model", "flux2", "model"),
        ("model", "krea-dev", "model"),
        ("df11_revision", "main", "df11_revision"),  # not a 40-hex SHA
        ("base_revision", "main", "base_revision"),
        ("size", 1000, "size"),  # not a multiple of 16
        ("size", 0, "size"),
        ("steps", 0, "steps"),
        ("warmup", 0, "warmup"),
        ("rounds", 0, "rounds"),
        ("cache_limit_bytes", 0, "cache_limit_bytes"),
        ("wall_budget_s", 0, "wall_budget_s"),
        ("conditions", ["df11", "df11"], "conditions"),  # duplicate
        ("conditions", ["bf16"], "conditions"),  # unknown condition
        ("conditions", [], "conditions"),
        ("seed", "42", "seed"),  # wrong type
        ("seed", True, "seed"),  # bool is not an int here
    ],
)
def test_every_invalid_field_is_refused_by_name(field, value, fragment):
    # Bug caught: a validator that checks types but not ranges (size 1000 accepted), or one that
    # ignores unknown keys, or one that raises a bare ValueError without the field name.
    data = {**GOOD, field: value}
    with pytest.raises(DFloatFormatError, match=fragment):
        scenario_from_mapping(data)


def test_a_missing_required_field_is_refused_by_name():
    data = dict(GOOD)
    del data["prompt"]
    with pytest.raises(DFloatFormatError, match="prompt"):
        scenario_from_mapping(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", "another"),
        ("cache_limit_bytes", 1),
        ("base_revision", "0" * 40),
        ("conditions", ["df11"]),
        ("steps", 6),
        ("size", 512),
        ("seed", 43),
        ("name", "x"),
        ("wall_budget_s", 1),
    ],
)
def test_the_hash_changes_with_every_field(field, value):
    # Bug caught: a field left out of the hashed dict (its change would not invalidate a result).
    a = scenario_from_mapping(GOOD)
    b = scenario_from_mapping({**GOOD, field: value})
    assert scenario_hash(a) != scenario_hash(b)
    assert len(scenario_hash(a)) == 64


def test_the_hash_does_not_depend_on_key_order():
    a = scenario_from_mapping(GOOD)
    b = scenario_from_mapping(dict(reversed(list(GOOD.items()))))
    assert scenario_hash(a) == scenario_hash(b)


def _toml(data):
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k} = [" + ", ".join(f'"{x}"' for x in v) + "]")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def test_load_scenario_reads_toml_and_names_the_file_on_a_bad_field(tmp_path):
    good = tmp_path / "s.toml"
    good.write_text(_toml(GOOD))
    assert load_scenario(good).name == "flux1-schnell-1024"
    bad = tmp_path / "bad.toml"
    bad.write_text(good.read_text().replace("size = 1024", "size = 1000"))
    with pytest.raises(DFloatFormatError, match=r"bad\.toml"):
        load_scenario(bad)
    broken = tmp_path / "broken.toml"
    broken.write_text("name = [unterminated\n")
    with pytest.raises(DFloatFormatError, match=r"broken\.toml"):
        load_scenario(broken)


@pytest.mark.parametrize("name", ["flux1-schnell-1024", "flux1-dev-1024"])
def test_the_committed_scenarios_load_and_pin_full_shas(name):
    # Bug caught: a committed scenario drifting from the validator (e.g. a branch name in a
    # revision field), which the bench would only discover at launch.
    s = load_scenario(REPO / "bench" / "scenarios" / f"{name}.toml")
    assert s.name == name
    assert len(s.df11_revision) == 40
    assert len(s.base_revision) == 40
    assert set(s.conditions) <= set(CONDITIONS)
    assert s.model in MODELS
    assert "q8" in s.conditions
    assert isinstance(s, Scenario)
