"""A bench scenario: the pinned recipe a result file is keyed on.

The file is TOML. Every field is validated here with the field name in the error, and unknown keys
are refused, so a mis-typed key can never run the bench on a default while the result claims the
scenario. ``scenario_hash`` is the sha256 of the canonical JSON of the fields, so two files with the
same content hash the same whatever their key order or spacing.
"""

import dataclasses
import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

from mlx_dfloat.errors import DFloatFormatError

CONDITIONS: tuple[str, ...] = (
    "df11",
    "control",
    "df11-depth2",
    "control-depth2",
    "control-noeval",
    "q8",
)
MODELS: tuple[str, ...] = ("schnell", "dev")
_SHA = re.compile(r"^[0-9a-f]{40}$")
_STR_FIELDS = (
    "name",
    "model",
    "df11_repo",
    "df11_revision",
    "base_repo",
    "base_revision",
    "prompt",
)
_INT_FIELDS = ("seed", "steps", "warmup", "size", "rounds", "cache_limit_bytes")
_REQUIRED = (*_STR_FIELDS, *_INT_FIELDS, "conditions", "wall_budget_s")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Scenario:
    """One pinned bench recipe (see the module docstring)."""

    name: str
    model: str
    df11_repo: str
    df11_revision: str
    base_repo: str
    base_revision: str
    prompt: str
    seed: int
    steps: int
    warmup: int
    size: int
    rounds: int
    cache_limit_bytes: int
    conditions: tuple[str, ...]
    wall_budget_s: float


def _fail(source: str, field: str, why: str) -> DFloatFormatError:
    return DFloatFormatError(f"{source}: {field}: {why}")


def _str(data: Mapping[str, object], field: str, source: str) -> str:
    value = data[field]
    if not isinstance(value, str):
        raise _fail(source, field, f"expected a string, got {type(value).__name__}")
    return value


def _int(data: Mapping[str, object], field: str, source: str) -> int:
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(source, field, f"expected an integer, got {type(value).__name__}")
    return value


def _float(data: Mapping[str, object], field: str, source: str) -> float:
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, field, f"expected a number, got {type(value).__name__}")
    return float(value)


def scenario_from_mapping(data: Mapping[str, object], *, source: str = "<mapping>") -> Scenario:
    """Validate a parsed mapping into a ``Scenario``.

    Raises:
        DFloatFormatError: An unknown key, a missing field, a wrong type, a value out of range, a
            revision that is not a full SHA, a duplicate, unknown or empty condition list.
    """
    unknown = sorted(set(data) - set(_REQUIRED))
    if unknown:
        raise _fail(source, unknown[0], "unknown key")
    for field in _REQUIRED:
        if field not in data:
            raise _fail(source, field, "missing")
    strings = {f: _str(data, f, source) for f in _STR_FIELDS}
    ints = {f: _int(data, f, source) for f in _INT_FIELDS}
    if strings["model"] not in MODELS:
        raise _fail(source, "model", f"choose from {MODELS}")
    for field in ("df11_revision", "base_revision"):
        if not _SHA.match(strings[field]):
            raise _fail(source, field, "must be a full 40-hex commit SHA")
    for field in ("steps", "warmup", "rounds", "cache_limit_bytes"):
        if ints[field] < 1:
            raise _fail(source, field, "must be >= 1")
    if ints["size"] < 16 or ints["size"] % 16:
        raise _fail(source, "size", "must be a positive multiple of 16")
    wall = _float(data, "wall_budget_s", source)
    if wall <= 0:
        raise _fail(source, "wall_budget_s", "must be positive")
    raw = data["conditions"]
    if not isinstance(raw, list) or not all(isinstance(c, str) for c in raw):
        raise _fail(source, "conditions", "expected a list of strings")
    conditions = tuple(raw)
    if not conditions:
        raise _fail(source, "conditions", "must not be empty")
    if len(set(conditions)) != len(conditions):
        raise _fail(source, "conditions", "duplicate entry")
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise _fail(source, "conditions", f"unknown {bad[0]!r}; choose from {CONDITIONS}")
    return Scenario(
        name=strings["name"],
        model=strings["model"],
        df11_repo=strings["df11_repo"],
        df11_revision=strings["df11_revision"],
        base_repo=strings["base_repo"],
        base_revision=strings["base_revision"],
        prompt=strings["prompt"],
        seed=ints["seed"],
        steps=ints["steps"],
        warmup=ints["warmup"],
        size=ints["size"],
        rounds=ints["rounds"],
        cache_limit_bytes=ints["cache_limit_bytes"],
        conditions=conditions,
        wall_budget_s=wall,
    )


def load_scenario(path: Path) -> Scenario:
    """Read and validate a scenario TOML file.

    Raises:
        DFloatFormatError: The file is not valid TOML or a field is invalid (the message names the file).
    """
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DFloatFormatError(f"{path}: cannot read the scenario: {exc}") from exc
    return scenario_from_mapping(data, source=str(path))


def scenario_hash(scenario: Scenario) -> str:
    """sha256 hex of the canonical JSON of the scenario's fields."""
    canonical = json.dumps(dataclasses.asdict(scenario), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


__all__ = [
    "CONDITIONS",
    "MODELS",
    "Scenario",
    "load_scenario",
    "scenario_from_mapping",
    "scenario_hash",
]
