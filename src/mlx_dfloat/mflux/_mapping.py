"""Name maps derived from an mflux ``WeightMapping``: block-matrix tables, non-block renames, extra transforms."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.names import StaticNameMap, Transform

INDEX_TOKENS: tuple[str, ...] = ("{layer}", "{block}")


def _sources(target: Any) -> list[str]:
    src = target.from_pattern
    return list(src) if isinstance(src, list | tuple) else [src]


def _templated(pattern: str) -> bool:
    return any(token in pattern for token in INDEX_TOKENS)


def _concrete(pattern: str) -> str:
    for token in INDEX_TOKENS:
        pattern = pattern.replace(token, "0")
    return pattern


def name_map_from_mapping(
    targets: Iterable[Any], *, matrix_subs: Mapping[str, Sequence[str]]
) -> StaticNameMap:
    """The map for the block kinds in ``matrix_subs`` (each kind's compressed sub-paths, from the DF11 pattern_dict).

    Block matrices must be pure renames (one source, no transform). Non-block targets whose source differs from the
    target become renames; their transforms are kept under the target name. Every target of a covered block kind and
    every non-block target is then checked: the derived map must load its source where mflux loads it.

    Raises:
        DFloatIntegrationError: A compressed sub with no mflux target; a block target with a transform or several
            sources, or outside its block list; one source mapped to two targets; or a derived placement that
            differs from mflux's.
    """
    targets = list(targets)
    by_source: dict[str, Any] = {}
    for target in targets:
        for source in _sources(target):
            seen = by_source.get(source)
            if seen is not None and seen.to_pattern != target.to_pattern:
                raise DFloatIntegrationError(
                    f"{source}: mflux maps it to two targets ({seen.to_pattern}, {target.to_pattern})"
                )
            by_source[source] = target
    tables: dict[str, dict[str, str]] = {}
    for kind, subs in matrix_subs.items():
        tables[kind] = {}
        for sub in subs:
            keys = [f"{kind}.{token}.{sub}.weight" for token in INDEX_TOKENS]
            target = next((by_source[k] for k in keys if k in by_source), None)
            if target is None:
                raise DFloatIntegrationError(
                    f"{kind}.{sub}: a compressed matrix with no mflux target"
                )
            if target.transform is not None or len(_sources(target)) != 1:
                raise DFloatIntegrationError(
                    f"{target.to_pattern}: a block matrix with a transform or several sources; not a pure rename"
                )
            token = next(
                (t for t in INDEX_TOKENS if target.to_pattern.startswith(f"{kind}.{t}.")), None
            )
            if token is None:
                raise DFloatIntegrationError(
                    f"{target.to_pattern}: mflux loads the compressed {kind}.{sub} outside the {kind} block list"
                )
            tables[kind][sub] = target.to_pattern.removeprefix(f"{kind}.{token}.").removesuffix(
                ".weight"
            )
    renames: dict[str, str] = {}
    transforms: dict[str, Transform] = {}
    for target in targets:
        for source in _sources(target):
            if _templated(source) or _templated(target.to_pattern):
                if target.transform is not None and source.split(".", 1)[0] in matrix_subs:
                    raise DFloatIntegrationError(
                        f"{target.to_pattern}: a transform on a block extra"
                    )
                continue
            if source != target.to_pattern:
                renames[source] = target.to_pattern
            if target.transform is not None:
                transforms[target.to_pattern] = target.transform
    name_map = StaticNameMap(tables, renames=renames, transforms=transforms)
    for target in targets:
        for source in _sources(target):
            if _templated(source) and source.split(".", 1)[0] not in matrix_subs:
                continue  # a block list this map does not cover
            got = name_map.param_name(_concrete(source))
            if got != _concrete(target.to_pattern):
                raise DFloatIntegrationError(
                    f"mflux maps {source} to {target.to_pattern}; this map would load it into {got}"
                )
    return name_map
