"""FLUX.1 naming: the map derived from mflux's own weight mapping, and the group-kind check."""

from typing import Any

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.integrate.names import StaticNameMap

DOUBLE_PREFIX = "transformer_blocks"
SINGLE_PREFIX = "single_transformer_blocks"
MATRICES_PER_KIND = {DOUBLE_PREFIX: 14, SINGLE_PREFIX: 6}
# mflux 0.20 builds `norm_out.linear` with bias=False and drops this bias through update(strict=False);
# the seam follows mflux so its output equals mflux's.
DROPPED_EXTRAS: frozenset[str] = frozenset({"norm_out.linear.bias"})


def flux_name_map() -> StaticNameMap:
    """The block-matrix map read from mflux's ``FluxWeightMapping`` (a pure rename; imports mflux).

    Raises:
        DFloatIntegrationError: A block matrix target has several source patterns or a transform, which this
            adapter cannot express.
    """
    from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

    tables: dict[str, dict[str, str]] = {DOUBLE_PREFIX: {}, SINGLE_PREFIX: {}}
    for target in FluxWeightMapping.get_transformer_mapping():
        to = target.to_pattern
        kind = next((k for k in tables if to.startswith(f"{k}.{{block}}.")), None)
        if kind is None or not to.endswith(".weight"):
            continue
        sources = (
            target.from_pattern if isinstance(target.from_pattern, list) else [target.from_pattern]
        )
        if len(sources) != 1 or getattr(target, "transform", None) is not None:
            raise DFloatIntegrationError(
                f"{to}: mflux maps it from {sources} with a transform; not a pure rename"
            )
        prefix = f"{kind}.{{block}}."
        sub = sources[0].removeprefix(prefix).removesuffix(".weight")
        attr = to.removeprefix(prefix).removesuffix(".weight")
        tables[kind][sub] = attr
    keep = {k: {s: a for s, a in v.items() if _is_matrix(k, s)} for k, v in tables.items()}
    return StaticNameMap(keep)


def _is_matrix(kind: str, sub: str) -> bool:
    """Whether ``sub`` (a checkpoint sub-path of ``kind``) is a DF11 matrix rather than a norm scale."""
    del kind  # kept for a readable call site; every FLUX kind uses the same norm-scale rule
    # Norm weights (`attn.norm_q`, `norm_k`, `norm_added_q`, ...) are RMSNorm scales, not DF11 matrices.
    return not any(part.startswith("norm_") for part in sub.split("."))


def check_flux_groups(ckpt: Any) -> tuple[int, int]:
    """Count the double and single blocks; the groups must be exactly those two contiguous families.

    Raises:
        DFloatFormatError: A group outside the two kinds, a hole in an index sequence, a missing kind, or a
            group with the wrong number of matrices.
    """
    seen: dict[str, set[int]] = {DOUBLE_PREFIX: set(), SINGLE_PREFIX: set()}
    for name, group in ckpt.groups.items():
        kind, _dot, idx = name.partition(".")
        if kind not in seen or not idx.isdigit():
            raise DFloatFormatError(f"{name}: not a FLUX block group")
        n = len(group.matrix_names)
        if n != MATRICES_PER_KIND[kind]:
            raise DFloatFormatError(
                f"{name}: {n} matrices, a {kind} group has {MATRICES_PER_KIND[kind]}"
            )
        seen[kind].add(int(idx))
    for kind, indices in seen.items():
        if not indices:
            raise DFloatFormatError(f"no {kind} groups in the checkpoint")
        if indices != set(range(len(indices))):
            raise DFloatFormatError(f"{kind} groups are not contiguous: {sorted(indices)}")
    return len(seen[DOUBLE_PREFIX]), len(seen[SINGLE_PREFIX])
