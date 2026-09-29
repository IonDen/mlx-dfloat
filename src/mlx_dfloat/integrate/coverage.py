"""Coverage bookkeeping: the extras plan, block-extra filtering, the resident set, full-block decode."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import mlx.core as mx

from mlx_dfloat._safetensors import TensorInfo
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint, MxGroup, load_group_mx
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.providers import WeightProvider, read_bf16


def load_resident_set(
    ckpt: DF11Checkpoint, names: Iterable[str] | None = None
) -> dict[str, MxGroup]:
    """Load every group (or the named ones) as evaluated ``MxGroup``s: the resident set of a run.

    Raises:
        DFloatIntegrationError: A requested name is not a group of the checkpoint.
    """
    wanted = list(ckpt.groups) if names is None else list(names)
    missing = [n for n in wanted if n not in ckpt.groups]
    if missing:
        raise DFloatIntegrationError(f"not groups of the checkpoint: {missing}")
    return {name: load_group_mx(ckpt.groups[name]) for name in wanted}


def extras_plan(
    ckpt: DF11Checkpoint,
    name_map: NameMap,
    *,
    counts: Mapping[str, int],
    dropped: frozenset[str] = frozenset(),
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
) -> list[tuple[str, Path, TensorInfo]]:
    """The extras to load as (module parameter name, file, tensor info), sorted by checkpoint name.

    Block extras at an index at or beyond ``counts[kind]`` and names in ``dropped`` are left out.
    ``extras`` overrides the checkpoint's own extras (a BF16 base's index, for the reference side of
    an image identity check) and defaults to ``ckpt.extras``.

    Raises:
        DFloatFormatError: Two checkpoint names map to the same module parameter name.
    """
    plan: list[tuple[str, Path, TensorInfo]] = []
    source_of: dict[str, str] = {}
    source = ckpt.extras if extras is None else extras
    for name, (path, info) in sorted(source.items()):
        if name in dropped:
            continue
        block = _block_of(name, name_map)
        if block is not None:
            kind, idx = block
            if idx >= counts.get(kind, 0):
                continue
        param = name_map.param_name(name)
        if param in source_of:
            raise DFloatFormatError(
                f"extras {source_of[param]!r} and {name!r} both map to the parameter {param!r}"
            )
        source_of[param] = name
        plan.append((param, path, info))
    return plan


def _block_of(name: str, name_map: NameMap) -> tuple[str, int] | None:
    head, _dot, _rest = name.partition(".")
    if head not in name_map.kinds:
        return None
    parts = name.split(".")
    index = parts[1] if len(parts) > 2 else ""
    return (head, int(index)) if index.isascii() and index.isdigit() else None


def read_extra(path: Path, info: TensorInfo) -> mx.array:
    """Read one BF16 extra as a bf16 ``mx.array``.

    Raises:
        DFloatFormatError: The tensor is not BF16.
    """
    return read_bf16(path, info)


def check_extras_cover(
    params: Iterable[str], extras: Iterable[str], matrices: Iterable[str]
) -> None:
    """Every non-matrix parameter gets exactly one extra, and every planned extra has a parameter.

    Raises:
        DFloatIntegrationError: A parameter nobody sets, or an extra with no target.
    """
    param_set, extra_set, matrix_set = set(params), set(extras), set(matrices)
    missing = sorted(param_set - matrix_set - extra_set)
    unexpected = sorted(extra_set - param_set)
    if missing or unexpected:
        raise DFloatIntegrationError(
            f"extras do not cover the transformer: parameters without an extra {missing}; "
            f"extras without a parameter {unexpected}"
        )


def decode_resident(provider: WeightProvider, shapes: Shapes) -> dict[str, dict[str, mx.array]]:
    """Decode every block's group once into resident BF16 dicts, one block evaluated before the next.

    One lazy eval over every block would allocate every decode up front (the run-ahead the per-block
    eval policy exists to prevent), so each block's dict is evaluated as soon as it is cut. The
    deferred status words are checked at the end.

    Raises:
        DFloatFormatError: A block's decode reported an error (``WeightProvider.verify``).
        DFloatIntegrationError: A block is not resident, or a matrix's size does not match its shape.
    """
    per_block: dict[str, dict[str, mx.array]] = {}
    for name, per in shapes.items():
        weights = provider.weights_for(name, per)
        seam._eval(weights)
        per_block[name] = weights
    provider.verify()
    return per_block
