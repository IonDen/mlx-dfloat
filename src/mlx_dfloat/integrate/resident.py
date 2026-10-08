"""Non-block DF11 groups (a matrix outside every block, like an embedder): decoded once when the set loads.

The decoded weight stays installed while the set is resident; ``clear_nonblock`` puts the placeholder back when the
set drops. The status word is read at once (one host sync per group, at load time), unlike a block's.
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import mlx.core as mx

from mlx_dfloat.decode import DecodeResult, check_status, decode_group, split_matrices
from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.format import MxGroup
from mlx_dfloat.integrate.names import NameMap
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, get_attr_path

NonBlockShapes = dict[str, tuple[int, ...]]


# Called through this name so a test can count the evaluations.
_eval = mx.eval


def _owner(module: Any, param: str) -> Any:
    if not param.endswith(".weight"):
        raise DFloatIntegrationError(f"{param}: a compressed matrix must land on a .weight")
    return get_attr_path(module, param.removesuffix(".weight"))


def install_nonblock_placeholders(
    module: Any, matrix_names: Iterable[str], name_map: NameMap
) -> NonBlockShapes:
    """Put placeholders in place of the non-block matrices; return parameter -> shape.

    Raises:
        DFloatIntegrationError: A matrix whose parameter is not a matrix module's weight, or one with a transform.
    """
    shapes: NonBlockShapes = {}
    for matrix in matrix_names:
        param = name_map.param_name(matrix)
        owner = _owner(module, param)
        if not name_map.is_matrix_module(owner):
            raise DFloatIntegrationError(f"{param}: not a matrix module ({type(owner).__name__})")
        if name_map.transform_of(param) is not None:
            raise DFloatIntegrationError(f"{param}: a transform on a compressed matrix")
        shapes[param] = tuple(int(d) for d in owner.weight.shape)
        owner.weight = PLACEHOLDER
    return shapes


def decode_nonblock(
    groups: Mapping[str, MxGroup],
    matrix_names: Mapping[str, Sequence[str]],
    shapes: NonBlockShapes,
    name_map: NameMap,
    *,
    decode: Callable[[MxGroup], DecodeResult] | None = None,
    eval_together: bool = False,
) -> dict[str, mx.array]:
    """Decode each non-block group (Metal by default), check its status, cut it into its parameters' shapes.

    Each group is evaluated and its status checked before the next is queued; with ``eval_together`` every group is
    queued first and evaluated in one ``mx.eval`` (a per-call decode pays one synchronisation, not one per group), then
    the status words are checked in group order.

    Raises:
        DFloatIntegrationError: A group is not resident, or its matrices do not fit their parameters.
        DFloatFormatError: A group's decode reported an error (the message names the group).
    """
    run = decode if decode is not None else partial(decode_group, backend="metal")
    out: dict[str, mx.array] = {}
    queued: list[tuple[str, mx.array, list[mx.array]]] = []
    for group_name, names in matrix_names.items():
        group = groups.get(group_name)
        if group is None:
            raise DFloatIntegrationError(f"{group_name}: no resident DF11 group")
        result = run(group)
        parts = split_matrices(result.bits, group.split_positions)
        if len(parts) != len(names):
            raise DFloatIntegrationError(
                f"{group_name}: {len(parts)} matrices decoded for {len(names)} names"
            )
        mine: list[mx.array] = []
        for matrix, part in zip(names, parts, strict=True):
            param = name_map.param_name(matrix)
            shape = shapes.get(param)
            if shape is None or part.size != math.prod(shape):
                raise DFloatIntegrationError(
                    f"{matrix}: decoded {part.size} elements for {param} {shape}"
                )
            out[param] = part.view(mx.bfloat16).reshape(shape)
            mine.append(out[param])
        if eval_together:
            queued.append((group_name, result.status, mine))
            continue
        _eval(result.status, *mine)
        check_status(result.status, name=group_name)
    if queued:
        _eval(*(a for _name, status, mine in queued for a in (status, *mine)))
        for group_name, status, _mine in queued:
            check_status(status, name=group_name)
    return out


def install_nonblock(module: Any, weights: Mapping[str, mx.array]) -> None:
    """Assign the decoded non-block weights."""
    for param, weight in weights.items():
        _owner(module, param).weight = weight


def clear_nonblock(module: Any, shapes: NonBlockShapes) -> None:
    """Put the placeholders back (the set is dropping)."""
    for param in shapes:
        _owner(module, param).weight = PLACEHOLDER
