"""Zero-size placeholders in place of every block matrix, and the shapes they had."""

from collections.abc import Iterable, Sequence
from typing import Any

import mlx.core as mx

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.names import BlockShapes, NameMap, Shapes

PLACEHOLDER = mx.zeros((0,), dtype=mx.bfloat16)


def _is_index(part: str) -> bool:
    """An ASCII-digit component (``"0"``); ``"²".isdigit()`` is true too, but ``int("²")`` raises."""
    return part.isascii() and part.isdigit()


def get_attr_path(module: Any, path: str) -> Any:
    """Resolve a dotted path on a module; a digit component indexes a list (``attn.to_out.0``).

    Raises:
        DFloatIntegrationError: A component does not exist.
    """
    node = module
    for part in path.split("."):
        try:
            node = node[int(part)] if _is_index(part) else getattr(node, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise DFloatIntegrationError(f"{path!r}: no {part!r} on {type(node).__name__}") from exc
    return node


def set_attr_path(module: Any, path: str, value: Any) -> None:
    """Assign ``value`` at a dotted path (see ``get_attr_path``)."""
    head, _dot, leaf = path.rpartition(".")
    parent = get_attr_path(module, head) if head else module
    if _is_index(leaf):
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def install_placeholders(
    block_lists: Iterable[tuple[str, Sequence[Any]]], name_map: NameMap
) -> Shapes:
    """Replace every block matrix weight with ``PLACEHOLDER``; return block name -> attr -> shape.

    Every mapped attribute must be a matrix module of the block and every matrix module of the block must
    be mapped, so an added or renamed layer is caught here, not at the first matmul.

    Raises:
        DFloatIntegrationError: A block's matrix-module set differs from the map.
    """
    shapes: Shapes = {}
    for kind, blocks in block_lists:
        mapped = set(name_map.attrs_of(kind))
        for idx, block in enumerate(blocks):
            name = f"{kind}.{idx}"
            present = {p for p, m in block.named_modules() if name_map.is_matrix_module(m)}
            if present != mapped:
                raise DFloatIntegrationError(
                    f"{name}: mapped matrices missing from the block: {sorted(mapped - present)}; "
                    f"Linear layers the map does not cover: {sorted(present - mapped)}"
                )
            per: BlockShapes = {}
            for attr in name_map.attrs_of(kind):
                module = get_attr_path(block, attr)
                per[attr] = tuple(int(d) for d in module.weight.shape)
                module.weight = PLACEHOLDER
            shapes[name] = per
    return shapes
