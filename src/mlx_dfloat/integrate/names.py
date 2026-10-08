"""Name maps: how a checkpoint's matrix names land on a module's attributes."""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn

from mlx_dfloat.errors import DFloatIntegrationError

Transform = Callable[[mx.array], mx.array]
BlockShapes = dict[str, tuple[int, ...]]
Shapes = dict[str, BlockShapes]


@dataclass(frozen=True, slots=True, kw_only=True)
class Placement:
    """Where one checkpoint matrix lands: the block name and the dotted attribute path of its module."""

    block: str
    attr: str


class NameMap(Protocol):
    """The adapter-specific naming an integration needs; everything else in this package is generic."""

    kinds: tuple[str, ...]

    def attrs_of(self, kind: str) -> tuple[str, ...]:
        """Attribute paths of every matrix module of a block of ``kind``, in the map's own order.

        The order is the map's construction order, not the checkpoint's: a checkpoint group's own
        concatenation order lives in its ``matrix_names``, recovered per matrix through ``place()``.
        """
        ...

    def place(self, matrix_name: str) -> Placement:
        """The block and attribute path of a checkpoint matrix name."""
        ...

    def kind_of(self, block_name: str) -> str:
        """The kind of a block name."""
        ...

    def param_name(self, checkpoint_name: str) -> str:
        """The module parameter name of a non-matrix checkpoint tensor (biases, norms, embedders)."""
        ...

    def transform_of(self, param: str) -> Transform | None:
        """The transform applied to the extra loaded into module parameter ``param`` (None for most)."""
        ...

    def is_matrix_module(self, module: Any) -> bool:
        """Whether ``module`` is one the seam swaps weights on."""
        ...


class StaticNameMap:
    """A name map from explicit tables: kind -> {checkpoint sub-path: attribute path}.

    Block names are ``<kind>.<index>``; a matrix name is ``<block>.<sub-path>.weight``; matrix modules are
    ``nn.Linear``.
    """

    def __init__(
        self,
        tables: Mapping[str, Mapping[str, str]],
        *,
        renames: Mapping[str, str] | None = None,
        transforms: Mapping[str, Transform] | None = None,
    ) -> None:
        """Keep one table per kind, checkpoint sub-path -> attribute path, in the given order.

        ``renames`` maps non-block checkpoint names to module parameter names; ``transforms`` maps a module
        parameter name to the transform applied to the extra loaded into it.

        Raises:
            DFloatIntegrationError: A rename's source is a block name.
        """
        self._tables = {k: dict(v) for k, v in tables.items()}
        self.kinds = tuple(self._tables)
        # `[0-9]`, not `\d`: `\d` matches any Unicode digit, and int("٣") == 3 would alias block 3.
        self._block = re.compile(
            rf"^({'|'.join(re.escape(k) for k in self.kinds)})\.([0-9]+)\.(.+)$"
        )
        self._renames = dict(renames or {})
        self._transforms = dict(transforms or {})
        blocky = sorted(s for s in self._renames if self._split(s) is not None)
        if blocky:
            raise DFloatIntegrationError(
                f"renames {blocky} are block names; block extras are renamed through the block tables"
            )

    def attrs_of(self, kind: str) -> tuple[str, ...]:
        """Attribute paths of every matrix module of a block of ``kind``, in table order."""
        return tuple(self._tables[kind].values())

    def _split(self, name: str) -> tuple[str, int, str] | None:
        match = self._block.match(name)
        return None if match is None else (match.group(1), int(match.group(2)), match.group(3))

    def place(self, matrix_name: str) -> Placement:
        """The block and attribute path of a checkpoint matrix name.

        Raises:
            DFloatIntegrationError: The name is not a ``<kind>.<index>.<sub-path>.weight`` name the
                table covers.
        """
        parsed = self._split(matrix_name)
        if parsed is None or not parsed[2].endswith(".weight"):
            raise DFloatIntegrationError(f"{matrix_name!r} is not a block matrix name")
        kind, idx, rest = parsed
        attr = self._tables[kind].get(rest.removesuffix(".weight"))
        if attr is None:
            raise DFloatIntegrationError(f"{matrix_name!r}: not a {kind} matrix the map covers")
        return Placement(block=f"{kind}.{idx}", attr=attr)

    def kind_of(self, block_name: str) -> str:
        """The kind of a ``<kind>.<index>`` block name.

        Raises:
            DFloatIntegrationError: ``block_name`` is not a block name of a known kind.
        """
        kind, _dot, idx = block_name.partition(".")
        if kind not in self._tables or not (idx.isascii() and idx.isdigit()):
            raise DFloatIntegrationError(f"{block_name!r} is not a block name")
        return kind

    def param_name(self, checkpoint_name: str) -> str:
        """The module parameter name of a non-matrix checkpoint tensor.

        A block extra is renamed through the same table as its block's matrices; a non-block name follows
        ``renames`` and is otherwise returned unchanged.
        """
        parsed = self._split(checkpoint_name)
        if parsed is None:
            return self._renames.get(checkpoint_name, checkpoint_name)
        kind, idx, rest = parsed
        head, _dot, leaf = rest.rpartition(".")
        mapped = self._tables[kind].get(head)
        return checkpoint_name if mapped is None else f"{kind}.{idx}.{mapped}.{leaf}"

    def transform_of(self, param: str) -> Transform | None:
        """The transform for the extra loaded into module parameter ``param`` (None when it has none)."""
        return self._transforms.get(param)

    def is_matrix_module(self, module: Any) -> bool:
        """Whether ``module`` is an ``nn.Linear``, the only matrix module this map swaps weights on."""
        return isinstance(module, nn.Linear)  # type: ignore[attr-defined]
