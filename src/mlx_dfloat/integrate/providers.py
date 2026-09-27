"""Weight providers: what the seam asks for one block's matrices."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Protocol, cast

import mlx.core as mx

from mlx_dfloat.decode import DecodeResult, check_status, decode_group, split_matrices
from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.format import MxGroup
from mlx_dfloat.integrate.names import BlockShapes, NameMap

# "per-block": evaluate each block's output. "depth2": async_eval it, then eval the previous
# block's (hides host-side allocation while the GPU runs the block; not a decode/compute overlap).
# "none": no evaluation inside the step (a non-launching provider only, or a few blocks).
EVAL_POLICIES: tuple[str, ...] = ("per-block", "depth2", "none")


class WeightProvider(Protocol):
    """Hands the seam one block's matrices as ``{attribute path: bf16 array of the block's shape}``."""

    launches: int
    launching: bool
    policies: tuple[str, ...]

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """Weights for ``block_name``; ``shapes`` is that block's entry from ``install_placeholders``."""
        ...

    def verify(self) -> None:
        """Check what ``weights_for`` deferred (decode status words); call it after the step's eval."""
        ...

    def reset(self) -> None:
        """Drop per-step state after a step that raised."""
        ...


class DF11Provider:
    """Decodes each block's DF11 group when asked: one decode launch per call.

    The status words are not read inside ``weights_for`` (a host read there would sync before every
    block); they queue in ``pending`` and ``verify`` reads them after the step's final eval.
    """

    launching = True
    policies = EVAL_POLICIES

    def __init__(
        self,
        resident: Mapping[str, MxGroup],
        matrix_names: Mapping[str, Sequence[str]],
        name_map: NameMap,
        *,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        """Hold the resident groups, each group's matrix names, and the name map that places them.

        ``decode`` defaults to the Metal backend; tests inject a counting reference decode.
        """
        self._resident = resident
        self._matrix_names = matrix_names
        self._name_map = name_map
        self._decode = decode if decode is not None else partial(decode_group, backend="metal")
        self.launches = 0
        self.pending: list[tuple[str, mx.array]] = []

    def verify(self) -> None:
        """Read every pending status word on the host and clear the list.

        Raises:
            DFloatFormatError: A block's decode reported an error; the message names the block.
        """
        pending, self.pending = self.pending, []
        for block_name, status in pending:
            check_status(status, name=block_name)

    def reset(self) -> None:
        """Forget queued status words (their decodes belong to a step that did not finish)."""
        self.pending = []

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """Decode the block's group and cut it into views of the block's shapes (no copies, no host read).

        Raises:
            DFloatIntegrationError: The block is not resident, or a matrix's size does not match its shape.
        """
        group = self._resident.get(block_name)
        if group is None:
            raise DFloatIntegrationError(f"{block_name}: no resident DF11 group")
        result = self._decode(group)
        self.pending.append((block_name, result.status))
        parts = split_matrices(result.bits, group.split_positions)
        names = self._matrix_names[block_name]
        if len(names) != len(parts):
            raise DFloatIntegrationError(
                f"{block_name}: {len(parts)} matrices decoded for {len(names)} names"
            )
        weights: dict[str, mx.array] = {}
        for matrix_name, part in zip(names, parts, strict=True):
            placement = self._name_map.place(matrix_name)
            if placement.block != block_name or placement.attr not in shapes:
                raise DFloatIntegrationError(
                    f"{matrix_name}: not a matrix of {block_name} with a known shape"
                )
            shape = shapes[placement.attr]
            n = 1
            for d in shape:
                n *= d
            if part.size != n:
                raise DFloatIntegrationError(
                    f"{matrix_name}: decoded {part.size} elements, expected {shape} = {n}"
                )
            weights[placement.attr] = part.view(mx.bfloat16).reshape(shape)
        self.launches += 1
        return weights


class ReuseProvider:
    """One pre-decoded dict per block kind, returned for every block of that kind; never launches."""

    launching = False
    policies = EVAL_POLICIES

    def __init__(self, per_kind: Mapping[str, Mapping[str, mx.array]], name_map: NameMap) -> None:
        """Keep one dict per kind, handed back as-is (by identity) for every block of that kind."""
        self._per_kind: dict[str, Mapping[str, mx.array]] = dict(per_kind)
        self._name_map = name_map
        self.launches = 0

    def verify(self) -> None:
        """Nothing deferred: no decode happened."""

    def reset(self) -> None:
        """No per-step state."""

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The kind's dict, checked against the block's shapes.

        Raises:
            DFloatIntegrationError: No dict for the block's kind, or a shape differs.
        """
        kind = self._name_map.kind_of(block_name)
        weights = self._per_kind.get(kind)
        if weights is None:
            raise DFloatIntegrationError(f"{block_name}: no reusable block of kind {kind!r}")
        _check_shapes(block_name, weights, shapes)
        return cast("dict[str, mx.array]", weights)


class ResidentProvider:
    """Per-block pre-decoded dicts (every block resident at once); never launches."""

    launching = False
    policies = EVAL_POLICIES

    def __init__(self, per_block: Mapping[str, Mapping[str, mx.array]]) -> None:
        """Keep the per-block dicts, handed back as-is (by identity) for their own block."""
        self._per_block: dict[str, Mapping[str, mx.array]] = dict(per_block)
        self.launches = 0

    def verify(self) -> None:
        """Nothing deferred: no decode happened."""

    def reset(self) -> None:
        """No per-step state."""

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The block's own dict, checked against its shapes.

        Raises:
            DFloatIntegrationError: The block is not resident, or a shape differs.
        """
        weights = self._per_block.get(block_name)
        if weights is None:
            raise DFloatIntegrationError(f"{block_name}: not resident")
        _check_shapes(block_name, weights, shapes)
        return cast("dict[str, mx.array]", weights)


def _check_shapes(block_name: str, weights: Mapping[str, mx.array], shapes: BlockShapes) -> None:
    for attr, shape in shapes.items():
        if attr not in weights:
            raise DFloatIntegrationError(f"{block_name}: no weight for {attr!r}")
        if tuple(weights[attr].shape) != tuple(shape):
            raise DFloatIntegrationError(
                f"{block_name}.{attr}: weight has shape {tuple(weights[attr].shape)}, the block needs {shape}"
            )
