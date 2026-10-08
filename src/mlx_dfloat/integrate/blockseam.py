"""The block seam by class swap, for transformers that call their blocks inline.

Each block instance's class is swapped for a subclass of its own class whose ``__call__`` runs the block
through ``seam.run_block``. The parameter paths do not change, so the placeholders, ``load_weights`` and the
extras coverage work as they do for a hook-based seam. Compose ``StepSeam`` in front of the transformer's own
class for the step boundaries; ``seam_blocks`` binds every block to the transformer's ``seam_cell``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.names import Shapes
from mlx_dfloat.integrate.providers import WeightProvider

_BINDING = "_dfloat_block"
_CELL = "_dfloat_cell"


class SeamCell:
    """The attached seam state, shared by a transformer and its blocks (a plain object, never a parameter)."""

    __slots__ = ("state",)

    def __init__(self) -> None:
        """Start detached."""
        self.state: seam.SeamState | None = None


@dataclass(frozen=True, slots=True)
class BlockBinding:
    """A seamed block's name and the cell it reads its state from."""

    name: str
    cell: SeamCell


class BlockSeam:
    """Placed in front of a block's own class: every call goes through the seam."""

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the block with its weights assigned, evaluated per policy, placeholders restored.

        The call's arguments reach the provider's optional ``before_block`` hook (``seam.BeforeBlock``).

        Raises:
            DFloatIntegrationError: Nothing is attached.
        """
        binding: BlockBinding = vars(self)[_BINDING]
        state = binding.cell.state
        if state is None:
            raise DFloatIntegrationError(
                f"{binding.name}: call attach(provider, shapes) before running a step"
            )
        parent: Any = super()
        return seam.run_block(
            state,
            binding.name,
            self,
            lambda: parent.__call__(*args, **kwargs),
            inputs=(args, kwargs),
        )


@cache
def seamed_block_class(cls: type) -> type:
    """``BlockSeam`` composed in front of ``cls`` (one class per block class)."""
    return type(f"Seam{cls.__name__}", (BlockSeam, cls), {"__slots__": (), "__module__": __name__})


def seam_blocks(
    block_lists: Sequence[tuple[str, Sequence[Any]]], cell: SeamCell
) -> tuple[str, ...]:
    """Swap every block's class for its seamed class and bind it to ``cell`` as ``<kind>.<idx>``; the names in run order.

    Raises:
        DFloatIntegrationError: A block is already seamed.
    """
    names: list[str] = []
    for kind, blocks in block_lists:
        for idx, block in enumerate(blocks):
            name = f"{kind}.{idx}"
            if _BINDING in vars(block):
                raise DFloatIntegrationError(f"{name}: already seamed")
            block_cls: type = type(block)
            block.__class__ = seamed_block_class(block_cls)
            vars(block)[_BINDING] = BlockBinding(name=name, cell=cell)
            names.append(name)
    return tuple(names)


class StepSeam:
    """Compose in front of a transformer class whose blocks ``seam_blocks`` bound to ``seam_cell``.

    A step is ``out = transformer(...)``; with ``verify_in_call`` the decode status words are checked before the call
    returns (required when one step calls the transformer more than once), otherwise call ``verify_step()`` after
    the step's final eval.
    """

    @property
    def seam_cell(self) -> SeamCell:
        """The cell this transformer's blocks read their state from (created on first use)."""
        cell = vars(self).get(_CELL)
        if cell is None:
            cell = SeamCell()
            vars(self)[_CELL] = cell
        return cell

    def attach(
        self,
        provider: WeightProvider,
        shapes: Shapes,
        *,
        eval_policy: str = "per-block",
        tracer: seam.TraceHook | None = None,
        verify_in_call: bool = False,
    ) -> None:
        """Bind the provider, the block shapes and the eval policy (see ``seam.attach_state``)."""
        self.seam_cell.state = seam.attach_state(
            provider, shapes, eval_policy=eval_policy, tracer=tracer, verify_in_call=verify_in_call
        )

    def _state(self) -> seam.SeamState:
        state = self.seam_cell.state
        if state is None:
            raise DFloatIntegrationError("call attach(provider, shapes) before running a step")
        return state

    def detach(self) -> None:
        """Forget the provider and the shapes; the next step needs a new ``attach``."""
        state = self.seam_cell.state
        if state is not None:
            state.provider.reset()
            self.seam_cell.state = None

    def verify_step(self) -> None:
        """Run the provider's deferred checks; call it after the step's final ``mx.eval``."""
        seam.verify_step(self._state())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the transformer's own call as one seam step; a failed step leaves no stale seam state."""
        state = self._state()
        seam.begin_step(state)
        try:
            out = super().__call__(*args, **kwargs)  # type: ignore[misc]
            return seam.end_step(state, out)
        except BaseException:
            seam.abort_step(state)
            raise
