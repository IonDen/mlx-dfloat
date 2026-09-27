"""The block seam: assign one block's weights, run it, evaluate per policy, restore the placeholders."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import mlx.core as mx

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.names import Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, get_attr_path
from mlx_dfloat.integrate.providers import EVAL_POLICIES, WeightProvider

# "none" with a launching provider keeps every decoded group alive
MAX_NONE_POLICY_LAUNCHING_BLOCKS = 2

# The seam calls MLX through these names so a test can record the evaluation order.
_eval = mx.eval
_async_eval = mx.async_eval


class TraceHook(Protocol):
    """Receives one event per block per step and the step boundaries; four clock reads per block, no sync."""

    def begin_step(self) -> None:
        """Open a new step (called once, before its first block runs)."""
        ...

    def end_step(self) -> None:
        """Close the step (called once, after its last block's eval)."""
        ...

    def record(
        self,
        block: str,
        *,
        decode_start: float,
        decode_end: float,
        encode_end: float,
        eval_end: float,
    ) -> None:
        """Record one block's four clock reads: decode, the mflux call, and the policy's eval."""
        ...


@dataclass(slots=True)
class SeamState:
    """Mutable per-attachment state the seam threads through a step: never rebuilt mid-step."""

    provider: WeightProvider
    shapes: Shapes
    policy: str
    tracer: TraceHook | None = None
    verify_in_call: bool = False
    prev: Any = None  # depth2: the previous block's output, evaluated after the next is queued


def attach_state(
    provider: WeightProvider,
    shapes: Shapes,
    *,
    eval_policy: str = "per-block",
    tracer: TraceHook | None = None,
    verify_in_call: bool = False,
) -> SeamState:
    """Validate the policy for this provider and build the seam's state.

    Raises:
        DFloatIntegrationError: Unknown policy; a policy the provider does not run under; ``"none"`` with a
            launching provider over more than ``MAX_NONE_POLICY_LAUNCHING_BLOCKS`` blocks; or
            ``verify_in_call`` with ``"none"`` and a launching provider (reading the status words at the
            end of the call would force every decode before the caller's own eval).
    """
    if eval_policy not in EVAL_POLICIES:
        raise DFloatIntegrationError(
            f"unknown eval policy {eval_policy!r}; choose from {EVAL_POLICIES}"
        )
    if eval_policy not in provider.policies:
        raise DFloatIntegrationError(
            f"{type(provider).__name__} runs only under {provider.policies}, not {eval_policy!r}"
        )
    if (
        eval_policy == "none"
        and provider.launching
        and len(shapes) > MAX_NONE_POLICY_LAUNCHING_BLOCKS
    ):
        raise DFloatIntegrationError(
            f"eval policy 'none' with a launching provider over {len(shapes)} blocks would keep every decoded "
            f"group resident until the final eval; 'none' is for non-launching providers (or at most "
            f"{MAX_NONE_POLICY_LAUNCHING_BLOCKS} blocks)"
        )
    if verify_in_call and eval_policy == "none" and provider.launching:
        raise DFloatIntegrationError(
            "verify_in_call with eval policy 'none' and a launching provider would read the status "
            "words before the caller's eval, forcing every decode of the step; verify_step() after "
            "the step's eval instead"
        )
    return SeamState(
        provider=provider,
        shapes=shapes,
        policy=eval_policy,
        tracer=tracer,
        verify_in_call=verify_in_call,
    )


def begin_step(state: SeamState) -> None:
    """Refuse a step while the last one's status words are unchecked; open the tracer's step.

    Raises:
        DFloatIntegrationError: The provider still holds status words from an earlier step (nobody
            called ``verify_step()`` after it); they are kept, so a ``verify_step()`` can still check them.
    """
    pending = getattr(state.provider, "pending", None)
    if pending:
        raise DFloatIntegrationError(
            f"{len(pending)} decode status words from an earlier step are unchecked; call "
            "verify_step() after each step's eval, or attach with verify_in_call=True"
        )
    if state.tracer is not None:
        state.tracer.begin_step()


def end_step(state: SeamState, out: Any) -> Any:
    """Drain depth-2's tail, verify when asked, close the trace; returns ``out``."""
    if state.prev is not None:
        _eval(state.prev)
        state.prev = None
    if state.verify_in_call:
        state.provider.verify()
    if state.tracer is not None:
        state.tracer.end_step()
    return out


def abort_step(state: SeamState) -> None:
    """After a step that raised: no stale look-ahead, no stale status words, no depth-2 tail."""
    state.prev = None
    state.provider.reset()


def verify_step(state: SeamState) -> None:
    """Run the provider's deferred checks (an explicit call, for a step run without ``verify_in_call``)."""
    state.provider.verify()


def _seam_eval(state: SeamState, out: Any) -> None:
    if state.policy == "per-block":
        _eval(out)
    elif state.policy == "depth2":
        _async_eval(out)  # type: ignore[no-untyped-call]  # mlx's async_eval stub carries no annotations
        if state.prev is not None:
            _eval(state.prev)
        state.prev = out


def run_block(state: SeamState, block_name: str, block: Any, run: Callable[[], Any]) -> Any:
    """Assign the block's weights, run it, evaluate per policy, restore the placeholders (also on a raise).

    Raises:
        DFloatIntegrationError: The provider's dict does not cover the block's matrices, or a weight's
            shape does not match the block.
    """
    shapes = state.shapes[block_name]
    t_decode_start = time.perf_counter()
    weights = state.provider.weights_for(block_name, shapes)
    t_decode_end = time.perf_counter()
    if weights.keys() != shapes.keys():
        raise DFloatIntegrationError(
            f"{block_name}: provider returned {sorted(weights)}; the block needs {sorted(shapes)} "
            f"(missing {sorted(shapes.keys() - weights.keys())})"
        )
    for attr, shape in shapes.items():
        if tuple(weights[attr].shape) != tuple(shape):
            raise DFloatIntegrationError(
                f"{block_name}.{attr}: weight has shape {tuple(weights[attr].shape)}, the block needs {shape}"
            )
    try:
        for attr, weight in weights.items():
            get_attr_path(block, attr).weight = weight
        out = run()
        t_encode_end = time.perf_counter()
        _seam_eval(state, out)
        if state.tracer is not None:
            state.tracer.record(
                block_name,
                decode_start=t_decode_start,
                decode_end=t_decode_end,
                encode_end=t_encode_end,
                eval_end=time.perf_counter(),
            )
        return out
    finally:
        for attr in shapes:
            get_attr_path(block, attr).weight = PLACEHOLDER
