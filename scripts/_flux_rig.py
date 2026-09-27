"""Bench glue over the package's integration API, plus the research providers the benches measure.

``Tracer``, ``BlockEvent`` and ``summarize_trace`` are the per-block timing bookkeeping the benches
attach to a step; ``PrefetchProvider`` is a research provider (a look-ahead decode, not part of the
package's shipped integration surface), so it stays here next to the benches that measure it.
``build_transformer`` adapts the package's depth-aware builder to the benches' short model names; it
imports mflux, so the dev venv (which lacks mflux) can still import everything else in this module.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import mlx.core as mx

from mlx_dfloat.errors import DFloatIntegrationError as RigError  # the benches' historical name
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.coverage import (
    check_extras_cover,
    decode_resident,
    extras_plan,
    load_resident_set,
    read_extra,
)
from mlx_dfloat.integrate.names import BlockShapes, Shapes, StaticNameMap
from mlx_dfloat.integrate.placeholders import (
    PLACEHOLDER,
    get_attr_path,
    install_placeholders,
    set_attr_path,
)
from mlx_dfloat.integrate.providers import (
    EVAL_POLICIES,
    DF11Provider,
    ResidentProvider,
    ReuseProvider,
    WeightProvider,
)
from mlx_dfloat.integrate.seam import MAX_NONE_POLICY_LAUNCHING_BLOCKS, _async_eval, _eval
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux1.names import DOUBLE_PREFIX, DROPPED_EXTRAS, SINGLE_PREFIX, flux_name_map
from mlx_dfloat.mflux.flux1.transformer import (
    MAX_BUILD_ACTIVE_BYTES,
    SeamMixin,
    seam_transformer_class,
)
from mlx_dfloat.mflux.flux1.transformer import build_transformer as _build_transformer

# Unchanged: the benches' recorded default MLX buffer-cache limit (see bench_flux_step --cache-limit).
FLUX_CACHE_LIMIT = int(1.4e9)


# --- trace ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockEvent:
    """One block of one step: ``perf_counter`` stamps at the seam's phase boundaries.

    ``t_decode_start`` → ``t_decode_end`` is ``weights_for`` (the provider's host work and launch
    queuing); → ``t_encode_end`` is the block's graph build (``run()``); → ``t_eval_end`` is the
    eval policy's wait. The placeholder restore comes after ``t_eval_end``.
    """

    step: int
    block: str
    t_decode_start: float
    t_decode_end: float
    t_encode_end: float
    t_eval_end: float


class Tracer:
    """Collects ``BlockEvent``s and the ``(start, end)`` stamps of every traced step.

    Pass one to ``SeamMixin.attach``; the seam records into it. The stamps are cheap (four
    ``perf_counter`` calls per block) and no host-device sync is added: the eval-wait phase is
    whatever the policy already waited for.
    """

    def __init__(self) -> None:
        """Start with no events and no open step."""
        self.events: list[BlockEvent] = []
        self.steps: list[tuple[float, float]] = []
        self._step_start: float | None = None

    @property
    def step(self) -> int:
        """Index of the step being recorded (or of the next one, between steps)."""
        return len(self.steps)

    def begin_step(self) -> None:
        """Open a new step (called once, before its first block runs)."""
        self._step_start = time.perf_counter()

    def end_step(self) -> None:
        """Close the step (called once, after its last block's eval).

        Raises:
            RigError: There is no open step (``begin_step`` was not called first).
        """
        if self._step_start is None:
            raise RigError("end_step without begin_step")
        self.steps.append((self._step_start, time.perf_counter()))
        self._step_start = None

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
        self.events.append(
            BlockEvent(
                step=self.step,
                block=block,
                t_decode_start=decode_start,
                t_decode_end=decode_end,
                t_encode_end=encode_end,
                t_eval_end=eval_end,
            )
        )


def summarize_trace(events: Sequence[BlockEvent], *, step: tuple[float, float]) -> dict[str, Any]:
    """Split one step's events into named seconds.

    ``host_decode_s``, ``host_encode_s`` and ``eval_wait_s`` sum the three phases over the blocks.
    ``restore_gap_s`` sums, from the second block on, the time between the previous block's eval
    end and this block's decode start (the placeholder restore and the hook's own overhead);
    ``gpu_idle_gap_s`` sums the previous eval end to this block's encode end, which under the
    per-block policy is host work done while the GPU has nothing queued. ``head_s`` and ``tail_s``
    are the step's time before the first decode and after the last eval; ``step_s`` is the whole
    ``step`` window.

    Raises:
        RigError: The events come from more than one step.
    """
    if len({e.step for e in events}) > 1:
        raise RigError("summarize_trace takes the events of one step")
    start, end = step
    return {
        "n_blocks": len(events),
        "host_decode_s": sum(e.t_decode_end - e.t_decode_start for e in events),
        "host_encode_s": sum(e.t_encode_end - e.t_decode_end for e in events),
        "eval_wait_s": sum(e.t_eval_end - e.t_encode_end for e in events),
        "restore_gap_s": sum(e.t_decode_start - prev.t_eval_end for prev, e in pairwise(events)),
        "gpu_idle_gap_s": sum(e.t_encode_end - prev.t_eval_end for prev, e in pairwise(events)),
        "head_s": events[0].t_decode_start - start if events else 0.0,
        "tail_s": end - events[-1].t_eval_end if events else 0.0,
        "step_s": end - start,
    }


# --- the research providers -------------------------------------------------------------------------


class PrefetchProvider:
    """Decodes block i+1 while block i runs: ``DF11Provider`` with one group of look-ahead.

    ``shapes`` fixes the step's block order (its keys); the last block prefetches the first, so the
    next step's first block is ready as well. ``stream`` is where the look-ahead decode is
    submitted: a second GPU stream (``mx.new_stream(mx.gpu)``), so the kernel may overlap the
    current block's compute, or None for the default stream, where the look-ahead can only hide
    host work. Each look-ahead is submitted with ``mx.async_eval`` (through the package seam
    module, so a test that patches its evaluation order sees the prefetch's submissions too); MLX
    orders the streams (the block's matmuls wait on the decode's event on the GPU, not on the
    host). The look-ahead is submitted at the top of the block, after the previous block's
    per-block eval returned, so its output allocation runs while the GPU is idle; under the
    per-block policy exactly one decoded group beyond the current block stays resident, which is
    why ``policies`` allows no other (depth-2 would hold three). ``launches`` counts the
    steady-state decodes (one per block per step); the single cold inline decode is
    ``cold_launches``.
    """

    launching = True
    policies = ("per-block",)

    def __init__(
        self, inner: DF11Provider, shapes: Shapes, *, stream: mx.Stream | None = None
    ) -> None:
        """Hold the wrapped provider, the step's block order, and the look-ahead's stream."""
        self._inner = inner
        self._shapes = shapes
        self._order = list(shapes)
        self.stream = stream
        self.cold_launches = 0
        self._ready: tuple[str, dict[str, mx.array]] | None = None

    @property
    def launches(self) -> int:
        """Steady-state decodes so far (the one cold inline decode is not counted)."""
        return self._inner.launches - self.cold_launches

    @property
    def pending(self) -> list[tuple[str, mx.array]]:
        """The wrapped provider's queued status words (the current step's and the look-ahead's)."""
        return self._inner.pending

    def verify(self) -> None:
        """Read every pending status word (the current step's and the look-ahead's)."""
        self._inner.verify()

    def reset(self) -> None:
        """Drop the look-ahead and the inner status words (after a step that raised); start cold again."""
        self._ready = None
        self._inner.reset()

    def _submit(self, block_name: str) -> dict[str, mx.array]:
        shapes = self._shapes[block_name]
        if self.stream is None:
            weights = self._inner.weights_for(block_name, shapes)
        else:
            with mx.stream(self.stream):
                weights = self._inner.weights_for(block_name, shapes)
        seam._async_eval(*weights.values())  # looked up on the module so a patched hook is honored
        return weights

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The weights submitted for ``block_name`` earlier, then submit the next block's.

        Raises:
            RigError: ``block_name`` is not a block of the step, or not the one the look-ahead
                was submitted for (blocks must be requested in the order of ``shapes``), or its
                shapes differ.
        """
        expected = self._shapes.get(block_name)
        if expected is None:
            raise RigError(f"{block_name}: no such block in the prefetch order")
        if shapes.keys() != expected.keys():
            raise RigError(f"{block_name}: shapes differ from the ones the prefetch was built for")
        if self._ready is None:
            self.cold_launches += 1
            weights = self._submit(block_name)
        else:
            ready_name, weights = self._ready
            if ready_name != block_name:
                raise RigError(
                    f"{block_name} requested out of order; expected {ready_name} (the prefetch "
                    "follows the block order)"
                )
        self._ready = None
        nxt = self._order[(self._order.index(block_name) + 1) % len(self._order)]
        self._ready = (nxt, self._submit(nxt))
        return weights


# --- the benches' entry ------------------------------------------------------------------------------


def build_transformer(
    model: str, ckpt: DF11Checkpoint, *, n_double: int = 19, n_single: int = 38
) -> tuple[Any, Shapes]:
    """The benches' entry: mflux config by short name, reduced depth for the validation rigs.

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        RigError: ``model`` is not ``"schnell"`` or ``"dev"``.
    """
    require_mflux()
    from mflux.models.common.config.model_config import ModelConfig

    factories = {"schnell": ModelConfig.schnell, "dev": ModelConfig.dev}
    if model not in factories:
        raise RigError(f"unknown model {model!r}; choose from {sorted(factories)}")
    return _build_transformer(factories[model](), ckpt, n_double=n_double, n_single=n_single)


__all__ = [
    "DOUBLE_PREFIX",
    "DROPPED_EXTRAS",
    "EVAL_POLICIES",
    "FLUX_CACHE_LIMIT",
    "MAX_BUILD_ACTIVE_BYTES",
    "MAX_NONE_POLICY_LAUNCHING_BLOCKS",
    "PLACEHOLDER",
    "SINGLE_PREFIX",
    "BlockEvent",
    "BlockShapes",
    "DF11Provider",
    "PrefetchProvider",
    "ResidentProvider",
    "ReuseProvider",
    "RigError",
    "SeamMixin",
    "Shapes",
    "StaticNameMap",
    "Tracer",
    "WeightProvider",
    "_async_eval",
    "_eval",
    "build_transformer",
    "check_extras_cover",
    "decode_resident",
    "extras_plan",
    "flux_name_map",
    "get_attr_path",
    "install_placeholders",
    "load_resident_set",
    "read_extra",
    "seam_transformer_class",
    "set_attr_path",
    "summarize_trace",
]
