"""The FLUX measurement rig: mflux's ``Transformer`` driven block by block from DF11 groups.

Pure logic (name maps, attribute paths, the placeholder step, the seam mixin, the providers, the
eval policies, the extras plan and its coverage check) is importable and testable without mflux.
Everything that touches mflux (``build_transformer``, ``seam_transformer_class``) imports it inside
the function, so the dev venv can import this module.

How a step runs: every block matrix weight of the transformer is a zero-size placeholder. The seam
(``SeamMixin``) overrides mflux 0.20.0's two per-block hooks; before a block runs it asks the
``WeightProvider`` for that block's weights and assigns them as ``(out, in)`` bf16 views, runs the
block, applies the eval policy to the whole returned value, and restores the placeholders, also
when the block raises. ``DF11Provider`` decodes the block's group on demand (one kernel launch per
block), ``ReuseProvider`` and ``ResidentProvider`` hand back pre-decoded weights and never launch.
"""

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, partial
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_dfloat._safetensors import TensorInfo, read_array
from mlx_dfloat.decode import DecodeResult, check_status, decode_group, split_matrices
from mlx_dfloat.format import DF11Checkpoint, MxGroup, load_group_mx

DOUBLE_PREFIX = "transformer_blocks"
SINGLE_PREFIX = "single_transformer_blocks"
# DF11 (diffusers) sub-path -> mflux 0.20.0 attribute path, in pattern_dict order. Only the
# feed-forward names differ (`ff.net.0.proj` -> `ff.linear1`, `ff.net.2` -> `ff.linear2`).
DOUBLE_MAP: dict[str, str] = {
    "norm1.linear": "norm1.linear",
    "norm1_context.linear": "norm1_context.linear",
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.add_k_proj": "attn.add_k_proj",
    "attn.add_v_proj": "attn.add_v_proj",
    "attn.add_q_proj": "attn.add_q_proj",
    "attn.to_out.0": "attn.to_out.0",
    "attn.to_add_out": "attn.to_add_out",
    "ff.net.0.proj": "ff.linear1",
    "ff.net.2": "ff.linear2",
    "ff_context.net.0.proj": "ff_context.linear1",
    "ff_context.net.2": "ff_context.linear2",
}
SINGLE_MAP: dict[str, str] = {
    "norm.linear": "norm.linear",
    "proj_mlp": "proj_mlp",
    "proj_out": "proj_out",
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
}
_MAPS = {DOUBLE_PREFIX: DOUBLE_MAP, SINGLE_PREFIX: SINGLE_MAP}
# Extras the checkpoint carries but mflux 0.20.0 has no parameter for: its AdaLayerNormContinuous
# builds `norm_out.linear` with bias=False, and its own WeightApplier drops the bias through
# update(strict=False). The rig follows mflux; the bias is non-zero in FLUX.1-schnell-DF11.
DROPPED_EXTRAS: frozenset[str] = frozenset({"norm_out.linear.bias"})
PLACEHOLDER = mx.zeros((0,), dtype=mx.bfloat16)
MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3
# MLX cache limit every FLUX rig process sets before building (the smoke and the step bench share it).
FLUX_CACHE_LIMIT = int(1.4e9)
EvalPolicy = Literal["per-block", "depth2", "none"]
# "per-block": mx.eval each block's output. "depth2": async_eval it, then eval the previous block's.
# The depth2 look-ahead is bounded by MLX's command-buffer window: the encoding thread blocks once
# enough committed buffers are in flight, so async_eval(out_i) returns only near the end of block i,
# and the next block's decode never overlaps block i's matmuls on the same in-order stream. What it
# saves is the host-side gap between blocks, on the DF11 and the control side alike; it is not a
# decode/compute overlap. "none": no evaluation inside the step; for ReuseProvider (the
# control-noeval run) only, since a launching provider would keep every decoded group alive until
# the final eval (~24 GB on FLUX.1).
EVAL_POLICIES: tuple[str, ...] = ("per-block", "depth2", "none")
MAX_NONE_POLICY_LAUNCHING_BLOCKS = (
    2  # the reduced-depth validation (1+1) may still launch under "none"
)
_BLOCK_NAME = re.compile(rf"^({DOUBLE_PREFIX}|{SINGLE_PREFIX})\.(\d+)\.(.+)$")

# The seam calls MLX through these names so a test can record the evaluation order.
_eval = mx.eval
_async_eval = mx.async_eval


class RigError(RuntimeError):
    """A rig invariant failed: a name outside the maps, a missing parameter, a shape mismatch."""


# --- names and attribute paths -----------------------------------------------------------------------


def split_block_name(name: str) -> tuple[str, int, str] | None:
    """Split ``transformer_blocks.4.ff.net.2.weight`` into (kind, 4, rest); None for non-block names."""
    match = _BLOCK_NAME.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def mflux_path(matrix_name: str) -> tuple[str, str]:
    """Map a DF11 matrix name to (block name, mflux attribute path of its ``nn.Linear``).

    Raises:
        RigError: The name is not a block matrix name the maps cover.
    """
    parsed = split_block_name(matrix_name)
    if parsed is None or not parsed[2].endswith(".weight"):
        raise RigError(f"{matrix_name!r} is not a FLUX block matrix name")
    kind, idx, rest = parsed
    sub = rest.removesuffix(".weight")
    mapping = _MAPS[kind]
    if sub not in mapping:
        raise RigError(f"{matrix_name!r}: {sub!r} is not a {kind} matrix the rig maps")
    return f"{kind}.{idx}", mapping[sub]


def mflux_param_name(df11_name: str) -> str:
    """The mflux parameter name of a checkpoint extra (biases, norms, non-block tensors).

    Block extras follow the matrix maps (``ff.net.0.proj.bias`` -> ``ff.linear1.bias``); every other
    name is already mflux's.
    """
    parsed = split_block_name(df11_name)
    if parsed is None:
        return df11_name
    kind, idx, rest = parsed
    head, _dot, leaf = rest.rpartition(".")
    mapped = _MAPS[kind].get(head)
    return df11_name if mapped is None else f"{kind}.{idx}.{mapped}.{leaf}"


def get_attr_path(module: Any, path: str) -> Any:
    """Resolve a dotted path on a module; a digit component indexes a list (``attn.to_out.0``).

    Raises:
        RigError: A component does not exist.
    """
    node = module
    for part in path.split("."):
        try:
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RigError(f"{path!r}: no {part!r} on {type(node).__name__}") from exc
    return node


def set_attr_path(module: Any, path: str, value: Any) -> None:
    """Assign ``value`` at a dotted path (see ``get_attr_path``)."""
    head, _dot, leaf = path.rpartition(".")
    parent = get_attr_path(module, head) if head else module
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


# --- the placeholder step ----------------------------------------------------------------------------

BlockShapes = dict[str, tuple[int, int]]
Shapes = dict[str, BlockShapes]


def _block_lists(transformer: Any) -> Iterable[tuple[str, Sequence[Any], Mapping[str, str]]]:
    yield DOUBLE_PREFIX, transformer.transformer_blocks, DOUBLE_MAP
    yield SINGLE_PREFIX, transformer.single_transformer_blocks, SINGLE_MAP


def install_placeholders(transformer: Any) -> Shapes:
    """Replace every block matrix weight with ``PLACEHOLDER`` and return the shapes it had.

    The result maps block name -> mflux attribute path -> ``(out, in)``. Every mapped path must be an
    ``nn.Linear`` of the block and every ``nn.Linear`` of the block must be mapped, so a renamed or
    added mflux layer is caught here, not at the first matmul.

    Raises:
        RigError: A block's ``nn.Linear`` set differs from the map.
    """
    shapes: Shapes = {}
    for kind, blocks, mapping in _block_lists(transformer):
        mapped = set(mapping.values())
        for idx, block in enumerate(blocks):
            name = f"{kind}.{idx}"
            linears = {p for p, m in block.named_modules() if isinstance(m, nn.Linear)}
            if linears != mapped:
                raise RigError(
                    f"{name}: mapped matrices missing from the block: {sorted(mapped - linears)}; "
                    f"Linear layers the maps do not cover: {sorted(linears - mapped)}"
                )
            per: BlockShapes = {}
            for attr in mapping.values():
                linear = get_attr_path(block, attr)
                out, inn = (int(d) for d in linear.weight.shape)
                per[attr] = (out, inn)
                linear.weight = PLACEHOLDER
            shapes[name] = per
    return shapes


# --- providers ---------------------------------------------------------------------------------------


class WeightProvider(Protocol):
    """Hands the seam one block's matrices as ``{mflux attribute path: bf16 (out, in) array}``.

    ``launching`` says whether ``weights_for`` decodes (so ``launches`` can grow); the seam uses it
    to refuse the ``"none"`` policy over more than ``MAX_NONE_POLICY_LAUNCHING_BLOCKS`` blocks.
    """

    launches: int
    launching: bool

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """Weights for ``block_name``; ``shapes`` is that block's entry from ``install_placeholders``."""
        ...

    def verify(self) -> None:
        """Check what ``weights_for`` deferred (the decode status words); call it after the step's eval."""
        ...


def _check_shapes(block_name: str, weights: Mapping[str, mx.array], shapes: BlockShapes) -> None:
    for attr, shape in shapes.items():
        got = weights.get(attr)
        if got is not None and tuple(got.shape) != shape:
            raise RigError(f"{block_name}.{attr}: weight shape {tuple(got.shape)} != {shape}")


class DF11Provider:
    """Decodes each block's DF11 group when asked: one decode launch per call.

    The kernel's status words are not read inside ``weights_for`` (a host read there would sync
    before every block and be counted as DF11 overhead); they are queued in ``pending`` and read by
    ``verify()``, which the step caller runs after the step's final ``mx.eval`` (the status comes
    out of the same launch as the bits, so by then it is already computed).
    """

    launching = True

    def __init__(
        self,
        resident: Mapping[str, MxGroup],
        matrix_names: Mapping[str, Sequence[str]],
        *,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        """Hold the resident groups (from ``load_resident_set``) and each group's matrix names.

        ``decode`` defaults to the Metal backend; tests inject a counting reference decode.
        """
        self._resident = resident
        self._matrix_names = matrix_names
        self._decode = decode if decode is not None else partial(decode_group, backend="metal")
        self.launches = 0
        self.pending: list[tuple[str, mx.array]] = []  # (block name, status words) not yet checked

    def verify(self) -> None:
        """Read every pending status word on the host and clear the list.

        Raises:
            DFloatFormatError: A block's decode reported an error; the message names the block.
        """
        pending, self.pending = self.pending, []
        for block_name, status in pending:
            check_status(status, name=block_name)

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """Decode the block's group and cut it into ``(out, in)`` bf16 views (no copies, no host read).

        Raises:
            RigError: The block is not resident, or a matrix's size does not match its shape.
        """
        group = self._resident.get(block_name)
        if group is None:
            raise RigError(f"{block_name}: no resident DF11 group")
        result = self._decode(group)
        self.pending.append((block_name, result.status))
        parts = split_matrices(result.bits, group.split_positions)
        names = self._matrix_names[block_name]
        if len(names) != len(parts):
            raise RigError(f"{block_name}: {len(parts)} matrices decoded for {len(names)} names")
        weights: dict[str, mx.array] = {}
        for matrix_name, part in zip(names, parts, strict=True):
            block, attr = mflux_path(matrix_name)
            if block != block_name or attr not in shapes:
                raise RigError(f"{matrix_name}: not a matrix of {block_name} with a known shape")
            out, inn = shapes[attr]
            if part.size != out * inn:
                raise RigError(
                    f"{matrix_name}: decoded {part.size} elements, expected {(out, inn)} = {out * inn}"
                )
            weights[attr] = part.view(mx.bfloat16).reshape(out, inn)
        self.launches += 1
        return weights


class PrefetchProvider:
    """Decodes block i+1 while block i runs: ``DF11Provider`` with one group of look-ahead.

    ``shapes`` fixes the step's block order (its keys); the last block prefetches the first, so the
    next step's first block is ready as well. ``stream`` is where the look-ahead decode is
    submitted: a second GPU stream (``mx.new_stream(mx.gpu)``), so the kernel may overlap the
    current block's compute, or None for the default stream, where the look-ahead can only hide
    host work. Each look-ahead is submitted with ``mx.async_eval``; MLX orders the streams. One
    decoded group beyond the current block stays resident. ``launches`` counts the steady-state
    decodes (one per block per step); the single cold inline decode is ``cold_launches``.
    """

    launching = True

    def __init__(
        self, inner: DF11Provider, shapes: Shapes, *, stream: mx.Stream | None = None
    ) -> None:
        self._inner = inner
        self._shapes = shapes
        self._order = list(shapes)
        self.stream = stream
        self.cold_launches = 0
        self._ready: tuple[str, dict[str, mx.array]] | None = None

    @property
    def launches(self) -> int:
        return self._inner.launches - self.cold_launches

    @property
    def pending(self) -> list[tuple[str, mx.array]]:
        return self._inner.pending

    def verify(self) -> None:
        """Read every pending status word (the current step's and the look-ahead's)."""
        self._inner.verify()

    def _submit(self, block_name: str) -> dict[str, mx.array]:
        shapes = self._shapes[block_name]
        if self.stream is None:
            weights = self._inner.weights_for(block_name, shapes)
        else:
            with mx.stream(self.stream):
                weights = self._inner.weights_for(block_name, shapes)
        mx.async_eval(*weights.values())
        return weights

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The weights submitted for ``block_name`` earlier, then submit the next block's.

        Raises:
            RigError: ``block_name`` is not the block the look-ahead was submitted for (blocks
                must be requested in the order of ``shapes``), or its shapes differ.
        """
        if shapes.keys() != self._shapes[block_name].keys():
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


class ReuseProvider:
    """One pre-decoded double-block dict and one single-block dict, returned for every block.

    The control provider: no launches, so it is the one the ``"none"`` policy is for.
    """

    launching = False

    def __init__(self, double: Mapping[str, mx.array], single: Mapping[str, mx.array]) -> None:
        """Keep the two dicts; they are handed back as-is."""
        self._dicts = {DOUBLE_PREFIX: dict(double), SINGLE_PREFIX: dict(single)}
        self.launches = 0

    def verify(self) -> None:
        """Nothing deferred: no decode happened."""

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The double or single dict, by the block's kind.

        Raises:
            RigError: The block name has no kind, or a weight's shape differs from the block's.
        """
        parsed = split_block_name(f"{block_name}.weight")
        if parsed is None:
            raise RigError(f"{block_name!r} is not a FLUX block name")
        weights = self._dicts[parsed[0]]
        _check_shapes(block_name, weights, shapes)
        return weights


class ResidentProvider:
    """Per-block pre-decoded BF16 dicts (every block resident at once); never launches.

    For the reduced-depth validation only: at full depth the resident set is the whole BF16 model.
    """

    launching = False

    def __init__(self, per_block: Mapping[str, Mapping[str, mx.array]]) -> None:
        """Keep the per-block dicts."""
        self._per_block = per_block
        self.launches = 0

    def verify(self) -> None:
        """Nothing deferred: no decode happened."""

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The block's resident dict.

        Raises:
            RigError: No dict for the block, or a weight's shape differs from the block's.
        """
        weights = self._per_block.get(block_name)
        if weights is None:
            raise RigError(f"{block_name}: no resident weights")
        _check_shapes(block_name, weights, shapes)
        return dict(weights)


def decode_resident(provider: DF11Provider, shapes: Shapes) -> dict[str, dict[str, mx.array]]:
    """Decode every block's group once into resident BF16 dicts, one block evaluated before the next.

    One lazy eval over every block would allocate every decode up front (the run-ahead the per-block
    eval policy exists to prevent), so each block's dict is evaluated as soon as it is cut. The
    deferred status words are checked at the end.

    Raises:
        DFloatFormatError: A block's decode reported an error (``DF11Provider.verify``).
        RigError: A block is not resident, or a matrix's size does not match its shape.
    """
    per_block: dict[str, dict[str, mx.array]] = {}
    for name, per in shapes.items():
        weights = provider.weights_for(name, per)
        _eval(weights)
        per_block[name] = weights
    provider.verify()
    return per_block


# --- the seam ----------------------------------------------------------------------------------------


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
        self.events: list[BlockEvent] = []
        self.steps: list[tuple[float, float]] = []
        self._step_start: float | None = None

    @property
    def step(self) -> int:
        """Index of the step being recorded (or of the next one, between steps)."""
        return len(self.steps)

    def begin_step(self) -> None:
        self._step_start = time.perf_counter()

    def end_step(self) -> None:
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
    out: dict[str, Any] = {
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
    return out


@dataclass(slots=True)
class _SeamState:
    provider: WeightProvider
    shapes: Shapes
    policy: str
    tracer: Tracer | None = None
    prev: Any = None  # depth2: the previous block's output, evaluated after the next is queued


class SeamMixin:
    """Overrides mflux's two per-block hooks to assign, run, evaluate and restore per block.

    Compose it in front of mflux's ``Transformer`` (``seam_transformer_class``) or a fake with the
    same hooks. Call ``attach`` before the first step. The state lives in a plain attribute so
    ``nn.Module`` never treats it as a parameter.

    A step is: ``out = transformer(...)``, ``mx.eval(out)``, then ``transformer.verify_step()``,
    which reads the decode status words the provider deferred. Nothing inside the step reads the
    device, so the eval policy alone decides where the host waits.
    """

    _seam: _SeamState

    def attach(
        self,
        provider: WeightProvider,
        shapes: Shapes,
        *,
        eval_policy: str = "per-block",
        tracer: Tracer | None = None,
    ) -> None:
        """Bind the provider, the block shapes (from ``install_placeholders``) and the eval policy.

        With a ``tracer`` every block of every step records a ``BlockEvent`` and every step its
        ``(start, end)``.

        Raises:
            RigError: Unknown eval policy, or ``"none"`` with a launching provider over more than
                ``MAX_NONE_POLICY_LAUNCHING_BLOCKS`` blocks (every decoded group would stay alive
                until the final eval).
        """
        if eval_policy not in EVAL_POLICIES:
            raise RigError(f"unknown eval policy {eval_policy!r}; choose from {EVAL_POLICIES}")
        if (
            eval_policy == "none"
            and provider.launching
            and len(shapes) > MAX_NONE_POLICY_LAUNCHING_BLOCKS
        ):
            raise RigError(
                f"eval policy 'none' with a launching provider over {len(shapes)} blocks would keep "
                f"every decoded group resident until the final eval; 'none' is for ReuseProvider "
                f"(or at most {MAX_NONE_POLICY_LAUNCHING_BLOCKS} blocks)"
            )
        self._seam = _SeamState(provider=provider, shapes=shapes, policy=eval_policy, tracer=tracer)

    def verify_step(self) -> None:
        """Run the provider's deferred checks; call it after the step's final ``mx.eval``.

        Raises:
            DFloatFormatError: A block's decode reported an error (``DF11Provider.verify``).
        """
        self._seam_state().provider.verify()

    def _seam_state(self) -> _SeamState:
        try:
            return self._seam
        except AttributeError as exc:
            raise RigError("call attach(provider, shapes) before running a step") from exc

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run mflux's step, then drain the depth2 tail; a failed step leaves no stale state."""
        state = self._seam_state()
        if state.tracer is not None:
            state.tracer.begin_step()
        try:
            out = super().__call__(*args, **kwargs)  # type: ignore[misc]
            if state.prev is not None:
                _eval(state.prev)
            if state.tracer is not None:
                state.tracer.end_step()
            return out
        finally:
            state.prev = None

    def _seam_eval(self, out: Any) -> None:
        state = self._seam
        if state.policy == "per-block":
            _eval(out)
        elif state.policy == "depth2":
            _async_eval(out)
            if state.prev is not None:
                _eval(state.prev)
            state.prev = out

    def _seam_run(self, block_name: str, block: Any, run: Callable[[], Any]) -> Any:
        state = self._seam_state()
        shapes = state.shapes[block_name]
        t_decode_start = time.perf_counter()
        weights = state.provider.weights_for(block_name, shapes)
        t_decode_end = time.perf_counter()
        if weights.keys() != shapes.keys():
            raise RigError(
                f"{block_name}: provider returned {sorted(weights)}; the block needs "
                f"{sorted(shapes)} (missing {sorted(shapes.keys() - weights.keys())})"
            )
        try:
            for attr, weight in weights.items():
                get_attr_path(block, attr).weight = weight
            out = run()
            t_encode_end = time.perf_counter()
            self._seam_eval(out)
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

    def _apply_joint_transformer_block(
        self,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: list[mx.array] | None,
    ) -> Any:
        """The joint-block hook of mflux, wrapped by the seam."""
        return self._seam_run(
            f"{DOUBLE_PREFIX}.{idx}",
            block,
            lambda: super(SeamMixin, self)._apply_joint_transformer_block(  # type: ignore[misc]
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_block_samples=controlnet_block_samples,
            ),
        )

    def _apply_single_transformer_block(
        self,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_single_block_samples: list[mx.array] | None,
    ) -> Any:
        """The single-block hook of mflux, wrapped by the seam."""
        return self._seam_run(
            f"{SINGLE_PREFIX}.{idx}",
            block,
            lambda: super(SeamMixin, self)._apply_single_transformer_block(  # type: ignore[misc]
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_single_block_samples=controlnet_single_block_samples,
            ),
        )


@cache
def seam_transformer_class() -> type:
    """``SeamMixin`` composed in front of mflux's ``Transformer`` (imports mflux)."""
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    return type("SeamTransformer", (SeamMixin, Transformer), {})


# --- checkpoint side ---------------------------------------------------------------------------------


def load_resident_set(
    ckpt: DF11Checkpoint, names: Iterable[str] | None = None
) -> dict[str, MxGroup]:
    """Load every group (or the named ones) as evaluated ``MxGroup``s: the resident set of a run.

    Raises:
        RigError: A requested name is not a group of the checkpoint.
    """
    wanted = list(ckpt.groups) if names is None else list(names)
    missing = [n for n in wanted if n not in ckpt.groups]
    if missing:
        raise RigError(f"not groups of the checkpoint: {missing}")
    return {name: load_group_mx(ckpt.groups[name]) for name in wanted}


def extras_plan(
    ckpt: DF11Checkpoint, *, n_double: int, n_single: int
) -> list[tuple[str, Path, TensorInfo]]:
    """The extras to load as (mflux parameter name, file, tensor info), sorted by DF11 name.

    Block extras beyond ``n_double`` / ``n_single`` and ``DROPPED_EXTRAS`` are left out.
    """
    plan: list[tuple[str, Path, TensorInfo]] = []
    for name, (path, info) in sorted(ckpt.extras.items()):
        if name in DROPPED_EXTRAS:
            continue
        parsed = split_block_name(name)
        if parsed is not None:
            kind, idx, _rest = parsed
            if idx >= (n_double if kind == DOUBLE_PREFIX else n_single):
                continue
        plan.append((mflux_param_name(name), path, info))
    return plan


def read_extra(path: Path, info: TensorInfo) -> mx.array:
    """Read one BF16 extra as a bf16 ``mx.array``.

    Raises:
        RigError: The tensor is not BF16.
    """
    if info.dtype != "BF16":
        raise RigError(f"{info.name}: extra is {info.dtype}, expected BF16")
    return mx.array(np.ascontiguousarray(read_array(path, info))).view(mx.bfloat16)


def check_extras_cover(
    param_paths: Iterable[str], planned: Iterable[str], matrix_paths: Iterable[str]
) -> None:
    """Every non-matrix parameter gets exactly one extra, and every planned extra has a parameter.

    Raises:
        RigError: A parameter nobody sets, or an extra with no target.
    """
    params, extras, matrices = set(param_paths), set(planned), set(matrix_paths)
    missing = sorted(params - matrices - extras)
    unexpected = sorted(extras - params)
    if missing or unexpected:
        raise RigError(
            f"extras do not cover the transformer: parameters without an extra {missing}; "
            f"extras without a parameter {unexpected}"
        )


def build_transformer(
    model: Literal["schnell", "dev"],
    ckpt: DF11Checkpoint,
    *,
    n_double: int = 19,
    n_single: int = 38,
) -> tuple[Any, Shapes]:
    """Construct mflux's ``Transformer`` (seamed) with placeholders for the matrices, extras loaded.

    Returns the transformer and the block shapes for ``attach``. Every block matrix is
    ``PLACEHOLDER`` and every other parameter comes from ``ckpt.extras``; the coverage of both is
    asserted, and so is the active MLX memory at the end (under ``MAX_BUILD_ACTIVE_BYTES``), before
    the caller loads the DF11 resident set.

    Raises:
        RigError: Unknown model, an uncovered parameter, an extra without a target or with the
            wrong shape, or too much active memory after the build.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mlx.utils import tree_flatten

    factories: dict[str, Callable[[], Any]] = {
        "schnell": ModelConfig.schnell,
        "dev": ModelConfig.dev,
    }
    if model not in factories:
        raise RigError(f"unknown model {model!r}; choose from {sorted(factories)}")
    transformer = seam_transformer_class()(
        factories[model](), num_transformer_blocks=n_double, num_single_transformer_blocks=n_single
    )
    shapes = install_placeholders(transformer)
    matrix_paths = {f"{block}.{attr}.weight" for block, per in shapes.items() for attr in per}
    params = dict(tree_flatten(transformer.parameters()))
    plan = extras_plan(ckpt, n_double=n_double, n_single=n_single)
    check_extras_cover(params, (name for name, _path, _info in plan), matrix_paths)
    weights: list[tuple[str, mx.array]] = []
    for name, path, info in plan:
        array = read_extra(path, info)
        if tuple(array.shape) != tuple(params[name].shape):
            raise RigError(
                f"{name}: extra has shape {tuple(array.shape)}, parameter {tuple(params[name].shape)}"
            )
        weights.append((name, array))
    transformer.load_weights(weights, strict=False)
    # Only what this function put in place: the extras and the placeholder. Never the whole
    # parameter tree, which would materialise anything a future mflux leaves as lazy random init.
    mx.eval([array for _name, array in weights], PLACEHOLDER)
    active = int(mx.get_active_memory())
    if active >= MAX_BUILD_ACTIVE_BYTES:
        raise RigError(
            f"{active / 1024**3:.2f} GiB active after the build; a block matrix is probably resident"
        )
    return transformer, shapes


__all__ = [
    "DOUBLE_MAP",
    "DROPPED_EXTRAS",
    "EVAL_POLICIES",
    "FLUX_CACHE_LIMIT",
    "MAX_BUILD_ACTIVE_BYTES",
    "MAX_NONE_POLICY_LAUNCHING_BLOCKS",
    "PLACEHOLDER",
    "SINGLE_MAP",
    "BlockEvent",
    "DF11Provider",
    "EvalPolicy",
    "PrefetchProvider",
    "ResidentProvider",
    "ReuseProvider",
    "RigError",
    "SeamMixin",
    "Tracer",
    "WeightProvider",
    "build_transformer",
    "check_extras_cover",
    "decode_resident",
    "extras_plan",
    "get_attr_path",
    "install_placeholders",
    "load_resident_set",
    "mflux_param_name",
    "mflux_path",
    "read_extra",
    "seam_transformer_class",
    "set_attr_path",
    "split_block_name",
    "summarize_trace",
]
