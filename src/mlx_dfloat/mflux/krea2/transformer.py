"""mflux's Krea 2 transformer with the class-swap seam.

mflux 0.20.0 ``Krea2Transformer.__call__`` runs its one block list, ``blocks``, inline, after the text fusion, the
timestep MLP and the text MLP have run once. The pipeline's ``_predict`` factory compiles the step function off base
and Pro M1/M2 chips (an M1 Max or Ultra, and every M3 or later, compile); the model runs it through
``mlx_dfloat.mflux._compile.uncompiled`` on every chip, because the block seam evaluates at block boundaries, which
cannot happen inside ``mx.compile``. A classifier-free guidance step is two transformer calls, so each block decodes
twice per step.

The build installs placeholders on the eight block matrices and on the seven non-block groups' matrices (the four
text-fusion blocks, ``tmlp``, ``tproj``, ``txtmlp``) and loads every other parameter from the checkpoint's extras (or a
BF16 base's). The non-block groups run once per call, before the blocks. They are either decoded once when the set
loads and kept installed (the resident class), or decoded at the start of each transformer call
(``seam_transformer_class(per_call=True)``, ``PerCallNonBlock``) at the cost of decoding them twice per guided step.

The per-call path empties MLX's buffer cache at two points, without changing any arithmetic: at the start of each
call (``_release_and_drain``), and before block 0's decode, once block 0's inputs are evaluated and the non-block
weights are back on their placeholders. Otherwise a guided step's second call would decode the non-block groups and
build its pre-block graph on top of the first call's freed activations, which MLX keeps cached until active plus
cached memory reaches the memory limit. Both points wait for the OS footprint to fall, because a released Metal
buffer leaves it only after the next GPU submission. At 1024² this keeps a guided Krea 2 Raw step's denoise peak at
21.14 GiB.
"""

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_dfloat._safetensors import TensorInfo, read_array, read_header
from mlx_dfloat._watchdog import phys_footprint
from mlx_dfloat.decode import DecodeResult
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint, MxGroup
from mlx_dfloat.integrate.blockseam import StepSeam, seam_blocks
from mlx_dfloat.integrate.coverage import check_extras_cover, extras_plan, load_extras
from mlx_dfloat.integrate.names import BlockShapes, NameMap, Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.providers import WeightProvider, read_bf16
from mlx_dfloat.integrate.resident import (
    NonBlockShapes,
    clear_nonblock,
    decode_nonblock,
    install_nonblock,
    install_nonblock_placeholders,
)
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux2.transformer import base_transformer_files_index
from mlx_dfloat.mflux.krea2.names import (
    DEFAULT_LAYERS,
    KIND,
    NONBLOCK_GROUPS,
    RUN_ORDER,
    check_krea2_groups,
    check_variant,
    krea2_name_map,
)

MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3

# mflux 0.20.0's Krea2Transformer builds no parameter of its own: the RoPE embedder holds no arrays, the timestep
# frequencies are computed per call, and the modulation tables and norm scales come from the checkpoint. Kept as a
# named constant so the build reads like the other families' and a future mflux table has one place to go.
COMPUTED_PARAMS: frozenset[str] = frozenset()


# The policies under which a call releases the non-block weights before block 0's decode. "none" promises no
# evaluation inside the call, so under it they stay installed until the call's own cleanup.
RELEASE_POLICIES: frozenset[str] = frozenset({"per-block", "depth2"})
_BINDING = "_dfloat_nonblock"

# The wrapper calls MLX through these names so a test can record the order of the evaluation and the drain.
_eval = mx.eval
_clear_cache = mx.clear_cache

# The drain: MLX's cache, the OS footprint and the clock are read through these names so a test can script them.
_synchronize = mx.synchronize
_cache_memory = mx.get_cache_memory
_footprint = phys_footprint
_clock = time.perf_counter
_sleep = time.sleep
# Wait for the footprint to fall by this fraction of the released bytes, at most this long, polling at this period.
DRAIN_FRACTION = 0.9
DRAIN_TIMEOUT_S = 0.25
DRAIN_POLL_S = 0.0005
# Below this many released bytes the drain clears and kicks but does not wait: a small release cannot stack into a
# peak, and a few MB may not show in the footprint at all (the wait would run to its cap).
DRAIN_MIN_BYTES = 64 * 1024**2


def _kick() -> None:
    """One tiny GPU evaluation: the driver frees released buffers after the next command submission."""
    mx.eval(mx.zeros((1,)) + 0)


def _release_and_drain() -> bool:
    """Empty MLX's buffer cache and wait until the OS footprint has dropped; touches no arithmetic.

    A buffer MLX releases leaves the process footprint only after the next GPU submission (about 20 ms with the GPU
    busy, 80-700 ms idle), so an allocation made in that window stacks on it. The drain waits for the GPU first, so
    every buffer whose free was still pending (MLX frees a buffer the GPU used from the command's completion handler)
    has reached the cache. Then it reads the cached bytes and the footprint, clears the cache, submits one tiny
    evaluation, and polls the footprint until it has fallen by ``DRAIN_FRACTION`` of the released bytes or
    ``DRAIN_TIMEOUT_S`` has passed. The tiny evaluation's own buffers reach the cache once its command completes, so
    the drain ends by waiting for the GPU again and clearing the cache once more. Returns whether the drop was seen (a
    release under ``DRAIN_MIN_BYTES`` counts as seen without a wait).
    """
    _synchronize()
    released = int(_cache_memory())
    waits = released >= DRAIN_MIN_BYTES
    before = _footprint() if waits else 0
    _clear_cache()
    _kick()
    dropped = True
    if waits:
        deadline = _clock() + DRAIN_TIMEOUT_S
        while before - _footprint() < DRAIN_FRACTION * released:
            if _clock() >= deadline:
                dropped = False
                break
            _sleep(DRAIN_POLL_S)
    _synchronize()
    _clear_cache()
    return dropped


class _NonBlockBinding:
    """The bound non-block groups and how to decode them (a plain object: a dict attribute would join the module tree)."""

    __slots__ = ("decode", "groups", "name_map", "shapes")

    def __init__(
        self,
        groups: Mapping[str, MxGroup],
        shapes: NonBlockShapes,
        name_map: NameMap,
        decode: Callable[[MxGroup], DecodeResult] | None,
    ) -> None:
        self.groups = dict(groups)
        self.shapes = dict(shapes)
        self.name_map = name_map
        self.decode = decode


class ReleaseAtFirstBlock:
    """Wraps the seam's provider for one call and releases the non-block weights before block 0's decode.

    Through the seam's ``before_block`` hook, at block 0: evaluate the block's inputs (the whole pre-block graph, so
    nothing lazy still references a decoded non-block array), put the non-block weights back on their placeholders,
    empty MLX's buffer cache and wait for the footprint to fall (``_release_and_drain``: a released buffer otherwise
    stays in the cache, and so in the footprint), and only then let block 0 request its weights. Everything else is
    the wrapped provider's. Under a policy outside ``RELEASE_POLICIES`` (``"none"``) nothing is released mid-call; the
    call's own cleanup does it.
    """

    def __init__(
        self, inner: WeightProvider, transformer: Any, shapes: NonBlockShapes, *, policy: str
    ) -> None:
        """Wrap ``inner`` for one call of ``transformer`` under ``policy``."""
        self._inner = inner
        self._transformer = transformer
        self._shapes = shapes
        self._releases = policy in RELEASE_POLICIES
        self._released = False
        self.launching = inner.launching
        self.policies = inner.policies

    @property
    def launches(self) -> int:
        """The wrapped provider's launch count."""
        return self._inner.launches

    @launches.setter
    def launches(self, value: int) -> None:
        self._inner.launches = value

    @property
    def pending(self) -> Any:
        """The wrapped provider's unchecked status words (None when it keeps none), for the seam's ``begin_step``."""
        return getattr(self._inner, "pending", None)

    def before_block(
        self, block_name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> None:
        """At the call's first block: evaluate its inputs, release the non-block weights, clear the cache."""
        if self._released or not self._releases:
            return
        flat: dict[str, Any] = dict(tree_flatten((args, dict(kwargs))))
        _eval(*(leaf for leaf in flat.values() if isinstance(leaf, mx.array)))
        clear_nonblock(self._transformer, self._shapes)
        _release_and_drain()
        self._released = True

    def weights_for(self, block_name: str, shapes: BlockShapes) -> dict[str, mx.array]:
        """The wrapped provider's weights."""
        return self._inner.weights_for(block_name, shapes)

    def verify(self) -> None:
        """The wrapped provider's deferred checks."""
        self._inner.verify()

    def reset(self) -> None:
        """The wrapped provider's reset."""
        self._inner.reset()


class PerCallNonBlock(StepSeam):
    """``StepSeam`` whose transformer decodes the bound non-block groups at the start of each call.

    Each call first empties MLX's buffer cache and waits for the footprint to fall (``_release_and_drain``), so it
    never starts on the previous call's cached activations. The decoded weights are installed for the call's pre-block
    graph and put back on their placeholders before block 0 requests its own weights, once that graph is evaluated
    (``ReleaseAtFirstBlock``), and again when the call ends however it ends. The call evaluates its output before it returns, so no lazy tail keeps this call's arrays alive
    into the next call of a guided step.
    """

    def bind_nonblock(
        self,
        groups: Mapping[str, MxGroup],
        shapes: NonBlockShapes,
        name_map: NameMap,
        *,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        """Keep the compressed non-block groups to decode per call (Metal by default); ``detach`` forgets them."""
        vars(self)[_BINDING] = _NonBlockBinding(groups, shapes, name_map, decode)

    def nonblock_bound(self) -> bool:
        """Whether non-block groups are bound."""
        return vars(self).get(_BINDING) is not None

    def detach(self) -> None:
        """Forget the provider, the shapes and the bound non-block groups (the set is dropping)."""
        super().detach()
        vars(self).pop(_BINDING, None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Decode and install the non-block weights, run the seam step with the release wrapper, evaluate, clear them.

        Raises:
            DFloatIntegrationError: No non-block groups are bound, or nothing is attached.
            DFloatFormatError: A non-block group's decode reported an error.
        """
        binding: _NonBlockBinding | None = vars(self).get(_BINDING)
        if binding is None:
            raise DFloatIntegrationError(
                "call bind_nonblock(groups, shapes, name_map) before running a step"
            )
        state = self._state()
        inner = state.provider
        # The previous call's freed activations leave MLX's cache before this call decodes and builds its pre-block
        # graph, so every call follows a first call's trajectory instead of stacking on the last one's cache.
        _release_and_drain()
        try:
            matrices = {g: NONBLOCK_GROUPS[g] for g in binding.groups}
            weights = decode_nonblock(
                binding.groups,
                matrices,
                binding.shapes,
                binding.name_map,
                decode=binding.decode,
                eval_together=True,
            )
            install_nonblock(self, weights)
            del weights
            state.provider = ReleaseAtFirstBlock(inner, self, binding.shapes, policy=state.policy)
            out = super().__call__(*args, **kwargs)
            _eval(out)
            return out
        finally:
            state.provider = inner
            clear_nonblock(self, binding.shapes)


@cache
def _seam_class(per_call: bool) -> type:
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    if per_call:
        return type("SeamKrea2TransformerPerCall", (PerCallNonBlock, Krea2Transformer), {})
    return type("SeamKrea2Transformer", (StepSeam, Krea2Transformer), {})


def seam_transformer_class(*, per_call: bool = False) -> type:
    """``StepSeam`` composed in front of mflux's ``Krea2Transformer`` (imports mflux); one class per mode.

    ``per_call`` puts ``PerCallNonBlock`` in front instead: the non-block groups are bound with ``bind_nonblock`` and
    decoded at each call. Importing the class also registers the ``er_sde`` and ``euler`` schedulers (mflux registers
    them in its Krea 2 package).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    return _seam_class(per_call)


def block_lists(transformer: Any) -> list[tuple[str, Sequence[Any]]]:
    """The one block list the forward pass runs, ``blocks``."""
    return [(kind, getattr(transformer, kind)) for kind in RUN_ORDER]


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2Build:
    """A built transformer, its block shapes, the non-block matrices' shapes and the block count built."""

    transformer: Any
    shapes: Shapes
    nonblock: NonBlockShapes
    counts: dict[str, int]


def build_transformer(
    ckpt: DF11Checkpoint,
    *,
    model: str | None = None,
    transformer_kwargs: Mapping[str, Any] | None = None,
    name_map: NameMap | None = None,
    n_layers: int | None = None,
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
    nonblock_from_extras: bool = False,
    transformer_class: Callable[..., Any] | None = None,
) -> Krea2Build:
    """Construct the seamed transformer: block and non-block matrices as placeholders, every other parameter loaded.

    ``model`` (``"krea-2-raw"`` or ``"krea-2"``), when given, refuses the other model's published checkpoint before
    anything is built. ``transformer_kwargs`` are the transformer's constructor arguments (none for the published
    model); the checkpoint's block count must equal the depth they build (``layers``; mflux's default 28 when they
    leave it out). ``n_layers`` then builds fewer blocks (a partial build for tests). ``extras`` replaces the
    checkpoint's own extras (a BF16 base's tensors, via ``base_extras``); with ``nonblock_from_extras`` the non-block
    matrices load from them as plain weights instead of placeholders (the BF16 side of an identity check). A
    non-block group the checkpoint stores as extras loads as plain weights either way. The active memory the build
    added is asserted.

    Raises:
        DFloatFormatError: The other Krea 2 model's checkpoint; the groups are not Krea 2's; the checkpoint's block
            count differs from the model's depth; an extra is not BF16 or has the wrong shape.
        DFloatIntegrationError: ``n_layers`` out of range, an uncovered parameter or an extra without one, or too
            much active memory added.
        DFloatDependencyError: The ``mflux`` extra is missing (default class and map only).
    """
    if model is not None:
        check_variant(model, ckpt)
    counts = check_krea2_groups(ckpt)
    kwargs = dict(transformer_kwargs or {})
    want = int(kwargs.get("layers", DEFAULT_LAYERS))
    if counts[KIND] != want:
        raise DFloatFormatError(f"checkpoint has {counts[KIND]} blocks; this model builds {want}")
    n = counts[KIND] if n_layers is None else n_layers
    if not 0 <= n <= counts[KIND]:
        raise DFloatIntegrationError(f"asked for {n} blocks; the checkpoint has {counts[KIND]}")
    names = krea2_name_map() if name_map is None else name_map
    make = seam_transformer_class() if transformer_class is None else transformer_class
    before = int(mx.get_active_memory())
    tf = make(**{**kwargs, "layers": n})
    lists = block_lists(tf)
    shapes = install_placeholders(lists, names)
    seam_blocks(lists, tf.seam_cell)
    nonblock_matrices = (
        []
        if nonblock_from_extras
        else [m for g in NONBLOCK_GROUPS if g in ckpt.groups for m in ckpt.groups[g].matrix_names]
    )
    nonblock = install_nonblock_placeholders(tf, nonblock_matrices, names)
    matrix_paths = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per} | set(nonblock)
    built = {KIND: n}
    plan = extras_plan(ckpt, names, counts=built, extras=extras)
    params = set(dict(tree_flatten(tf.parameters()))) - COMPUTED_PARAMS
    check_extras_cover(params, (p for p, _f, _i in plan), matrix_paths)
    load_extras(tf, plan, names)
    mx.eval(PLACEHOLDER)
    added = int(mx.get_active_memory()) - before
    if added >= MAX_BUILD_ACTIVE_BYTES:
        raise DFloatIntegrationError(f"extras added {added / 1024**3:.2f} GiB of active memory")
    return Krea2Build(transformer=tf, shapes=shapes, nonblock=nonblock, counts=built)


def base_extras(
    index: Mapping[str, tuple[Path, TensorInfo]], ckpt: DF11Checkpoint
) -> dict[str, tuple[Path, TensorInfo]]:
    """The BF16 base's tensors minus the block groups' matrices (the non-block matrices stay, as plain weights)."""
    block = {
        m for name, g in ckpt.groups.items() if name not in NONBLOCK_GROUPS for m in g.matrix_names
    }
    return {name: entry for name, entry in index.items() if name not in block}


def base_bf16_weights(path: Path, names: Iterable[str]) -> dict[str, mx.array]:
    """The named tensors of a base's native single file (``raw.safetensors``) as BF16, as mflux loads them.

    A BF16 tensor is read as it is (a bit view); an FP32 one is cast with ``astype(mx.bfloat16)``, mflux's own load
    cast, which rounds to nearest even. The published Krea 2 files hold the transformer's matrices in BF16 and its
    extras and five non-block matrices in FP32.

    Raises:
        DFloatFormatError: A name the file does not hold, a tensor of another dtype, or an unreadable header.
    """
    header = read_header(path)
    out: dict[str, mx.array] = {}
    for name in names:
        info = header.get(name)
        if info is None:
            raise DFloatFormatError(f"{path.name} has no tensor {name!r}")
        if info.dtype == "BF16":
            out[name] = read_bf16(path, info)
        elif info.dtype == "F32":
            cast = mx.array(np.ascontiguousarray(read_array(path, info))).astype(mx.bfloat16)
            _eval(cast)  # one FP32 source alive at a time, not all of them until the caller's eval
            out[name] = cast
        else:
            raise DFloatFormatError(f"{name} is {info.dtype}; the base must hold BF16 or F32")
    return out


def install_base_weights(
    transformer: Any,
    ckpt: DF11Checkpoint,
    path: Path,
    name_map: NameMap,
    nonblock: NonBlockShapes,
) -> list[str]:
    """Replace every extra the transformer was built with, and every non-block matrix, with the base file's.

    The reference side of an image identity check then reads nothing of the DF11 file but its group shapes: the
    extras the build loaded from the checkpoint and the non-block placeholders (``nonblock``, the build's shapes) all
    take the base's tensors (``base_bf16_weights``); the block matrices stay placeholders for a streaming provider.
    Returns the checkpoint names replaced.

    Raises:
        DFloatFormatError: A base tensor missing, of another dtype, or of another shape than the model's parameter.
    """
    params: dict[str, Any] = dict(tree_flatten(transformer.parameters()))
    extras = [n for n in ckpt.extras if name_map.param_name(n) in params]
    matrices = [
        m
        for g in NONBLOCK_GROUPS
        if g in ckpt.groups
        for m in ckpt.groups[g].matrix_names
        if name_map.param_name(m) in nonblock
    ]
    weights = base_bf16_weights(path, [*extras, *matrices])
    plain: list[tuple[str, mx.array]] = []
    installed: dict[str, mx.array] = {}
    for name, weight in weights.items():
        param = name_map.param_name(name)
        want = nonblock[param] if param in nonblock else tuple(params[param].shape)
        if tuple(weight.shape) != tuple(want):
            raise DFloatFormatError(
                f"{name} has shape {tuple(weight.shape)}; the model needs {tuple(want)}"
            )
        if param in nonblock:
            installed[param] = weight
        else:
            plain.append((param, weight))
    transformer.load_weights(plain, strict=False)
    install_nonblock(transformer, installed)
    return [*extras, *matrices]


__all__ = [
    "COMPUTED_PARAMS",
    "MAX_BUILD_ACTIVE_BYTES",
    "RELEASE_POLICIES",
    "Krea2Build",
    "PerCallNonBlock",
    "ReleaseAtFirstBlock",
    "base_bf16_weights",
    "base_extras",
    "base_transformer_files_index",
    "block_lists",
    "build_transformer",
    "install_base_weights",
    "seam_transformer_class",
]
