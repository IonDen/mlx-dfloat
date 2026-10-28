"""mflux's Qwen-Image 2.1 transformer with the block seam, its forward passes uncompiled and its projections packed per block.

mflux 0.23.0 ``Qwen21Transformer.__call__`` runs, for a prompt without padding, the text-prefix cache: each block's
``text_forward`` once per prompt, then each block's ``image_forward`` every step, compiled into ``_image_step_fn``;
padded prompts run the joint ``_forward`` compiled into ``_step_fn``. The block seam evaluates at block boundaries,
which cannot happen inside ``mx.compile``, so ``EagerForward`` sets both step functions to the plain bound methods
first and refuses any other value, and every block entry point (``__call__``, ``text_forward``, ``image_forward``)
runs through the seam.

mflux 0.23.0 ``inference_projections()`` packs every block's Q/K/V and gate/up into one matrix each, once per
generation, and the blocks then run one matmul against it; on this machine that changes output bits against the
separate matmuls at the text-prefix shapes. Our blocks hold placeholders until their decode, so ``PackPerBlock``
turns packing on for the window and each block packs its decoded weights with mflux's own ``_pack_linears`` for its
call, then drops the packed matrices.

The build installs placeholders on the seven block matrices and on ``modulation.1`` (a non-block group, decoded once
when the set loads), loads every other parameter from the checkpoint's extras (or a BF16 base's) and leaves the
parameters mflux computes at construction as they are.
"""

import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_dfloat._safetensors import TensorInfo
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.blockseam import SeamCell, StepSeam
from mlx_dfloat.integrate.coverage import check_extras_cover, extras_plan, load_extras
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.resident import NonBlockShapes, install_nonblock_placeholders
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux2.transformer import base_transformer_files_index
from mlx_dfloat.mflux.qwen21.names import (
    DEFAULT_LAYERS,
    KIND,
    NONBLOCK_GROUPS,
    RUN_ORDER,
    check_qwen21_groups,
    qwen21_name_map,
)

MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3


_BINDING = "_dfloat_qwen21_block"
_SWITCH = "_dfloat_pack_switch"
# The block methods mflux 0.23's transformer calls: ``__call__`` (the joint pass, padded prompts or the text cache
# off), ``text_forward`` (the text-prefix K/V cache, built once per prompt) and ``image_forward`` (every step after).
ENTRY_POINTS: tuple[str, ...] = ("__call__", "text_forward", "image_forward")

Packer = Callable[[tuple[Any, ...]], Any]


class PackSwitch:
    """Whether the blocks pack their projections: on inside ``inference_projections()`` (shared, never a parameter)."""

    __slots__ = ("on",)

    def __init__(self) -> None:
        """Start off: outside ``inference_projections()`` mflux runs the separate projections."""
        self.on = False


@dataclass(frozen=True, slots=True)
class Qwen21Binding:
    """A seamed block's name, the seam cell it reads its state from, the packing switch and the packer."""

    name: str
    cell: SeamCell
    switch: PackSwitch
    pack: Packer


def _pack(block: Any, pack: Packer) -> None:
    """Pack the block's decoded weights as mflux 0.23 ``inference_projections()`` packs them, one block at a time."""
    attention, mlp = block.attn, block.img_mlp
    attention._qkv_weight = pack((attention.to_q, attention.to_k, attention.to_v))
    mlp._gate_proj_weight = pack((mlp.gate_layer, mlp.proj))


def _unpack(block: Any) -> None:
    """Drop the packed matrices (mflux 0.23 ``_clear_inference_projections`` per block)."""
    block.attn._qkv_weight = None
    block.img_mlp._gate_proj_weight = None


class Qwen21BlockSeam:
    """Placed in front of mflux's block class: every entry point runs through the seam, packed when switched on."""

    __slots__ = ()

    def _dfloat_run(self, entry: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        binding: Qwen21Binding = vars(self)[_BINDING]
        state = binding.cell.state
        if state is None:
            raise DFloatIntegrationError(
                f"{binding.name}: call attach(provider, shapes) before running a step"
            )
        own = getattr(super(), entry)

        def run() -> Any:
            # The weights are assigned by now; the packed matrices hold copies of them and go with the block's call.
            try:
                if binding.switch.on:
                    _pack(self, binding.pack)
                return own(*args, **kwargs)
            finally:
                _unpack(self)

        return seam.run_block(state, binding.name, self, run, inputs=(args, kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """The joint pass through the seam."""
        return self._dfloat_run("__call__", args, kwargs)

    def text_forward(self, *args: Any, **kwargs: Any) -> Any:
        """The text-prefix pass through the seam."""
        return self._dfloat_run("text_forward", args, kwargs)

    def image_forward(self, *args: Any, **kwargs: Any) -> Any:
        """The image-token pass through the seam."""
        return self._dfloat_run("image_forward", args, kwargs)


@cache
def seamed_block_class(cls: type) -> type:
    """``Qwen21BlockSeam`` composed in front of ``cls`` (one class per block class)."""
    return type(
        f"Seam{cls.__name__}", (Qwen21BlockSeam, cls), {"__slots__": (), "__module__": __name__}
    )


def seam_qwen21_blocks(
    block_lists: Sequence[tuple[str, Sequence[Any]]],
    cell: SeamCell,
    switch: PackSwitch,
    *,
    pack: Packer,
) -> tuple[str, ...]:
    """Swap every block's class for its seamed class and bind it as ``<kind>.<idx>``; the names in run order.

    ``pack`` packs a tuple of ``nn.Linear`` layers into one matrix or returns None (mflux 0.23's
    ``Qwen21Transformer._pack_linears``); a block packs only while ``switch.on``.

    Raises:
        DFloatIntegrationError: A block is already seamed.
    """
    names: list[str] = []
    for kind, blocks in block_lists:
        for idx, block in enumerate(blocks):
            name = f"{kind}.{idx}"
            if isinstance(block, Qwen21BlockSeam):
                raise DFloatIntegrationError(f"{name}: already seamed")
            block_cls: type = type(block)
            block.__class__ = seamed_block_class(block_cls)
            vars(block)[_BINDING] = Qwen21Binding(name=name, cell=cell, switch=switch, pack=pack)
            names.append(name)
    return tuple(names)


# The tables mflux computes when it builds the transformer, which no checkpoint holds: the RoPE tables, one per axis
# (mflux 0.23.0 qwen21_rope.py:21-26, private lists), and the timestep frequencies (qwen21_time_text_embed.py:16, a
# private array). MLX leaves underscore attributes out of ``parameters()``, so the extras coverage never sees them
# and the build leaves them as mflux built them; one that turned into a parameter again would be refused as
# uncovered.
COMPUTED_TABLES: tuple[str, ...] = (
    "pos_embed._cos_tables",
    "pos_embed._sin_tables",
    "time_text_embed.time_proj._freqs",
)

# The parameters mflux computes at construction, as krea2 and ernie export them: none here, the computed tables above
# are not parameters.
COMPUTED_PARAMS: frozenset[str] = frozenset()


def computed_tables(transformer: Any) -> dict[str, mx.array]:
    """The computed tables by path, a list's entries as ``<path>.<i>``.

    Raises:
        DFloatIntegrationError: A table is missing or is neither an array nor a list of arrays (mflux moved it).
    """
    tables: dict[str, mx.array] = {}
    for path in COMPUTED_TABLES:
        owner_path, _, attr = path.rpartition(".")
        owner = transformer
        for part in owner_path.split("."):
            owner = getattr(owner, part, None)
        value = getattr(
            owner, attr, None
        )  # nn.Module keeps arrays and lists as its dict items, not in __dict__
        items = value if isinstance(value, list) else [value]
        if value is None or not items or not all(isinstance(t, mx.array) for t in items):
            raise DFloatIntegrationError(f"{path}: not a computed table on this mflux transformer")
        if isinstance(value, list):
            tables.update({f"{path}.{i}": t for i, t in enumerate(value)})
        else:
            tables[path] = value
    return tables


# The compiled step functions mflux 0.23 ``Qwen21Transformer.__call__`` builds on first use, and the plain methods
# they compile: ``_step_fn`` the joint pass (padded prompts, or ``use_text_cache`` off), ``_image_step_fn`` the
# image-token pass behind the text-prefix cache (the default).
STEP_FNS: tuple[tuple[str, str], ...] = (
    ("_step_fn", "_forward"),
    ("_image_step_fn", "_image_forward"),
)


def _plain(transformer: Any, fn: Any, method: str) -> bool:
    """Whether ``fn`` is unset or ``transformer``'s own plain ``method``.

    ``mx.compile`` returns ``mlx.gc_func``, a
    ``FunctionType`` subclass, so only an exact type check tells a compiled function from a bound method.
    """
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    return fn is None or (
        type(fn) is types.MethodType
        and fn.__func__ is getattr(Qwen21Transformer, method)
        and fn.__self__ is transformer
    )


def is_eager(transformer: Any) -> bool:
    """Whether both step functions are unset or the transformer's own plain, uncompiled methods (imports mflux)."""
    require_mflux()
    return all(_plain(transformer, getattr(transformer, attr), method) for attr, method in STEP_FNS)


def refuse_compiled(transformer: Any) -> None:
    """Refuse a step function that is set and is not the transformer's own plain method (imports mflux).

    Raises:
        DFloatIntegrationError: A step function is set to anything else (a compiled forward pass).
    """
    require_mflux()
    for attr, method in STEP_FNS:
        fn = getattr(transformer, attr)
        if not _plain(transformer, fn, method):
            raise DFloatIntegrationError(
                f"Qwen21Transformer.{attr} is {type(fn).__module__}.{type(fn).__qualname__}: mflux compiled the "
                "forward pass, and the block seam cannot evaluate inside mx.compile"
            )


class EagerForward:
    """Compose in front of ``Qwen21Transformer``: both forward passes run as the plain methods, never compiled."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse a compiled step function, set each unset one to its plain method, then run mflux's own call.

        Raises:
            DFloatIntegrationError: A step function is set to anything else (a compiled forward pass).
        """
        refuse_compiled(self)
        for attr, method in STEP_FNS:
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(self, method))
        return super().__call__(*args, **kwargs)  # type: ignore[misc]


class PackPerBlock:
    """Compose in front of ``Qwen21Transformer``: ``inference_projections()`` packs per block, at each block's decode.

    mflux 0.23 packs every block's Q/K/V and gate/up once on entry; here the blocks hold placeholders, so entry only
    turns the blocks' packing on (when ``use_packed_projections`` is set, as mflux reads it on entry) and each seamed
    block packs its decoded weights with mflux's own ``_pack_linears`` for its call. Entry and exit clear what mflux
    clears (packed matrices, step functions, text caches).

    On mflux 0.23 ``_pack_linears`` evaluates the packed matrices synchronously inside each block, so the ``depth2``
    policy's look-ahead decode does not overlap the block before it: ``depth2`` runs as ``per-block`` here (not
    measured).
    """

    use_packed_projections: bool
    _clear_inference_projections: Callable[[], None]

    @property
    def pack_switch(self) -> PackSwitch:
        """The switch this transformer's blocks read (created on first use)."""
        switch = vars(self).get(_SWITCH)
        if switch is None:
            switch = PackSwitch()
            vars(self)[_SWITCH] = switch
        return switch

    @contextmanager
    def inference_projections(self) -> Iterator[None]:
        """The inference-only window of mflux 0.23, with the packing deferred to each block's decode.

        Raises:
            DFloatIntegrationError: A step function is compiled on entry, before mflux's clear would hide it.
        """
        refuse_compiled(self)
        self._clear_inference_projections()
        self.pack_switch.on = bool(self.use_packed_projections)
        try:
            yield
        finally:
            self.pack_switch.on = False
            self._clear_inference_projections()


def mflux_packer() -> Packer:
    """The packer of mflux 0.23 itself, ``Qwen21Transformer._pack_linears`` (imports mflux)."""
    require_mflux()
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    pack: Packer = Qwen21Transformer._pack_linears
    return pack


@cache
def seam_transformer_class() -> type:
    """``StepSeam``, ``EagerForward`` and ``PackPerBlock`` in front of mflux's ``Qwen21Transformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    return type(
        "SeamQwen21Transformer", (StepSeam, EagerForward, PackPerBlock, Qwen21Transformer), {}
    )


def block_lists(transformer: Any) -> list[tuple[str, Sequence[Any]]]:
    """The one block list, the one ``_forward`` iterates."""
    return [(kind, getattr(transformer, kind)) for kind in RUN_ORDER]


@dataclass(frozen=True, slots=True, kw_only=True)
class Qwen21Build:
    """A built transformer, its block shapes, the non-block matrices' shapes and the block count built."""

    transformer: Any
    shapes: Shapes
    nonblock: NonBlockShapes
    counts: dict[str, int]


def build_transformer(
    ckpt: DF11Checkpoint,
    *,
    transformer_kwargs: Mapping[str, Any] | None = None,
    name_map: NameMap | None = None,
    n_layers: int | None = None,
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
    nonblock_from_extras: bool = False,
    transformer_class: Callable[..., Any] | None = None,
) -> Qwen21Build:
    """Construct the seamed transformer: block and non-block matrices as placeholders, every other parameter loaded.

    ``transformer_kwargs`` are the transformer's constructor arguments (none for the published model); the
    checkpoint's block count must equal the depth they build (mflux's default 32 when they leave it out).
    ``n_layers`` then builds fewer blocks (a partial build for tests). ``extras`` replaces the checkpoint's own extras
    (a BF16 base's tensors, via ``base_extras``); with ``nonblock_from_extras`` ``modulation.1`` loads from them as a
    plain weight instead of a placeholder (the BF16 side of an identity check). A checkpoint that stores
    ``modulation.1`` as an extra loads it as a plain weight either way. The parameters mflux computes at construction
    (``COMPUTED_TABLES``) are not parameters, so the build never loads them. The active memory the build added is
    asserted.

    Raises:
        DFloatFormatError: The groups are not Qwen-Image 2.1's; the checkpoint's block count differs from the
            model's depth; an extra is not BF16 or has the wrong shape.
        DFloatIntegrationError: ``n_layers`` out of range, an uncovered parameter or an extra without one, or too
            much active memory added.
        DFloatDependencyError: The ``mflux`` extra is missing (default class and map only).
    """
    names = qwen21_name_map() if name_map is None else name_map
    counts = check_qwen21_groups(ckpt)
    kwargs = dict(transformer_kwargs or {})
    want = int(kwargs.get("num_layers", DEFAULT_LAYERS))
    if counts[KIND] != want:
        raise DFloatFormatError(f"checkpoint has {counts[KIND]} blocks; this model builds {want}")
    n = counts[KIND] if n_layers is None else n_layers
    if not 0 <= n <= counts[KIND]:
        raise DFloatIntegrationError(f"asked for {n} blocks; the checkpoint has {counts[KIND]}")
    make = seam_transformer_class() if transformer_class is None else transformer_class
    before = int(mx.get_active_memory())
    tf = make(**{**kwargs, "num_layers": n})
    lists = block_lists(tf)
    shapes = install_placeholders(lists, names)
    seam_qwen21_blocks(lists, tf.seam_cell, tf.pack_switch, pack=mflux_packer())
    nonblock_matrices = (
        []
        if nonblock_from_extras
        else [m for g in NONBLOCK_GROUPS if g in ckpt.groups for m in ckpt.groups[g].matrix_names]
    )
    nonblock = install_nonblock_placeholders(tf, nonblock_matrices, names)
    matrix_paths = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per} | set(nonblock)
    built = {KIND: n}
    plan = extras_plan(ckpt, names, counts=built, extras=extras)
    params = set(dict(tree_flatten(tf.parameters())))
    check_extras_cover(params, (p for p, _f, _i in plan), matrix_paths)
    load_extras(tf, plan, names)
    mx.eval(PLACEHOLDER)
    added = int(mx.get_active_memory()) - before
    if added >= MAX_BUILD_ACTIVE_BYTES:
        raise DFloatIntegrationError(f"extras added {added / 1024**3:.2f} GiB of active memory")
    return Qwen21Build(transformer=tf, shapes=shapes, nonblock=nonblock, counts=built)


def base_extras(
    index: Mapping[str, tuple[Path, TensorInfo]], ckpt: DF11Checkpoint
) -> dict[str, tuple[Path, TensorInfo]]:
    """The BF16 base's tensors minus the block groups' matrices, by their decoded names (``modulation.1`` stays).

    The block groups' ``matrix_names`` are the decoded ones (``img_mlp.gate_layer``, ``img_mlp.proj``), which is how
    the base names them, so a fused stored matrix never leaves its halves behind as extras.
    """
    block = {
        m for name, g in ckpt.groups.items() if name not in NONBLOCK_GROUPS for m in g.matrix_names
    }
    return {name: entry for name, entry in index.items() if name not in block}


__all__ = [
    "COMPUTED_PARAMS",
    "COMPUTED_TABLES",
    "ENTRY_POINTS",
    "MAX_BUILD_ACTIVE_BYTES",
    "STEP_FNS",
    "EagerForward",
    "PackPerBlock",
    "PackSwitch",
    "Qwen21Binding",
    "Qwen21BlockSeam",
    "Qwen21Build",
    "base_extras",
    "base_transformer_files_index",
    "block_lists",
    "build_transformer",
    "computed_tables",
    "is_eager",
    "mflux_packer",
    "refuse_compiled",
    "seam_qwen21_blocks",
    "seam_transformer_class",
    "seamed_block_class",
]
