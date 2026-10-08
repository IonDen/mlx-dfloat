"""mflux's Qwen-Image 2.1 transformer with the class-swap seam and its forward pass kept uncompiled.

mflux 0.20.0 ``Qwen21Transformer.__call__`` compiles its ``_forward`` on the first call, on every chip, and keeps it
in ``_step_fn``. The block seam evaluates at block boundaries, which cannot happen inside ``mx.compile``, so
``EagerForward`` sets ``_step_fn`` to the plain bound ``_forward`` first and refuses any other value.

The build installs placeholders on the seven block matrices and on ``modulation.1`` (a non-block group, decoded once
when the set loads), loads every other parameter from the checkpoint's extras (or a BF16 base's) and leaves the
parameters mflux computes at construction as they are.
"""

import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_dfloat._safetensors import TensorInfo
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.blockseam import StepSeam, seam_blocks
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

# Parameters mflux computes when it builds the transformer; no checkpoint holds them, so the extras coverage leaves
# them out: the RoPE tables (mflux 0.20.0 qwen21_rope.py:21-26) and the timestep frequencies
# (qwen21_time_text_embed.py:16).
COMPUTED_PARAMS: frozenset[str] = frozenset(
    {
        "pos_embed.cos_tables.0",
        "pos_embed.cos_tables.1",
        "pos_embed.cos_tables.2",
        "pos_embed.sin_tables.0",
        "pos_embed.sin_tables.1",
        "pos_embed.sin_tables.2",
        "time_text_embed.time_proj.freqs",
    }
)


def is_eager(transformer: Any) -> bool:
    """Whether the transformer's ``_step_fn`` is unset or its own plain, uncompiled ``_forward`` (imports mflux).

    ``mx.compile`` returns ``mlx.gc_func``, a ``FunctionType`` subclass, so only an exact type check tells a compiled
    function from a bound method.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    fn = transformer._step_fn
    return fn is None or (
        type(fn) is types.MethodType
        and fn.__func__ is Qwen21Transformer._forward
        and fn.__self__ is transformer
    )


class EagerForward:
    """Compose in front of ``Qwen21Transformer``: its forward pass runs as the plain method, never compiled."""

    _step_fn: Any
    _forward: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Set ``_step_fn`` to the plain ``_forward`` when unset, then run mflux's own call.

        Raises:
            DFloatIntegrationError: ``_step_fn`` is set to anything else (a compiled forward pass).
        """
        if self._step_fn is None:
            self._step_fn = self._forward
        elif not is_eager(self):
            fn = self._step_fn
            raise DFloatIntegrationError(
                f"Qwen21Transformer._step_fn is {type(fn).__module__}.{type(fn).__qualname__}: mflux compiled the "
                "forward pass, and the block seam cannot evaluate inside mx.compile"
            )
        return super().__call__(*args, **kwargs)  # type: ignore[misc]


@cache
def seam_transformer_class() -> type:
    """``StepSeam`` and ``EagerForward`` composed in front of mflux's ``Qwen21Transformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    return type("SeamQwen21Transformer", (StepSeam, EagerForward, Qwen21Transformer), {})


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
    (``COMPUTED_PARAMS``) are left out of the coverage check and keep their values. The active memory the build added
    is asserted.

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
    "MAX_BUILD_ACTIVE_BYTES",
    "EagerForward",
    "Qwen21Build",
    "base_extras",
    "base_transformer_files_index",
    "block_lists",
    "build_transformer",
    "is_eager",
    "seam_transformer_class",
]
