"""mflux's ERNIE-Image transformer with the class-swap seam; the build from a DF11 checkpoint.

mflux 0.20.0 ``ErnieTransformer.__call__`` runs its one block list, ``layers``, inline. The pipeline's ``_predict``
factory compiles the step function off base and Pro M1/M2 chips; the model runs it through
``mlx_dfloat.mflux._compile.uncompiled`` on every chip, because the block seam evaluates at block boundaries, which
cannot happen inside ``mx.compile``.

The build installs placeholders on the seven block matrices and on the three non-block groups' four matrices
(``time_embedding``, ``adaLN_modulation.1``, ``final_norm.linear``; decoded once when the set loads) and loads every
other parameter from the checkpoint's extras (or a BF16 base's), the patch convolution transposed from torch layout.
"""

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
from mlx_dfloat.mflux.ernie.names import (
    DEFAULT_LAYERS,
    KIND,
    NONBLOCK_GROUPS,
    RUN_ORDER,
    check_ernie_groups,
    ernie_name_map,
)
from mlx_dfloat.mflux.flux2.transformer import base_transformer_files_index

MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3

# mflux 0.20.0's ErnieTransformer builds no parameter of its own: the RoPE embedder holds no arrays, the timestep
# frequencies are computed per call, and ``_pos_cache`` (an underscore attribute) is not a parameter. Kept as a named
# constant so the build reads like the other families' and a future mflux table has one place to go.
COMPUTED_PARAMS: frozenset[str] = frozenset()


@cache
def seam_transformer_class() -> type:
    """``StepSeam`` composed in front of mflux's ``ErnieTransformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    return type("SeamErnieTransformer", (StepSeam, ErnieTransformer), {})


def block_lists(transformer: Any) -> list[tuple[str, Sequence[Any]]]:
    """The one block list the forward pass runs, ``layers``."""
    return [(kind, getattr(transformer, kind)) for kind in RUN_ORDER]


@dataclass(frozen=True, slots=True, kw_only=True)
class ErnieBuild:
    """A built transformer, its block shapes, the non-block matrices' shapes and the block count built."""

    transformer: Any
    shapes: Shapes
    nonblock: NonBlockShapes
    counts: dict[str, int]


def build_transformer(
    ckpt: DF11Checkpoint,
    *,
    transformer_overrides: Mapping[str, Any] | None = None,
    name_map: NameMap | None = None,
    n_layers: int | None = None,
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
    nonblock_from_extras: bool = False,
    transformer_class: Callable[..., Any] | None = None,
) -> ErnieBuild:
    """Construct the seamed transformer: block and non-block matrices as placeholders, every other parameter loaded.

    ``transformer_overrides`` are the model's (``ModelConfig.transformer_overrides``); the checkpoint's block count
    must equal the depth they build (mflux's default 36 when they leave it out). ``n_layers`` then builds fewer
    blocks (a partial build for tests). ``extras`` replaces the checkpoint's own extras (a BF16 base's tensors, via
    ``base_extras``); with ``nonblock_from_extras`` the non-block matrices load from them as plain weights instead of
    placeholders (the BF16 side of an identity check). A non-block group the checkpoint stores as extras loads as
    plain weights either way. The active memory the build added is asserted.

    Raises:
        DFloatFormatError: The groups are not ERNIE-Image's; the checkpoint's block count differs from the model's
            depth; an extra is not BF16 or has the wrong shape.
        DFloatIntegrationError: ``n_layers`` out of range, an uncovered parameter or an extra without one, or too
            much active memory added.
        DFloatDependencyError: The ``mflux`` extra is missing (default class and map only).
    """
    names = ernie_name_map() if name_map is None else name_map
    counts = check_ernie_groups(ckpt)
    kwargs = dict(transformer_overrides or {})
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
    return ErnieBuild(transformer=tf, shapes=shapes, nonblock=nonblock, counts=built)


def base_extras(
    index: Mapping[str, tuple[Path, TensorInfo]], ckpt: DF11Checkpoint
) -> dict[str, tuple[Path, TensorInfo]]:
    """The BF16 base's tensors minus the block groups' matrices (the non-block matrices stay, as plain weights)."""
    block = {
        m for name, g in ckpt.groups.items() if name not in NONBLOCK_GROUPS for m in g.matrix_names
    }
    return {name: entry for name, entry in index.items() if name not in block}


__all__ = [
    "COMPUTED_PARAMS",
    "MAX_BUILD_ACTIVE_BYTES",
    "ErnieBuild",
    "base_extras",
    "base_transformer_files_index",
    "block_lists",
    "build_transformer",
    "seam_transformer_class",
]
