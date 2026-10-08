"""mflux's Z-Image transformer with the class-swap seam; the build from a DF11 checkpoint."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_dfloat._safetensors import TensorInfo
from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.blockseam import StepSeam, seam_blocks
from mlx_dfloat.integrate.coverage import check_extras_cover, extras_plan, load_extras
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.resident import NonBlockShapes, install_nonblock_placeholders
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.zimage.names import (
    CONTEXT_REFINER,
    LAYERS,
    NOISE_REFINER,
    NONBLOCK_GROUPS,
    RUN_ORDER,
    check_zimage_groups,
    zimage_name_map,
)

ZIMAGE_BASE_INDEX = "diffusion_pytorch_model.safetensors.index.json"
MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3


@cache
def seam_transformer_class() -> type:
    """``StepSeam`` composed in front of mflux's ``ZImageTransformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

    return type("SeamZImageTransformer", (StepSeam, ZImageTransformer), {})


def block_lists(transformer: Any) -> list[tuple[str, Sequence[Any]]]:
    """The three block lists in run order (noise refiner, context refiner, main layers)."""
    return [(kind, getattr(transformer, kind)) for kind in RUN_ORDER]


@dataclass(frozen=True, slots=True, kw_only=True)
class ZImageBuild:
    """A built transformer, its block shapes, the non-block matrices' shapes and the block counts built."""

    transformer: Any
    shapes: Shapes
    nonblock: NonBlockShapes
    counts: dict[str, int]


def build_transformer(
    ckpt: DF11Checkpoint,
    *,
    name_map: NameMap | None = None,
    n_refiner: int | None = None,
    n_layers: int | None = None,
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
    nonblock_from_extras: bool = False,
    transformer_class: Callable[..., Any] | None = None,
) -> ZImageBuild:
    """Construct the seamed transformer: block and non-block matrices as placeholders, every other parameter loaded.

    ``extras`` replaces the checkpoint's own extras (a BF16 base's tensors, via ``base_extras``); with
    ``nonblock_from_extras`` the non-block matrices are loaded from them as plain weights instead of placeholders
    (the BF16 side of an identity check). The coverage of both is asserted, and the active memory the build added.

    Raises:
        DFloatFormatError: The groups are not Z-Image's; an extra is not BF16 or has the wrong shape.
        DFloatIntegrationError: A depth override out of range, an uncovered parameter or an extra without one, or
            too much active memory added.
        DFloatDependencyError: The ``mflux`` extra is missing (default class and map only).
    """
    names = zimage_name_map() if name_map is None else name_map
    counts = check_zimage_groups(ckpt)
    n_ref = counts[NOISE_REFINER] if n_refiner is None else n_refiner
    n_lay = counts[LAYERS] if n_layers is None else n_layers
    if not (0 <= n_ref <= counts[NOISE_REFINER] and 0 <= n_lay <= counts[LAYERS]):
        raise DFloatIntegrationError(
            f"asked for {n_ref} refiner / {n_lay} main blocks; the checkpoint has "
            f"{counts[NOISE_REFINER]} / {counts[LAYERS]}"
        )
    make = seam_transformer_class() if transformer_class is None else transformer_class
    before = int(mx.get_active_memory())
    tf = make(n_layers=n_lay, n_refiner_layers=n_ref)
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
    built = {NOISE_REFINER: n_ref, CONTEXT_REFINER: n_ref, LAYERS: n_lay}
    plan = extras_plan(ckpt, names, counts=built, extras=extras)
    check_extras_cover(dict(tree_flatten(tf.parameters())), (n for n, _p, _i in plan), matrix_paths)
    load_extras(tf, plan, names)
    mx.eval(PLACEHOLDER)
    added = int(mx.get_active_memory()) - before
    if added >= MAX_BUILD_ACTIVE_BYTES:
        raise DFloatIntegrationError(f"extras added {added / 1024**3:.2f} GiB of active memory")
    return ZImageBuild(transformer=tf, shapes=shapes, nonblock=nonblock, counts=built)


def base_extras(
    index: Mapping[str, tuple[Path, TensorInfo]], ckpt: DF11Checkpoint
) -> dict[str, tuple[Path, TensorInfo]]:
    """The BF16 base's tensors minus the block groups' matrices (non-block matrices stay: loaded as plain weights)."""
    block = {
        m for name, g in ckpt.groups.items() if name not in NONBLOCK_GROUPS for m in g.matrix_names
    }
    return {name: entry for name, entry in index.items() if name not in block}
