"""mflux's FLUX.2 Klein transformer with the class-swap seam; the build from a DF11 checkpoint."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_dfloat._safetensors import TensorInfo, read_header
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.blockseam import StepSeam, seam_blocks
from mlx_dfloat.integrate.coverage import check_extras_cover, extras_plan, load_extras
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.resident import NonBlockShapes, install_nonblock_placeholders
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux1.transformer import base_transformer_index
from mlx_dfloat.mflux.flux2.names import (
    DOUBLE,
    NONBLOCK_GROUPS,
    RUN_ORDER,
    SINGLE,
    check_klein_groups,
    klein_name_map,
)

# mflux 0.20.0 ``Flux2Transformer.__init__`` defaults (the 4B depth), used when the overrides leave them out.
DEFAULT_DOUBLE, DEFAULT_SINGLE = 5, 20
# ``ModelConfig.transformer_overrides["num_attention_heads"]`` per Klein size (24 heads x 128 = 3072 wide; 32 = 4096).
KLEIN_SIZES: dict[int, str] = {24: "4b", 32: "9b"}
MAX_BUILD_ACTIVE_BYTES = 2 * 1024**3


@cache
def seam_transformer_class() -> type:
    """``StepSeam`` composed in front of mflux's ``Flux2Transformer`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    require_mflux()
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    return type("SeamFlux2Transformer", (StepSeam, Flux2Transformer), {})


def block_lists(transformer: Any) -> list[tuple[str, Sequence[Any]]]:
    """The two block lists in run order (double blocks, then single blocks)."""
    return [(kind, getattr(transformer, kind)) for kind in RUN_ORDER]


def size_key(model_config: Any) -> str:
    """``"4b"`` or ``"9b"``: the Klein size of an mflux ``ModelConfig``, read from its head count.

    Raises:
        DFloatUnsupportedError: A head count with no Klein size (no memory constants for it).
    """
    heads = model_config.transformer_overrides.get("num_attention_heads")
    key = KLEIN_SIZES.get(heads) if isinstance(heads, int) else None
    if key is None:
        raise DFloatUnsupportedError(
            f"a FLUX.2 Klein transformer with num_attention_heads={heads!r}; supported: "
            f"{sorted(KLEIN_SIZES)} (4B and 9B)"
        )
    return key


@dataclass(frozen=True, slots=True, kw_only=True)
class Flux2Build:
    """A built transformer, its block shapes, the non-block matrices' shapes and the block counts built."""

    transformer: Any
    shapes: Shapes
    nonblock: NonBlockShapes
    counts: dict[str, int]


def build_transformer(
    ckpt: DF11Checkpoint,
    *,
    transformer_overrides: Mapping[str, Any],
    name_map: NameMap | None = None,
    n_double: int | None = None,
    n_single: int | None = None,
    extras: Mapping[str, tuple[Path, TensorInfo]] | None = None,
    nonblock_from_extras: bool = False,
    transformer_class: Callable[..., Any] | None = None,
) -> Flux2Build:
    """Construct the seamed transformer: block and non-block matrices as placeholders, every other parameter loaded.

    ``transformer_overrides`` are the model's (``ModelConfig.transformer_overrides``); the checkpoint's block counts
    must equal the depth they build. ``n_double``/``n_single`` then build fewer blocks (a partial build for tests).
    ``extras`` replaces the checkpoint's own extras (a BF16 base's tensors, via ``base_extras``); with
    ``nonblock_from_extras`` the non-block matrices are loaded from them as plain weights instead of placeholders
    (the BF16 side of an identity check). Non-block matrices that the checkpoint stores as extras load as plain
    weights either way. The coverage of both is asserted, and the active memory the build added.

    Raises:
        DFloatFormatError: The groups are not FLUX.2 Klein's; the checkpoint's block counts differ from the model's
            depth; an extra is not BF16 or has the wrong shape.
        DFloatIntegrationError: A depth override out of range, an uncovered parameter or an extra without one, or
            too much active memory added.
        DFloatDependencyError: The ``mflux`` extra is missing (default class and map only).
    """
    names = klein_name_map() if name_map is None else name_map
    counts = check_klein_groups(ckpt)
    want_d = int(transformer_overrides.get("num_layers", DEFAULT_DOUBLE))
    want_s = int(transformer_overrides.get("num_single_layers", DEFAULT_SINGLE))
    if (counts[DOUBLE], counts[SINGLE]) != (want_d, want_s):
        raise DFloatFormatError(
            f"checkpoint has {counts[DOUBLE]} double / {counts[SINGLE]} single blocks; this model builds "
            f"{want_d} / {want_s}"
        )
    n_d = counts[DOUBLE] if n_double is None else n_double
    n_s = counts[SINGLE] if n_single is None else n_single
    if not (0 <= n_d <= counts[DOUBLE] and 0 <= n_s <= counts[SINGLE]):
        raise DFloatIntegrationError(
            f"asked for {n_d} double / {n_s} single blocks; the checkpoint has "
            f"{counts[DOUBLE]} / {counts[SINGLE]}"
        )
    make = seam_transformer_class() if transformer_class is None else transformer_class
    before = int(mx.get_active_memory())
    tf = make(**{**transformer_overrides, "num_layers": n_d, "num_single_layers": n_s})
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
    built = {DOUBLE: n_d, SINGLE: n_s}
    plan = extras_plan(ckpt, names, counts=built, extras=extras)
    check_extras_cover(dict(tree_flatten(tf.parameters())), (n for n, _p, _i in plan), matrix_paths)
    load_extras(tf, plan, names)
    mx.eval(PLACEHOLDER)
    added = int(mx.get_active_memory()) - before
    if added >= MAX_BUILD_ACTIVE_BYTES:
        raise DFloatIntegrationError(f"extras added {added / 1024**3:.2f} GiB of active memory")
    return Flux2Build(transformer=tf, shapes=shapes, nonblock=nonblock, counts=built)


def base_extras(
    index: Mapping[str, tuple[Path, TensorInfo]], ckpt: DF11Checkpoint
) -> dict[str, tuple[Path, TensorInfo]]:
    """The BF16 base's tensors minus the block groups' matrices (non-block matrices stay: loaded as plain weights)."""
    block = {
        m for name, g in ckpt.groups.items() if name not in NONBLOCK_GROUPS for m in g.matrix_names
    }
    return {name: entry for name, entry in index.items() if name not in block}


def base_transformer_files_index(root: Path) -> dict[str, tuple[Path, TensorInfo]]:
    """Every tensor of a BF16 transformer directory: name -> (file, tensor info).

    A directory with a ``*.safetensors.index.json`` is read through its index (FLUX.2 Klein base 9B: two shards);
    one without is read from its single ``*.safetensors`` file (base 4B ships one file and no index).

    Raises:
        DFloatFormatError: Several index files; no index and zero or several safetensors files; or an unusable
            index or header.
    """
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise DFloatFormatError(
            f"{root}: {len(indexes)} weight indexes ({[p.name for p in indexes]}); expected one"
        )
    if indexes:
        return base_transformer_index(root, index_file=indexes[0].name)
    files = sorted(root.glob("*.safetensors"))
    if len(files) != 1:
        raise DFloatFormatError(
            f"{root}: no weight index and {len(files)} safetensors files; expected exactly one"
        )
    return {name: (files[0], info) for name, info in read_header(files[0]).items()}
