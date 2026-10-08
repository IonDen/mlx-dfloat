"""Krea 2 naming: the block kind, the decoded sub-paths, the map derived from mflux, the group and variant checks.

The block kind and its run order: mflux 0.20.0 ``Krea2Transformer`` runs one list, ``blocks``, inline. The decoded
sub-paths and the seven non-block groups (the four text-fusion blocks, ``tmlp``, ``tproj``, ``txtmlp``) are the
``pattern_dict`` of the two ComfyUI single-file DF11 exports (Krea 2 Raw and Krea 2 Turbo, identical); their stored
order was checked against the originals' bytes. mflux's ``Krea2WeightMapping`` templates its block targets with
``{layer}`` and lists the text-fusion blocks by concrete index as identities, so the shared derivation reads it as it
is. The text-fusion blocks are not a seamed kind: they run once per call before the blocks, like ``tmlp``, ``tproj``
and ``txtmlp``. The model decodes all seven at the start of each transformer call
(``mlx_dfloat.mflux.krea2.transformer.PerCallNonBlock``); the resident build keeps them decoded with the set.
"""

import re
from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._mapping import name_map_from_mapping

KIND = "blocks"
RUN_ORDER: tuple[str, ...] = (KIND,)
BLOCK_SUBS: tuple[str, ...] = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)
TEXT_SUBS: tuple[str, ...] = (
    BLOCK_SUBS  # a text-fusion block stores the same eight matrices, in the same order
)
MATRIX_SUBS: dict[str, tuple[str, ...]] = {KIND: BLOCK_SUBS}
TEXT_FUSION_GROUPS: tuple[str, ...] = (
    "txtfusion.layerwise_blocks.0",
    "txtfusion.layerwise_blocks.1",
    "txtfusion.refiner_blocks.0",
    "txtfusion.refiner_blocks.1",
)
NONBLOCK_GROUPS: dict[str, tuple[str, ...]] = {
    "tmlp": ("tmlp.0.weight", "tmlp.2.weight"),
    "tproj": ("tproj.1.weight",),
    "txtmlp": ("txtmlp.1.weight", "txtmlp.3.weight"),
    **{g: tuple(f"{g}.{s}.weight" for s in TEXT_SUBS) for g in TEXT_FUSION_GROUPS},
}
DEFAULT_LAYERS = 28  # mflux 0.20.0 ``Krea2Transformer.__init__(layers=28)``
LAYOUT_FOR_MODEL: dict[str, str] = {
    "krea-2-raw": "krea-2-raw-comfyui",
    "krea-2": "krea-2-turbo-comfyui",
}
_LABELS = {"krea-2-raw": "Krea 2 Raw", "krea-2": "Krea 2 Turbo"}
_SOURCE = re.compile(r"header and spot checks match layout (\S+) ")


def krea2_name_map() -> StaticNameMap:
    """The map read from mflux's ``Krea2WeightMapping`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: mflux's mapping no longer shapes the way this adapter expects.
    """
    require_mflux()
    from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping

    return name_map_from_mapping(
        Krea2WeightMapping.get_transformer_mapping(), matrix_subs=MATRIX_SUBS
    )


def check_krea2_groups(ckpt: Any) -> dict[str, int]:
    """The block count: block groups contiguous from 0, each holding the eight matrices, plus non-block groups.

    Each non-block group is optional: a checkpoint that stores it as BF16 extras loads it as plain weights. mflux
    builds the block list from its own count, so a hole would leave a block on placeholders.

    Raises:
        DFloatFormatError: A group of another name, a hole in the block indices, no block groups, a block whose
            matrices differ from the eight, or a non-block group with other matrices.
    """
    indices: set[int] = set()
    for name, group in ckpt.groups.items():
        if name in NONBLOCK_GROUPS:
            if tuple(group.matrix_names) != NONBLOCK_GROUPS[name]:
                raise DFloatFormatError(
                    f"{name}: holds {list(group.matrix_names)}, expected {list(NONBLOCK_GROUPS[name])}"
                )
            continue
        kind, _dot, idx = name.partition(".")
        # A zero-padded index ("01") is refused: the seam asks for the group by its canonical name.
        if kind != KIND or not (idx.isascii() and idx.isdigit() and idx == str(int(idx))):
            raise DFloatFormatError(
                f"{name}: not a Krea 2 group (blocks: {KIND}.<n>; non-block groups accepted: "
                f"{', '.join(NONBLOCK_GROUPS)})"
            )
        if tuple(group.matrix_names) != tuple(f"{name}.{sub}.weight" for sub in BLOCK_SUBS):
            raise DFloatFormatError(
                f"{name}: holds {len(group.matrix_names)} matrices {list(group.matrix_names)}; a Krea 2 block "
                f"holds {', '.join(BLOCK_SUBS)}"
            )
        indices.add(int(idx))
    if not indices:
        raise DFloatFormatError(f"no {KIND} groups in the checkpoint")
    if indices != set(range(len(indices))):
        raise DFloatFormatError(f"{KIND} groups are not contiguous: {sorted(indices)}")
    return {KIND: len(indices)}


def check_variant(model: str, ckpt: Any) -> None:
    """Refuse the other Krea 2 model's published checkpoint (Raw and Turbo share every name and shape).

    The checkpoint's ``config_source`` names the pinned layout it opened through; a source that names no layout (a
    ``config.json``) or a layout of neither Krea model passes.

    Raises:
        DFloatFormatError: The checkpoint is the other Krea 2 model's published file.
    """
    found = _SOURCE.match(ckpt.config_source)
    if found is None:
        return
    key = found.group(1)
    for other, layout in LAYOUT_FOR_MODEL.items():
        if other != model and layout == key:
            raise DFloatFormatError(
                f"{ckpt.root.name}: this is the {_LABELS[other]} DF11 checkpoint (layout {key}); run it with "
                f"--model {other}"
            )
