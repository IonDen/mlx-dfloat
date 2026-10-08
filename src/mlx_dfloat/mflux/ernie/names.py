"""ERNIE-Image naming: the block kind, the decoded sub-paths, the map derived from mflux, the group check.

The block kind and its run order: mflux 0.20.0 ``ErnieTransformer`` runs one list, ``layers``. The decoded sub-paths
and the three non-block groups are the ``pattern_dict`` of both mingyi456 DF11 checkpoints (ERNIE-Image and
ERNIE-Image-Turbo, identical). mflux's ``ErnieWeightMapping`` already templates its block targets with ``{layer}``,
so the shared derivation reads it as it is.
"""

from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._mapping import name_map_from_mapping

KIND = "layers"
RUN_ORDER: tuple[str, ...] = (KIND,)
MATRIX_SUBS: dict[str, tuple[str, ...]] = {
    KIND: (
        "self_attention.to_q",
        "self_attention.to_k",
        "self_attention.to_v",
        "self_attention.to_out.0",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.linear_fc2",
    )
}
NONBLOCK_GROUPS: dict[str, tuple[str, ...]] = {
    "time_embedding": ("time_embedding.linear_1.weight", "time_embedding.linear_2.weight"),
    "adaLN_modulation.1": ("adaLN_modulation.1.weight",),
    "final_norm.linear": ("final_norm.linear.weight",),
}
DEFAULT_LAYERS = 36  # mflux 0.20.0 ``ErnieTransformer.__init__(num_layers=36)``


def ernie_name_map() -> StaticNameMap:
    """The map read from mflux's ``ErnieWeightMapping`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: mflux's mapping no longer shapes the way this adapter expects.
    """
    require_mflux()
    from mflux.models.ernie_image.weights.ernie_weight_mapping import ErnieWeightMapping

    return name_map_from_mapping(
        ErnieWeightMapping.get_transformer_mapping(), matrix_subs=MATRIX_SUBS
    )


def check_ernie_groups(ckpt: Any) -> dict[str, int]:
    """The block count: block groups contiguous from 0, each holding the seven matrices, plus non-block groups.

    Each non-block group (``time_embedding``, ``adaLN_modulation.1``, ``final_norm.linear``) is optional: a checkpoint
    that stores it as BF16 extras loads it as plain weights. mflux builds the block list from its own count, so a hole
    would leave a block on placeholders.

    Raises:
        DFloatFormatError: A group of another name, a hole in the block indices, no block groups, a block whose
            matrices differ from the seven, or a non-block group with other matrices.
    """
    expected = MATRIX_SUBS[KIND]
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
                f"{name}: not an ERNIE-Image group (blocks: {KIND}.<n>; non-block groups accepted: "
                f"{', '.join(NONBLOCK_GROUPS)})"
            )
        if tuple(group.matrix_names) != tuple(f"{name}.{sub}.weight" for sub in expected):
            raise DFloatFormatError(
                f"{name}: holds {len(group.matrix_names)} matrices {list(group.matrix_names)}; an ERNIE-Image block "
                f"holds {', '.join(expected)}"
            )
        indices.add(int(idx))
    if not indices:
        raise DFloatFormatError(f"no {KIND} groups in the checkpoint")
    if indices != set(range(len(indices))):
        raise DFloatFormatError(f"{KIND} groups are not contiguous: {sorted(indices)}")
    return {KIND: len(indices)}
