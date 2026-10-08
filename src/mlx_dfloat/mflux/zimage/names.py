"""Z-Image naming: the block kinds in run order, the compressed sub-paths, the map derived from mflux, the group check.

Block kinds and run order: mflux 0.20.0 ``ZImageTransformer.__call__``. Compressed
sub-paths: the ``pattern_dict`` of both mingyi456 DF11 checkpoints (Z-Image and Z-Image-Turbo, identical).
"""

from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._mapping import name_map_from_mapping

NOISE_REFINER, CONTEXT_REFINER, LAYERS = "noise_refiner", "context_refiner", "layers"
RUN_ORDER: tuple[str, ...] = (NOISE_REFINER, CONTEXT_REFINER, LAYERS)
_BLOCK = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
MATRIX_SUBS: dict[str, tuple[str, ...]] = {
    NOISE_REFINER: (*_BLOCK, "adaLN_modulation.0"),
    CONTEXT_REFINER: _BLOCK,
    LAYERS: (*_BLOCK, "adaLN_modulation.0"),
}
NONBLOCK_GROUPS: dict[str, tuple[str, ...]] = {"cap_embedder": ("cap_embedder.1.weight",)}


def zimage_name_map() -> StaticNameMap:
    """The map read from mflux's ``ZImageWeightMapping`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: mflux's mapping no longer shapes the way this adapter expects.
    """
    require_mflux()
    from mflux.models.z_image.weights.z_image_weight_mapping import ZImageWeightMapping

    return name_map_from_mapping(
        ZImageWeightMapping.get_transformer_mapping(), matrix_subs=MATRIX_SUBS
    )


def check_zimage_groups(ckpt: Any) -> dict[str, int]:
    """Block counts per kind: the groups must be the three kinds (contiguous, the right matrices) plus non-block groups.

    Raises:
        DFloatFormatError: A group of another kind, a hole in an index sequence, a missing kind, a group whose
            matrices differ from its kind's, a non-block group with other matrices, or refiner counts that differ
            (mflux builds both refiner lists from one count).
    """
    seen: dict[str, set[int]] = {kind: set() for kind in RUN_ORDER}
    for name, group in ckpt.groups.items():
        if name in NONBLOCK_GROUPS:
            if tuple(group.matrix_names) != NONBLOCK_GROUPS[name]:
                raise DFloatFormatError(
                    f"{name}: holds {group.matrix_names}, expected {NONBLOCK_GROUPS[name]}"
                )
            continue
        kind, _dot, idx = name.partition(".")
        # A zero-padded index ("01") is refused: the seam asks for the group by its canonical name.
        if kind not in seen or not (idx.isascii() and idx.isdigit() and idx == str(int(idx))):
            raise DFloatFormatError(f"{name}: not a Z-Image block group")
        want = {f"{name}.{sub}.weight" for sub in MATRIX_SUBS[kind]}
        if set(group.matrix_names) != want or len(group.matrix_names) != len(want):
            raise DFloatFormatError(
                f"{name}: matrices {sorted(group.matrix_names)} are not a {kind} block's"
            )
        seen[kind].add(int(idx))
    for kind, indices in seen.items():
        if not indices:
            raise DFloatFormatError(f"no {kind} groups in the checkpoint")
        if indices != set(range(len(indices))):
            raise DFloatFormatError(f"{kind} groups are not contiguous: {sorted(indices)}")
    counts = {kind: len(indices) for kind, indices in seen.items()}
    if counts[NOISE_REFINER] != counts[CONTEXT_REFINER]:
        raise DFloatFormatError(f"refiner counts differ: {counts}")
    return counts
