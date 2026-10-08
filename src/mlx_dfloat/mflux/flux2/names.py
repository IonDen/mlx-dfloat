"""FLUX.2 Klein naming: the block kinds in run order, the compressed sub-paths, the map derived from mflux, the check.

Block kinds and run order: mflux 0.20.0 ``Flux2Transformer.__call__`` (double blocks, then single blocks). Compressed
sub-paths: the ``pattern_dict`` of the mingyi456 FLUX.2 Klein DF11 checkpoints (base and distilled, 4B and 9B). An
empty pattern list there means the group is the module itself, so its one matrix is ``<group>.weight``.
"""

from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._mapping import name_map_from_mapping

DOUBLE, SINGLE = "transformer_blocks", "single_transformer_blocks"
RUN_ORDER: tuple[str, ...] = (DOUBLE, SINGLE)
MATRIX_SUBS: dict[str, tuple[str, ...]] = {
    DOUBLE: (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
    ),
    SINGLE: ("attn.to_qkv_mlp_proj", "attn.to_out"),
}
NONBLOCK_GROUPS: dict[str, tuple[str, ...]] = {
    g: (f"{g}.weight",)
    for g in (
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "context_embedder",
        "norm_out.linear",
    )
}


def klein_name_map() -> StaticNameMap:
    """The map read from mflux's ``Flux2WeightMapping`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: mflux's mapping no longer shapes the way this adapter expects.
    """
    require_mflux()
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    return name_map_from_mapping(
        Flux2WeightMapping.get_transformer_mapping(), matrix_subs=MATRIX_SUBS
    )


def check_klein_groups(ckpt: Any) -> dict[str, int]:
    """Block counts per kind: the groups must be both kinds (contiguous, the right matrices) plus non-block groups.

    The non-block groups are optional: a checkpoint that stores those matrices as BF16 extras loads them as plain
    weights. mflux builds both block lists from its own counts (``Flux2Transformer.__init__``), so a hole would leave a
    block on placeholders.

    Raises:
        DFloatFormatError: A group of another kind, a hole in an index sequence, a missing kind, a group whose
            matrices differ from its kind's, or a non-block group with other matrices.
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
            raise DFloatFormatError(
                f"{name}: not a FLUX.2 Klein block group (blocks: {', '.join(RUN_ORDER)}; non-block groups "
                f"accepted: {', '.join(NONBLOCK_GROUPS)})"
            )
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
    return {kind: len(indices) for kind, indices in seen.items()}
