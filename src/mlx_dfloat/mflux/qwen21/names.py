"""Qwen-Image 2.1 naming: the block kind, the decoded sub-paths, the map derived from mflux, the group check.

The block kind and its run order: mflux 0.20.0 ``Qwen21Transformer`` runs one list, ``transformer_blocks``. The
decoded sub-paths are those of the ComfyUI single-file DF11 export after its fused ``img_mlp.gate_up`` matrix is cut
into ``img_mlp.gate_layer`` and ``img_mlp.proj`` (the reader does the cut), so the map never sees ``gate_up``. mflux's
``Qwen21WeightMapping`` lists the block targets by concrete index rather than a ``{block}`` template;
``templated_targets`` folds them into one template per sub-path before the shared derivation runs.
"""

import re
import types
from collections.abc import Iterable, Sequence
from typing import Any

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._mapping import name_map_from_mapping

KIND = "transformer_blocks"
RUN_ORDER: tuple[str, ...] = (KIND,)
MATRIX_SUBS: dict[str, tuple[str, ...]] = {
    KIND: (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.gate_layer",
        "img_mlp.proj",
        "img_mlp.out",
    )
}
NONBLOCK_GROUPS: dict[str, tuple[str, ...]] = {"modulation.1": ("modulation.1.weight",)}
DEFAULT_LAYERS = 32  # mflux 0.20.0 ``Qwen21Transformer.__init__(num_layers=32)``


def _sources(target: Any) -> list[str]:
    src = target.from_pattern
    return list(src) if isinstance(src, list | tuple) else [src]


def templated_targets(targets: Iterable[Any], *, kinds: Sequence[str], n_blocks: int) -> list[Any]:
    """Fold concrete-index block targets into one ``{block}``-templated target per sub-path.

    A target whose target and single source both name ``<kind>.<i>.<sub>`` (the same ``i``) is folded with the
    other blocks' targets of the same source sub-path. Every block ``0 .. n_blocks - 1`` must have one, each with the
    same target sub-path and no transform. Other targets pass through unchanged, in their order.

    Raises:
        DFloatIntegrationError: A block target with several sources, a transform, another index or kind on one side,
            or another rename than block 0's; a sub-path missing for a block, given twice, or given beyond
            ``n_blocks``.
    """
    pattern = re.compile(rf"^({'|'.join(re.escape(k) for k in kinds)})\.([0-9]+)\.(.+)$")
    out: list[Any] = []
    folded: dict[tuple[str, str], dict[int, str]] = {}
    slot: dict[tuple[str, str], int] = {}
    for target in targets:
        sources = _sources(target)
        to = pattern.match(target.to_pattern)
        frm = [pattern.match(s) for s in sources]
        if to is None and not any(frm):
            out.append(target)
            continue
        if len(sources) != 1 or to is None or frm[0] is None:
            raise DFloatIntegrationError(
                f"{target.to_pattern} <- {sources}: a block target must be one block source in its own block"
            )
        source = frm[0]
        kind, index, sub = source.group(1), int(source.group(2)), source.group(3)
        if (to.group(1), int(to.group(2))) != (kind, index):
            raise DFloatIntegrationError(
                f"{sources[0]}: mflux loads it into {target.to_pattern}, another block"
            )
        if target.transform is not None:
            raise DFloatIntegrationError(f"{kind}.{sub}: a transform on block {index}'s target")
        if index >= n_blocks:
            raise DFloatIntegrationError(
                f"{kind}.{sub}: a target for block {index}, beyond the {n_blocks} blocks"
            )
        per = folded.setdefault((kind, sub), {})
        if index in per:
            raise DFloatIntegrationError(f"{kind}.{sub}: two targets for block {index}")
        per[index] = to.group(3)
        if (kind, sub) not in slot:
            slot[(kind, sub)] = len(out)
            out.append(None)
    for (kind, sub), per in folded.items():
        for index in range(n_blocks):
            if index not in per:
                raise DFloatIntegrationError(f"{kind}.{sub}: no target for block {index}")
            if per[index] != per[0]:
                raise DFloatIntegrationError(
                    f"{kind}.{sub}: block {index} loads it into {per[index]}, block 0 into {per[0]}"
                )
        out[slot[(kind, sub)]] = types.SimpleNamespace(
            to_pattern=f"{kind}.{{block}}.{per[0]}",
            from_pattern=[f"{kind}.{{block}}.{sub}"],
            transform=None,
        )
    return out


def qwen21_name_map() -> StaticNameMap:
    """The map read from mflux's ``Qwen21WeightMapping`` (imports mflux).

    Raises:
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
        DFloatIntegrationError: mflux's mapping no longer shapes the way this adapter expects.
    """
    require_mflux()
    from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping

    return name_map_from_mapping(
        templated_targets(
            Qwen21WeightMapping.get_transformer_mapping(),
            kinds=RUN_ORDER,
            n_blocks=Qwen21WeightMapping.NUM_TRANSFORMER_BLOCKS,
        ),
        matrix_subs=MATRIX_SUBS,
    )


def check_qwen21_groups(ckpt: Any) -> dict[str, int]:
    """The block count: block groups contiguous from 0, each holding the seven matrices, plus ``modulation.1``.

    ``modulation.1`` is optional: a checkpoint that stores it as a BF16 extra loads it as a plain weight. mflux builds
    the block list from its own count, so a hole would leave a block on placeholders.

    Raises:
        DFloatFormatError: A group of another name, a hole in the block indices, no block groups, a block whose
            matrices differ from the seven, or a ``modulation.1`` group with other matrices.
    """
    expected = MATRIX_SUBS[KIND]
    indices: set[int] = set()
    for name, group in ckpt.groups.items():
        if name in NONBLOCK_GROUPS:
            if tuple(group.matrix_names) != NONBLOCK_GROUPS[name]:
                raise DFloatFormatError(
                    f"{name}: holds {group.matrix_names}, expected {NONBLOCK_GROUPS[name]}"
                )
            continue
        kind, _dot, idx = name.partition(".")
        # A zero-padded index ("01") is refused: the seam asks for the group by its canonical name.
        if kind != KIND or not (idx.isascii() and idx.isdigit() and idx == str(int(idx))):
            raise DFloatFormatError(
                f"{name}: not a Qwen-Image 2.1 group (blocks: {KIND}.<n>; non-block groups accepted: "
                f"{', '.join(NONBLOCK_GROUPS)})"
            )
        if tuple(group.matrix_names) != tuple(f"{name}.{sub}.weight" for sub in expected):
            raise DFloatFormatError(
                f"{name}: holds {len(group.matrix_names)} matrices {list(group.matrix_names)}; a Qwen-Image 2.1 "
                f"block holds {', '.join(expected)} (the fused img_mlp.gate_up cut into its halves)"
            )
        indices.add(int(idx))
    if not indices:
        raise DFloatFormatError(f"no {KIND} groups in the checkpoint")
    if indices != set(range(len(indices))):
        raise DFloatFormatError(f"{KIND} groups are not contiguous: {sorted(indices)}")
    return {KIND: len(indices)}
