"""Qwen-Image 2.1 naming: the folded mapping and the group check (offline); the real mapping (mflux lane)."""

from types import SimpleNamespace

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux._mapping import name_map_from_mapping
from mlx_dfloat.mflux.qwen21.names import (
    KIND,
    MATRIX_SUBS,
    RUN_ORDER,
    check_qwen21_groups,
    templated_targets,
)

# The seven block matrices in the order a block decodes into after the gate_up split (S0: the stored order is
# to_q, to_k, to_v, to_out.0, gate_up = [gate_layer; proj], out). Literal, so a reordered MATRIX_SUBS shows.
SEVEN = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)


def _t(to, frm, transform=None):
    return SimpleNamespace(to_pattern=to, from_pattern=[frm], transform=transform)


def _mapping(n_blocks=3, *, odd_block=None):
    # A copy of Qwen21WeightMapping.get_transformer_mapping() (mflux 0.20.0
    # models/qwen21/weights/qwen21_weight_mapping.py:14-53) for n_blocks blocks, in its order: nine non-block targets,
    # one rename (modulation.1 -> modulation.layers.1, :18), then per block by concrete index (:32-52).
    # test_the_real_mapping_matches_the_copy compares it with mflux's.
    targets = [_t("img_in.weight", "img_in.weight"), _t("proj_out.weight", "proj_out.weight")]
    targets.append(_t("modulation.layers.1.weight", "modulation.1.weight"))
    targets += [
        _t(n, n)
        for n in (
            "norm_out.linear.weight",
            "txt_in.text_norm.weight",
            "txt_in.in_layer.weight",
            "txt_in.out_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_2.weight",
        )
    ]
    for b in range(n_blocks):
        p = f"transformer_blocks.{b}."
        targets += [
            _t(f"{p}attn.{s}.weight", f"{p}attn.{s}.weight")
            for s in ("to_q", "to_k", "to_v", "norm_q", "norm_k")
        ]
        to_out = f"{p}attn.to_out.0.weight"
        targets.append(_t(f"{p}attn.out_proj.weight" if b == odd_block else to_out, to_out))
        targets += [
            _t(f"{p}img_mlp.{s}.weight", f"{p}img_mlp.{s}.weight")
            for s in ("proj", "out", "gate_layer")
        ]
    return targets


def _folded(targets=None):
    targets = _mapping() if targets is None else targets
    return name_map_from_mapping(
        templated_targets(targets, kinds=RUN_ORDER, n_blocks=3), matrix_subs=MATRIX_SUBS
    )


def test_the_folded_map_places_every_block_matrix_on_its_own_path_in_decoded_order():
    # Bug caught: a concrete-index mapping not folded (no {block} target: "a compressed matrix with no mflux target"),
    # or the table order not the decoded order (gate_layer before proj, S0).
    m = _folded()
    assert m.attrs_of("transformer_blocks") == SEVEN
    assert m.place("transformer_blocks.2.img_mlp.proj.weight").attr == "img_mlp.proj"
    assert m.place("transformer_blocks.0.attn.to_out.0.weight").attr == "attn.to_out.0"
    assert (
        m.param_name("transformer_blocks.1.attn.norm_k.weight")
        == "transformer_blocks.1.attn.norm_k.weight"
    )


def test_modulation_is_renamed_into_the_sequential():
    # Bug caught: the one non-block rename lost (modulation.1.weight has no parameter of that name in mflux).
    assert _folded().param_name("modulation.1.weight") == "modulation.layers.1.weight"


def test_non_block_targets_pass_through_unchanged():
    # Bug caught: the fold swallowing (or templating) a non-block target, so img_in or the rename disappears.
    out = templated_targets(_mapping(), kinds=RUN_ORDER, n_blocks=3)
    flat = {(t.to_pattern, tuple(t.from_pattern)) for t in out}
    assert ("img_in.weight", ("img_in.weight",)) in flat
    assert ("modulation.layers.1.weight", ("modulation.1.weight",)) in flat
    assert (
        "transformer_blocks.{block}.img_mlp.gate_layer.weight",
        ("transformer_blocks.{block}.img_mlp.gate_layer.weight",),
    ) in flat
    assert len(out) == 9 + 9  # nine non-block targets, nine block subs (seven matrices, two norms)


def test_a_block_whose_rename_differs_is_refused():
    # Bug caught: block 0's rename applied to every block when block 1 maps elsewhere.
    with pytest.raises(DFloatIntegrationError, match=r"attn\.to_out\.0.*block 1\b"):
        templated_targets(_mapping(odd_block=1), kinds=RUN_ORDER, n_blocks=3)


def test_a_missing_block_target_is_refused():
    # Bug caught: a mapping that stops at block 30 folded as if it covered 32.
    with pytest.raises(DFloatIntegrationError, match=r"no target for block 2\b"):
        templated_targets(_mapping(n_blocks=2), kinds=RUN_ORDER, n_blocks=3)


def test_a_block_target_beyond_the_count_is_refused():
    # Bug caught: a mapping for 33 blocks folded against 32, the extra block's target silently dropped.
    with pytest.raises(DFloatIntegrationError, match=r"block 3\b"):
        templated_targets(_mapping(n_blocks=4), kinds=RUN_ORDER, n_blocks=3)


def test_a_block_target_with_a_transform_is_refused():
    # Bug caught: a transform on one block's target dropped by the fold (the BF16 bytes would load untransformed).
    targets = _mapping()
    i = next(
        i
        for i, t in enumerate(targets)
        if t.to_pattern == "transformer_blocks.2.attn.norm_q.weight"
    )
    targets[i] = _t(targets[i].to_pattern, targets[i].from_pattern[0], transform=lambda a: a)
    with pytest.raises(DFloatIntegrationError, match=r"attn\.norm_q.*block 2\b"):
        templated_targets(targets, kinds=RUN_ORDER, n_blocks=3)


def test_a_block_source_mapped_to_another_block_index_is_refused():
    # Bug caught: block 1's source loaded into block 2's parameter folded as if it were the identity.
    targets = _mapping()
    i = next(
        i for i, t in enumerate(targets) if t.to_pattern == "transformer_blocks.1.attn.to_q.weight"
    )
    targets[i] = _t(
        "transformer_blocks.2.attn.to_q.weight", "transformer_blocks.1.attn.to_q.weight"
    )
    with pytest.raises(DFloatIntegrationError, match=r"transformer_blocks\.1\.attn\.to_q"):
        templated_targets(targets, kinds=RUN_ORDER, n_blocks=3)


# --- the group check, on written tiny checkpoints -------------------------------------------------------------------

_BLOCK = r"transformer_blocks\.\d+"


def _write(root, *, blocks=(0, 1), subs=SEVEN, modulation=False, extra_groups=()):
    rng = np.random.default_rng(3)
    groups = {f"transformer_blocks.{b}": [random_bf16(rng, (2, 4)) for _ in subs] for b in blocks}
    patterns = {_BLOCK: subs}
    if modulation:
        groups["modulation.1"] = [random_bf16(rng, (8, 4))]
        patterns[r"modulation\.1"] = ()
    for name in extra_groups:
        groups[name] = [random_bf16(rng, (2, 4))]
        patterns[name.replace(".", r"\.")] = ()
    write_checkpoint(root, groups=groups, patterns=patterns)
    return open_checkpoint(root)


@pytest.mark.parametrize("modulation", [False, True])
def test_the_block_count_is_returned_with_and_without_the_modulation_group(tmp_path, modulation):
    # Bug caught: the optional non-block group counted as a block (or refused), or the count off by one.
    assert check_qwen21_groups(_write(tmp_path, modulation=modulation)) == {"transformer_blocks": 2}


def test_a_hole_in_the_block_indices_is_refused(tmp_path):
    # Bug caught: blocks 0 and 2 accepted, block 1 left on placeholders (a zero-size matmul at run time).
    with pytest.raises(DFloatFormatError, match="not contiguous"):
        check_qwen21_groups(_write(tmp_path, blocks=(0, 2)))


def test_a_zero_padded_block_index_is_refused(tmp_path):
    # Bug caught (fable M3): transformer_blocks.01 counted as block 1 (int("01") == 1), then the seam asks the
    # provider for transformer_blocks.1, a group the checkpoint does not have: a failure at the first block after the
    # encoder and set loads.
    with pytest.raises(
        DFloatFormatError, match=r"transformer_blocks\.01: not a Qwen-Image 2\.1 group"
    ):
        check_qwen21_groups(_write(tmp_path, blocks=("0", "01")))


def test_a_checkpoint_without_blocks_is_refused(tmp_path):
    # Bug caught: a modulation-only checkpoint accepted as zero blocks.
    rng = np.random.default_rng(4)
    write_checkpoint(
        tmp_path,
        groups={"modulation.1": [random_bf16(rng, (8, 4))]},
        patterns={r"modulation\.1": ()},
    )
    with pytest.raises(DFloatFormatError, match="no transformer_blocks groups"):
        check_qwen21_groups(open_checkpoint(tmp_path))


def test_a_qwen_image_1_block_is_refused_naming_the_group_and_the_seven_matrices(tmp_path):
    # Bug caught (Review Focus 2): the Qwen-Image 1 DF11 checkpoint (dual-stream blocks, 14 matrices, same group
    # names) passed for 2.1 and failing with a shape error at the first block instead of at build.
    qwen_image_1 = (
        "img_mod.1",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mod.1",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    )
    with pytest.raises(DFloatFormatError, match=r"transformer_blocks\.0.*img_mlp\.gate_layer"):
        check_qwen21_groups(_write(tmp_path, subs=qwen_image_1))


def test_a_stray_group_is_refused_naming_the_accepted_non_block_group(tmp_path):
    # Bug caught: an unknown group (another model's) silently ignored, its matrices never placed.
    with pytest.raises(DFloatFormatError, match=r"txt_mod\.1.*modulation\.1"):
        check_qwen21_groups(_write(tmp_path, extra_groups=("txt_mod.1",)))


def test_a_modulation_group_with_other_matrices_is_refused(tmp_path):
    # Bug caught: a modulation.1 group of several matrices accepted, the extra ones never installed.
    rng = np.random.default_rng(5)
    groups = {
        "transformer_blocks.0": [random_bf16(rng, (2, 4)) for _ in SEVEN],
        "modulation.1": [random_bf16(rng, (8, 4)), random_bf16(rng, (8, 4))],
    }
    write_checkpoint(
        tmp_path,
        groups=groups,
        patterns={_BLOCK: SEVEN, r"modulation\.1": ("a", "b")},
    )
    with pytest.raises(DFloatFormatError, match=r"modulation\.1"):
        check_qwen21_groups(open_checkpoint(tmp_path))


def _shape_of(targets):
    return [(t.to_pattern, tuple(t.from_pattern), t.transform) for t in targets]


@pytest.mark.mflux
def test_the_real_mapping_matches_the_copy():
    # Bug caught: the offline tests running on a copy that no longer is mflux's mapping (a target renamed, added,
    # reordered or given a transform upstream), so they pass while the real fold would differ. mflux's 32 blocks are
    # cut to the copy's three; everything else must be equal, in order.
    from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping

    real = Qwen21WeightMapping.get_transformer_mapping()
    kept = [
        t
        for t in real
        if not t.to_pattern.startswith("transformer_blocks.") or int(t.to_pattern.split(".")[1]) < 3
    ]
    assert _shape_of(kept) == _shape_of(_mapping(3))
    assert len(real) == 9 + 32 * 9


@pytest.mark.mflux
def test_the_default_layer_count_is_mfluxs_constructor_default():
    # Bug caught: an mflux bump changing Qwen21Transformer's default depth (what the published model is built with)
    # while the count check keeps 32: every real checkpoint refused, or one of another depth accepted.
    import inspect

    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.mflux.qwen21.names import DEFAULT_LAYERS

    default = inspect.signature(Qwen21Transformer.__init__).parameters["num_layers"].default
    assert default == 32
    assert DEFAULT_LAYERS == 32


@pytest.mark.mflux
def test_the_real_mapping_folds_and_every_matrix_lands_on_a_linear():
    # Bug caught: a decoded matrix with no nn.Linear to land on in a real tiny Qwen21Transformer, or the real mapping
    # not folding into the seven block matrices and the modulation rename.
    import mlx.nn as nn
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.integrate.placeholders import get_attr_path
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    m = qwen21_name_map()
    assert m.attrs_of(KIND) == SEVEN
    assert m.param_name("modulation.1.weight") == "modulation.layers.1.weight"
    tf = Qwen21Transformer(
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=32,
        axes_dims_rope=(4, 6, 6),
    )
    for attr in m.attrs_of(KIND):
        assert isinstance(get_attr_path(tf.transformer_blocks[0], attr), nn.Linear), attr
    modulation = get_attr_path(tf, m.param_name("modulation.1.weight").removesuffix(".weight"))
    assert isinstance(modulation, nn.Linear)


QWEN21_MODULES = (
    "mlx_dfloat.mflux.qwen21",
    "mlx_dfloat.mflux.qwen21.names",
    "mlx_dfloat.mflux.qwen21.transformer",
)


@pytest.mark.parametrize("module", QWEN21_MODULES)
def test_importing_the_adapter_does_not_import_mflux(module):
    # Bug caught: a module-level mflux import in the adapter (a user without the extra gets a bare ImportError on
    # import instead of the dependency error naming it, and every import pays for mflux).
    import subprocess
    import sys

    code = f"import sys, {module}; print('mflux' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        # Bug caught: a non-block parameter loaded from a block's tensor passed through the fold untouched (the
        # shared derivation would place the block source by the table, not where mflux loads it).
        (
            _t("img_in.weight", "transformer_blocks.0.attn.to_q.weight"),
            "one block source in its own block",
        ),
        # Bug caught: a second target for block 1's source folded over the first (one of them silently lost).
        (
            _t("transformer_blocks.1.attn.to_q.weight", "transformer_blocks.1.attn.to_q.weight"),
            r"two targets for block 1\b",
        ),
    ],
)
def test_a_block_source_that_does_not_fold_cleanly_is_refused(extra, message):
    with pytest.raises(DFloatIntegrationError, match=message):
        templated_targets([*_mapping(), extra], kinds=RUN_ORDER, n_blocks=3)
