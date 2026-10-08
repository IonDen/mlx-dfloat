"""ERNIE-Image naming: the map derived from mflux's weight mapping, and the DF11 group check."""

from types import SimpleNamespace

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux._mapping import name_map_from_mapping
from mlx_dfloat.mflux.ernie.names import KIND, MATRIX_SUBS, NONBLOCK_GROUPS, check_ernie_groups

SEVEN = (
    "self_attention.to_q",
    "self_attention.to_k",
    "self_attention.to_v",
    "self_attention.to_out.0",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.linear_fc2",
)


def _conv(w):  # stands in for WeightTransforms.transpose_conv2d_weight
    return w.transpose(0, 2, 3, 1)


def _t(to, frm, transform=None):
    return SimpleNamespace(to_pattern=to, from_pattern=[frm], transform=transform)


def _mapping():
    # A copy of ErnieWeightMapping.get_transformer_mapping() (mflux 0.20.0 ernie_weight_mapping.py:11-109).
    targets = [
        _t("adaln_modulation.weight", "adaLN_modulation.1.weight"),
        _t("adaln_modulation.bias", "adaLN_modulation.1.bias"),
        _t("x_embedder.proj.weight", "x_embedder.proj.weight", _conv),
        _t("x_embedder.proj.bias", "x_embedder.proj.bias"),
    ]
    targets += [
        _t(n, n)
        for n in (
            "text_proj.weight",
            "time_embedding.linear_1.weight",
            "time_embedding.linear_1.bias",
            "time_embedding.linear_2.weight",
            "time_embedding.linear_2.bias",
            "final_norm.linear.weight",
            "final_norm.linear.bias",
            "final_linear.weight",
            "final_linear.bias",
        )
    ]
    for sub in (
        "adaLN_sa_ln",
        "adaLN_mlp_ln",
        "self_attention.to_q",
        "self_attention.to_k",
        "self_attention.to_v",
        "self_attention.to_out.0",
        "self_attention.norm_q",
        "self_attention.norm_k",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.linear_fc2",
    ):
        src = f"layers.{{layer}}.{sub}.weight"
        targets.append(_t(src, src))
    return targets


# The DF11 config's pattern_dict (mingyi456/ERNIE-Image{,-Turbo}-DF11 config.json, byte-identical in both).
PATTERN = {
    "time_embedding": ("linear_1", "linear_2"),
    "adaLN_modulation.1": (),
    r"layers\.\d+": SEVEN,
    "final_norm.linear": (),
}


def test_the_map_places_every_block_matrix_on_its_own_path_in_decoded_order():
    # Bug caught: MATRIX_SUBS out of the checkpoint's pattern_dict order (gate_proj and up_proj swapped: the block
    # would run up(x) * gelu(up(x)) on the wrong halves), or a block sub-path renamed.
    m = name_map_from_mapping(_mapping(), matrix_subs=MATRIX_SUBS)
    assert m.attrs_of(KIND) == SEVEN
    assert m.place("layers.35.mlp.up_proj.weight").attr == "mlp.up_proj"
    assert m.param_name("layers.3.adaLN_sa_ln.weight") == "layers.3.adaLN_sa_ln.weight"


def test_adaln_modulation_is_renamed_and_the_patch_conv_keeps_its_transform():
    # Bug caught: the one non-block rename lost (adaLN_modulation.1.weight has no parameter of that name in mflux:
    # an extras-coverage refusal), or the Conv2d transpose dropped (a shape refusal at load, or a wrong layout).
    m = name_map_from_mapping(_mapping(), matrix_subs=MATRIX_SUBS)
    assert m.param_name("adaLN_modulation.1.weight") == "adaln_modulation.weight"
    assert m.param_name("adaLN_modulation.1.bias") == "adaln_modulation.bias"
    assert m.transform_of("x_embedder.proj.weight") is _conv


def test_the_adapters_tables_are_the_checkpoints_matrix_names():
    # Bug caught: a sub-path missing, misspelt or reordered against the published checkpoint (the group check would
    # refuse every real ERNIE checkpoint, or accept the matrices in the wrong order), or a non-block group's matrices
    # not what the reader derives from the pattern ("adaLN_modulation.1": [] is the group's own weight).
    from mlx_dfloat.format import matrix_names_for

    assert matrix_names_for("layers.7", PATTERN) == tuple(
        f"layers.7.{s}.weight" for s in MATRIX_SUBS[KIND]
    )
    assert NONBLOCK_GROUPS == {
        "time_embedding": ("time_embedding.linear_1.weight", "time_embedding.linear_2.weight"),
        "adaLN_modulation.1": ("adaLN_modulation.1.weight",),
        "final_norm.linear": ("final_norm.linear.weight",),
    }
    for group, matrices in NONBLOCK_GROUPS.items():
        assert matrix_names_for(group, PATTERN) == matrices


def test_a_sub_path_mflux_does_not_load_is_refused():
    # Bug caught: MATRIX_SUBS naming a matrix mflux's mapping has no target for (it would stay a placeholder).
    targets = [t for t in _mapping() if t.from_pattern != ["layers.{layer}.mlp.up_proj.weight"]]
    with pytest.raises(
        DFloatIntegrationError, match=r"mlp\.up_proj: a compressed matrix with no mflux target"
    ):
        name_map_from_mapping(targets, matrix_subs=MATRIX_SUBS)


# --- the group check, on written tiny checkpoints -------------------------------------------------------------------

_BLOCK = r"layers\.\d+"
_NONBLOCK_PATTERNS = {
    r"time_embedding": ("linear_1", "linear_2"),
    r"adaLN_modulation\.1": (),
    r"final_norm\.linear": (),
}


def _write(root, *, blocks=(0, 1), subs=SEVEN, nonblock=True, extra_groups=()):
    rng = np.random.default_rng(3)
    groups = {f"layers.{b}": [random_bf16(rng, (4, 4)) for _ in subs] for b in blocks}
    patterns = {_BLOCK: subs}
    if nonblock:
        groups["time_embedding"] = [random_bf16(rng, (4, 4)), random_bf16(rng, (4, 4))]
        groups["adaLN_modulation.1"] = [random_bf16(rng, (8, 4))]
        groups["final_norm.linear"] = [random_bf16(rng, (8, 4))]
        patterns.update(_NONBLOCK_PATTERNS)
    for name in extra_groups:
        groups[name] = [random_bf16(rng, (4, 4))]
        patterns[name.replace(".", r"\.")] = ()
    write_checkpoint(root, groups=groups, patterns=patterns)
    return open_checkpoint(root)


@pytest.mark.parametrize("nonblock", [False, True])
def test_the_block_count_is_returned_with_and_without_the_nonblock_groups(tmp_path, nonblock):
    # Bug caught: an optional non-block group counted as a block (or refused when present or absent), or the count
    # off by one.
    assert check_ernie_groups(_write(tmp_path, nonblock=nonblock)) == {"layers": 2}


def test_a_hole_in_the_block_indices_is_refused(tmp_path):
    # Bug caught: layers.0 and layers.2 accepted, layers.1 left on placeholders (a zero-size matmul at run time).
    with pytest.raises(DFloatFormatError, match="layers groups are not contiguous"):
        check_ernie_groups(_write(tmp_path, blocks=(0, 2)))


def test_a_zero_padded_block_index_is_refused(tmp_path):
    # Bug caught: layers.01 counted as block 1 (int("01") == 1), then the seam asks the provider for layers.1, a
    # group the checkpoint does not have: a failure at the first block after the encoder and set loads.
    with pytest.raises(DFloatFormatError, match=r"layers\.01: not an ERNIE-Image group"):
        check_ernie_groups(_write(tmp_path, blocks=("0", "01")))


def test_a_checkpoint_without_blocks_is_refused(tmp_path):
    # Bug caught: a non-block-only checkpoint accepted as zero blocks.
    with pytest.raises(DFloatFormatError, match="no layers groups in the checkpoint"):
        check_ernie_groups(_write(tmp_path, blocks=()))


def test_a_stray_group_is_refused_naming_the_accepted_nonblock_groups(tmp_path):
    # Bug caught: an unknown group (another model's) silently ignored, its matrices never placed.
    with pytest.raises(
        DFloatFormatError,
        match=r"cap_embedder: not an ERNIE-Image group .*time_embedding, adaLN_modulation\.1, final_norm\.linear",
    ):
        check_ernie_groups(_write(tmp_path, extra_groups=("cap_embedder",)))


def test_a_time_embedding_group_with_one_matrix_is_refused(tmp_path):
    # Bug caught: a non-block group of the wrong matrices accepted; time_embedding.linear_2 never installed (a
    # zero-size placeholder in the timestep MLP before the first block).
    rng = np.random.default_rng(5)
    write_checkpoint(
        tmp_path,
        groups={
            "layers.0": [random_bf16(rng, (4, 4)) for _ in SEVEN],
            "time_embedding": [random_bf16(rng, (4, 4))],
        },
        patterns={_BLOCK: SEVEN, "time_embedding": ("linear_1",)},
    )
    with pytest.raises(DFloatFormatError, match=r"time_embedding: holds"):
        check_ernie_groups(open_checkpoint(tmp_path))


def test_a_zimage_block_under_the_same_group_name_is_refused_naming_the_seven_matrices(tmp_path):
    # Bug caught: a Z-Image DF11 checkpoint (its main blocks are also named layers.<n>) passed for
    # ERNIE and failing with a shape error at the first block instead of at build.
    zimage = (
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    )
    with pytest.raises(
        DFloatFormatError, match=r"layers\.0: holds 8 matrices .*self_attention\.to_q"
    ):
        check_ernie_groups(_write(tmp_path, subs=zimage, nonblock=False))


@pytest.mark.mflux
def test_the_real_mapping_matches_the_copy_and_every_matrix_lands_on_a_linear():
    # Bug caught: mflux's real mapping no longer matching the copy (a rename or transform added upstream), or a
    # decoded matrix with no nn.Linear to land on in a real tiny ErnieTransformer.
    import mlx.nn as nn
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
    from mflux.models.ernie_image.weights.ernie_weight_mapping import ErnieWeightMapping
    from tests._ernie_tiny import TINY

    from mlx_dfloat.integrate.placeholders import get_attr_path
    from mlx_dfloat.mflux.ernie.names import ernie_name_map

    real = [
        (t.to_pattern, list(t.from_pattern), t.transform is not None)
        for t in ErnieWeightMapping.get_transformer_mapping()
    ]
    copy = [(t.to_pattern, list(t.from_pattern), t.transform is not None) for t in _mapping()]
    assert sorted(real) == sorted(copy)
    m = ernie_name_map()
    assert m.param_name("adaLN_modulation.1.weight") == "adaln_modulation.weight"
    tf = ErnieTransformer(**TINY)
    for attr in m.attrs_of(KIND):
        assert isinstance(get_attr_path(tf.layers[0], attr), nn.Linear), attr
    for matrices in NONBLOCK_GROUPS.values():
        for name in matrices:
            module = get_attr_path(tf, m.param_name(name).removesuffix(".weight"))
            assert isinstance(module, nn.Linear), name


@pytest.mark.mflux
def test_the_default_layer_count_is_mfluxs_constructor_default():
    # Bug caught: an mflux bump changing ErnieTransformer's default depth (what the published model is built with)
    # while the count check keeps 36: every real checkpoint refused, or one of another depth accepted.
    import inspect

    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    from mlx_dfloat.mflux.ernie.names import DEFAULT_LAYERS

    default = inspect.signature(ErnieTransformer.__init__).parameters["num_layers"].default
    assert default == 36
    assert DEFAULT_LAYERS == 36


@pytest.mark.parametrize("module", ["mlx_dfloat.mflux.ernie", "mlx_dfloat.mflux.ernie.names"])
def test_importing_the_adapter_does_not_import_mflux(module):
    # Bug caught: a module-level mflux import in the adapter (a user without the extra gets a bare ImportError on
    # import instead of the dependency error naming it, and every import pays for mflux).
    import subprocess
    import sys

    code = f"import sys, {module}; print('mflux' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
