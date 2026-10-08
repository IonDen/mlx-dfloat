"""Krea 2 naming: the map derived from mflux's weight mapping, the DF11 group check and the variant check."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat._layouts import KREA_2_RAW_COMFYUI
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import matrix_names_for, open_checkpoint
from mlx_dfloat.mflux._mapping import name_map_from_mapping
from mlx_dfloat.mflux.krea2.names import (
    KIND,
    LAYOUT_FOR_MODEL,
    MATRIX_SUBS,
    NONBLOCK_GROUPS,
    TEXT_FUSION_GROUPS,
    check_krea2_groups,
    check_variant,
)

# The stored order a byte comparison with the originals confirmed in both published files (registration order of
# mflux's modules; see the Krea 2 layout comment in mlx_dfloat/_layouts.py).
EIGHT = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")


def _t(to, frm):
    return SimpleNamespace(to_pattern=to, from_pattern=[frm], transform=None)


def _ids(*keys):
    return [_t(k, k) for k in keys]


def _attn(p):
    return _ids(*(f"{p}.attn.{s}.weight" for s in ("wq", "wk", "wv", "wo", "gate"))) + _ids(
        f"{p}.attn.qknorm.qnorm.scale", f"{p}.attn.qknorm.knorm.scale"
    )


def _mlp(p):
    return _ids(f"{p}.mlp.gate.weight", f"{p}.mlp.up.weight", f"{p}.mlp.down.weight")


def _norms(p):
    return _ids(f"{p}.prenorm.scale", f"{p}.postnorm.scale")


def _mapping():
    # A copy of Krea2WeightMapping.get_transformer_mapping() (mflux 0.20.0 krea2_weight_mapping.py:41-95), in its
    # order: patch embed; the {layer}-templated block targets; the six timestep renames; the text fusion by concrete
    # index (two layerwise blocks, the projector, two refiner blocks); the five text-MLP renames; the final layer.
    # test_the_real_mapping_matches_the_copy_and_every_matrix_lands_on_a_linear compares it with mflux's.
    targets = _ids("first.weight", "first.bias")
    b = "blocks.{layer}"
    targets += _norms(b) + _attn(b) + _mlp(b) + _ids(f"{b}.mod.lin")
    targets += [
        _t("tmlp.linear_in.weight", "tmlp.0.weight"),
        _t("tmlp.linear_in.bias", "tmlp.0.bias"),
        _t("tmlp.linear_out.weight", "tmlp.2.weight"),
        _t("tmlp.linear_out.bias", "tmlp.2.bias"),
        _t("tproj.linear.weight", "tproj.1.weight"),
        _t("tproj.linear.bias", "tproj.1.bias"),
    ]
    for i in range(2):
        p = f"txtfusion.layerwise_blocks.{i}"
        targets += _norms(p) + _attn(p) + _mlp(p)
    targets += _ids("txtfusion.projector.weight")
    for i in range(2):
        p = f"txtfusion.refiner_blocks.{i}"
        targets += _norms(p) + _attn(p) + _mlp(p)
    targets += [
        _t("txtmlp.norm.scale", "txtmlp.0.scale"),
        _t("txtmlp.linear_in.weight", "txtmlp.1.weight"),
        _t("txtmlp.linear_in.bias", "txtmlp.1.bias"),
        _t("txtmlp.linear_out.weight", "txtmlp.3.weight"),
        _t("txtmlp.linear_out.bias", "txtmlp.3.bias"),
    ]
    targets += _ids(
        "last.norm.scale", "last.linear.weight", "last.linear.bias", "last.modulation.lin"
    )
    return targets


def test_the_map_places_every_block_matrix_on_its_own_path_in_decoded_order():
    # Bug caught: attn.gate and attn.wo swapped against the checkpoint (equal shapes: the block runs, the image is
    # noise), or a block sub-path renamed on its way to the module.
    m = name_map_from_mapping(_mapping(), matrix_subs=MATRIX_SUBS)
    assert m.attrs_of("blocks") == EIGHT
    assert m.place("blocks.27.mlp.down.weight").attr == "mlp.down"
    assert m.place("blocks.27.mlp.down.weight").block == "blocks.27"
    assert m.param_name("blocks.3.mod.lin") == "blocks.3.mod.lin"


def test_the_nonblock_renames_are_mfluxs():
    # Bug caught: a rename lost (tmlp.0.weight has no parameter of that name in mflux: an extras-coverage refusal at
    # build), or a text-fusion name swallowed by the block table.
    m = name_map_from_mapping(_mapping(), matrix_subs=MATRIX_SUBS)
    assert m.param_name("tmlp.0.weight") == "tmlp.linear_in.weight"
    assert m.param_name("tmlp.2.bias") == "tmlp.linear_out.bias"
    assert m.param_name("tproj.1.weight") == "tproj.linear.weight"
    assert m.param_name("txtmlp.0.scale") == "txtmlp.norm.scale"
    assert m.param_name("txtmlp.3.weight") == "txtmlp.linear_out.weight"
    name = "txtfusion.refiner_blocks.1.attn.wq.weight"
    assert m.param_name(name) == name


def test_the_tables_are_the_published_layouts_matrix_names():
    # Bug caught: the adapter's order or group names drifting from the layout's (the group check would then refuse
    # both published files, or install a matrix under another's name).
    pattern = KREA_2_RAW_COMFYUI.raw_config["pattern_dict"]
    assert matrix_names_for("blocks.9", pattern) == tuple(
        f"blocks.9.{s}.weight" for s in MATRIX_SUBS[KIND]
    )
    assert MATRIX_SUBS == {"blocks": EIGHT}
    assert TEXT_FUSION_GROUPS == (
        "txtfusion.layerwise_blocks.0",
        "txtfusion.layerwise_blocks.1",
        "txtfusion.refiner_blocks.0",
        "txtfusion.refiner_blocks.1",
    )
    assert {
        "tmlp": ("tmlp.0.weight", "tmlp.2.weight"),
        "tproj": ("tproj.1.weight",),
        "txtmlp": ("txtmlp.1.weight", "txtmlp.3.weight"),
        **{g: tuple(f"{g}.{s}.weight" for s in EIGHT) for g in TEXT_FUSION_GROUPS},
    } == NONBLOCK_GROUPS
    for group, matrices in NONBLOCK_GROUPS.items():
        assert matrix_names_for(group, pattern) == matrices


# --- the group check, on written tiny checkpoints (the published layout's pattern_dict, 4 x 4 matrices) ------------

_PATTERNS = KREA_2_RAW_COMFYUI.raw_config["pattern_dict"]


def _write(root, *, blocks=(0, 1), nonblock=True, extra_groups=(), subs=EIGHT):
    rng = np.random.default_rng(3)
    groups = {f"blocks.{b}": [random_bf16(rng, (4, 4)) for _ in subs] for b in blocks}
    patterns = {r"blocks\.\d+": subs}
    if nonblock:
        for name, matrices in NONBLOCK_GROUPS.items():
            groups[name] = [random_bf16(rng, (4, 4)) for _ in matrices]
        patterns.update({k: v for k, v in _PATTERNS.items() if k != r"blocks\.\d+"})
    for name in extra_groups:
        groups[name] = [random_bf16(rng, (4, 4))]
        patterns[name.replace(".", r"\.")] = ()
    write_checkpoint(root, groups=groups, patterns=patterns)
    return open_checkpoint(root)


@pytest.mark.parametrize("nonblock", [False, True])
def test_the_block_count_is_returned_with_and_without_the_nonblock_groups(tmp_path, nonblock):
    # Bug caught: an optional non-block group (a text-fusion block is named like a block list) counted as a block or
    # refused, or the count off by one.
    assert check_krea2_groups(_write(tmp_path, nonblock=nonblock)) == {"blocks": 2}


def test_a_hole_in_the_block_indices_is_refused(tmp_path):
    # Bug caught: blocks.0 and blocks.2 accepted, blocks.1 left on placeholders (a zero-size matmul at run time).
    with pytest.raises(DFloatFormatError, match=r"^blocks groups are not contiguous: \[0, 2\]$"):
        check_krea2_groups(_write(tmp_path, blocks=(0, 2)))


def test_a_zero_padded_block_index_is_refused(tmp_path):
    # Bug caught: blocks.01 counted as block 1 (int("01") == 1); the seam then asks for blocks.1, a group the
    # checkpoint does not have, after the encoder and the set have loaded.
    with pytest.raises(DFloatFormatError, match=r"^blocks\.01: not a Krea 2 group"):
        check_krea2_groups(_write(tmp_path, blocks=("0", "01")))


def test_a_checkpoint_without_blocks_is_refused(tmp_path):
    # Bug caught: a non-block-only checkpoint accepted as zero blocks.
    with pytest.raises(DFloatFormatError, match=r"^no blocks groups in the checkpoint$"):
        check_krea2_groups(_write(tmp_path, blocks=()))


def test_a_stray_group_is_refused_naming_the_accepted_nonblock_groups(tmp_path):
    # Bug caught: an unknown group (another model's) silently ignored, its matrices never placed.
    with pytest.raises(DFloatFormatError) as err:
        check_krea2_groups(_write(tmp_path, extra_groups=("cap_embedder",)))
    assert str(err.value) == (
        "cap_embedder: not a Krea 2 group (blocks: blocks.<n>; non-block groups accepted: tmlp, tproj, txtmlp, "
        "txtfusion.layerwise_blocks.0, txtfusion.layerwise_blocks.1, txtfusion.refiner_blocks.0, "
        "txtfusion.refiner_blocks.1)"
    )


def test_a_tmlp_group_with_one_matrix_is_refused(tmp_path):
    # Bug caught: a non-block group of the wrong matrices accepted; tmlp.linear_out never installed (a zero-size
    # placeholder in the timestep MLP before the first block).
    rng = np.random.default_rng(5)
    write_checkpoint(
        tmp_path,
        groups={
            "blocks.0": [random_bf16(rng, (4, 4)) for _ in EIGHT],
            "tmlp": [random_bf16(rng, (4, 4))],
        },
        patterns={r"blocks\.\d+": EIGHT, "tmlp": ("0",)},
    )
    with pytest.raises(DFloatFormatError, match=r"^tmlp: holds \['tmlp\.0\.weight'\], expected"):
        check_krea2_groups(open_checkpoint(tmp_path))


def test_a_block_of_other_matrices_is_refused_naming_the_eight(tmp_path):
    # Bug caught: a block whose stored matrices are not Krea's (another model's group under the same name) accepted
    # and failing with a shape error at the first block instead of at build.
    seven = EIGHT[:-1]
    with pytest.raises(DFloatFormatError) as err:
        check_krea2_groups(_write(tmp_path, nonblock=False, subs=seven))
    assert str(err.value) == (
        "blocks.0: holds 7 matrices "
        + str([f"blocks.0.{s}.weight" for s in seven])
        + "; a Krea 2 block holds attn.wq, attn.wk, attn.wv, attn.gate, attn.wo, mlp.gate, mlp.up, mlp.down"
    )


@pytest.mark.mflux
def test_a_qwen_image_21_config_less_checkpoint_is_refused_by_the_group_check(tmp_path):
    # Bug caught: another family's config-less file accepted for Krea 2 and failing with a shape
    # error at the first block (or silently decoding into the wrong modules) instead of at build.
    from tests._qwen21_tiny import write_tiny_checkpoint

    _m, _c, layout = write_tiny_checkpoint(
        tmp_path, np.random.default_rng(2), modulation_as_group=False
    )
    ckpt = open_checkpoint(tmp_path, layouts=(layout,))
    with pytest.raises(
        DFloatFormatError,
        match=r"^transformer_blocks\.0: not a Krea 2 group \(blocks: blocks\.<n>;",
    ):
        check_krea2_groups(ckpt)


def _ckpt(source):
    return SimpleNamespace(root=Path("snap"), config_source=source)


def test_the_turbo_file_is_refused_for_raw_and_the_raw_file_for_turbo():
    # Bug caught: Raw's 25-step default (or the card's 52 steps at 3.5) run silently on Turbo's
    # distilled weights, and a README row labelled Raw measuring Turbo; or the reverse.
    turbo = _ckpt(
        "header and spot checks match layout krea-2-turbo-comfyui (mingyi456/Krea-2-Turbo-DF11-ComfyUI@978d)"
    )
    raw = _ckpt(
        "header and spot checks match layout krea-2-raw-comfyui (mingyi456/Krea-2-Raw-DF11-ComfyUI@8320)"
    )
    with pytest.raises(DFloatFormatError) as err:
        check_variant("krea-2-raw", turbo)
    assert str(err.value) == (
        "snap: this is the Krea 2 Turbo DF11 checkpoint (layout krea-2-turbo-comfyui); run it with --model krea-2"
    )
    with pytest.raises(DFloatFormatError) as err:
        check_variant("krea-2", raw)
    assert str(err.value) == (
        "snap: this is the Krea 2 Raw DF11 checkpoint (layout krea-2-raw-comfyui); run it with --model krea-2-raw"
    )
    check_variant("krea-2-raw", raw)
    check_variant("krea-2", turbo)


@pytest.mark.parametrize("model", ["krea-2-raw", "krea-2"])
@pytest.mark.parametrize(
    "source",
    [
        "config.json",
        "header and spot checks match layout qwen-image-2.1-comfyui (mingyi456/Qwen-Image-2.1-DF11-ComfyUI@1b22)",
        "header and spot checks match layout tiny (t/t@0000)",
    ],
)
def test_a_checkpoint_of_neither_krea_layout_passes_the_variant_check(model, source):
    # Bug caught: a user's own config.json export (or a test's pinned tiny file) refused as "the other model", or any
    # layout but the model's own read as the other Krea model's.
    check_variant(model, _ckpt(source))


def test_each_model_names_its_own_layout():
    # Bug caught: the two keys swapped (every published file refused for its own model) or a layout key that no
    # layout carries.
    from mlx_dfloat._layouts import KREA_2_TURBO_COMFYUI

    assert {
        "krea-2-raw": KREA_2_RAW_COMFYUI.key,
        "krea-2": KREA_2_TURBO_COMFYUI.key,
    } == LAYOUT_FOR_MODEL
    assert LAYOUT_FOR_MODEL == {
        "krea-2-raw": "krea-2-raw-comfyui",
        "krea-2": "krea-2-turbo-comfyui",
    }


@pytest.mark.mflux
def test_the_real_mapping_matches_the_copy_and_every_matrix_lands_on_a_linear():
    # Bug caught: mflux's real mapping no longer matching the copy (a rename or transform added upstream), or a
    # decoded matrix with no nn.Linear to land on in a real tiny Krea2Transformer.
    import mlx.nn as nn
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping
    from tests._krea2_tiny import TINY

    from mlx_dfloat.integrate.placeholders import get_attr_path
    from mlx_dfloat.mflux.krea2.names import krea2_name_map

    real = [
        (t.to_pattern, list(t.from_pattern), t.transform is not None)
        for t in Krea2WeightMapping.get_transformer_mapping()
    ]
    copy = [(t.to_pattern, list(t.from_pattern), t.transform is not None) for t in _mapping()]
    assert sorted(real) == sorted(copy)
    m = krea2_name_map()
    assert m.attrs_of(KIND) == EIGHT
    assert m.param_name("tproj.1.weight") == "tproj.linear.weight"
    tf = Krea2Transformer(**TINY)
    for attr in m.attrs_of(KIND):
        assert isinstance(get_attr_path(tf.blocks[0], attr), nn.Linear), attr
    for matrices in NONBLOCK_GROUPS.values():
        for name in matrices:
            module = get_attr_path(tf, m.param_name(name).removesuffix(".weight"))
            assert isinstance(module, nn.Linear), name


@pytest.mark.mflux
def test_the_default_layer_count_is_mfluxs_constructor_default():
    # Bug caught: an mflux bump changing Krea2Transformer's default depth while the count check keeps 28: every real
    # checkpoint refused, or one of another depth accepted.
    import inspect

    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    from mlx_dfloat.mflux.krea2.names import DEFAULT_LAYERS

    assert inspect.signature(Krea2Transformer.__init__).parameters["layers"].default == 28
    assert DEFAULT_LAYERS == 28


@pytest.mark.parametrize("module", ["mlx_dfloat.mflux.krea2", "mlx_dfloat.mflux.krea2.names"])
def test_importing_the_adapter_does_not_import_mflux(module):
    # Bug caught: a module-level mflux import in the adapter (a user without the extra gets a bare ImportError on
    # import instead of the dependency error naming it, and every import pays for mflux).
    import subprocess
    import sys

    code = f"import sys, {module}; print('mflux' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
