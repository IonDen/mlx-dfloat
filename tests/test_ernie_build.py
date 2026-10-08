"""The ERNIE-Image build from a DF11 checkpoint: base extras offline, the real tiny build (mflux lane)."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._ernie_tiny import TINY, write_tiny_checkpoint

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux.ernie.transformer import base_extras, build_transformer

# The seven block matrices at TINY size (hidden 32, ffn 48), weights (out, in): ErnieAttention's four Linear(32, 32)
# (mflux 0.20.0 attention.py:15-25), ErnieFeedForward's gate_proj/up_proj Linear(32, 48) and linear_fc2 Linear(48, 32)
# (feed_forward.py:8-13).
SEVEN = {
    "self_attention.to_q": (32, 32),
    "self_attention.to_k": (32, 32),
    "self_attention.to_v": (32, 32),
    "self_attention.to_out.0": (32, 32),
    "mlp.gate_proj": (48, 32),
    "mlp.up_proj": (48, 32),
    "mlp.linear_fc2": (32, 48),
}
# The four non-block matrices at hidden 32 (transformer.py:36-76): time_embedding.linear_{1,2} Linear(32, 32),
# adaln_modulation Linear(32, 6 x 32), final_norm.linear Linear(32, 2 x 32).
NONBLOCK = {
    "time_embedding.linear_1.weight": (32, 32),
    "time_embedding.linear_2.weight": (32, 32),
    "adaln_modulation.weight": (192, 32),
    "final_norm.linear.weight": (64, 32),
}


def test_base_extras_drop_the_block_matrices_and_keep_the_nonblock_ones():
    # Bug caught: the norms dropped with the block (an extras-coverage refusal on the BF16 side) or the block matrices
    # kept as extras (the BF16 side would load them as plain weights next to its decoded ones), or the non-block
    # matrices dropped (the BF16 side loads them as plain weights).
    groups = {
        "layers.0": SimpleNamespace(
            matrix_names=("layers.0.mlp.up_proj.weight", "layers.0.self_attention.to_q.weight")
        ),
        "adaLN_modulation.1": SimpleNamespace(matrix_names=("adaLN_modulation.1.weight",)),
    }
    index = {
        name: (Path(name), None)
        for name in (
            "layers.0.mlp.up_proj.weight",
            "layers.0.adaLN_sa_ln.weight",
            "adaLN_modulation.1.weight",
            "x_embedder.proj.weight",
        )
    }
    assert sorted(base_extras(index, SimpleNamespace(groups=groups))) == [
        "adaLN_modulation.1.weight",
        "layers.0.adaLN_sa_ln.weight",
        "x_embedder.proj.weight",
    ]


def _open(tmp_path, **kwargs):
    root = tmp_path / "df11"
    matrices, constants, conv = write_tiny_checkpoint(root, np.random.default_rng(3), **kwargs)
    return open_checkpoint(root), matrices, constants, conv


def _params(build):
    return dict(tree_flatten(build.transformer.parameters()))


@pytest.mark.mflux
def test_the_build_installs_placeholders_and_loads_every_extra_by_name(tmp_path):
    # Bug caught: the modulation rename lost (an extras-coverage refusal), a compressed matrix left live (memory), or
    # an extra loaded onto the wrong parameter (a constant in the wrong place).
    ckpt, _m, constants, _conv = _open(tmp_path)
    build = build_transformer(ckpt, transformer_overrides=TINY)
    params = _params(build)
    assert build.nonblock == NONBLOCK
    for name in NONBLOCK:
        assert params[name].size == 0, name
    assert build.shapes == {f"layers.{i}": SEVEN for i in range(2)}
    for block in build.shapes:
        for attr in SEVEN:
            assert params[f"{block}.{attr}.weight"].size == 0, (block, attr)
    # Every parameter outside the compressed groups and the patch conv (mflux 0.20.0 ErnieTransformer at 2 layers).
    assert sorted(constants) == [
        "adaln_modulation.bias",
        "final_linear.bias",
        "final_linear.weight",
        "final_norm.linear.bias",
        "layers.0.adaLN_mlp_ln.weight",
        "layers.0.adaLN_sa_ln.weight",
        "layers.0.self_attention.norm_k.weight",
        "layers.0.self_attention.norm_q.weight",
        "layers.1.adaLN_mlp_ln.weight",
        "layers.1.adaLN_sa_ln.weight",
        "layers.1.self_attention.norm_k.weight",
        "layers.1.self_attention.norm_q.weight",
        "text_proj.weight",
        "time_embedding.linear_1.bias",
        "time_embedding.linear_2.bias",
        "x_embedder.proj.bias",
    ]
    for name, value in constants.items():
        assert np.all(np.array(params[name].view(mx.uint16)) == value), name
    assert build.counts == {"layers": 2}


@pytest.mark.mflux
def test_the_patch_conv_is_transposed_from_torch_layout(tmp_path):
    # Bug caught: the transform dropped (a shape refusal) or applied on the wrong axes ((0, 3, 2, 1) lands (3, 5)
    # elsewhere; every value is distinct, so no axis mix-up can match). Bits by hand: 0x5000 + o * 128 + i.
    ckpt, _m, _c, _conv = _open(tmp_path)
    weight = _params(build_transformer(ckpt, transformer_overrides=TINY))["x_embedder.proj.weight"]
    assert weight.shape == (32, 1, 1, 128)
    for (o, i), bits in {(0, 0): 20480, (3, 5): 20869, (31, 127): 24575}.items():
        assert int(weight[o, 0, 0, i].view(mx.uint16).item()) == bits, (o, i)


@pytest.mark.mflux
def test_nonblock_groups_stored_as_extras_load_as_plain_weights(tmp_path):
    # Bug caught: a checkpoint without the non-block groups refused, or its weights left on placeholders (a zero-size
    # matmul before the first block).
    ckpt, _m, constants, _conv = _open(tmp_path, nonblock_as_groups=False)
    assert "adaLN_modulation.1.weight" in ckpt.extras
    build = build_transformer(ckpt, transformer_overrides=TINY)
    weight = _params(build)["adaln_modulation.weight"]
    assert build.nonblock == {}
    assert weight.shape == (192, 32)
    assert np.all(np.array(weight.view(mx.uint16)) == constants["adaln_modulation.weight"])


@pytest.mark.mflux
def test_a_checkpoint_with_other_block_counts_is_refused_naming_both(tmp_path):
    # Bug caught: a 3-block checkpoint built as 2 (a shape error at block 2, or the block silently dropped).
    ckpt, _m, _c, _conv = _open(tmp_path, n_layers=3)
    with pytest.raises(DFloatFormatError, match=r"checkpoint has 3 blocks; this model builds 2\b"):
        build_transformer(ckpt, transformer_overrides=TINY)


@pytest.mark.mflux
def test_the_count_check_defaults_to_mfluxs_depth(tmp_path):
    # Bug caught: overrides without num_layers checked against something other than mflux's default 36
    # (transformer.py:41): the published model's overrides carry only rope_axes_dim.
    ckpt, _m, _c, _conv = _open(tmp_path)
    no_depth = {k: v for k, v in TINY.items() if k != "num_layers"}
    with pytest.raises(DFloatFormatError, match=r"checkpoint has 2 blocks; this model builds 36\b"):
        build_transformer(ckpt, transformer_overrides=no_depth)


@pytest.mark.mflux
def test_n_layers_builds_fewer_blocks_and_is_range_checked(tmp_path):
    # Bug caught: a partial build loading block 1's norms into a transformer without block 1 (an extra without a
    # parameter), or a request beyond the checkpoint built from nothing.
    ckpt, _m, _c, _conv = _open(tmp_path)
    build = build_transformer(ckpt, transformer_overrides=TINY, n_layers=1)
    assert build.counts == {"layers": 1}
    assert len(build.transformer.layers) == 1
    with pytest.raises(DFloatIntegrationError, match=r"asked for 3 blocks; the checkpoint has 2"):
        build_transformer(ckpt, transformer_overrides=TINY, n_layers=3)


@pytest.mark.mflux
def test_a_checkpoint_of_another_width_is_refused_by_the_extras_shape_check(tmp_path):
    # Bug caught: the same block count at another hidden width (a hand-made checkpoint) accepted.
    ckpt, _m, _c, _conv = _open(tmp_path, hidden_override=16)
    with pytest.raises(DFloatFormatError, match="shape"):
        build_transformer(ckpt, transformer_overrides=TINY)


@pytest.mark.mflux
def test_the_decoded_block_matrices_land_in_pattern_order(tmp_path):
    # Bug caught: the order swapped between the checkpoint, the reader and the map (gate_proj and up_proj have the
    # same shape, so only their values show it: the block would run up(x) * gelu(gate(x)) on the wrong halves).
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.integrate.providers import DF11Provider
    from mlx_dfloat.mflux.ernie.names import NONBLOCK_GROUPS, ernie_name_map

    ckpt, matrices, _c, _conv = _open(tmp_path)
    build = build_transformer(ckpt, transformer_overrides=TINY)
    blocks = [g for g in ckpt.groups if g not in NONBLOCK_GROUPS]
    provider = DF11Provider(
        load_resident_set(ckpt, blocks),
        {g: ckpt.groups[g].matrix_names for g in blocks},
        ernie_name_map(),
        decode=partial(decode_group, backend="reference"),
    )
    got = provider.weights_for("layers.1", build.shapes["layers.1"])
    provider.verify()
    for attr in SEVEN:
        want = matrices["layers.1"][attr]
        assert np.array_equal(np.array(got[attr].view(mx.uint16)), want), attr
