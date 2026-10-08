"""The Z-Image build from a DF11 checkpoint, on fakes with Z-Image's parameter layout (no mflux)."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._family_fakes import (
    ZIMAGE_FULL_TABLE,
    D,
    FakeSeamZImageFull,
    FakeZImageFull,
    write_fake_zimage_checkpoint,
)
from tests._flux_fakes import Recorder

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.mflux.zimage.transformer import base_extras, block_lists, build_transformer


def _fake(**kw):
    return FakeSeamZImageFull(Recorder(), **kw)


def _checkpoint(tmp_path, *, n_refiner=1, n_layers=2):
    probe = FakeZImageFull(None, n_refiner_layers=n_refiner, n_layers=n_layers)
    shapes = install_placeholders(block_lists(probe), ZIMAGE_FULL_TABLE)
    _groups, constants = write_fake_zimage_checkpoint(tmp_path, shapes, np.random.default_rng(5))
    return open_checkpoint(tmp_path), shapes, constants


def _params(tf):
    return dict(tree_flatten(tf.parameters()))


def test_the_build_loads_every_extra_under_mflux_names_and_leaves_matrices_as_placeholders(
    tmp_path,
):
    # Bug caught: a diffusers-named extra (t_embedder.mlp.0.weight) not renamed (coverage refuses) or loaded into the
    # wrong module; a block matrix or the cap_embedder matrix left at its random init instead of a placeholder.
    ckpt, shapes, constants = _checkpoint(tmp_path)
    build = build_transformer(ckpt, name_map=ZIMAGE_FULL_TABLE, transformer_class=_fake)
    params = _params(build.transformer)
    assert constants["t_embedder.linear1.weight"] != constants["t_embedder.linear2.weight"]
    for param, constant in constants.items():
        assert np.all(np.array(params[param].view(mx.uint16)) == constant), param
    for block, per in shapes.items():
        for attr in per:
            assert params[f"{block}.{attr}.weight"].size == 0, (block, attr)
    assert params["cap_embedder.1.weight"].size == 0
    assert build.nonblock == {"cap_embedder.1.weight": (D, D)}
    assert build.counts == {"noise_refiner": 1, "context_refiner": 1, "layers": 2}
    assert build.shapes == shapes


def test_a_missing_extra_and_an_extra_without_a_parameter_are_both_refused(tmp_path):
    # Bug caught: a parameter silently left at mflux's random init; an extra silently ignored.
    ckpt, _shapes, _constants = _checkpoint(tmp_path)
    some = dict(ckpt.extras)
    name = "t_embedder.mlp.0.weight"
    without = {k: v for k, v in some.items() if k != name}
    with pytest.raises(
        DFloatIntegrationError, match=r"parameters without an extra .*linear1\.weight"
    ):
        build_transformer(ckpt, name_map=ZIMAGE_FULL_TABLE, extras=without, transformer_class=_fake)
    with pytest.raises(
        DFloatIntegrationError, match=r"extras without a parameter .*no_such_tensor"
    ):
        build_transformer(
            ckpt,
            name_map=ZIMAGE_FULL_TABLE,
            extras={**some, "no_such_tensor": some[name]},
            transformer_class=_fake,
        )


def test_depth_overrides_are_checked_and_the_refiners_share_one_count(tmp_path):
    # Bug caught: n_refiner past the checkpoint (extras plan and shapes disagree) or a negative depth accepted.
    ckpt, _shapes, _constants = _checkpoint(tmp_path)
    for kw in ({"n_refiner": 2}, {"n_layers": 3}, {"n_layers": -1}, {"n_refiner": -1}):
        with pytest.raises(DFloatIntegrationError, match="the checkpoint has 1 / 2"):
            build_transformer(ckpt, name_map=ZIMAGE_FULL_TABLE, transformer_class=_fake, **kw)
    one = build_transformer(ckpt, name_map=ZIMAGE_FULL_TABLE, n_layers=1, transformer_class=_fake)
    # Block 1's extras are left out (not "extras without a parameter"), and both refiner lists follow one count.
    assert one.counts == {"noise_refiner": 1, "context_refiner": 1, "layers": 1}
    assert len(one.transformer.layers) == 1
    none = build_transformer(
        ckpt, name_map=ZIMAGE_FULL_TABLE, n_refiner=0, n_layers=0, transformer_class=_fake
    )
    assert len(none.transformer.noise_refiner) == len(none.transformer.context_refiner) == 0


def test_the_cap_embedder_matrix_loads_from_extras_when_asked(tmp_path):
    # Bug caught: nonblock_from_extras still putting a placeholder on cap_embedder.1.weight (the BF16 side of an
    # identity check would run on an empty weight).
    ckpt, _shapes, _constants = _checkpoint(tmp_path)
    extras = dict(ckpt.extras)
    # any BF16 (D, D) tensor stands in for the base repository's cap_embedder.1.weight
    path, info = extras["all_x_embedder.2-1.weight"]
    assert info.shape == (D, D)
    extras["cap_embedder.1.weight"] = (path, info)
    build = build_transformer(
        ckpt,
        name_map=ZIMAGE_FULL_TABLE,
        extras=extras,
        nonblock_from_extras=True,
        transformer_class=_fake,
    )
    assert build.nonblock == {}
    assert _params(build.transformer)["cap_embedder.1.weight"].shape == (D, D)


def test_base_extras_keeps_the_nonblock_matrix_and_drops_block_matrices():
    # Bug caught: the BF16 side building without cap_embedder.1.weight (coverage refuses) or with block matrices as
    # extras (the whole BF16 transformer resident).
    groups = {
        "layers.0": SimpleNamespace(matrix_names=("layers.0.attention.to_q.weight",)),
        "cap_embedder": SimpleNamespace(matrix_names=("cap_embedder.1.weight",)),
    }
    index = {
        name: (Path(name), None)
        for name in (
            "layers.0.attention.to_q.weight",
            "layers.0.attention_norm1.weight",
            "cap_embedder.1.weight",
            "cap_embedder.1.bias",
            "x_pad_token",
        )
    }
    assert sorted(base_extras(index, SimpleNamespace(groups=groups))) == [
        "cap_embedder.1.bias",
        "cap_embedder.1.weight",
        "layers.0.attention_norm1.weight",
        "x_pad_token",
    ]
