"""The tiny Z-Image checkpoint helper itself: it must match mflux's real parameter layout and the real name map."""

from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._zimage_tiny import TINY, stub_components, write_tiny_checkpoint

from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux.zimage.transformer import build_transformer, seam_transformer_class

pytestmark = pytest.mark.mflux


def test_the_tiny_checkpoint_builds_a_real_mflux_transformer_with_every_extra_in_its_place(
    tmp_path,
):
    # Bug caught: a helper whose diffusers-named extras do not invert the real map's renames (the build refuses an
    # uncovered parameter), a matrix written as an extra, or an extra routed to the wrong parameter.
    groups, constants = write_tiny_checkpoint(tmp_path, np.random.default_rng(7))
    ckpt = open_checkpoint(tmp_path)
    build = build_transformer(ckpt, transformer_class=partial(seam_transformer_class(), **TINY))
    params = dict(tree_flatten(build.transformer.parameters()))
    assert build.counts == {"noise_refiner": 1, "context_refiner": 1, "layers": 2}
    assert set(groups) == set(ckpt.groups)
    # The checkpoint carries the diffusers names (the real DF11 files' spelling), not mflux's.
    assert "t_embedder.mlp.0.weight" in ckpt.extras
    assert "t_embedder.linear1.weight" not in ckpt.extras
    assert "all_final_layer.2-1.adaLN_modulation.1.bias" in ckpt.extras
    assert constants["t_embedder.linear1.weight"] != constants["t_embedder.linear2.weight"]
    for param, constant in constants.items():
        assert np.all(np.array(params[param].view(mx.uint16)) == constant), param
    assert params["cap_embedder.1.weight"].size == 0
    assert build.nonblock == {"cap_embedder.1.weight": (64, 32)}


def test_the_stub_components_encode_through_mflux_s_prompt_encoder():
    # Bug caught: stubs whose shapes mflux's PromptEncoder rejects (it slices the first sequence by the mask's sum).
    from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder

    parts = stub_components()
    out = PromptEncoder.encode_prompt(
        prompt="p", tokenizer=parts.tokenizers["z_image"], text_encoder=parts.text_encoder
    )
    assert out.shape == (8, TINY["cap_feat_dim"])
