"""Tiny Z-Image transformer helpers for the mflux lane. Every mflux import happens inside a function."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tests._df11_fixtures import random_bf16, write_checkpoint
from tests._family_fakes import FULL_SUBS, ZIMAGE_RENAMES
from tests._family_fakes import compress_blocks as tiny_groups

__all__ = [
    "TINY",
    "TINY_SEED",
    "StubTextEncoder",
    "StubTokenizer",
    "fake_model",
    "stub_components",
    "tiny_groups",
    "tiny_inputs",
    "tiny_seamed",
    "write_tiny_checkpoint",
]

# head_dim = dim / n_heads = 32 = sum(axes_dims); one refiner block per list, ``n_layers`` main blocks.
TINY_SEED = 3  # ``fake_model`` writes its checkpoint with this seed: a test regenerates the source matrices from it
TINY = {
    "dim": 64,
    "n_heads": 2,
    "cap_feat_dim": 32,
    "axes_dims": [8, 12, 12],
    "axes_lens": [64, 32, 32],
}


def tiny_seamed(n_layers=2):
    """A seamed mflux ``ZImageTransformer`` at ``TINY`` size with placeholders installed; returns (tf, shapes)."""
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.zimage.names import zimage_name_map
    from mlx_dfloat.mflux.zimage.transformer import block_lists, seam_transformer_class

    tf = seam_transformer_class()(n_layers=n_layers, n_refiner_layers=1, **TINY)
    shapes = install_placeholders(block_lists(tf), zimage_name_map())
    seam_blocks(block_lists(tf), tf.seam_cell)
    return tf, shapes


def tiny_inputs():
    """The keyword inputs of mflux's ``predict`` apart from ``guidance`` (text and negative encodings included)."""
    keys = mx.random.split(mx.random.key(0), 3)
    return {
        "latents": mx.random.normal((16, 1, 4, 4), key=keys[0]).astype(mx.bfloat16),
        "timestep": mx.array([0.5]),
        "sigmas": mx.array([1.0, 0.0]),
        "text_encodings": mx.random.normal((8, 32), key=keys[1]),
        "negative_encodings": mx.random.normal((8, 32), key=keys[2]),
    }


class StubTokenizer:
    """mflux's tokenizer protocol for Z-Image: ``tokenize(prompt)`` -> ids and attention mask of shape (1, 8)."""

    def __init__(self, max_length):
        self.max_length = max_length

    def tokenize(self, prompt, **_kwargs):
        return SimpleNamespace(
            input_ids=mx.ones((1, 8), dtype=mx.int32),
            attention_mask=mx.ones((1, 8), dtype=mx.int32),
        )


class StubTextEncoder(nn.Module):
    """Stands in for the Qwen3 encoder: ``(ids, mask)`` -> (1, 8, 32) features (``TINY``'s ``cap_feat_dim``)."""

    def __init__(self):
        super().__init__()
        self.weight = mx.ones((8, 32), dtype=mx.bfloat16)

    def __call__(self, input_ids, attention_mask):
        return mx.broadcast_to(self.weight[None], (1, 8, 32))


def stub_components():
    """Base components as ``ZImageComponents`` with a stub encoder, a one-weight VAE and a stub tokenizer."""
    from mlx_dfloat.mflux.zimage.init import ZImageComponents

    return ZImageComponents(
        vae=nn.Linear(1, 1),
        text_encoder=StubTextEncoder(),
        tokenizers={"z_image": StubTokenizer(512)},
    )


def write_tiny_checkpoint(root, rng, *, n_layers=2, n_refiner_layers=1):
    """A DF11 checkpoint for mflux's ``ZImageTransformer`` at ``TINY`` size.

    One DF11 group per block (``MATRIX_SUBS`` order), the ``cap_embedder`` group, and every other parameter as a BF16
    extra under its diffusers name (the inverse of the six renames), each filled with a distinct constant
    ``0x3F80 + i`` so a mis-routed extra shows. Returns ``(matrices per group, {mflux parameter: constant})``.
    """
    from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.zimage.names import zimage_name_map
    from mlx_dfloat.mflux.zimage.transformer import block_lists

    probe = ZImageTransformer(**TINY, n_layers=n_layers, n_refiner_layers=n_refiner_layers)
    cap_shape = tuple(probe.cap_embedder[1].weight.shape)
    shapes = install_placeholders(block_lists(probe), zimage_name_map())
    groups = {
        block: [random_bf16(rng, per[sub]) for sub in FULL_SUBS[block.partition(".")[0]]]
        for block, per in shapes.items()
    }
    groups["cap_embedder"] = [random_bf16(rng, cap_shape)]
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    matrices.add("cap_embedder.1.weight")
    inverse = {param: ckpt for ckpt, param in ZIMAGE_RENAMES.items()}
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(tree_flatten(probe.parameters()))):
        if param in matrices:
            continue
        constants[param] = 0x3F80 + i
        extras[inverse.get(param, param)] = mx.full(
            array.shape, 0x3F80 + i, dtype=mx.uint16
        ).__array__()
    write_checkpoint(
        root,
        groups=groups,
        patterns={
            r"noise_refiner\.\d+": FULL_SUBS["noise_refiner"],
            r"context_refiner\.\d+": FULL_SUBS["context_refiner"],
            r"layers\.\d+": FULL_SUBS["layers"],
            "cap_embedder": ["1"],
        },
        extras=extras,
    )
    return groups, constants


def fake_model(tmp_path, monkeypatch, model="z-image-turbo", **overrides):
    """A ``DFloatZImage`` over the tiny real transformer and stub base parts; no weights, no network.

    Mirrors ``tests/test_flux1_model.py:_fake_model``: the budget is fixed at 23 GiB, the encoder reload builds a
    fresh stub, and every decode runs on the NumPy reference.
    """
    from functools import partial

    import numpy as np
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux.zimage import init as zinit
    from mlx_dfloat.mflux.zimage import model as model_module
    from mlx_dfloat.mflux.zimage.memory import ZImageSizes
    from mlx_dfloat.mflux.zimage.transformer import build_transformer, seam_transformer_class

    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    ckpt = open_checkpoint(tmp_path / "df11")
    build = build_transformer(ckpt, transformer_class=partial(seam_transformer_class(), **TINY))
    (tmp_path / "base").mkdir(exist_ok=True)
    monkeypatch.setattr(zinit, "load_text_encoder", lambda root: StubTextEncoder())
    # A fixed budget: the CI runner's device reports a 7 GiB working set; the tests are about the arithmetic.
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * 1024**3)
    parts = {
        "model": model,
        "model_config": ModelConfig.from_name(model_name=model, base_model=None),
        "ckpt": ckpt,
        "df11": ResolvedRepo(root=tmp_path / "df11", repo_id="mingyi456/tiny", revision="a" * 40),
        "base": ResolvedRepo(root=tmp_path / "base", repo_id="Tongyi-MAI/tiny", revision="b" * 40),
        "components": stub_components(),
        "build": build,
        "sizes": ZImageSizes(compressed=1_000, extras=100, nonblock=10, encoders=1_000, vae=10),
        "decode": partial(decode_group, backend="reference"),
    }
    parts.update(overrides)
    return model_module.DFloatZImage._from_parts(**parts)
