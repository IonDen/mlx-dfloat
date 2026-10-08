"""Tiny FLUX.2 Klein transformer helpers for the mflux lane. Every mflux import happens inside a function."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tests._df11_fixtures import random_bf16, write_checkpoint
from tests._family_fakes import compress_blocks

__all__ = [
    "KLEIN_RENAMES",
    "TINY",
    "TINY_OVERRIDES",
    "TINY_SEED",
    "StubTextEncoder",
    "StubTokenizer",
    "StubVAE",
    "fake_model",
    "stub_components",
    "tiny_groups",
    "tiny_inputs",
    "tiny_seamed",
    "write_tiny_checkpoint",
]

# head dim 16 = sum of the four RoPE axes (mflux 0.20.0 flux2_transformer/pos_embed.py:15-21); inner dim 2 x 16 = 32.
# in_channels stays 128: mflux's Flux2LatentCreator always packs 32 x 4 = 128 channels
# (latent_creator/flux2_latent_creator.py:60-77), and the model-level tests run mflux's own loop.
TINY = {
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "in_channels": 128,
    "timestep_guidance_channels": 16,
    "axes_dims_rope": (4, 4, 4, 4),
}
TINY_OVERRIDES = dict(TINY)  # plays a real ModelConfig.transformer_overrides in the build
TINY_SEED = 5  # ``fake_model`` writes its checkpoint with this seed: a test regenerates the source matrices from it

# The two non-block renames of mflux 0.20.0's Flux2WeightMapping (weights/flux2_weight_mapping.py:19-26): checkpoint
# (diffusers) name -> mflux parameter.
KLEIN_RENAMES = {
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_guidance_embed.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_guidance_embed.linear_2.weight",
}


def tiny_seamed(n_double=1, n_single=2):
    """A seamed mflux ``Flux2Transformer`` at ``TINY`` size with placeholders installed; returns (tf, shapes)."""
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.flux2.names import klein_name_map
    from mlx_dfloat.mflux.flux2.transformer import block_lists, seam_transformer_class

    tf = seam_transformer_class()(**{**TINY, "num_layers": n_double, "num_single_layers": n_single})
    shapes = install_placeholders(block_lists(tf), klein_name_map())
    seam_blocks(block_lists(tf), tf.seam_cell)
    return tf, shapes


def tiny_groups(shapes, rng):
    """One real DF11 group per block of ``shapes``, its matrices named by the checkpoint's sub-paths.

    ``compress_blocks`` names a matrix by its mflux attribute (``attn.to_out``); a FLUX.2 Klein checkpoint and its
    name map use the diffusers sub-path (``attn.to_out.0``). ``install_placeholders`` keeps the map's table order,
    which is ``MATRIX_SUBS`` order, so the two pair up position by position. Returns groups, names, sources.
    """
    from mlx_dfloat.mflux.flux2.names import MATRIX_SUBS

    groups, names, source = compress_blocks(shapes, rng)
    for block, per in shapes.items():
        subs = MATRIX_SUBS[block.partition(".")[0]]
        assert len(subs) == len(per), block
        renamed = tuple(f"{block}.{sub}.weight" for sub in subs)
        source[block] = dict(zip(renamed, source[block].values(), strict=True))
        names[block] = renamed
    return groups, names, source


def tiny_inputs():
    """The keyword inputs of Klein's ``predict`` apart from ``guidance`` (the negative pair included).

    A (1, 128, 4, 4) packed latent is 16 tokens of 128 channels; the prompt is 8 tokens of ``joint_attention_dim``.
    """
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

    keys = mx.random.split(mx.random.key(0), 3)
    prompt = mx.random.normal((1, 8, 32), key=keys[1]).astype(mx.bfloat16)
    negative = mx.random.normal((1, 8, 32), key=keys[2]).astype(mx.bfloat16)
    return {
        "latents": mx.random.normal((1, 16, 128), key=keys[0]).astype(mx.bfloat16),
        "latent_ids": Flux2LatentCreator.prepare_grid_ids(mx.zeros((1, 128, 4, 4)), t_coord=0),
        "prompt_embeds": prompt,
        "text_ids": Flux2PromptEncoder.prepare_text_ids(prompt),
        "negative_prompt_embeds": negative,
        "negative_text_ids": Flux2PromptEncoder.prepare_text_ids(negative),
        "timestep": mx.array(0.5),
    }


def write_tiny_checkpoint(
    root, rng, *, n_double=1, n_single=2, nonblock_as_groups=True, inner_override=None
):
    """A DF11 checkpoint for mflux's ``Flux2Transformer`` at ``TINY`` size (or ``inner_override`` heads wide).

    One DF11 group per block (``MATRIX_SUBS`` order, the checkpoint's sub-names such as ``attn.to_out.0``), the five
    non-block groups (or, with ``nonblock_as_groups=False``, those five matrices as BF16 extras), and every other
    parameter as a BF16 extra under its diffusers name (the inverse of ``KLEIN_RENAMES``), each filled with a
    distinct constant ``0x3F80 + i`` so a mis-routed extra shows. Returns ``(matrices per group, {mflux parameter:
    constant})``.
    """
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    from mlx_dfloat.integrate.placeholders import get_attr_path, install_placeholders
    from mlx_dfloat.mflux.flux2.names import (
        DOUBLE,
        MATRIX_SUBS,
        NONBLOCK_GROUPS,
        SINGLE,
        klein_name_map,
    )
    from mlx_dfloat.mflux.flux2.transformer import block_lists

    dims = {**TINY, "num_layers": n_double, "num_single_layers": n_single}
    if inner_override is not None:
        dims["num_attention_heads"] = inner_override
    probe = Flux2Transformer(**dims)
    nonblock_shapes = {g: tuple(get_attr_path(probe, g).weight.shape) for g in NONBLOCK_GROUPS}
    shapes = install_placeholders(block_lists(probe), klein_name_map())
    groups = {
        block: [
            random_bf16(rng, shape)
            for _sub, shape in zip(MATRIX_SUBS[block.partition(".")[0]], per.values(), strict=True)
        ]
        for block, per in shapes.items()
    }
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    if nonblock_as_groups:
        for g, shape in nonblock_shapes.items():
            groups[g] = [random_bf16(rng, shape)]
            matrices.add(f"{g}.weight")
    inverse = {param: ckpt for ckpt, param in KLEIN_RENAMES.items()}
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(tree_flatten(probe.parameters()))):
        if param in matrices:
            continue
        constants[param] = 0x3F80 + i
        extras[inverse.get(param, param)] = np.full(array.shape, 0x3F80 + i, dtype=np.uint16)
    patterns = {
        r"transformer_blocks\.\d+": MATRIX_SUBS[DOUBLE],
        r"single_transformer_blocks\.\d+": MATRIX_SUBS[SINGLE],
    }
    if nonblock_as_groups:
        patterns |= dict.fromkeys(NONBLOCK_GROUPS, ())
    write_checkpoint(root, groups=groups, patterns=patterns, extras=extras)
    return groups, constants


class StubTokenizer:
    """mflux's tokenizer protocol as ``Flux2PromptEncoder`` calls it: ids and attention mask of shape (1, 8)."""

    def tokenize(self, prompt, max_length=512, **_kwargs):
        return SimpleNamespace(
            input_ids=mx.ones((1, 8), dtype=mx.int32),
            attention_mask=mx.ones((1, 8), dtype=mx.int32),
        )


class StubTextEncoder(nn.Module):
    """Stands in for Qwen3: ``get_prompt_embeds`` -> (1, 8, 32) features (``TINY``'s ``joint_attention_dim``).

    ``Flux2PromptEncoder._get_qwen3_prompt_embeds`` calls ``get_prompt_embeds(input_ids=, attention_mask=,
    hidden_state_layers=)`` (mflux 0.20.0 prompt_encoder.py:38-42).
    """

    def __init__(self):
        super().__init__()
        self.weight = mx.full((8, 32), 0.25, dtype=mx.bfloat16)

    def get_prompt_embeds(self, input_ids, attention_mask=None, hidden_state_layers=(9, 18, 27)):
        return mx.broadcast_to(self.weight[None], (1, 8, 32))


class StubVAE(nn.Module):
    """One weight (so the retained bound can size it) and the decode ``Flux2Klein.generate_image`` calls inline.

    ``decode_packed_latents`` returns a zero image of 16 pixels per packed latent cell (mflux 0.20.0
    flux2_klein.py:129; the packed latents are (1, 128, h / 16, w / 16)).
    """

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)

    def decode_packed_latents(self, packed, tiling_config=None):
        return mx.zeros((1, 3, 16 * packed.shape[2], 16 * packed.shape[3]))


def stub_components():
    """Base components as ``Flux2Components`` with a stub encoder, the stub VAE and a stub tokenizer."""
    from mlx_dfloat.mflux.flux2.init import Flux2Components

    return Flux2Components(
        vae=StubVAE(), text_encoder=StubTextEncoder(), tokenizers={"qwen3": StubTokenizer()}
    )


def fake_model(tmp_path, monkeypatch, model="flux2-klein-4b", **overrides):
    """A ``DFloatFlux2Klein`` over the tiny real transformer (1 double + 2 single blocks) and stub base parts.

    No weights, no network. Mirrors ``tests/_zimage_tiny.py:fake_model``: the budget is fixed at 23 GiB, the encoder
    reload builds a fresh stub, every decode runs on the NumPy reference. ``model`` picks the mflux ``ModelConfig``
    (and so the memory constants: a 4B name keys ``"4b"``, a 9B name ``"9b"``); the transformer stays tiny.
    """
    from functools import partial

    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux._phases import FamilySizes
    from mlx_dfloat.mflux.flux2 import init as finit
    from mlx_dfloat.mflux.flux2 import model as model_module
    from mlx_dfloat.mflux.flux2.transformer import build_transformer

    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    ckpt = open_checkpoint(tmp_path / "df11")
    build = build_transformer(ckpt, transformer_overrides=TINY_OVERRIDES)
    (tmp_path / "base").mkdir(exist_ok=True)
    monkeypatch.setattr(finit, "load_text_encoder", lambda root, model_config: StubTextEncoder())
    # A fixed budget: the CI runner's device reports a 7 GiB working set; the tests are about the arithmetic.
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * 1024**3)
    parts = {
        "model": model,
        "model_config": ModelConfig.from_name(model_name=model, base_model=None),
        "ckpt": ckpt,
        "df11": ResolvedRepo(root=tmp_path / "df11", repo_id="mingyi456/tiny", revision="a" * 40),
        "base": ResolvedRepo(
            root=tmp_path / "base", repo_id="black-forest-labs/tiny", revision="b" * 40
        ),
        "components": stub_components(),
        "build": build,
        "sizes": FamilySizes(compressed=1_000, extras=100, nonblock=10, encoders=1_000, vae=10),
        "decode": partial(decode_group, backend="reference"),
    }
    parts.update(overrides)
    return model_module.DFloatFlux2Klein._from_parts(**parts)
