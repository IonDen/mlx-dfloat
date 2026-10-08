"""Tiny ERNIE-Image transformer helpers for the mflux lane. Every mflux import happens inside a function."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tests._df11_fixtures import random_bf16, write_checkpoint

__all__ = [
    "PATTERNS",
    "TINY",
    "TINY_SEED",
    "StubTextEncoder",
    "StubTokenizer",
    "StubVAE",
    "fake_model",
    "stub_components",
    "tiny_groups",
    "tiny_inputs",
    "tiny_predict",
    "tiny_seamed",
    "write_tiny_checkpoint",
]

# Head dim 16 = 32 / 2 heads = the sum of the three RoPE axes (4 + 6 + 6, each even). 128 channels: ErnieLatentCreator
# always makes 128-channel latents (mflux 0.20.0 ernie_latent_creator.py:7). Text width 3072: build_text_batch pads
# every prompt with zeros of width 3072, a literal default (prompt_encoder.py:27).
TINY = {
    "hidden_size": 32,
    "num_attention_heads": 2,
    "num_layers": 2,
    "ffn_hidden_size": 48,
    "in_channels": 128,
    "out_channels": 128,
    "patch_size": 1,
    "text_in_dim": 3072,
    "rope_axes_dim": [4, 6, 6],
}


def tiny_seamed(n_layers=2):
    """A seamed mflux ``ErnieTransformer`` at ``TINY`` size with placeholders installed; returns (tf, shapes)."""
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.ernie.names import ernie_name_map
    from mlx_dfloat.mflux.ernie.transformer import block_lists, seam_transformer_class

    tf = seam_transformer_class()(**{**TINY, "num_layers": n_layers})
    shapes = install_placeholders(block_lists(tf), ernie_name_map())
    seam_blocks(block_lists(tf), tf.seam_cell)
    return tf, shapes


def tiny_inputs(batch=1, seed=0):
    """A 4 x 4 latent of 128 channels, ``batch`` prompts of 8 tokens at width 3072, and a sigma.

    ``batch`` 2 is a CFG batch (mflux runs it as one batch-2 transformer call, ernie_image.py:239-250).
    """
    keys = mx.random.split(mx.random.key(seed), 2)
    return {
        "hidden_states": mx.random.normal((1, 128, 4, 4), key=keys[0]).astype(mx.bfloat16),
        "text_bth": mx.random.normal((batch, 8, 3072), key=keys[1]).astype(mx.bfloat16),
        "text_lens": mx.array([8] * batch, dtype=mx.int32),
        "sigma": mx.array([0.5]),
    }


def tiny_predict(tf, inputs, *, compiled=False, guidance=1.0):
    """One call of mflux's ``ErnieImage._predict`` on ``tf`` with the chip check answering "not M1/M2".

    Unless ``compiled``, the factory runs through ``_compile.uncompiled`` (the adapter's bypass); with ``compiled`` it is
    the stock factory, which compiles off M1/M2 (ernie_image.py:252-254).
    """
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
    from mflux.utils.apple_silicon import AppleSiliconUtil

    from mlx_dfloat.mflux._compile import uncompiled

    original = AppleSiliconUtil.__dict__["is_m1_or_m2"]
    setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))  # noqa: B010
    try:
        args = (tf, inputs["text_bth"], inputs["text_lens"], inputs["hidden_states"])
        fn = ErnieImage._predict(*args) if compiled else uncompiled(ErnieImage._predict, *args)
    finally:
        setattr(AppleSiliconUtil, "is_m1_or_m2", original)  # noqa: B010
    return fn(
        inputs["hidden_states"],
        inputs["sigma"],
        inputs["text_bth"],
        inputs["text_lens"],
        guidance,
    )


def tiny_groups(shapes, rng):
    """One real DF11 group per block of ``shapes``; ERNIE's sub-paths are its attribute paths, so the names match.

    Returns groups, names, sources (``tests/_family_fakes.py:compress_blocks``).
    """
    from tests._family_fakes import compress_blocks

    return compress_blocks(shapes, rng)


# The DF11 config's pattern_dict (mingyi456/ERNIE-Image{,-Turbo}-DF11 config.json, byte-identical in both).
BLOCK_PATTERN = r"layers\.\d+"
PATTERNS = {
    "time_embedding": ("linear_1", "linear_2"),
    "adaLN_modulation.1": (),
    BLOCK_PATTERN: (
        "self_attention.to_q",
        "self_attention.to_k",
        "self_attention.to_v",
        "self_attention.to_out.0",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.linear_fc2",
    ),
    "final_norm.linear": (),
}
# mflux parameter -> checkpoint name, the inverse of ErnieWeightMapping's two renames (ernie_weight_mapping.py:12-19).
_CHECKPOINT_NAME = {
    "adaln_modulation.weight": "adaLN_modulation.1.weight",
    "adaln_modulation.bias": "adaLN_modulation.1.bias",
}
_NONBLOCK_PARAMS = {
    "time_embedding": ("time_embedding.linear_1.weight", "time_embedding.linear_2.weight"),
    "adaLN_modulation.1": ("adaln_modulation.weight",),
    "final_norm.linear": ("final_norm.linear.weight",),
}


def write_tiny_checkpoint(
    root, rng, *, n_layers=2, nonblock_as_groups=True, hidden_override=None, random_extras=None
):
    """A DF11 checkpoint (config.json + safetensors) for mflux's ``ErnieTransformer`` at ``TINY`` size.

    One group per block in ``MATRIX_SUBS`` order; the three non-block groups (or, with ``nonblock_as_groups=False``,
    BF16 extras); every other parameter a BF16 extra under its checkpoint name (``adaln_modulation.*`` stored as
    ``adaLN_modulation.1.*``), filled with a distinct constant ``0x3F80 + i``, except ``x_embedder.proj.weight``: torch
    layout ``(hidden, 128, 1, 1)`` holding the bit patterns ``0x5000 + flat index`` (every value distinct, so any axis
    mix-up of the transpose shows). ``hidden_override`` builds the probe that wide. ``random_extras`` (a NumPy generator)
    fills every extra, the patch conv included, with random BF16 instead: constant weights make the forward pass blind
    to its inputs, which an end-to-end comparison cannot use; the constants' values are then the uint16 arrays in mflux's
    layout (the conv transposed to (hidden, 1, 1, 128)).

    Returns ``(matrices, constants, conv_bits)``: ``{group: {sub-path or parameter: uint16 matrix}}``,
    ``{mflux parameter: constant}`` and the uint16 conv array written.
    """
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.ernie.names import ernie_name_map
    from mlx_dfloat.mflux.ernie.transformer import block_lists

    dims = {**TINY, "num_layers": n_layers}
    if hidden_override is not None:
        dims["hidden_size"] = hidden_override
    probe = ErnieTransformer(**dims)
    probe_params = dict(tree_flatten(probe.parameters()))
    shapes = install_placeholders(block_lists(probe), ernie_name_map())
    matrices, groups = {}, {}
    for block, per in shapes.items():
        matrices[block] = {attr: random_bf16(rng, shape) for attr, shape in per.items()}
        groups[block] = [matrices[block][sub] for sub in PATTERNS[BLOCK_PATTERN]]
    taken = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    patterns = {BLOCK_PATTERN: PATTERNS[BLOCK_PATTERN]}
    if nonblock_as_groups:
        for group, params in _NONBLOCK_PARAMS.items():
            matrices[group] = {p: random_bf16(rng, tuple(probe_params[p].shape)) for p in params}
            groups[group] = [matrices[group][p] for p in params]
            patterns[group] = PATTERNS[group]
            taken |= set(params)
    hidden = dims["hidden_size"]
    conv_bits = (0x5000 + np.arange(hidden * 128, dtype=np.uint16)).reshape(hidden, 128, 1, 1)
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(probe_params.items())):
        if param in taken:
            continue
        if random_extras is not None:
            if param == "x_embedder.proj.weight":
                torch_layout = random_bf16(random_extras, (hidden, 128, 1, 1))
                extras[param] = torch_layout
                constants[param] = torch_layout.transpose(0, 2, 3, 1)
            else:
                bits = random_bf16(random_extras, tuple(array.shape))
                extras[_CHECKPOINT_NAME.get(param, param)] = constants[param] = bits
            continue
        if param == "x_embedder.proj.weight":
            extras[param] = conv_bits
            continue
        constants[param] = 0x3F80 + i
        extras[_CHECKPOINT_NAME.get(param, param)] = np.full(
            array.shape, 0x3F80 + i, dtype=np.uint16
        )
    write_checkpoint(root, groups=groups, patterns=patterns, extras=extras)
    return matrices, constants, conv_bits


TINY_SEED = 5  # ``fake_model`` writes its checkpoint with this seed: a test regenerates the source matrices from it


class StubTokenizer:
    """mflux's tokenizer protocol as ``ErniePromptEncoder`` uses it (mflux 0.20.0 prompt_encoder.py:15-17).

    ``tokenize(prompt, max_length=None)`` gives ids that depend on the prompt (``1 + ord(c) % 7`` per character, a
    blank prompt one ``"_"``) and an all-ones attention mask; ``lengths`` overrides a prompt's token count (the ids then
    cycle through its characters).
    """

    def __init__(self, lengths=None):
        self.lengths = dict(lengths or {})

    def tokenize(self, prompt, max_length=None):
        del max_length
        chars = prompt or "_"
        n = self.lengths.get(prompt, len(chars))
        ids = [1 + ord(chars[i % len(chars)]) % 7 for i in range(n)]
        return SimpleNamespace(
            input_ids=mx.array([ids], dtype=mx.int32),
            attention_mask=mx.ones((1, n), dtype=mx.int32),
        )


class StubTextEncoder(nn.Module):
    """Stands in for the Mistral text stack: ``__call__(input_ids, attention_mask)`` -> (1, L, 3072) bf16 hidden states,
    each position the weight times its id, so a wrong or swapped prompt shows.

    3072 is the width ``build_text_batch`` pads with (prompt_encoder.py:27); mflux calls it positionally (:16).
    """

    def __init__(self):
        super().__init__()
        self.weight = mx.full((1, 3072), 0.01, dtype=mx.bfloat16)
        self.calls = 0

    def __call__(self, input_ids, attention_mask=None):
        self.calls += 1
        return self.weight[None] * input_ids[..., None].astype(mx.bfloat16)


class StubVAE(nn.Module):
    """One weight (so the retained bound can size it) and the decode ``ErnieImage._decode_latents`` calls
    (ernie_image.py:193): a zero image of 16 pixels per latent cell."""

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)

    def decode_packed_latents(self, latents, tiling_config=None):
        del tiling_config
        return mx.zeros((1, 3, 16 * latents.shape[-2], 16 * latents.shape[-1]))


def stub_components(tokenizer=None):
    """Base components as ``ErnieComponents`` with the stub encoder, VAE and tokenizer."""
    from mlx_dfloat.mflux.ernie.init import ErnieComponents

    return ErnieComponents(
        vae=StubVAE(),
        text_encoder=StubTextEncoder(),
        tokenizers={"ernie": tokenizer or StubTokenizer()},
    )


def fake_model(tmp_path, monkeypatch, *, model="ernie-image-turbo", tokenizer=None, **overrides):
    """A ``DFloatErnieImage`` over the tiny real transformer (2 blocks, config.json checkpoint) and stub base parts.

    No weights, no network. As the other families' fakes: the budget is fixed at 23 GiB, the encoder reload builds a
    fresh stub, every decode runs on the NumPy reference.
    """
    from functools import partial

    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux._phases import FamilySizes
    from mlx_dfloat.mflux.ernie import init as einit
    from mlx_dfloat.mflux.ernie import model as model_module
    from mlx_dfloat.mflux.ernie.transformer import build_transformer

    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    ckpt = open_checkpoint(tmp_path / "df11")
    build = build_transformer(ckpt, transformer_overrides=TINY)
    (tmp_path / "base").mkdir(exist_ok=True)
    monkeypatch.setattr(einit, "load_text_encoder", lambda root, **kwargs: StubTextEncoder())
    # A fixed budget: the CI runner's device reports a 7 GiB working set; the tests are about the arithmetic.
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * 1024**3)
    parts = {
        "model": model,
        "model_config": ModelConfig.from_name(model_name=model, base_model=None),
        "ckpt": ckpt,
        "df11": ResolvedRepo(root=tmp_path / "df11", repo_id="mingyi456/tiny", revision="a" * 40),
        "base": ResolvedRepo(root=tmp_path / "base", repo_id="baidu/tiny", revision="b" * 40),
        "components": stub_components(tokenizer),
        "build": build,
        "sizes": FamilySizes(compressed=1_000, extras=100, nonblock=10, encoders=1_000, vae=10),
        "decode": partial(decode_group, backend="reference"),
    }
    parts.update(overrides)
    return model_module.DFloatErnieImage._from_parts(**parts)
