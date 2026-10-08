"""Tiny Qwen-Image 2.1 transformer helpers for the mflux lane. Every mflux import happens inside a function."""

import zlib
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tests._df11_fixtures import pin_layout, random_bf16, write_checkpoint

__all__ = [
    "PREFIX_IDS",
    "STORED_SUBS",
    "TINY",
    "TINY_ROW_SPLITS",
    "TINY_SEED",
    "StubTextEncoder",
    "StubTokenizer",
    "StubVAE",
    "fake_model",
    "stub_components",
    "tiny_config",
    "tiny_groups",
    "tiny_inputs",
    "tiny_seamed",
    "write_tiny_checkpoint",
]

# Head dim 16 = the sum of the three RoPE axes (4 + 6 + 6, each even: mflux 0.20.0 qwen21_rope.py:23-26 builds one
# table per axis at half its width); inner dim 2 x 16 = 32. in_channels stays the default 64: Qwen21LatentCreator
# always makes 64-channel latents (latent_creator/qwen21_latent_creator.py:13-14), and the model-level tests run
# mflux's own loop.
TINY = {
    "num_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "context_in_dim": 32,
    "mlp_ratio": 3,
    "axes_dims_rope": (4, 6, 6),
}


def tiny_config(steps=2, size=64):
    """mflux's ``Config`` for Qwen-Image 2.1 at ``size`` pixels square (4 x 4 = 16 latent tokens at 64)."""
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    return Config(
        model_config=ModelConfig.qwen_image_21(),
        num_inference_steps=steps,
        height=size,
        width=size,
        guidance=1.0,
        scheduler="linear",
    )


def tiny_inputs(seed=0):
    """The keyword inputs of ``Qwen21Transformer.__call__``: 16 image tokens of 64 channels, 8 prompt tokens of 32.

    The mask is all ones, so mflux takes its padding-free path (``qwen21_transformer.py:117-123``).
    """
    keys = mx.random.split(mx.random.key(seed), 2)
    return {
        "t": 0,
        "config": tiny_config(),
        "hidden_states": mx.random.normal((1, 16, 64), key=keys[0]).astype(mx.bfloat16),
        "encoder_hidden_states": mx.random.normal((1, 8, 32), key=keys[1]).astype(mx.bfloat16),
        "encoder_hidden_states_mask": mx.ones((1, 8), dtype=mx.int32),
    }


def tiny_seamed(n_layers=2):
    """A seamed mflux ``Qwen21Transformer`` at ``TINY`` size with placeholders installed; returns (tf, shapes)."""
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map
    from mlx_dfloat.mflux.qwen21.transformer import block_lists, seam_transformer_class

    tf = seam_transformer_class()(**{**TINY, "num_layers": n_layers})
    shapes = install_placeholders(block_lists(tf), qwen21_name_map())
    seam_blocks(block_lists(tf), tf.seam_cell)
    return tf, shapes


def tiny_groups(shapes, rng):
    """One real DF11 group per block of ``shapes``; Qwen's sub-paths are its attribute paths, so the names match.

    Returns groups, names, sources (``tests/_family_fakes.py:compress_blocks``).
    """
    from tests._family_fakes import compress_blocks

    return compress_blocks(shapes, rng)


# The ComfyUI export's stored order (S0): gate_layer and proj stacked by rows as img_mlp.gate_up, gate first.
STORED_SUBS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_up",
    "img_mlp.out",
)
TINY_ROW_SPLITS = {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}


def write_tiny_checkpoint(
    root, rng, *, n_layers=2, modulation_as_group=True, inner_override=None, random_extras=None
):
    """A config-less single-file DF11 checkpoint for mflux's ``Qwen21Transformer`` at ``TINY`` size, as ComfyUI stores it.

    One group per block in the stored order (``STORED_SUBS``: ``img_mlp.gate_up`` = ``concat(gate_layer, proj)`` by
    rows), ``modulation.1`` as a one-matrix group (or, with ``modulation_as_group=False``, a BF16 extra), and every
    other parameter mflux does not compute as a BF16 extra under its checkpoint name, each filled with a distinct
    constant ``0x3F80 + i`` so a mis-routed extra shows. ``inner_override`` builds the probe that many heads wide.
    ``random_extras`` (a NumPy generator) fills the extras with random BF16 instead: constant matrices make the forward
    pass blind to its inputs (a LayerNorm's output sums to zero across channels, so a constant ``proj_out`` maps every
    input to its bias), which an end-to-end comparison cannot use.

    Returns ``(matrices, constants, layout)``: ``{group: {sub-path: uint16 matrix}}`` (the seven decoded sub-paths,
    or ``"weight"`` for ``modulation.1``), ``{mflux parameter: constant}``, and the ``SynthesizedLayout`` pinned to
    the written file, for ``open_checkpoint(root, layouts=(layout,))``. With ``random_extras`` the constants'
    values are the uint16 arrays written.
    """
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map
    from mlx_dfloat.mflux.qwen21.transformer import COMPUTED_PARAMS, block_lists

    dims = {**TINY, "num_layers": n_layers}
    if inner_override is not None:
        dims["num_attention_heads"] = inner_override
    probe = Qwen21Transformer(**dims)
    modulation_shape = tuple(probe.modulation.layers[1].weight.shape)
    shapes = install_placeholders(block_lists(probe), qwen21_name_map())
    matrices, groups = {}, {}
    for block, per in shapes.items():
        mats = {attr: random_bf16(rng, shape) for attr, shape in per.items()}
        matrices[block] = mats
        gate_up = np.concatenate([mats["img_mlp.gate_layer"], mats["img_mlp.proj"]])
        stored = {**mats, "img_mlp.gate_up": gate_up}
        groups[block] = [stored[sub] for sub in STORED_SUBS]
    taken = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per} | set(COMPUTED_PARAMS)
    patterns = {r"transformer_blocks\.\d+": STORED_SUBS}
    if modulation_as_group:
        matrices["modulation.1"] = {"weight": random_bf16(rng, modulation_shape)}
        groups["modulation.1"] = [matrices["modulation.1"]["weight"]]
        patterns[r"modulation\.1"] = ()
        taken.add("modulation.layers.1.weight")
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(tree_flatten(probe.parameters()))):
        if param in taken:
            continue
        name = "modulation.1.weight" if param == "modulation.layers.1.weight" else param
        if random_extras is None:
            constants[param] = 0x3F80 + i
            extras[name] = np.full(array.shape, 0x3F80 + i, dtype=np.uint16)
        else:
            extras[name] = constants[param] = random_bf16(random_extras, tuple(array.shape))
    write_checkpoint(
        root, groups=groups, patterns=patterns, extras=extras, single_file=True, write_config=False
    )
    layout = pin_layout(
        root / "model.safetensors",
        pattern_dict=patterns,
        row_splits=TINY_ROW_SPLITS,
        groups=len(groups),
        extras=len(extras),
        key="tiny-qwen",
    )
    return matrices, constants, layout


TINY_SEED = 5  # ``fake_model`` writes its checkpoint with this seed: a test regenerates the source matrices from it
PREFIX_IDS = 3  # the stub tokenizer's system prefix: mflux drops this many ids (qwen21_prompt_encoder.py:28-31)


class StubTokenizer:
    """mflux's tokenizer protocol as ``Qwen21PromptEncoder`` uses it (mflux 0.20.0 qwen21_prompt_encoder.py:26-38).

    ``tokenize(prompt)`` gives ids and an attention mask of shape (1, L), L = ``lengths.get(prompt, 8)``; the wrapped
    tokenizer encodes the system prefix to ``PREFIX_IDS`` ids. The ids depend on the prompt (its CRC-32 and the
    position, in 1..251 so they stay exact in bfloat16), so the prompts the tests use get different ids.
    """

    def __init__(self, lengths=None):
        self.lengths = dict(lengths or {})
        self.tokenizer = lambda text, add_special_tokens=True: {"input_ids": [1, 2, 3]}

    def tokenize(self, prompt, **_kwargs):
        n = self.lengths.get(prompt, 8)
        seed = zlib.crc32(prompt.encode())
        ids = [(seed + 7 * i) % 251 + 1 for i in range(n)]
        return SimpleNamespace(
            input_ids=mx.array([ids], dtype=mx.int32),
            attention_mask=mx.ones((1, n), dtype=mx.int32),
        )


class StubTextEncoder(nn.Module):
    """Stands in for the Qwen3-VL text stack: ``__call__(input_ids, attention_mask)`` -> (1, L, 32) hidden states,
    each position the weight times its id.

    32 is ``TINY``'s ``context_in_dim``; mflux calls it with keyword arguments (qwen21_prompt_encoder.py:27).
    """

    def __init__(self):
        super().__init__()
        self.weight = mx.full((1, 32), 0.25, dtype=mx.bfloat16)
        self.calls = 0

    def __call__(self, input_ids, attention_mask=None):
        self.calls += 1
        # Each position's hidden state is its id x the weight: the output depends on the prompt's ids.
        return self.weight[None] * input_ids[..., None].astype(mx.bfloat16)


class StubVAE(nn.Module):
    """One weight (so the retained bound can size it) and the decode ``VAEUtil.decode`` calls (vae_util.py:66).

    mflux unpacks the latents to (1, 64, 1, h / 16, w / 16) first (qwen21_latent_creator.py:26-29); the stub returns
    a zero image of 16 pixels per latent cell.
    """

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)

    def decode(self, latents):
        return mx.zeros((1, 3, 16 * latents.shape[-2], 16 * latents.shape[-1]))


def stub_components(tokenizer=None):
    """Base components as ``Qwen21Components`` with the stub encoder, VAE and tokenizer."""
    from mlx_dfloat.mflux.qwen21.init import Qwen21Components

    return Qwen21Components(
        vae=StubVAE(),
        text_encoder=StubTextEncoder(),
        tokenizers={"qwen21": tokenizer or StubTokenizer()},
    )


def fake_model(tmp_path, monkeypatch, *, tokenizer=None, **overrides):
    """A ``DFloatQwenImage21`` over the tiny real transformer (2 blocks, config-less checkpoint) and stub base parts.

    No weights, no network. As ``tests/_flux2_tiny.py:fake_model``: the budget is fixed at 23 GiB, the encoder reload
    builds a fresh stub, every decode runs on the NumPy reference.
    """
    from functools import partial

    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux._phases import FamilySizes
    from mlx_dfloat.mflux.qwen21 import init as qinit
    from mlx_dfloat.mflux.qwen21 import model as model_module
    from mlx_dfloat.mflux.qwen21.transformer import build_transformer

    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    ckpt = open_checkpoint(tmp_path / "df11", layouts=(layout,))
    build = build_transformer(ckpt, transformer_kwargs=TINY)
    (tmp_path / "base").mkdir(exist_ok=True)
    monkeypatch.setattr(qinit, "load_text_encoder", lambda root, **kwargs: StubTextEncoder())
    # A fixed budget: the CI runner's device reports a 7 GiB working set; the tests are about the arithmetic.
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * 1024**3)
    parts = {
        "model": "qwen-image-2.1",
        "model_config": ModelConfig.qwen_image_21(),
        "ckpt": ckpt,
        "df11": ResolvedRepo(root=tmp_path / "df11", repo_id="mingyi456/tiny", revision="a" * 40),
        "base": ResolvedRepo(root=tmp_path / "base", repo_id="Qwen/tiny", revision="b" * 40),
        "components": stub_components(tokenizer),
        "build": build,
        "sizes": FamilySizes(compressed=1_000, extras=100, nonblock=10, encoders=1_000, vae=10),
        "decode": partial(decode_group, backend="reference"),
    }
    parts.update(overrides)
    return model_module.DFloatQwenImage21._from_parts(**parts)
