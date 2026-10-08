"""Tiny Krea 2 transformer helpers for the mflux lane. Every mflux import happens inside a function."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tests._df11_fixtures import pin_layout, random_bf16, write_checkpoint

__all__ = [
    "CHECKPOINT_NAME",
    "TEMPLATE_IDS",
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

# Head dim 16 = 32 / 2 heads = the sum of the RoPE axes mflux derives from it ([16 - 12, 6, 6] = [4, 6, 6],
# transformer.py:39-40); one KV head (wk / wv 16 x 32, a GQA repeat of 2). SwiGLU width int(2 * 32 / 3) * 4 = 84,
# rounded up to 128 (feed_forward.py:7-9). 16 channels and patch 2: Krea2LatentCreator always makes 16-channel
# latents. txtlayers 12 at txtdim 32: the context the transformer unpacks is 12 x 32 = 384 wide (transformer.py:93-101).
TINY = {
    "features": 32,
    "tdim": 32,
    "txtdim": 32,
    "heads": 2,
    "kvheads": 1,
    "multiplier": 4,
    "layers": 2,
    "patch": 2,
    "channels": 16,
    "txtlayers": 12,
    "txtheads": 2,
    "txtkvheads": 2,
}


def tiny_seamed(n_layers=2):
    """A seamed mflux ``Krea2Transformer`` at ``TINY`` size with placeholders installed; returns (tf, shapes)."""
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.krea2.names import krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import block_lists, seam_transformer_class

    tf = seam_transformer_class()(**{**TINY, "layers": n_layers})
    shapes = install_placeholders(block_lists(tf), krea2_name_map())
    seam_blocks(block_lists(tf), tf.seam_cell)
    return tf, shapes


def tiny_inputs(seed=0):
    """An 8 x 8 latent of 16 channels in float32 (as ``Krea2LatentCreator.create_noise``), a sigma, and two prompts.

    The prompts are 6 and 3 tokens at the 384-wide context (12 taps x 32), bf16 as the text encoder returns them;
    ``neg_embeds`` is the negative a CFG step encodes. 8 x 8 latents at patch 2 are 16 image tokens.
    """
    keys = mx.random.split(mx.random.key(seed), 3)
    return {
        "latents": mx.random.normal((1, 16, 8, 8), key=keys[0]),
        "timestep": mx.array([0.5]),
        "embeds": mx.random.normal((1, 6, 384), key=keys[1]).astype(mx.bfloat16),
        "neg_embeds": mx.random.normal((1, 3, 384), key=keys[2]).astype(mx.bfloat16),
    }


def tiny_predict(tf, inputs, *, guidance, compiled=False):
    """One call of mflux's ``Krea2._predict`` on ``tf`` with the chip check answering "not M1/M2".

    ``inputs["neg_embeds"]`` None runs one transformer call, as mflux at guidance 1.0. Unless ``compiled``, the factory
    runs through ``_compile.uncompiled`` (the adapter's bypass); with ``compiled`` it is the stock factory, which
    compiles off base and Pro M1/M2 (krea2.py:202-204).
    """
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2
    from mflux.utils.apple_silicon import AppleSiliconUtil

    from mlx_dfloat.mflux._compile import uncompiled

    original = AppleSiliconUtil.__dict__["is_m1_or_m2"]
    setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))  # noqa: B010
    try:
        args = (tf, inputs["embeds"], inputs["neg_embeds"], guidance)
        fn = Krea2._predict(*args) if compiled else uncompiled(Krea2._predict, *args)
    finally:
        setattr(AppleSiliconUtil, "is_m1_or_m2", original)  # noqa: B010
    return fn(latents=inputs["latents"], timestep=inputs["timestep"])


def tiny_groups(shapes, rng):
    """One real DF11 group per block of ``shapes``; Krea's sub-paths are its attribute paths, so the names match.

    Returns groups, names, sources (``tests/_family_fakes.py:compress_blocks``).
    """
    from tests._family_fakes import compress_blocks

    return compress_blocks(shapes, rng)


# mflux parameter -> checkpoint name: the inverse of Krea2WeightMapping's eleven non-block renames
# (krea2_weight_mapping.py:59-66, 75-81). Every other parameter is stored under its own name.
CHECKPOINT_NAME = {
    "tmlp.linear_in.weight": "tmlp.0.weight",
    "tmlp.linear_in.bias": "tmlp.0.bias",
    "tmlp.linear_out.weight": "tmlp.2.weight",
    "tmlp.linear_out.bias": "tmlp.2.bias",
    "tproj.linear.weight": "tproj.1.weight",
    "tproj.linear.bias": "tproj.1.bias",
    "txtmlp.norm.scale": "txtmlp.0.scale",
    "txtmlp.linear_in.weight": "txtmlp.1.weight",
    "txtmlp.linear_in.bias": "txtmlp.1.bias",
    "txtmlp.linear_out.weight": "txtmlp.3.weight",
    "txtmlp.linear_out.bias": "txtmlp.3.bias",
}


def write_tiny_checkpoint(
    root, rng, *, n_layers=2, nonblock_as_groups=True, random_extras=None, key="tiny-krea"
):
    """A config-less single-file DF11 checkpoint for mflux's ``Krea2Transformer`` at ``TINY`` size, as published.

    The published layout's ``pattern_dict``; one group per block in ``BLOCK_SUBS`` order; the seven non-block groups
    (``tmlp``, ``tproj``, ``txtmlp``, the four text-fusion blocks) or, with ``nonblock_as_groups=False``, their
    matrices as BF16 extras; every other parameter a BF16 extra under its checkpoint name (``CHECKPOINT_NAME``),
    filled with a distinct constant ``0x3F80 + i``. ``random_extras`` (a NumPy generator) fills the extras with random
    BF16 instead: constant matrices blind an RMSNorm-fed forward pass to its inputs. ``key`` names the pinned layout
    (a published layout's key exercises the variant check).

    Returns ``(matrices, constants, layout)``: ``{group: {sub-path or mflux parameter: uint16 matrix}}``, ``{mflux
    parameter: constant}`` (with ``random_extras``, the uint16 arrays written) and the ``SynthesizedLayout`` pinned
    to the written file, for ``open_checkpoint(root, layouts=(layout,))``.
    """
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    from mlx_dfloat._layouts import KREA_2_RAW_COMFYUI
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.mflux.krea2.names import BLOCK_SUBS, NONBLOCK_GROUPS, krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import COMPUTED_PARAMS, block_lists

    names = krea2_name_map()
    probe = Krea2Transformer(**{**TINY, "layers": n_layers})
    probe_params = dict(tree_flatten(probe.parameters()))
    shapes = install_placeholders(block_lists(probe), names)
    matrices, groups = {}, {}
    for block, per in shapes.items():
        matrices[block] = {attr: random_bf16(rng, shape) for attr, shape in per.items()}
        groups[block] = [matrices[block][sub] for sub in BLOCK_SUBS]
    taken = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per} | set(COMPUTED_PARAMS)
    patterns = dict(KREA_2_RAW_COMFYUI.raw_config["pattern_dict"])
    if nonblock_as_groups:
        for group, stored in NONBLOCK_GROUPS.items():
            params = [names.param_name(m) for m in stored]
            matrices[group] = {p: random_bf16(rng, tuple(probe_params[p].shape)) for p in params}
            groups[group] = [matrices[group][p] for p in params]
            taken |= set(params)
    else:
        patterns = {k: v for k, v in patterns.items() if k == r"blocks\.\d+"}
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(probe_params.items())):
        if param in taken:
            continue
        name = CHECKPOINT_NAME.get(param, param)
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
        row_splits={},
        groups=len(groups),
        extras=len(extras),
        key=key,
    )
    return matrices, constants, layout


TINY_SEED = 5  # ``fake_model`` writes its checkpoint with this seed: a test regenerates the source matrices from it
# The stub's chat-template prefix: <|im_start|>, a system token, <|im_start|>, "user", newline. mflux's
# Krea2TextEncoder._template_end finds the second <|im_start|> (151644) followed by "user" (872) and a newline (198)
# and strips everything before the prompt (text_encoder.py:117-127): 5 ids.
TEMPLATE_IDS = (151644, 9, 151644, 872, 198)


class StubTokenizer:
    """mflux's tokenizer protocol as ``Krea2PromptEncoder.encode_prompt`` uses it (prompt_encoder.py:9-15).

    ``tokenize(prompt)`` gives the template prefix ``TEMPLATE_IDS`` and then one id per character, ``1 + ord(c) % 7``
    (``lengths`` overrides a prompt's id count; the ids then cycle through its characters), and an all-ones mask. The
    empty prompt gives no ids at all, as the real Qwen tokenizer does.
    """

    def __init__(self, lengths=None):
        self.lengths = dict(lengths or {})

    def tokenize(self, prompt, **_kwargs):
        if not prompt:
            ids = []
        else:
            n = self.lengths.get(prompt, len(prompt))
            ids = [*TEMPLATE_IDS, *(1 + ord(prompt[i % len(prompt)]) % 7 for i in range(n))]
        return SimpleNamespace(
            input_ids=mx.array([ids], dtype=mx.int32).reshape(1, len(ids)),
            attention_mask=mx.ones((1, len(ids)), dtype=mx.int32),
        )


class StubTextEncoder(nn.Module):
    """Stands in for the Qwen3-VL stack: ``get_prompt_embeds(input_ids, attention_mask)`` -> (1, L', 384) bf16 rows.

    384 = TINY's 12 taps x 32 (the context the transformer unpacks). Each row's direction depends on its id (a sine of
    the id at 384 frequencies; a row that is one vector scaled by the id would look the same to every RMSNorm), and the
    template prefix is stripped as mflux strips it (``Krea2TextEncoder._template_end``).
    """

    def __init__(self):
        super().__init__()
        self.freq = mx.arange(1, 385, dtype=mx.float32) / 7
        self.calls = 0

    def get_prompt_embeds(self, input_ids, attention_mask=None):
        from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

        self.calls += 1
        rows = mx.sin(input_ids[..., None].astype(mx.float32) * self.freq).astype(mx.bfloat16)
        return rows[:, Krea2TextEncoder._template_end(input_ids) :, :]


class StubVAE(nn.Module):
    """One weight (so the retained bound can size it) and the decode ``VAEUtil.decode`` calls (vae_util.py:66): a
    zero image of 8 pixels per latent cell (Krea 2's latents are (1, 16, h / 8, w / 8), unpacked)."""

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)
        self.seen = []

    def decode(self, latents):
        self.seen.append(latents)
        return mx.zeros((1, 3, 8 * latents.shape[-2], 8 * latents.shape[-1]))


def stub_components(tokenizer=None):
    """Base components as ``Krea2Components`` with the stub encoder, VAE and tokenizer."""
    from mlx_dfloat.mflux.krea2.init import Krea2Components

    return Krea2Components(
        vae=StubVAE(),
        text_encoder=StubTextEncoder(),
        tokenizers={"qwen3vl": tokenizer or StubTokenizer()},
    )


def fake_model(
    tmp_path, monkeypatch, *, model="krea-2", tokenizer=None, nonblock="per-call", **overrides
):
    """A ``DFloatKrea2`` over the tiny real transformer (2 blocks, config-less checkpoint) and stub base parts.

    No weights, no network. As the other families' fakes: the budget is fixed at 23 GiB, the encoder reload builds a
    fresh stub, every decode runs on the NumPy reference. ``nonblock`` is ``"per-call"`` (the model's own build: the
    non-block groups decoded at each transformer call) or ``"resident"`` (decoded once at set load).
    """
    from functools import partial

    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux._phases import FamilySizes
    from mlx_dfloat.mflux.krea2 import init as kinit
    from mlx_dfloat.mflux.krea2 import model as model_module
    from mlx_dfloat.mflux.krea2.transformer import build_transformer, seam_transformer_class

    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    ckpt = open_checkpoint(tmp_path / "df11", layouts=(layout,))
    build = build_transformer(
        ckpt,
        transformer_kwargs=TINY,
        transformer_class=seam_transformer_class(per_call=nonblock == "per-call"),
    )
    (tmp_path / "base").mkdir(exist_ok=True)
    monkeypatch.setattr(kinit, "load_text_encoder", lambda root, **kwargs: StubTextEncoder())
    # A fixed budget: the CI runner's device reports a 7 GiB working set; the tests are about the arithmetic.
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * 1024**3)
    parts = {
        "model": model,
        "model_config": ModelConfig.from_name(model_name=model, base_model=None),
        "ckpt": ckpt,
        "df11": ResolvedRepo(root=tmp_path / "df11", repo_id="mingyi456/tiny", revision="a" * 40),
        "base": ResolvedRepo(root=tmp_path / "base", repo_id="krea/tiny", revision="b" * 40),
        "components": stub_components(tokenizer),
        "build": build,
        # A per-call model keeps no decoded non-block bytes with the set.
        "sizes": FamilySizes(
            compressed=1_000,
            extras=100,
            nonblock=0 if nonblock == "per-call" else 10,
            encoders=1_000,
            vae=10,
        ),
        "decode": partial(decode_group, backend="reference"),
    }
    parts.update(overrides)
    return model_module.DFloatKrea2._from_parts(**parts)


def stock_krea2(tokenizer, text_encoder, vae, matrices, extras):
    """Stock mflux ``Krea2`` over a plain tiny ``Krea2Transformer`` holding the BF16 source of a tiny checkpoint.

    ``write_tiny_checkpoint`` returns the block matrices by sub-path and the non-block matrices and the extras by
    mflux's parameter name already, so no rename is needed here.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.krea2.krea2_initializer import Krea2Initializer
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2
    from mlx.utils import tree_flatten, tree_unflatten

    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS

    transformer = Krea2Transformer(**TINY)
    shapes = {name: p.shape for name, p in tree_flatten(transformer.parameters())}
    weights = [
        (
            name if group in NONBLOCK_GROUPS else f"{group}.{name}.weight",
            mx.array(m).view(mx.bfloat16),
        )
        for group, per in matrices.items()
        for name, m in per.items()
    ]
    weights += [(name, mx.array(bits).view(mx.bfloat16)) for name, bits in extras.items()]
    assert sorted(n for n, _ in weights) == sorted(
        shapes
    )  # every parameter from the source, none twice
    assert all(tuple(shapes[n]) == w.shape for n, w in weights)
    transformer.update(tree_unflatten(weights))
    stock = Krea2.__new__(Krea2)
    nn.Module.__init__(stock)
    Krea2Initializer._init_config(stock, ModelConfig.krea2_raw())
    stock.vae, stock.text_encoder, stock.transformer = vae, text_encoder, transformer
    stock.tokenizers = {"qwen3vl": tokenizer}
    stock.bits = None
    stock.lora_paths, stock.lora_scales = None, None
    return stock
