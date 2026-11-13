"""The FLUX.1 release gate: a DF11 checkpoint run through our seam gives stock mflux's latents, bit for bit.

The gate is composed of two halves and a third leg that joins them, and the claim is the three together, never any
one alone.

mflux's FLUX.1 transformer has no small config: every ``nn.Linear`` in ``flux_transformer/*.py`` is a literal width
3072, so one double plus one single block is about 481 M weights, and the vendored DF11 encoder (per-element
Python) cannot compress that in test time. So no single test can run a DF11 checkpoint through mflux's real loop.

(a) The DF11 set feeds the seam the source bits, at a fake width: a real DF11 checkpoint of the fake FLUX-shaped
    transformer, attached through ``DF11Provider``, puts on every block exactly the BF16 matrices that were
    compressed, and gives the same forward as the same seam holding those source matrices resident.
(b) The seam holding those bits gives stock mflux's latents, at full width: mflux's real ``Transformer`` with 1 double
    and 1 single block, seamed and run through ``DFloatFlux1.generate_image`` with the block matrices resident,
    against stock ``Flux1.generate_image`` over a plain ``Transformer`` holding the same arrays.

(c) The two halves place matrices with the same names: (a) uses the test table ``FLUX_TABLE`` and (b) the map
    derived from mflux's weight mapping (``flux_name_map``). ``tests/test_flux1_names.py:132``
    (``test_flux_name_map_derived_from_mflux_equals_the_measured_table``) checks that the derived map equals
    ``FLUX_TABLE``, kind by kind and matrix by matrix.

(a) says the DF11 path and the resident path hand the seam the same bits; (b) says the seam with the resident bits
is stock mflux; (c) says (a)'s placements are (b)'s. Together: a DF11 checkpoint through our seam is stock mflux over
the source weights. No real checkpoint loads: the weights are random, the shapes are mflux's own.
"""

import gc

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tests._flux_fakes import (
    DOUBLE_SUBS,
    FLUX_TABLE,
    SINGLE_SUBS,
    FakeDoubleBlock,
    FakeSeamTransformer,
    FakeSingleBlock,
    Recorder,
    block_lists,
    inputs,
    write_flux_checkpoint,
)

from mlx_dfloat.integrate.placeholders import get_attr_path, install_placeholders

GIB = 1024**3


# --- (a) the DF11 set feeds the seam the source bits (fake width) ----------------------------------------------------


def _bits_seen_by_each_block(monkeypatch):
    """Patch the fake blocks to record, as each one runs, the BF16 bits of every mapped matrix it holds."""
    seen = []
    for cls, prefix, subs in (
        (FakeDoubleBlock, "transformer_blocks", DOUBLE_SUBS),
        (FakeSingleBlock, "single_transformer_blocks", SINGLE_SUBS),
    ):
        real = cls.__call__

        def recording(self, *args, _real=real, _prefix=prefix, _subs=subs, **kwargs):
            seen.append(
                [
                    np.array(
                        get_attr_path(
                            self, FLUX_TABLE.place(f"{_prefix}.0.{sub}.weight").attr
                        ).weight.view(mx.uint16)
                    )
                    for sub in _subs
                ]
            )
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__call__", recording)
    return seen


def _run(tf, provider, shapes):
    tf.attach(provider, shapes, eval_policy="per-block")
    try:
        out = tf(*inputs())
        mx.eval(out)
        return np.array(out.view(mx.uint16))
    finally:
        tf.detach()


def test_the_df11_set_feeds_the_seam_the_source_bits_and_the_resident_sources_forward(
    tmp_path, monkeypatch
):
    # Bug caught: the DF11 path handing a block other bits than were compressed: a group decoded onto another block,
    # a split position off by one (one matrix's tail lands in the next), a matrix name placed on the wrong attribute.
    # Expected bits are the source matrices the checkpoint was written from, not anything our code returns.
    from functools import partial

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider

    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=2)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    source = write_flux_checkpoint(tmp_path / "df11", shapes, np.random.default_rng(3))
    ckpt = open_checkpoint(tmp_path / "df11")
    resident = load_resident_set(ckpt)
    df11 = DF11Provider(
        resident,
        {n: ckpt.groups[n].matrix_names for n in resident},
        FLUX_TABLE,
        decode=partial(decode_group, backend="reference"),
    )
    held = ResidentProvider(
        {
            block: {
                FLUX_TABLE.place(f"{block}.{sub}.weight").attr: mx.array(a).view(mx.bfloat16)
                for sub, a in zip(
                    DOUBLE_SUBS if block.startswith("transformer_blocks.") else SINGLE_SUBS,
                    source[block],
                    strict=True,
                )
            }
            for block in shapes
        }
    )
    seen = _bits_seen_by_each_block(monkeypatch)

    through_df11 = _run(tf, df11, shapes)
    order = [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
    ]
    assert len(seen) == len(order)
    for block, bits in zip(order, seen, strict=True):
        assert len(bits) == len(source[block])
        for got, want in zip(bits, source[block], strict=True):
            assert np.array_equal(got, want), block
    through_source = _run(tf, held, shapes)
    assert np.array_equal(through_df11, through_source)


# --- (b) the seam holding those bits gives stock mflux's latents (full width, 1 + 1 blocks) --------------------------


class _RecordingVAE(nn.Module):
    """A stub VAE that keeps the unpacked latents mflux hands its decoder and returns a 64x64 zero image."""

    def __init__(self):
        super().__init__()
        self.scale = mx.zeros((1,))
        self.seen = []

    def decode(self, latents):
        self.seen.append(latents)
        return mx.zeros((1, 3, 1, 64, 64)) + self.scale


class _T5(nn.Module):
    """FLUX.1's T5 output shape (1, 256, 4096), varying over tokens and channels and with the prompt's ids."""

    def __init__(self):
        super().__init__()
        self.freq = mx.arange(1, 4097, dtype=mx.float32) / 4096

    def __call__(self, ids):
        pos = mx.arange(256, dtype=mx.float32)[:, None]
        return mx.sin(pos * self.freq + ids.astype(mx.float32).sum())[None]


class _Clip(nn.Module):
    """FLUX.1's pooled CLIP output shape (1, 768), varying over channels and with the prompt's ids."""

    def __init__(self):
        super().__init__()
        self.freq = mx.arange(1, 769, dtype=mx.float32) / 97

    def __call__(self, ids):
        return mx.cos(self.freq * (1 + ids.astype(mx.float32).sum()))[None]


class _Tokenizer:
    """``tokenize(prompt)`` with ids that depend on the prompt."""

    def __init__(self, max_length):
        self.max_length = max_length

    def tokenize(self, prompt):
        from tests._flux_fakes import StubTokens

        return StubTokens(mx.array([[ord(c) % 97 + 1 for c in prompt] + [1, 2]]))


def _full_width_latents(tmp_path, monkeypatch):
    """Half (b)'s two runs: (our latents, stock mflux's latents), each the array mflux handed its VAE decoder.

    The full-width weights (about 1 GiB), both transformers and the model live in this frame and in the patches,
    which are undone before it returns: once it has returned, nothing but MLX's buffer cache holds those buffers.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.flux_initializer import FluxInitializer
    from mflux.models.flux.model.flux_transformer.transformer import Transformer
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mlx.utils import tree_flatten, tree_unflatten
    from tests._qwen21_tiny import mflux_uncompiled

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.integrate.providers import ResidentProvider
    from mlx_dfloat.mflux.flux1 import init as base_init
    from mlx_dfloat.mflux.flux1 import model as model_module
    from mlx_dfloat.mflux.flux1.init import BaseComponents, ResolvedRepo
    from mlx_dfloat.mflux.flux1.memory import FluxSizes
    from mlx_dfloat.mflux.flux1.names import flux_name_map
    from mlx_dfloat.mflux.flux1.transformer import seam_transformer_class

    dims = {"num_transformer_blocks": 1, "num_single_transformer_blocks": 1}
    stock_tf = Transformer(ModelConfig.schnell(), **dims)
    # Every parameter random BF16 with small exponents (2^-15 .. 2^-6) so the forward stays finite. The arrays are
    # made before anything evaluates the transformers, so their fp32 initial weights never materialise. The bits are
    # drawn on the device, one parameter at a time, and each one's temporaries are handed back before the next: drawn
    # in NumPy, the freed temporaries stayed in this process's footprint (1.08 GiB on top of the 1.0 GiB of weights).
    keys = iter(mx.random.split(mx.random.key(11), 2 * len(tree_flatten(stock_tf.parameters()))))
    weights = {}
    for name, p in tree_flatten(stock_tf.parameters()):
        # sm: bit 7 becomes the sign, bits 0-6 the mantissa
        sm = mx.random.randint(0, 256, p.shape, dtype=mx.uint16, key=next(keys))
        exponent = mx.random.randint(112, 122, p.shape, dtype=mx.uint16, key=next(keys))
        weights[name] = (((sm & 0x80) << 8) | (exponent << 7) | (sm & 0x7F)).view(mx.bfloat16)
        mx.eval(weights[name])
        mx.clear_cache()  # a buffer leaves the footprint at the next GPU submission: the next parameter's
    mx.eval(mx.zeros((1,)) + 1)  # and, after the last one, this one
    mx.synchronize()
    mx.clear_cache()
    stock_tf.update(tree_unflatten(list(weights.items())))

    ours_tf = seam_transformer_class()(ModelConfig.schnell(), **dims)
    shapes = install_placeholders(
        [
            ("transformer_blocks", ours_tf.transformer_blocks),
            ("single_transformer_blocks", ours_tf.single_transformer_blocks),
        ],
        flux_name_map(),
    )
    matrices = {f"{block}.{attr}.weight" for block, per in shapes.items() for attr in per}
    assert matrices < set(weights)
    ours_tf.update(tree_unflatten([(n, w) for n, w in weights.items() if n not in matrices]))
    resident = {
        block: {attr: weights[f"{block}.{attr}.weight"] for attr in per}
        for block, per in shapes.items()
    }

    ours_vae, stock_vae = _RecordingVAE(), _RecordingVAE()
    tokenizers = {"t5": _Tokenizer(256), "clip": _Tokenizer(77)}
    monkeypatch.setattr(base_init, "load_encoders", lambda root: (_T5(), _Clip()))
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * GIB)
    # The planner reads the largest group sizes off a checkpoint; a fake-width one keeps the fit check trivial.
    fake_shapes = install_placeholders(
        block_lists(FakeSeamTransformer(Recorder(), n_double=1, n_single=1)), FLUX_TABLE
    )
    write_flux_checkpoint(tmp_path / "df11", fake_shapes, np.random.default_rng(3))
    (tmp_path / "base").mkdir()
    model = model_module.DFloatFlux1._from_parts(
        model="schnell",
        model_config=ModelConfig.schnell(),
        ckpt=open_checkpoint(tmp_path / "df11"),
        df11=ResolvedRepo(root=tmp_path / "df11", repo_id="DFloat11/tiny", revision="a" * 40),
        base=ResolvedRepo(
            root=tmp_path / "base", repo_id="black-forest-labs/tiny", revision="b" * 40
        ),
        components=BaseComponents(vae=ours_vae, t5=_T5(), clip=_Clip(), tokenizers=tokenizers),
        transformer=ours_tf,
        shapes=shapes,
        name_map=flux_name_map(),
        sizes=FluxSizes(compressed=1_000, extras=100, encoders=1_000, vae=10),
    )

    def load_resident():
        model.transformer.attach(
            ResidentProvider(resident), shapes, eval_policy=model._policy, verify_in_call=True
        )

    monkeypatch.setattr(model._lifecycle, "_load_set", load_resident)

    stock = Flux1.__new__(Flux1)
    nn.Module.__init__(stock)
    FluxInitializer._init_config(stock, ModelConfig.schnell())
    stock.vae = stock_vae
    stock.t5_text_encoder, stock.clip_text_encoder = _T5(), _Clip()
    stock.tokenizers = tokenizers
    stock.transformer = stock_tf
    stock.bits = None
    stock.lora_paths, stock.lora_scales = [], []

    kwargs = {"seed": 3, "prompt": "p", "num_inference_steps": 2, "height": 64, "width": 64}
    with mflux_uncompiled():
        model.generate_image(**kwargs)
    with mflux_uncompiled():
        stock.generate_image(**kwargs)
    (ours,), (theirs,) = ours_vae.seen, stock_vae.seen
    monkeypatch.undo()  # the patched lifecycle holds the resident set; undone, the caller's cache clear reaches it
    return ours, theirs


@pytest.mark.mflux
def test_a_full_width_generation_through_the_seam_gives_stock_mfluxs_latents_bit_for_bit(
    tmp_path, monkeypatch
):
    # Bug caught: the SeamMixin hooks diverging from mflux's _apply_joint_transformer_block /
    # _apply_single_transformer_block (an argument dropped or swapped, a block's output pair in the wrong order, a
    # block skipped or run twice), a matrix of the resident set placed on the wrong attribute, or our prompt-cache
    # prelude giving mflux's loop other embeddings than mflux computes itself. The reference is stock mflux's
    # Flux1.generate_image over a plain Transformer holding the same mx.array objects (no copy), both sides
    # uncompiled, seed 3, schnell, 2 steps, 64x64; equality is bit for bit (an integer view of the latents).
    try:
        ours, theirs = _full_width_latents(tmp_path, monkeypatch)
        assert bool(mx.all(mx.isfinite(theirs)))
        assert (ours.dtype, ours.shape) == (theirs.dtype, theirs.shape)
        view = mx.uint32 if ours.dtype == mx.float32 else mx.uint16
        assert np.array_equal(np.array(ours.view(view)), np.array(theirs.view(view)))
    finally:
        # The run's full-width buffers (about 1 GiB) would otherwise stay with MLX for the rest of the mflux lane: the
        # model and its lifecycle form a reference cycle, so they are freed only by the cycle collector, and the freed
        # buffers then sit in MLX's buffer cache.
        gc.collect()
        mx.clear_cache()
