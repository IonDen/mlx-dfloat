"""The compile bypass on FLUX.2 Klein's ``_predict`` and the CFG step's call count, on a real tiny transformer."""

import types

import mlx.core as mx
import numpy as np
import pytest
from tests._flux2_tiny import tiny_groups, tiny_inputs, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider
from mlx_dfloat.mflux._compile import uncompiled

pytestmark = pytest.mark.mflux

PLAIN = types.FunctionType
COMPILED = type(
    mx.compile(lambda x: x)
)  # mlx.gc_func, a FunctionType subclass: only `type(f) is` tells them apart
BLOCKS = ["transformer_blocks.0", "single_transformer_blocks.0", "single_transformer_blocks.1"]


def _as_max_chip(monkeypatch):
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    return AppleSiliconUtil


def _counting_provider(shapes, calls):
    from mlx_dfloat.mflux.flux2.names import klein_name_map

    groups, names, _src = tiny_groups(shapes, np.random.default_rng(9))

    def count(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    return DF11Provider(groups, names, klein_name_map(), decode=count)


def test_on_a_max_chip_klein_predict_compiles_and_the_bypass_returns_the_plain_function(
    monkeypatch,
):
    # Bug caught: the bypass not reaching Flux2Klein's factory (it reads is_m1_or_m2 at call time,
    # flux2_klein.py:292-294), so the seam's per-block mx.eval would run inside mx.compile. First assert is the control.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    util = _as_max_chip(monkeypatch)
    assert type(Flux2Klein._predict(object())) is COMPILED
    assert type(uncompiled(Flux2Klein._predict, object())) is PLAIN
    assert util.is_m1_or_m2() is False  # restored
    assert type(Flux2Klein._predict(object())) is COMPILED  # stock mflux still compiles


def test_an_uncompiled_cfg_step_decodes_once_per_block_per_call(monkeypatch):
    # Bug caught: the negative branch (flux2_klein.py:280-289) skipping the seam, a block decoded twice in one call, or
    # the second call refused for the first call's unchecked status words. 2 calls x 3 blocks = 6 decodes.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    _as_max_chip(monkeypatch)
    tf, shapes = tiny_seamed(n_double=1, n_single=2)
    calls = []
    tf.attach(_counting_provider(shapes, calls), shapes, verify_in_call=True)
    noise = uncompiled(Flux2Klein._predict, tf)(guidance=4.0, **tiny_inputs())
    mx.eval(noise)
    assert calls == BLOCKS * 2
    assert noise.shape == (1, 16, 128)
    assert bool(mx.isfinite(noise).all().item())


def test_without_a_negative_pair_the_step_is_one_call(monkeypatch):
    # Bug caught: a distilled step (guidance 1.0, no negative encodings, flux2_klein.py:280) running the transformer
    # twice.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    _as_max_chip(monkeypatch)
    tf, shapes = tiny_seamed(n_double=1, n_single=2)
    calls = []
    tf.attach(_counting_provider(shapes, calls), shapes, verify_in_call=True)
    inputs = {**tiny_inputs(), "negative_prompt_embeds": None, "negative_text_ids": None}
    noise = uncompiled(Flux2Klein._predict, tf)(guidance=1.0, **inputs)
    mx.eval(noise)
    assert calls == BLOCKS


def test_the_stock_compiled_klein_predict_cannot_run_the_seam(monkeypatch):
    # Bug caught (the reason for the bypass, as a control): if the compiled predict could run per-block evals, the
    # bypass and its M3+ cost would be unjustified.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    _as_max_chip(monkeypatch)
    tf, shapes = tiny_seamed(n_double=1, n_single=1)
    zeros = {
        b: {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in per.items()} for b, per in shapes.items()
    }
    tf.attach(ResidentProvider(zeros), shapes)
    inputs = {**tiny_inputs(), "negative_prompt_embeds": None, "negative_text_ids": None}
    with pytest.raises(ValueError, match=r"\[eval\] Attempting to eval"):
        mx.eval(Flux2Klein._predict(tf)(guidance=1.0, **inputs))
