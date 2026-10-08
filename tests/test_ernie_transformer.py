"""The ERNIE-Image transformer: the class-swap seam and the compile bypass, on a real tiny mflux transformer."""

import importlib.util
import types
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from tests._ernie_tiny import TINY, tiny_groups, tiny_inputs, tiny_predict, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatDependencyError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.placeholders import get_attr_path
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider
from mlx_dfloat.mflux._compile import uncompiled
from mlx_dfloat.mflux.ernie.transformer import block_lists, seam_transformer_class

COMPILED = type(
    mx.compile(lambda x: x)
)  # mlx.gc_func, a FunctionType subclass: only `type(f) is` tells them apart
BLOCKS = ["layers.0", "layers.1"]


@pytest.fixture
def not_m1(monkeypatch):
    """mflux's chip check answers "not a base or Pro M1/M2": stock mflux compiles ERNIE's predict there.

    On this M1 Max the unpatched check is already False; CI's base M1/M2 runners answer True, where mflux would hand
    back the plain function without our override and a bypass test would pass vacuously.
    """
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    return AppleSiliconUtil


def _stock_args(inputs):
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    tf = ErnieTransformer(**TINY)
    return (tf, inputs["text_bth"], inputs["text_lens"], inputs["hidden_states"])


def _random_weights(shapes, seed=11):
    """Every block matrix its own random bfloat16 values (small, so the tiny forward stays finite)."""
    rng = np.random.default_rng(seed)
    return {
        block: {
            attr: mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32)).astype(
                mx.bfloat16
            )
            for attr, shape in per.items()
        }
        for block, per in shapes.items()
    }


def _constant_weights(shapes):
    """Block i's matrices filled with the constant i + 1."""
    return {
        block: {
            attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()
        }
        for i, (block, per) in enumerate(shapes.items())
    }


def _call_inputs(inputs):
    """``ErnieTransformer.__call__`` keyword inputs from ``tiny_inputs``: the timestep is sigma x 1000, as mflux."""
    return {
        "hidden_states": inputs["hidden_states"],
        "timestep": mx.broadcast_to(inputs["sigma"] * 1000, (inputs["text_bth"].shape[0],)),
        "text_bth": inputs["text_bth"],
        "text_lens": inputs["text_lens"],
    }


def test_block_lists_is_the_one_list_the_forward_runs():
    # Bug caught: another attribute than the list __call__ iterates (transformer.py:157-159), so no block would run
    # through the seam.
    tf = SimpleNamespace(layers=[1, 2])
    assert block_lists(tf) == [("layers", [1, 2])]


def test_the_seamed_class_needs_the_mflux_extra(monkeypatch):
    # Bug caught: the adapter importing mflux without the guard (a bare ImportError instead of the package's own
    # dependency error naming the extra). The class factory is cached, so the cache is emptied around the call.
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "mflux" else real(name, *a, **k),
    )
    seam_transformer_class.cache_clear()
    try:
        with pytest.raises(DFloatDependencyError, match=r"mlx-dfloat\[mflux\]"):
            seam_transformer_class()
    finally:
        seam_transformer_class.cache_clear()


@pytest.mark.mflux
def test_off_m1_m2_the_stock_factory_compiles_and_on_them_it_does_not(not_m1, monkeypatch):
    # Bug caught (the control): the reason for the bypass gone (an mflux change: ernie_image.py:252-254 no longer
    # compiling off M1/M2), or a test of the bypass that passes vacuously on a base M1/M2 CI runner (where the
    # unpatched chip check is True). On this M1 Max the unpatched check is already False: stock mflux compiles here.
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

    args = _stock_args(tiny_inputs())
    assert type(ErnieImage._predict(*args)) is COMPILED
    monkeypatch.setattr(not_m1, "is_m1_or_m2", classmethod(lambda cls: True))
    assert type(ErnieImage._predict(*args)) is types.FunctionType


@pytest.mark.mflux
def test_uncompiled_returns_mfluxs_plain_predict_on_any_chip(not_m1):
    # Bug caught: the chip override not reaching ErnieImage._predict (the seam's per-block mx.eval would then run
    # inside mx.compile), or leaking past the factory call (every later mflux factory would run uncompiled).
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

    fn = uncompiled(ErnieImage._predict, *_stock_args(tiny_inputs()))
    assert type(fn) is types.FunctionType
    assert not_m1.is_m1_or_m2() is False


@pytest.mark.mflux
def test_the_compiled_predict_cannot_run_the_seam():
    # Bug caught: none in our code; it pins why the bypass exists (if the compiled predict could run per-block evals,
    # the bypass and its cost on Max/Ultra and M3+ chips would be unjustified).
    tf, shapes = tiny_seamed()
    tf.attach(ResidentProvider(_constant_weights(shapes)), shapes)
    with pytest.raises(ValueError, match=r"\[eval\] Attempting to eval"):
        mx.eval(tiny_predict(tf, tiny_inputs(), compiled=True))


@pytest.mark.mflux
def test_the_swap_keeps_parameter_paths_and_nothing_is_computed():
    # Bug caught: the class swap breaking on ErnieTransformerBlock or moving parameters under another path; a
    # parameter mflux computes at construction added upstream (no checkpoint holds it: every build would be refused by
    # the extras coverage), or a stale exemption hiding a parameter no checkpoint fills.
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
    from mflux.models.ernie_image.model.ernie_transformer.transformer_block import (
        ErnieTransformerBlock,
    )
    from mflux.models.ernie_image.weights.ernie_weight_mapping import ErnieWeightMapping

    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.mflux.ernie.transformer import COMPUTED_PARAMS

    tf = seam_transformer_class()(**TINY)
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert seam_blocks(block_lists(tf), tf.seam_cell) == ("layers.0", "layers.1")
    after = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert after == before
    assert isinstance(tf.layers[1], ErnieTransformerBlock)

    params = {k for k, _v in tree_flatten(ErnieTransformer(**TINY).parameters())}
    targets = set()
    for t in ErnieWeightMapping.get_transformer_mapping():
        if "{layer}" in t.to_pattern:
            targets |= {t.to_pattern.replace("{layer}", str(i)) for i in range(2)}
        else:
            targets.add(t.to_pattern)
    # mflux 0.20.0 computes nothing at construction: every parameter is a mapping target.
    assert params - targets == set()
    assert len(COMPUTED_PARAMS) == 0


def _eager_reference(tf, shapes, weights):
    """Stock mflux holding tf's non-block parameters and ``weights`` on its blocks."""
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    plain = ErnieTransformer(**TINY)
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    plain.update(
        tree_unflatten([(k, v) for k, v in tree_flatten(tf.parameters()) if k not in matrices])
    )
    for block, per in weights.items():
        idx = int(block.partition(".")[2])
        for attr, w in per.items():
            get_attr_path(plain.layers[idx], attr).weight = w
    return plain


@pytest.mark.mflux
def test_a_forward_runs_each_block_with_its_own_weights_and_ends_on_placeholders(monkeypatch):
    # Bug caught: a block run on another block's weights, or a matrix landing on another attribute of its own block
    # (to_q and to_k swapped: same shape, so only per-matrix values show it), either way the output differs from
    # stock mflux holding the same weights; a block output not evaluated at its boundary; or the placeholders not
    # restored afterwards.
    tf, shapes = tiny_seamed()
    weights = _random_weights(shapes)
    evals = []
    real_eval = seam._eval

    def recording_eval(*args):
        evals.append(args[0])
        real_eval(*args)

    monkeypatch.setattr(seam, "_eval", recording_eval)
    tf.attach(ResidentProvider(weights), shapes)
    inputs = _call_inputs(tiny_inputs())
    out = tf(**inputs)
    mx.eval(out)
    tf.verify_step()
    want = _eager_reference(tf, shapes, weights)(**inputs)
    mx.eval(want)

    assert len(evals) == 2
    # Each block returns one array: the joint [16 image | 8 text] sequence at hidden size 32.
    assert [tuple(e.shape) for e in evals] == [(1, 24, 32), (1, 24, 32)]
    assert out.shape == (1, 128, 4, 4)
    assert bool(mx.isfinite(out).all().item())
    # float32: the tiny transformer's non-block parameters are mflux's float32 init (the published ones are BF16).
    assert out.dtype == want.dtype == mx.float32
    assert mx.array_equal(out.view(mx.uint32), want.view(mx.uint32)).item()
    for block, per in shapes.items():
        idx = int(block.partition(".")[2])
        for attr in per:
            assert get_attr_path(tf.layers[idx], attr).weight.size == 0, (block, attr)


@pytest.mark.mflux
def test_depth2_drains_the_last_block(monkeypatch):
    # Bug caught: end_step not draining the tail (the last block's output never evaluated inside the step), or the
    # look-ahead evaluating in the wrong order. Expected (seam._seam_eval + end_step): async r0, async r1, eval r0,
    # eval r1.
    tf, shapes = tiny_seamed()
    events = []
    real_eval, real_async = seam._eval, seam._async_eval

    def recording_eval(*args):
        events.append(("eval", args[0]))
        real_eval(*args)

    def recording_async(*args):
        events.append(("async", args[0]))
        real_async(*args)

    monkeypatch.setattr(seam, "_eval", recording_eval)
    monkeypatch.setattr(seam, "_async_eval", recording_async)
    tf.attach(ResidentProvider(_constant_weights(shapes)), shapes, eval_policy="depth2")
    mx.eval(tf(**_call_inputs(tiny_inputs())))
    tf.verify_step()

    outputs = []
    for _kind, obj in events:
        if not any(obj is seen for seen in outputs):
            outputs.append(obj)
    order = [(kind, next(i for i, o in enumerate(outputs) if o is obj)) for kind, obj in events]
    assert order == [("async", 0), ("async", 1), ("eval", 0), ("eval", 1)]


@pytest.mark.mflux
def test_a_cfg_batch_decodes_each_block_once():
    # Bug caught: CFG split into two transformer calls (four decodes, as Klein and Qwen do), or a block decoded twice
    # in one call. mflux runs a CFG step as one batch-2 call (ernie_image.py:239-250): 2 decodes per step.
    from mlx_dfloat.mflux.ernie.names import ernie_name_map

    tf, shapes = tiny_seamed()
    groups, names, _src = tiny_groups(shapes, np.random.default_rng(9))
    calls = []

    def count(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    tf.attach(
        DF11Provider(groups, names, ernie_name_map(), decode=count), shapes, verify_in_call=True
    )
    out = tiny_predict(tf, tiny_inputs(batch=2), guidance=4.0)
    mx.eval(out)
    assert calls == BLOCKS
    assert out.shape == (1, 128, 4, 4)
    assert bool(mx.isfinite(out).all().item())


def test_importing_the_transformer_module_does_not_import_mflux():
    # Bug caught: a module-level mflux import in the transformer module (a user without the extra gets a bare
    # ImportError on import).
    import subprocess
    import sys

    code = "import sys, mlx_dfloat.mflux.ernie.transformer; print('mflux' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.mflux
def test_a_cfg_batch_through_the_seam_gives_stock_mfluxs_prediction_bit_for_bit(monkeypatch):
    # Bug caught (T2): the batch-2 call wrong through the seam where batch 1 is right: the two prompts' rows mixed,
    # their unequal text lengths (5 and 8 tokens, masked differently) mishandled, or the guidance combination fed the
    # wrong half. The reference is mflux's own predict over a stock transformer holding the same weights, both
    # uncompiled; equality is bit for bit.
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
    from mflux.utils.apple_silicon import AppleSiliconUtil

    tf, shapes = tiny_seamed()
    weights = _random_weights(shapes)
    tf.attach(ResidentProvider(weights), shapes, verify_in_call=True)
    inputs = {**tiny_inputs(batch=2), "text_lens": mx.array([5, 8], dtype=mx.int32)}
    ours = tiny_predict(tf, inputs, guidance=4.0)
    mx.eval(ours)
    plain = _eager_reference(tf, shapes, weights)
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    predict = ErnieImage._predict(
        plain, inputs["text_bth"], inputs["text_lens"], inputs["hidden_states"]
    )
    assert type(predict) is types.FunctionType  # stock, uncompiled like ours
    theirs = predict(
        inputs["hidden_states"], inputs["sigma"], inputs["text_bth"], inputs["text_lens"], 4.0
    )
    mx.eval(theirs)
    assert ours.shape == theirs.shape == (1, 128, 4, 4)
    assert ours.dtype == theirs.dtype
    view = mx.uint32 if ours.dtype == mx.float32 else mx.uint16
    assert mx.array_equal(ours.view(view), theirs.view(view)).item()
