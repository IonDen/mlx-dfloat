"""The FLUX.2 Klein transformer with the class-swap seam: the pure parts offline, real forwards in the mflux lane."""

import importlib.util
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from tests._flux2_tiny import TINY, tiny_inputs, tiny_seamed

from mlx_dfloat.errors import DFloatDependencyError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.placeholders import get_attr_path
from mlx_dfloat.integrate.providers import ResidentProvider
from mlx_dfloat.mflux.flux2.transformer import block_lists, seam_transformer_class


def test_block_lists_follow_the_run_order_of_the_transformer_call():
    # Bug caught: the lists in another order than Flux2Transformer.__call__ runs them (double blocks first,
    # transformer.py:132-163), which would run the eval/status bookkeeping against the wrong sequence.
    tf = SimpleNamespace(single_transformer_blocks=[2, 3], transformer_blocks=[1])
    assert block_lists(tf) == [
        ("transformer_blocks", [1]),
        ("single_transformer_blocks", [2, 3]),
    ]


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


def _call(tf, inputs):
    return tf(
        hidden_states=inputs["latents"],
        encoder_hidden_states=inputs["prompt_embeds"],
        timestep=inputs["timestep"],
        img_ids=inputs["latent_ids"],
        txt_ids=inputs["text_ids"],
    )


def _constant_weights(shapes):
    """Block i's matrices filled with the constant i + 1 (1.0, 2.0, 3.0 for the tiny transformer)."""
    return {
        block: {
            attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()
        }
        for i, (block, per) in enumerate(shapes.items())
    }


@pytest.mark.mflux
def test_the_swap_keeps_parameter_paths_and_types_on_the_real_transformer():
    # Bug caught: the class swap breaking on Flux2's block classes (e.g. a __slots__ or __call__ signature clash), or
    # parameters moving under another path (placeholders, load_weights and coverage would miss them).
    from mflux.models.flux2.model.flux2_transformer.single_transformer_block import (
        Flux2SingleTransformerBlock,
    )
    from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock

    from mlx_dfloat.integrate.blockseam import seam_blocks

    tf = seam_transformer_class()(**TINY)
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    names = seam_blocks(block_lists(tf), tf.seam_cell)
    assert names == (
        "transformer_blocks.0",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
    )
    assert sorted(k for k, _v in tree_flatten(tf.parameters())) == before
    assert isinstance(tf.transformer_blocks[0], Flux2TransformerBlock)
    assert isinstance(tf.single_transformer_blocks[1], Flux2SingleTransformerBlock)


@pytest.mark.mflux
def test_a_forward_runs_each_block_with_its_own_weights_and_ends_on_placeholders(monkeypatch):
    # Bug caught: the double block's tuple output not evaluated at the boundary (the next block's decode allocated
    # before this one finished), a block run on another block's weights (the output would differ from a plain mflux
    # transformer holding the same weights), or placeholders not restored.
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    tf, shapes = tiny_seamed(n_double=1, n_single=2)
    weights = _constant_weights(shapes)
    evals = []
    real_eval = seam._eval

    def recording_eval(*args):
        evals.append(args[0])
        real_eval(*args)

    monkeypatch.setattr(seam, "_eval", recording_eval)
    tf.attach(ResidentProvider(weights), shapes)
    inputs = tiny_inputs()
    out = _call(tf, inputs)
    mx.eval(out)
    tf.verify_step()

    # The reference: stock mflux, same non-block parameters, block i's matrices set to the constant i + 1.
    plain = Flux2Transformer(**TINY)
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    plain.update(
        tree_unflatten([(k, v) for k, v in tree_flatten(tf.parameters()) if k not in matrices])
    )
    for block, per in weights.items():
        kind, _dot, idx = block.partition(".")
        for attr, w in per.items():
            get_attr_path(getattr(plain, kind)[int(idx)], attr).weight = w
    want = _call(plain, inputs)
    mx.eval(want)

    assert len(evals) == 3
    assert isinstance(evals[0], tuple)
    assert len(evals[0]) == 2  # (encoder_hidden_states, hidden_states), transformer_block.py:62
    assert out.shape == (1, 16, 128)  # 16 image tokens; proj_out back to in_channels 128
    assert bool(mx.isfinite(out).all().item())
    assert mx.array_equal(out.view(mx.uint16), want.view(mx.uint16)).item()
    for block, per in shapes.items():
        kind, _dot, idx = block.partition(".")
        for attr in per:
            assert get_attr_path(getattr(tf, kind)[int(idx)], attr).weight.size == 0, (block, attr)


@pytest.mark.mflux
def test_depth2_drains_the_last_single_block_and_handles_the_tuple_output(monkeypatch):
    # Bug caught: depth2's look-ahead dropping the tuple from the double block, or end_step not draining the tail.
    # Expected order (seam._seam_eval + end_step): async r0, async r1, eval r0, async r2, eval r1, eval r2.
    tf, shapes = tiny_seamed(n_double=1, n_single=2)
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
    out = _call(tf, tiny_inputs())
    mx.eval(out)
    tf.verify_step()

    outputs = []  # distinct objects in order of first appearance: r0, r1, r2
    for _kind, obj in events:
        if not any(obj is seen for seen in outputs):
            outputs.append(obj)
    assert len(outputs) == 3
    r0, r1, r2 = outputs
    order = [(kind, next(i for i, o in enumerate(outputs) if o is obj)) for kind, obj in events]
    assert order == [
        ("async", 0),
        ("async", 1),
        ("eval", 0),
        ("async", 2),
        ("eval", 1),
        ("eval", 2),
    ]
    assert isinstance(r0, tuple)
    # The double block returns (encoder 8 tokens, image 16 tokens) at inner dim 32; the single blocks see both joined.
    assert [tuple(a.shape) for a in r0] == [(1, 8, 32), (1, 16, 32)]
    assert tuple(r1.shape) == tuple(r2.shape) == (1, 24, 32)
