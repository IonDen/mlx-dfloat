"""The Qwen-Image 2.1 transformer: the class-swap seam and the eager-forward guard, on a real tiny mflux transformer."""

import importlib.util
import types
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from tests._qwen21_tiny import TINY, tiny_groups, tiny_inputs, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatDependencyError, DFloatIntegrationError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.placeholders import get_attr_path
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider
from mlx_dfloat.mflux.qwen21.transformer import block_lists, seam_transformer_class

COMPILED = type(
    mx.compile(lambda x: x)
)  # mlx.gc_func, a FunctionType subclass: only `type(f) is` tells them apart
BLOCKS = ["transformer_blocks.0", "transformer_blocks.1"]


def test_block_lists_is_the_one_list_the_forward_runs():
    # Bug caught: another attribute than the list _forward iterates (qwen21_transformer.py:96-100), so no block
    # would run through the seam.
    tf = SimpleNamespace(transformer_blocks=[1, 2])
    assert block_lists(tf) == [("transformer_blocks", [1, 2])]


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


def _constant_weights(shapes):
    """Block i's matrices filled with the constant i + 1."""
    return {
        block: {
            attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()
        }
        for i, (block, per) in enumerate(shapes.items())
    }


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


def _counting_provider(shapes, calls):
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    groups, names, _src = tiny_groups(shapes, np.random.default_rng(9))

    def count(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    return DF11Provider(groups, names, qwen21_name_map(), decode=count)


@pytest.mark.mflux
def test_the_stock_transformer_compiles_its_forward_on_the_first_call():
    # Bug caught (the control): mflux no longer compiling _forward in __call__ (qwen21_transformer.py:69-72), which
    # would leave the guard and its note unjustified. mflux compiles there on every chip, no chip check.
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    tf = Qwen21Transformer(**TINY)
    assert tf._step_fn is None
    mx.eval(tf(**tiny_inputs()))
    assert type(tf._step_fn) is COMPILED


@pytest.mark.mflux
def test_the_seamed_transformer_runs_the_plain_forward():
    # Bug caught: mflux's compile reached through the seamed class (the per-block eval would then raise inside it).
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    tf, shapes = tiny_seamed()
    tf.attach(ResidentProvider(_constant_weights(shapes)), shapes)
    mx.eval(tf(**tiny_inputs()))
    assert type(tf._step_fn) is types.MethodType
    assert tf._step_fn.__func__ is Qwen21Transformer._forward
    assert tf._step_fn.__self__ is tf
    assert is_eager(tf)


@pytest.mark.mflux
@pytest.mark.parametrize("step_fn", ["compiled", "another transformer's forward"])
def test_a_step_fn_other_than_this_transformers_plain_forward_is_refused_before_any_block(step_fn):
    # Bug caught: the guard checking only for None (a compiled _step_fn left from mflux, or set by a later mflux
    # elsewhere, runs the blocks inside mx.compile), or accepting any bound _forward (another transformer's weights).
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    tf, shapes = tiny_seamed()
    calls = []
    tf.attach(_counting_provider(shapes, calls), shapes, verify_in_call=True)
    tf._step_fn = (
        mx.compile(tf._forward) if step_fn == "compiled" else Qwen21Transformer(**TINY)._forward
    )
    assert not is_eager(tf)
    with pytest.raises(DFloatIntegrationError, match="compiled the forward pass"):
        tf(**tiny_inputs())
    assert calls == []


@pytest.mark.mflux
def test_the_compiled_forward_cannot_run_the_seam():
    # Bug caught (the reason for the guard, as a control): if mflux's compiled _forward could run per-block evals,
    # the guard and its uncompiled cost would be unjustified. The arguments are built as __call__ builds them
    # (qwen21_transformer.py:61-68).
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    tf, shapes = tiny_seamed()
    zeros = {
        b: {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in per.items()} for b, per in shapes.items()
    }
    tf.attach(ResidentProvider(zeros), shapes)
    inputs = tiny_inputs()
    timestep = Qwen21Transformer._compute_timestep(inputs["t"], inputs["config"])
    rows = mx.concatenate([timestep, mx.zeros((1,), dtype=timestep.dtype)])
    cos, sin, mask = tf._geometry(
        text_len=8,
        latent_height=4,
        latent_width=4,
        encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
    )
    compiled = mx.compile(tf._forward)
    with pytest.raises(ValueError, match=r"\[eval\] Attempting to eval"):
        mx.eval(
            compiled(inputs["hidden_states"], inputs["encoder_hidden_states"], rows, cos, sin, mask)
        )


@pytest.mark.mflux
def test_the_swap_keeps_parameter_paths_and_the_computed_tables():
    # Bug caught: the class swap breaking on Qwen21TransformerBlock, parameters moving under another path
    # (placeholders, load_weights and coverage would miss them), or a computed table renamed upstream (the extras
    # coverage exemption would go stale silently).
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import (
        Qwen21TransformerBlock,
    )

    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.mflux.qwen21.transformer import COMPUTED_PARAMS

    tf = seam_transformer_class()(**TINY)
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert seam_blocks(block_lists(tf), tf.seam_cell) == (
        "transformer_blocks.0",
        "transformer_blocks.1",
    )
    after = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert after == before
    assert isinstance(tf.transformer_blocks[1], Qwen21TransformerBlock)
    # The seven names mflux 0.20.0 computes at construction (qwen21_rope.py:21-26, qwen21_time_text_embed.py:16).
    assert set(COMPUTED_PARAMS) == {
        "pos_embed.cos_tables.0",
        "pos_embed.cos_tables.1",
        "pos_embed.cos_tables.2",
        "pos_embed.sin_tables.0",
        "pos_embed.sin_tables.1",
        "pos_embed.sin_tables.2",
        "time_text_embed.time_proj.freqs",
    }
    assert set(COMPUTED_PARAMS) <= set(after)


def _eager_reference(tf, shapes, weights):
    """Stock mflux holding tf's non-block parameters and ``weights`` on its blocks, running its plain forward.

    The seamed side runs uncompiled, so the reference runs mflux's own ``_forward`` uncompiled too.
    """
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    plain = Qwen21Transformer(**TINY)
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    plain.update(
        tree_unflatten([(k, v) for k, v in tree_flatten(tf.parameters()) if k not in matrices])
    )
    for block, per in weights.items():
        idx = int(block.partition(".")[2])
        for attr, w in per.items():
            get_attr_path(plain.transformer_blocks[idx], attr).weight = w
    plain._step_fn = plain._forward
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
    inputs = tiny_inputs()
    out = tf(**inputs)
    mx.eval(out)
    tf.verify_step()
    want = _eager_reference(tf, shapes, weights)(**inputs)
    mx.eval(want)

    assert len(evals) == 2
    # Each block returns one array: the joint [8 text | 16 image] sequence at inner dim 32.
    assert [tuple(e.shape) for e in evals] == [(1, 24, 32), (1, 24, 32)]
    assert out.shape == (1, 16, 64)  # the image tokens, projected back to 64 channels
    assert bool(mx.isfinite(out).all().item())
    assert out.dtype == want.dtype
    assert mx.array_equal(out.view(mx.uint32), want.view(mx.uint32)).item()
    for block, per in shapes.items():
        idx = int(block.partition(".")[2])
        for attr in per:
            assert get_attr_path(tf.transformer_blocks[idx], attr).weight.size == 0, (block, attr)


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
    mx.eval(tf(**tiny_inputs()))
    tf.verify_step()

    outputs = []
    for _kind, obj in events:
        if not any(obj is seen for seen in outputs):
            outputs.append(obj)
    order = [(kind, next(i for i, o in enumerate(outputs) if o is obj)) for kind, obj in events]
    assert order == [("async", 0), ("async", 1), ("eval", 0), ("eval", 1)]


@pytest.mark.mflux
def test_two_calls_per_step_decode_once_per_block_per_call():
    # Bug caught: the second (negative-prompt) call of a CFG step (qwen_image_21.py:104-118) refused for the first
    # call's unchecked status words, or a block decoded twice in one call. 2 calls x 2 blocks = 4 decodes.
    tf, shapes = tiny_seamed()
    calls = []
    tf.attach(_counting_provider(shapes, calls), shapes, verify_in_call=True)
    positive = tiny_inputs(seed=0)
    negative = {**positive, "encoder_hidden_states": tiny_inputs(seed=1)["encoder_hidden_states"]}
    pos = tf(**positive)
    neg = tf(**negative)
    mx.eval(pos, neg)
    assert calls == BLOCKS * 2
    assert pos.shape == neg.shape == (1, 16, 64)
    assert not mx.array_equal(pos, neg).item()
