"""The class-swap block seam on Z-Image-shaped fakes (no mflux)."""

from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._family_fakes import (
    ZIMAGE_TABLE,
    FakeRefinerBlock,
    FakeSeamZImage,
    compress_blocks,
    zimage_block_lists,
    zimage_inputs,
)
from tests._flux_fakes import Recorder, resident_dicts

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.blockseam import seam_blocks
from mlx_dfloat.integrate.placeholders import get_attr_path, install_placeholders
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider


def _seamed(rec, **kw):
    tf = FakeSeamZImage(rec, **kw)
    shapes = install_placeholders(zimage_block_lists(tf), ZIMAGE_TABLE)
    seam_blocks(zimage_block_lists(tf), tf.seam_cell)
    return tf, shapes


def test_the_swap_keeps_every_parameter_path_and_the_block_type():
    # Bug caught: a wrapper proxy moving each block's parameters under `.inner` (placeholders, load_weights and
    # the extras coverage would all miss them), or a swap that breaks isinstance for mflux code that checks it.
    tf = FakeSeamZImage(Recorder())
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    names = seam_blocks(zimage_block_lists(tf), tf.seam_cell)
    assert names == ("noise_refiner.0", "context_refiner.0", "layers.0", "layers.1")
    assert sorted(k for k, _v in tree_flatten(tf.parameters())) == before
    assert isinstance(tf.layers[0], FakeRefinerBlock)
    assert type(tf.layers[0]).__name__ == "SeamFakeRefinerBlock"


def test_each_block_runs_with_its_own_weights_and_ends_on_placeholders():
    # Bug caught: every block bound to one name (all would see block 1's weights), or the placeholders not put back
    # after the call (the decoded weights stay referenced by the module).
    rec = Recorder()
    tf, shapes = _seamed(rec)
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes)
    out = tf(*zimage_inputs())
    mx.eval(out)
    assert [float(w.reshape(-1)[0]) for w in rec.seen] == [1.0, 2.0, 3.0, 4.0]
    for block, per in shapes.items():
        kind, _dot, idx = block.partition(".")
        for attr in per:
            assert get_attr_path(getattr(tf, kind)[int(idx)], attr).weight.size == 0, (
                f"{block}.{attr}"
            )


def test_depth2_evaluates_every_block_output_across_the_kind_boundaries(monkeypatch):
    # Bug caught: the look-ahead `prev` dropped where the run crosses from noise_refiner to context_refiner to layers
    # (a block's output never evaluated in-step), or end_step not draining the last block.
    rec = Recorder()
    tf, shapes = _seamed(rec)
    monkeypatch.setattr(seam, "_eval", lambda *a: rec.events.append(("eval", id(a[0]))))
    monkeypatch.setattr(seam, "_async_eval", lambda *a: rec.events.append(("async", id(a[0]))))
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="depth2")
    tf(*zimage_inputs())
    runs = [i for kind, i in rec.events if kind == "run"]
    assert len(runs) == 4
    assert [e for e in rec.events if e[0] != "run"] == [
        ("async", runs[0]),
        ("async", runs[1]),
        ("eval", runs[0]),
        ("async", runs[2]),
        ("eval", runs[1]),
        ("async", runs[3]),
        ("eval", runs[2]),
        ("eval", runs[3]),
    ]


def test_a_block_called_before_attach_is_refused():
    # Bug caught: a seamed block silently running on its zero-size placeholders when nothing is attached.
    tf, _shapes = _seamed(Recorder())
    with pytest.raises(DFloatIntegrationError, match="attach"):
        tf.layers[0](mx.ones((1, 2, 4)), mx.ones((1, 4)))


def test_seaming_a_block_twice_is_refused():
    # Bug caught: a second swap nesting the seam (each call decodes twice and restores placeholders mid-block).
    tf, _shapes = _seamed(Recorder())
    with pytest.raises(DFloatIntegrationError, match="already seamed"):
        seam_blocks(zimage_block_lists(tf), tf.seam_cell)


def test_two_transformer_calls_per_step_need_verify_in_call():
    # Bug caught (the CFG invariant): a family whose step calls the transformer twice (Z-Image base with guidance)
    # attached without verify_in_call, so the second call's begin_step refuses the first call's unchecked status
    # words; and, with it on, a decode count other than one per block per call.
    rec = Recorder()
    tf, shapes = _seamed(rec)
    groups, names, _src = compress_blocks(shapes, np.random.default_rng(5))
    calls = []
    count = lambda g: calls.append(g.name) or decode_group(g, backend="reference")  # noqa: E731
    provider = DF11Provider(groups, names, ZIMAGE_TABLE, decode=count)
    tf.attach(provider, shapes, verify_in_call=False)
    mx.eval(tf(*zimage_inputs()))
    with pytest.raises(DFloatIntegrationError, match="unchecked"):
        tf(*zimage_inputs())
    tf.detach()
    calls.clear()
    tf.attach(DF11Provider(groups, names, ZIMAGE_TABLE, decode=count), shapes, verify_in_call=True)
    mx.eval(tf(*zimage_inputs()))
    mx.eval(tf(*zimage_inputs()))
    assert calls == ["noise_refiner.0", "context_refiner.0", "layers.0", "layers.1"] * 2


def test_a_raise_inside_a_block_restores_placeholders_and_clears_pending_words():
    # Bug caught: abort_step skipped on the class-swap path (stale status words refuse the retry) or the raising
    # block left holding its decoded weights.
    rec = Recorder()
    tf, shapes = _seamed(rec)
    groups, names, _src = compress_blocks(shapes, np.random.default_rng(6))
    provider = DF11Provider(
        groups, names, ZIMAGE_TABLE, decode=partial(decode_group, backend="reference")
    )
    tf.attach(provider, shapes)
    with pytest.raises(ValueError, match=r"\[matmul\] Last dimension"):
        tf(mx.ones((1, 2, 5)), mx.ones((1, 2, 4)), mx.ones((1, 4)))  # D=4 blocks given width 5
    assert provider.pending == []
    assert tf.noise_refiner[0].attention.to_q.weight.size == 0  # the block that raised


class _HookedProvider(ResidentProvider):
    """A resident provider with the optional ``before_block`` hook: records the hook and the request in order."""

    def __init__(self, per_block, rec):
        super().__init__(per_block)
        self.rec = rec

    def before_block(self, block_name, args, kwargs):
        self.rec.events.append(("before", block_name, tuple(id(a) for a in args), dict(kwargs)))

    def weights_for(self, block_name, shapes):
        self.rec.events.append(("weights", block_name))
        return super().weights_for(block_name, shapes)


def test_a_providers_before_block_hook_runs_before_each_request_with_the_blocks_own_inputs():
    # Bug caught: the hook called after the weights are requested (a provider releasing memory there would release
    # it after the block's decode, not before), called with another block's inputs, or not called for every block.
    # Block 0 of layers is called with (concat(x, cap), t_emb); layers.1 with layers.0's output.
    rec = Recorder()
    tf, shapes = _seamed(rec)
    x, cap, t_emb = zimage_inputs()
    tf.attach(_HookedProvider(resident_dicts(shapes), rec), shapes)
    mx.eval(tf(x, cap, t_emb))
    runs = [e[1] for e in rec.events if e[0] == "run"]
    order = [(e[0], e[1]) for e in rec.events if e[0] != "run"]
    names = ["noise_refiner.0", "context_refiner.0", "layers.0", "layers.1"]
    assert order == [step for n in names for step in (("before", n), ("weights", n))]
    hooks = {e[1]: e[2:] for e in rec.events if e[0] == "before"}
    assert hooks["noise_refiner.0"] == ((id(x), id(t_emb)), {})
    assert hooks["context_refiner.0"][0][0] != id(
        cap
    )  # cap went through the resident cap_embedder first
    assert hooks["layers.1"] == ((runs[2], id(t_emb)), {})


def test_without_the_hook_the_seam_requests_the_weights_and_nothing_else():
    # Bug caught: the seam calling something new on a provider that has no hook (every family before Krea 2 runs one
    # of these), e.g. a hook looked up with a default that is called, or the inputs evaluated for every provider.
    # The provider records every attribute the seam reads during a step.
    seen = []

    class Watching(ResidentProvider):
        def __getattribute__(self, name):
            if not name.startswith("_"):
                seen.append(name)
            return super().__getattribute__(name)

    rec = Recorder()
    tf, shapes = _seamed(rec)
    tf.attach(Watching(resident_dicts(shapes)), shapes)
    seen.clear()
    mx.eval(tf(*zimage_inputs()))
    assert seen == ["pending"] + ["before_block", "weights_for"] * 4
