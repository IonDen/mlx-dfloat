"""Tests for the block seam: run one FLUX block on the provider's weight, evaluate per the step's
eval policy, and restore the placeholder afterward -- even when the block raises.

`FakeSeamTransformer` composes the package's `SeamMixin` over the fake FLUX transformer, so these
tests exercise exactly the mixin `seam_transformer_class()` builds over mflux. No test here imports
mflux.
"""

import dataclasses
from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from tests._flux_fakes import (
    FLUX_TABLE,
    D,
    FakeDoubleBlock,
    FakeSeamTransformer,
    FakeTransformer,
    Recorder,
    all_block_weights,
    block_lists,
    df11_groups,
    inputs,
    resident_dicts,
)

from mlx_dfloat.decode import STATUS_INVALID_CODE, decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.integrate.providers import (
    EVAL_POLICIES,
    DF11Provider,
    ResidentProvider,
    ReuseProvider,
)
from mlx_dfloat.mflux.flux1.transformer import SeamMixin


def test_seam_runs_each_block_on_the_provider_weight_and_restores_the_placeholder():
    # Bug caught: the weight assigned to the wrong block or attribute, or the placeholder not
    # restored so decoded weights stay resident across blocks.
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    per_block = resident_dicts(shapes)
    provider = ResidentProvider(per_block)
    tf.attach(provider, shapes, eval_policy="per-block")
    out = tf(*inputs())
    mx.eval(out)
    assert out.shape == (1, 2, D)
    assert [
        w is per_block[name]["attn.to_q"] for w, name in zip(rec.seen, shapes, strict=True)
    ] == [True, True, True]
    assert all(w.size == 0 for w in all_block_weights(tf))
    assert provider.launches == 0


def _patch_eval_recorders(monkeypatch, rec):
    """Record which object the seam hands to mx.eval / mx.async_eval: for a joint block that must
    be the very tuple the block returned, not one of its elements."""
    real_eval, real_async = seam._eval, seam._async_eval

    def eval_(out):
        rec.events.append(("eval", id(out)))
        rec.keep.append(out)
        real_eval(out)

    def async_(out):
        rec.events.append(("async", id(out)))
        rec.keep.append(out)
        real_async(out)

    monkeypatch.setattr(seam, "_eval", eval_)
    monkeypatch.setattr(seam, "_async_eval", async_)


def test_seam_restores_the_placeholder_when_the_block_raises(monkeypatch):
    # Bug caught: a missing try/finally leaving a decoded weight assigned after a failed step, or
    # the failed step's depth2 `prev` (block 0's output, already queued with async_eval when block 1
    # raises) surviving into the next step, where the first block's eval would drain it.
    class BoomOnce(FakeDoubleBlock):
        def __init__(self, recorder):
            super().__init__(recorder)
            self._armed = True

        def __call__(self, **kwargs):
            if self._armed:
                self._armed = False
                self._recorder.seen.append(self.attn.to_q.weight)
                raise RuntimeError("boom")
            return super().__call__(**kwargs)

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=0)
    tf.transformer_blocks[1] = BoomOnce(rec)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="depth2")
    _patch_eval_recorders(monkeypatch, rec)
    with pytest.raises(RuntimeError, match="boom"):
        tf(*inputs())
    assert rec.seen[1].size > 0  # the weight was assigned when the raising block ran
    assert all(w.size == 0 for w in all_block_weights(tf))
    failed = [key for kind, key in rec.events if kind == "run"]
    # Block 0 ran and was queued; block 1 raised before any eval.
    assert rec.events == [("run", failed[0]), ("async", failed[0])]
    # The second step starts clean: the depth2 sequence of a fresh step, and the failed step's queued
    # block-0 output is never evaluated (a surviving `prev` would put ("eval", failed[0]) first).
    tf(*inputs())
    o = [key for kind, key in rec.events[2:] if kind == "run"]
    assert rec.events[2:] == [
        ("run", o[0]),
        ("async", o[0]),
        ("run", o[1]),
        ("async", o[1]),
        ("eval", o[0]),
        ("eval", o[1]),
    ]
    assert ("eval", failed[0]) not in rec.events


def test_seam_refuses_a_provider_dict_that_does_not_match_the_block():
    # Bug caught: a provider returning a subset (or a typo'd key) leaving a placeholder in place;
    # without the check the matmul would raise a shape error naming nothing useful.
    class Partial:
        launches = 0
        launching = False
        policies = EVAL_POLICIES

        def weights_for(self, block_name, shapes):
            return {k: mx.zeros(v, dtype=mx.bfloat16) for k, v in list(shapes.items())[:-1]}

        def verify(self):
            pass

        def reset(self):
            pass

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(Partial(), shapes)
    with pytest.raises(DFloatIntegrationError, match=r"transformer_blocks\.0.*ff_context\.linear2"):
        tf(*inputs())
    assert all(w.size == 0 for w in all_block_weights(tf))


def test_seam_refuses_an_unknown_eval_policy():
    # Bug caught: a typo'd policy string (e.g. "depth3") silently falling through with no
    # evaluation at all, instead of being refused before the first block runs.
    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match="depth3"):
        tf.attach(ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="depth3")


class _Launching:
    """A provider that decodes (launches grow), like DF11Provider, without any groups."""

    launches = 0
    launching = True
    policies = EVAL_POLICIES

    def weights_for(self, block_name, shapes):
        self.launches += 1
        return {k: mx.zeros(v, dtype=mx.bfloat16) for k, v in shapes.items()}

    def verify(self):
        pass

    def reset(self):
        pass


def test_none_policy_is_refused_for_a_launching_provider_over_more_than_two_blocks():
    # Bug caught: "none" + DF11Provider over the full model keeps every decoded group alive until
    # the final eval (about 24 GB on FLUX.1); the refusal must fire before any block is decoded.
    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match=r"none.*3 blocks|3 blocks.*none"):
        tf.attach(_Launching(), shapes, eval_policy="none")
    # Two blocks (the reduced-depth validation) and the non-launching control are fine.
    tf.attach(_Launching(), {k: shapes[k] for k in list(shapes)[:2]}, eval_policy="none")
    double = {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["transformer_blocks.0"].items()}
    single = {
        a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["single_transformer_blocks.0"].items()
    }
    reuse = ReuseProvider(
        {"transformer_blocks": double, "single_transformer_blocks": single}, FLUX_TABLE
    )
    tf.attach(reuse, shapes, eval_policy="none")
    mx.eval(tf(*inputs()))


def _run_with_policy(monkeypatch, policy, *, n_double=2, n_single=2):
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=n_double, n_single=n_single)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes, eval_policy=policy)
    _patch_eval_recorders(monkeypatch, rec)
    out = tf(*inputs())
    runs = [key for kind, key in rec.events if kind == "run"]
    return rec.events, runs, out


def test_per_block_policy_evaluates_each_block_before_the_next_runs(monkeypatch):
    # Bug caught: evaluating only the last block (all decoded weights resident at once), or
    # evaluating a tuple's second element only.
    events, o, _ = _run_with_policy(monkeypatch, "per-block")
    assert events == [
        ("run", o[0]),
        ("eval", o[0]),
        ("run", o[1]),
        ("eval", o[1]),
        ("run", o[2]),
        ("eval", o[2]),
        ("run", o[3]),
        ("eval", o[3]),
    ]


def test_depth2_policy_evaluates_block_i_minus_1_before_block_i_plus_1_runs(monkeypatch):
    # Bug caught: eval(prev) placed before async_eval(out) (the host would wait on block i-1 before
    # block i is even queued, losing the gap the policy hides), or prev never drained.
    events, o, _ = _run_with_policy(monkeypatch, "depth2")
    assert events == [
        ("run", o[0]),
        ("async", o[0]),
        ("run", o[1]),
        ("async", o[1]),
        ("eval", o[0]),
        ("run", o[2]),
        ("async", o[2]),
        ("eval", o[1]),
        ("run", o[3]),
        ("async", o[3]),
        ("eval", o[2]),
        ("eval", o[3]),  # the drain after the last block
    ]


def test_none_policy_never_evaluates(monkeypatch):
    # Bug caught: a stray eval in the seam turning the run-ahead control into a per-block run.
    events, o, out = _run_with_policy(monkeypatch, "none")
    assert events == [("run", k) for k in o]
    mx.eval(out)


def test_seam_refuses_a_weight_of_the_right_size_but_wrong_shape():
    # Bug caught: a transposed (in, out) weight reaching the matmul, which raises a shape error
    # naming nothing useful, or silently computing garbage when in == out. The provider here does
    # not self-validate (unlike ResidentProvider), so only the seam's own guard can catch this.
    class WrongShape:
        launches = 0
        launching = False
        policies = EVAL_POLICIES

        def weights_for(self, block_name, shapes):
            weights = {k: mx.zeros(v, dtype=mx.bfloat16) for k, v in shapes.items()}
            weights["norm1.linear"] = mx.zeros((4, 24), dtype=mx.bfloat16)  # (24, 4) expected
            return weights

        def verify(self):
            pass

        def reset(self):
            pass

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(WrongShape(), shapes)
    with pytest.raises(DFloatIntegrationError, match=r"norm1\.linear: weight has shape \(4, 24\)"):
        tf(*inputs())
    assert all(w.size == 0 for w in all_block_weights(tf))


def test_verify_in_call_reads_the_status_words_before_the_call_returns():
    # Bug caught: a corrupt block surfacing only at an explicit verify_step (the model class relies on
    # the seam to raise from inside the transformer call).
    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(5))
    bad = mx.array([STATUS_INVALID_CODE], dtype=mx.uint32)
    provider = DF11Provider(
        groups,
        names,
        FLUX_TABLE,
        decode=lambda g: dataclasses.replace(decode_group(g, backend="reference"), status=bad),
    )
    tf.attach(provider, shapes, verify_in_call=True)
    with pytest.raises(DFloatFormatError, match="invalid code"):
        tf(*inputs())
    assert provider.pending == []


@pytest.mark.metal
def test_df11_provider_with_the_metal_backend_feeds_the_seam_bit_exact_weights():
    # Bug caught: a Metal decode whose lazy views the seam mishandles (evaluated after the placeholder
    # is restored, or cut at the wrong split), or a status word the step never checks; the
    # reference-backed tests above cannot see either, since only this path launches the kernel
    # through the seam.
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, source = df11_groups(shapes, np.random.default_rng(11))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=partial(decode_group, backend="metal")
    )
    tf.attach(provider, shapes, eval_policy="per-block")
    mx.eval(tf(*inputs()))
    tf.verify_step()
    assert provider.launches == 3
    assert provider.pending == []
    for seen, block_name in zip(rec.seen, shapes, strict=True):
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert seen.shape == want.shape
        assert np.array_equal(np.array(seen.view(mx.uint16)), want)
    assert all(w.size == 0 for w in all_block_weights(tf))


def test_hooks_forward_unknown_keywords_to_mflux():
    # Bug caught: the seam dropping (or choking on) a keyword a later mflux adds to its per-block
    # hooks, instead of passing it through to mflux's own hook. The fake's hooks mirror mflux 0.20.0
    # exactly (no **kwargs), so this local subclass stands in for a future mflux that takes one more.
    forwarded = []

    class FutureTransformer(FakeTransformer):
        def _apply_joint_transformer_block(self, idx, block, **kwargs):
            forwarded.append(kwargs.pop("future_keyword", None))
            return super()._apply_joint_transformer_block(idx=idx, block=block, **kwargs)

    class FutureSeam(SeamMixin, FutureTransformer):
        pass

    tf = FutureSeam(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes)
    x = mx.ones((1, 2, D), dtype=mx.bfloat16)
    out = tf._apply_joint_transformer_block(
        idx=0,
        block=tf.transformer_blocks[0],
        hidden_states=x,
        encoder_hidden_states=x,
        text_embeddings=x,
        image_rotary_embeddings=x,
        controlnet_block_samples=None,
        future_keyword=1,
    )
    mx.eval(out)
    assert forwarded == [1]


def _reference_df11(n_double, n_single, seed):
    tf = FakeSeamTransformer(Recorder(), n_double=n_double, n_single=n_single)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(seed))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=partial(decode_group, backend="reference")
    )
    return tf, shapes, provider


def test_a_step_is_refused_while_the_previous_steps_status_words_are_unverified():
    # Bug caught: a caller that never calls verify_step() growing the provider's pending list
    # without bound and never checking a single decode (mflux's own denoise loop evaluates inside
    # the step, so nothing else would ever read them). The refusal fires before any block decodes
    # and keeps the pending words, so the caller can still verify them.
    tf, shapes, provider = _reference_df11(1, 1, 41)
    tf.attach(provider, shapes)
    mx.eval(tf(*inputs()))
    with pytest.raises(DFloatIntegrationError, match=r"verify_step\(\).*verify_in_call=True"):
        tf(*inputs())
    assert provider.launches == 2
    assert len(provider.pending) == 2
    tf.verify_step()
    mx.eval(tf(*inputs()))  # verified in between: the next step runs
    tf.verify_step()
    tf.attach(provider, shapes, verify_in_call=True)
    mx.eval(tf(*inputs()))
    mx.eval(tf(*inputs()))  # verify_in_call drains the words inside every step
    assert provider.pending == []
    assert provider.launches == 8


def test_verify_in_call_is_refused_under_the_none_policy_with_a_launching_provider():
    # Bug caught: verify_in_call reading the status words at the end of a "none" step, which forces
    # every decode of the step before the caller's own eval (the run-ahead "none" is there to
    # measure turns into a hidden full-step eval).
    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match=r"verify_in_call.*'none'"):
        tf.attach(_Launching(), shapes, eval_policy="none", verify_in_call=True)
    tf.attach(_Launching(), shapes, eval_policy="per-block", verify_in_call=True)
    tf.attach(_Launching(), shapes, eval_policy="none")
    tf.attach(
        ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="none", verify_in_call=True
    )


def test_a_step_that_raises_midway_clears_the_df11_providers_status_words():
    # Bug caught: the seam's abort path not resetting the provider, so the failed step's status
    # words (block 0's decode) stay pending and the next step is refused as unverified.
    class BoomOnce(FakeDoubleBlock):
        def __init__(self, recorder):
            super().__init__(recorder)
            self.boomed = False

        def __call__(self, **kwargs):
            if not self.boomed:
                self.boomed = True
                raise RuntimeError("boom")
            return super().__call__(**kwargs)

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=0)
    tf.transformer_blocks[1] = BoomOnce(rec)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(42))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=partial(decode_group, backend="reference")
    )
    tf.attach(provider, shapes)
    with pytest.raises(RuntimeError, match="boom"):
        tf(*inputs())
    assert provider.launches == 2
    assert provider.pending == []
    mx.eval(tf(*inputs()))
    tf.verify_step()


def test_detach_forgets_the_provider_and_a_step_is_refused_until_the_next_attach():
    # Bug caught: a detached transformer still running on the old provider (the dropped set's
    # groups kept alive through the seam state), or detach leaving pending status words.
    from tests._flux_fakes import (
        FLUX_TABLE,
        FakeSeamTransformer,
        Recorder,
        block_lists,
        inputs,
        resident_dicts,
    )

    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.integrate.providers import ResidentProvider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes)
    tf(*inputs())
    tf.detach()
    with pytest.raises(DFloatIntegrationError, match="attach"):
        tf(*inputs())
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes)
    mx.eval(tf(*inputs()))


def test_detach_clears_a_df11_providers_pending_status_words():
    # Bug caught: `detach` forgetting the provider without `reset()`, so a DF11 provider re-attached
    # later still holds the dropped step's status words (and the decoded groups they reference)
    # and the next step is refused as "unchecked".
    import numpy as np
    from tests._flux_fakes import (
        FLUX_TABLE,
        FakeSeamTransformer,
        Recorder,
        block_lists,
        df11_groups,
        inputs,
    )

    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.integrate.providers import DF11Provider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(11))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=lambda g: decode_group(g, backend="reference")
    )
    tf.attach(provider, shapes)
    mx.eval(tf(*inputs()))
    assert provider.pending != []  # verify_step was never called
    tf.detach()
    assert provider.pending == []


def test_a_seam_that_passes_no_inputs_hands_each_hook_call_its_own_empty_arguments():
    # Bug caught: one shared default argument object for every call of run_block, so a before_block hook that adds a
    # key to the kwargs it is given (the FLUX.1 seam passes none) leaks it into every later block's call, and into the
    # next step's.
    seen = []

    class Hooked(ResidentProvider):
        def before_block(self, block_name, args, kwargs):
            seen.append((block_name, args, dict(kwargs)))
            kwargs["touched"] = block_name

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tf.attach(Hooked(resident_dicts(shapes)), shapes, eval_policy="per-block")
    mx.eval(tf(*inputs()))
    mx.eval(tf(*inputs()))
    assert [(args, kwargs) for _name, args, kwargs in seen] == [((), {})] * 6
