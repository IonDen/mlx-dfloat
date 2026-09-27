"""The rig's own code: the re-exports the benches measure, the trace bookkeeping, and the prefetch
provider (a research provider that is not part of the package's shipped integration surface).

`FakeSeamTransformer` composes the package's `SeamMixin` over the fake FLUX transformer, so these
tests exercise exactly the mixin `seam_transformer_class()` builds over mflux. No test here imports
mflux.
"""

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from scripts import _flux_rig as rig
from scripts._flux_rig import DF11Provider, RigError
from tests._flux_fakes import (
    FLUX_TABLE,
    FakeDoubleBlock,
    FakeSeamTransformer,
    Recorder,
    all_block_weights,
    block_lists,
    df11_groups,
    inputs,
    resident_dicts,
)

from mlx_dfloat.decode import decode_group
from mlx_dfloat.integrate import providers, seam
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.integrate.providers import ResidentProvider
from mlx_dfloat.mflux.flux1 import transformer


def test_the_rig_re_exports_the_package_seam_and_providers():
    # Bug caught: the benches measuring a private copy of the seam instead of what ships.
    assert rig.DF11Provider is providers.DF11Provider
    assert rig.ReuseProvider is providers.ReuseProvider
    assert rig.SeamMixin is transformer.SeamMixin
    assert rig._eval is seam._eval


# --- trace ----------------------------------------------------------------------------------------


def test_seam_trace_records_one_ordered_event_per_block_per_step():
    # Bug caught: a block with no event (a hook path that skips the tracer), stamps taken out of
    # order (decode after encode), or events not tagged with the step they belong to.
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tracer = rig.Tracer()
    tf.attach(
        ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="per-block", tracer=tracer
    )
    mx.eval(tf(*inputs()))
    mx.eval(tf(*inputs()))
    assert [(e.step, e.block) for e in tracer.events] == [
        (s, name) for s in range(2) for name in shapes
    ]
    assert len(tracer.steps) == 2
    for e in tracer.events:
        assert e.t_decode_start <= e.t_decode_end <= e.t_encode_end <= e.t_eval_end
        start, end = tracer.steps[e.step]
        assert start <= e.t_decode_start
        assert e.t_eval_end <= end


def test_seam_trace_attributes_the_provider_time_to_the_decode_phase():
    # Bug caught: the decode stamp taken after run() so weights_for time lands in the encode phase.
    import time

    class Slow(ResidentProvider):
        def weights_for(self, block_name, shapes):
            time.sleep(0.01)
            return super().weights_for(block_name, shapes)

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tracer = rig.Tracer()
    tf.attach(Slow(resident_dicts(shapes)), shapes, eval_policy="per-block", tracer=tracer)
    mx.eval(tf(*inputs()))
    (event,) = tracer.events
    assert event.t_decode_end - event.t_decode_start >= 0.009


def test_seam_trace_attributes_the_block_and_the_eval_to_their_own_phases(monkeypatch):
    # Bug caught: t_encode_end stamped before run() (the graph build would land in eval wait) or
    # t_eval_end stamped before the policy's eval (the wait would vanish from the trace).
    import time

    class SlowBlock(FakeDoubleBlock):
        def __call__(self, **kwargs):
            time.sleep(0.01)
            return super().__call__(**kwargs)

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=1, n_single=0)
    tf.transformer_blocks = [SlowBlock(rec)]
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    real_eval = seam._eval
    # The seam (mlx_dfloat.integrate.seam) is what SeamMixin actually calls; patching the rig's own
    # copy of the name would not touch it.
    monkeypatch.setattr(seam, "_eval", lambda x: (time.sleep(0.01), real_eval(x)))
    tracer = rig.Tracer()
    tf.attach(
        ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="per-block", tracer=tracer
    )
    mx.eval(tf(*inputs()))
    (event,) = tracer.events
    assert event.t_encode_end - event.t_decode_end >= 0.009
    assert event.t_eval_end - event.t_encode_end >= 0.009


def test_seam_trace_under_depth2_keeps_the_drained_tail_inside_the_step(monkeypatch):
    # Bug caught: end_step stamped before the depth2 drain of the last block, so its wait would
    # fall outside the step window.
    import time

    evals = []
    real_eval = seam._eval
    monkeypatch.setattr(seam, "_eval", lambda x: (evals.append(time.perf_counter()), real_eval(x)))
    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    tracer = rig.Tracer()
    tf.attach(ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="depth2", tracer=tracer)
    mx.eval(tf(*inputs()))
    assert [e.block for e in tracer.events] == list(shapes)
    (start, end) = tracer.steps[0]
    assert len(evals) == 3  # two inside the blocks, one drain after the last block
    assert evals[-1] > tracer.events[-1].t_eval_end  # the drain came after the last event
    assert evals[-1] <= end  # and inside the step window
    assert all(start <= e.t_decode_start for e in tracer.events)


def test_summarize_trace_splits_a_step_into_named_seconds():
    # Bug caught: a phase summed from the wrong pair of stamps, the gap counted from the wrong
    # neighbour, or the head/tail measured against the wrong step boundary.
    ev = rig.BlockEvent
    events = [
        ev(
            step=0,
            block="a",
            t_decode_start=1.0,
            t_decode_end=1.2,
            t_encode_end=1.5,
            t_eval_end=2.5,
        ),
        ev(
            step=0,
            block="b",
            t_decode_start=2.6,
            t_decode_end=2.7,
            t_encode_end=2.9,
            t_eval_end=3.9,
        ),
    ]
    got = rig.summarize_trace(events, step=(0.5, 4.4))
    want = {
        "n_blocks": 2,
        "host_decode_s": 0.3,
        "host_encode_s": 0.5,
        "eval_wait_s": 2.0,
        "restore_gap_s": 0.1,  # eval end of a -> decode start of b
        "gpu_idle_gap_s": 0.4,  # eval end of a -> encode end of b (host work while the GPU waits)
        "head_s": 0.5,  # step start -> first decode start
        "tail_s": 0.5,  # last eval end -> step end
        "step_s": 3.9,
    }
    assert got.keys() == want.keys()
    for key, value in want.items():
        assert got[key] == pytest.approx(value), key


def test_summarize_trace_of_an_empty_step_has_zero_blocks_and_only_the_step_time():
    got = rig.summarize_trace([], step=(1.0, 3.0))
    assert got["n_blocks"] == 0
    assert got["step_s"] == pytest.approx(2.0)
    assert got["host_decode_s"] == 0.0
    assert got["gpu_idle_gap_s"] == 0.0


def test_summarize_trace_refuses_events_from_more_than_one_step():
    ev = rig.BlockEvent
    events = [
        ev(step=0, block="a", t_decode_start=1, t_decode_end=1, t_encode_end=1, t_eval_end=1),
        ev(step=1, block="a", t_decode_start=2, t_decode_end=2, t_encode_end=2, t_eval_end=2),
    ]
    with pytest.raises(RigError, match="one step"):
        rig.summarize_trace(events, step=(0.0, 3.0))


# --- prefetch -------------------------------------------------------------------------------------


def _prefetch_over(n_double, n_single, rng, *, decode=None, stream=None):
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=n_double, n_single=n_single)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, source = df11_groups(shapes, rng)
    inner = DF11Provider(groups, names, FLUX_TABLE, decode=decode)
    provider = rig.PrefetchProvider(inner, shapes, stream=stream)
    tf.attach(provider, shapes, eval_policy="per-block")
    return rec, tf, shapes, source, provider


def test_prefetch_provider_decodes_the_next_block_one_ahead_and_cycles_to_the_first():
    # Bug caught: the block decoded when it is asked for (no look-ahead), the wrong next block, the
    # last block not prefetching the next step's first, or the cold inline decode counted as a
    # steady-state launch (the bench's launches-per-step parity check would then refuse every run).
    calls = []

    def counting_decode(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    rec, tf, shapes, source, provider = _prefetch_over(
        2, 1, np.random.default_rng(21), decode=counting_decode
    )
    order = list(shapes)
    assert provider.launching is True
    mx.eval(tf(*inputs()))
    tf.verify_step()
    assert calls == [order[0], order[1], order[2], order[0]]
    assert provider.cold_launches == 1
    assert provider.launches == 3
    mx.eval(tf(*inputs()))
    tf.verify_step()
    assert calls[4:] == [order[1], order[2], order[0]]
    assert provider.launches == 6
    assert provider.pending == []
    for seen, block_name in zip(rec.seen, order * 2, strict=True):
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert seen.dtype == mx.bfloat16
        assert np.array_equal(np.array(seen.view(mx.uint16)), want)
    assert all(w.size == 0 for w in all_block_weights(tf))


def test_prefetch_provider_refuses_a_block_out_of_order():
    # Bug caught: a misordered request silently served by an inline decode (the prefetched group
    # would leak and the measurement would not be a prefetch).
    _rec, _tf, shapes, _source, provider = _prefetch_over(2, 0, np.random.default_rng(22))
    provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    with pytest.raises(RigError, match=r"transformer_blocks\.0 .* expected transformer_blocks\.1"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])


def test_prefetch_provider_defers_the_status_words_to_verify():
    # Bug caught: the prefetch reading a status word (a host sync inside the step).
    class Unread:
        def __array__(self, *args, **kwargs):
            raise AssertionError("status was read")

    _rec, tf, _shapes, _source, provider = _prefetch_over(
        1,
        1,
        np.random.default_rng(23),
        decode=lambda g: dataclasses.replace(decode_group(g, backend="reference"), status=Unread()),
    )
    mx.eval(tf(*inputs()))
    assert [name for name, _s in provider.pending] == [
        "transformer_blocks.0",
        "single_transformer_blocks.0",
        "transformer_blocks.0",
    ]
    with pytest.raises(AssertionError, match="status was read"):
        tf.verify_step()


@pytest.mark.metal
def test_prefetch_provider_on_a_second_stream_feeds_the_seam_bit_exact_weights():
    # Bug caught: a decode launched on another GPU stream whose result the block consumes before
    # it is complete (MLX must order the two streams), or views cut on the wrong stream.
    from functools import partial

    rec, tf, shapes, source, provider = _prefetch_over(
        2,
        1,
        np.random.default_rng(24),
        decode=partial(decode_group, backend="metal"),
        stream=mx.new_stream(mx.gpu),
    )
    mx.eval(tf(*inputs()))
    tf.verify_step()
    mx.eval(tf(*inputs()))
    tf.verify_step()
    assert provider.launches == 6
    for seen, block_name in zip(rec.seen, list(shapes) * 2, strict=True):
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert np.array_equal(np.array(seen.view(mx.uint16)), want)


def test_prefetch_provider_is_refused_under_any_policy_but_per_block():
    # Bug caught: prefetch attached under depth2, where block i-1's weights are still held by its
    # in-flight command buffers, so three decoded groups sit resident instead of two.
    _rec, tf, shapes, _source, provider = _prefetch_over(1, 0, np.random.default_rng(27))
    with pytest.raises(RigError, match="per-block"):
        tf.attach(provider, shapes, eval_policy="depth2")
    tf.attach(provider, shapes, eval_policy="per-block")
    tf.attach(
        ResidentProvider(resident_dicts(shapes)), shapes, eval_policy="depth2"
    )  # others: fine


def test_prefetch_provider_refuses_an_unknown_block_with_a_rig_error():
    _rec, _tf, shapes, _source, provider = _prefetch_over(1, 0, np.random.default_rng(25))
    with pytest.raises(RigError, match="no such block"):
        provider.weights_for("transformer_blocks.9", shapes["transformer_blocks.0"])


def test_a_step_that_raises_midway_leaves_no_stale_look_ahead():
    # Bug caught: the look-ahead submitted for block k+1 surviving a failed step, so the next
    # step's block 0 is refused as out of order.
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
    # The boom sits in block 0, whose look-ahead is block 1: without reset() the next step's block 0
    # is refused as out of order. (A boom in the last block would already have wrapped to block 0.)
    tf.transformer_blocks[0] = BoomOnce(rec)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(26))
    provider = rig.PrefetchProvider(
        DF11Provider(
            groups, names, FLUX_TABLE, decode=lambda g: decode_group(g, backend="reference")
        ),
        shapes,
    )
    tf.attach(provider, shapes, eval_policy="per-block")
    with pytest.raises(RuntimeError, match="boom"):
        tf(*inputs())
    mx.eval(tf(*inputs()))  # block 0 is served again, not refused
    tf.verify_step()
