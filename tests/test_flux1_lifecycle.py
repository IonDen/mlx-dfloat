import gc
import weakref

import mlx.core as mx
import pytest

from mlx_dfloat.errors import DFloatResourceError
from mlx_dfloat.mflux import lifecycle as lc
from mlx_dfloat.mflux.lifecycle import Lifecycle

MIB = 1024**2


class _Holder:
    """Stands in for the model: owns the encoder and set objects the lifecycle callbacks create and drop."""

    def __init__(self, events, *, leak_set=False):
        self.events, self.encoders, self.resident, self.leak_set = events, None, None, leak_set
        self.leaked = []

    def load_encoders(self):
        self.events.append("load_encoders")
        self.encoders = _Encoders()

    def unload_encoders(self):
        self.events.append("unload_encoders")
        self.encoders = None

    def encode(self, prompt):
        self.events.append(("encode", prompt))
        w = self.encoders.weight
        return w[:1, :4] * 2, w[
            :1, :2
        ] + 1  # lazy: the graph holds the 8 MiB weight until evaluated

    def load_set(self):
        self.events.append("load_set")
        self.resident = mx.zeros((2048, 2048), dtype=mx.bfloat16)  # 8 MiB
        mx.eval(self.resident)
        if self.leak_set:
            self.leaked.append(self.resident)

    def unload_set(self):
        self.events.append("unload_set")
        self.resident = None


class _Encoders:
    def __init__(self):
        self.weight = mx.ones(
            (2048, 2048), dtype=mx.bfloat16
        )  # 8 MiB, evaluated: what T5 stands for
        mx.eval(self.weight)


def _fixed(bound):
    return lambda: bound


def _lifecycle(holder, cache=None, *, slack=MIB, encoders_loaded=False):
    gc.collect()
    mx.clear_cache()
    return Lifecycle(
        load_encoders=holder.load_encoders,
        unload_encoders=holder.unload_encoders,
        encode=holder.encode,
        load_set=holder.load_set,
        unload_set=holder.unload_set,
        prompt_cache={} if cache is None else cache,
        retained_bound=_fixed(int(mx.get_active_memory()) + slack),
        encoders_loaded=encoders_loaded,
    )


def test_a_new_prompt_evaluates_the_pair_and_drops_the_encoders_and_their_memory(monkeypatch):
    # Bug caught: caching the lazy pair (its graph keeps every encoder weight alive past the drop,
    # so T5 lands in the first step next to the compressed set), or forgetting the drop.
    events = []
    monkeypatch.setattr(
        lc, "_eval", lambda *arrays: (events.append(("eval", len(arrays))), mx.eval(*arrays))
    )
    holder = _Holder(events)
    cache = {}
    life = _lifecycle(holder, cache)
    life.ensure_embeddings("a lighthouse")
    assert events == ["load_encoders", ("encode", "a lighthouse"), ("eval", 2), "unload_encoders"]
    assert holder.encoders is None
    assert not life.encoders_loaded
    assert set(cache) == {"a lighthouse"}
    assert int(mx.get_active_memory()) <= life.retained_bound()  # the 8 MiB weight is gone


def test_an_encoder_object_is_unreachable_after_the_drop():
    # Bug caught: `drop_encoders` keeping its own reference to the encoder object (e.g. storing
    # what `load_encoders` built on the lifecycle), so the T5 module outlives the drop and the
    # weakref stays alive.
    holder = _Holder([])
    life = _lifecycle(holder)
    life.ensure_embeddings("p")
    holder.load_encoders()
    ref = weakref.ref(holder.encoders)
    life.encoders_loaded = True
    life.drop_encoders()
    gc.collect()
    assert ref() is None


def test_a_new_prompt_with_the_set_resident_drops_the_set_first_and_a_later_step_reloads_it():
    # Bug caught: encoding with the set resident (T5 + set = over the fit rule), or not counting
    # the forced drop the report must show.
    events = []
    holder = _Holder(events)
    life = _lifecycle(holder)
    life.ensure_embeddings("first")
    life.ensure_set()
    events.clear()
    life.ensure_embeddings("second")
    assert events[:3] == ["unload_set", "load_encoders", ("encode", "second")]
    assert events[-1] == "unload_encoders"
    assert not life.set_resident
    life.ensure_set()
    assert life.set_resident
    assert events[-1] == "load_set"
    assert life.counters.forced_set_drops == 1
    assert life.counters.set_loads == 2
    assert life.counters.encoder_loads == 2


def test_a_cached_prompt_touches_neither_the_encoders_nor_the_set():
    # Bug caught: re-encoding a cached prompt (a 35 s reload for nothing), or dropping the set for it.
    events = []
    holder = _Holder(events)
    life = _lifecycle(holder)
    life.ensure_embeddings("p")
    life.ensure_set()
    events.clear()
    life.ensure_embeddings("p")
    life.ensure_set()
    assert events == []


def test_several_prompts_load_the_encoders_once():
    # Bug caught: one encoder load per prompt in `encode(*prompts)`, which is what pre-encoding exists to avoid.
    events = []
    holder = _Holder(events)
    life = _lifecycle(holder)
    life.ensure_embeddings("a", "b", "c")
    assert events.count("load_encoders") == 1
    assert events.count("unload_encoders") == 1
    encodes = [e for e in events if isinstance(e, tuple) and e[0] == "encode"]
    assert encodes == [("encode", "a"), ("encode", "b"), ("encode", "c")]


def test_memory_still_active_after_a_drop_is_an_error_not_a_paging_storm():
    # Bug caught: a leak (something else holding the set) passing silently; the next load would
    # then sit next to 15 GiB of stale buffers.
    holder = _Holder([], leak_set=True)
    life = _lifecycle(holder)
    life.ensure_set()
    with pytest.raises(DFloatResourceError, match="still active"):
        life.drop_set()
    assert not life.set_resident


def test_counters_serialise_for_the_report():
    # Bug caught: a counter missing from the report dict.
    life = _lifecycle(_Holder([]))
    life.ensure_embeddings("p")
    d = life.counters.as_dict()
    assert set(d) == {
        "encoder_loads",
        "encoder_seconds",
        "set_loads",
        "set_seconds",
        "forced_set_drops",
    }
    assert d["encoder_loads"] == 1
    assert d["encoder_seconds"] >= 0


def test_a_failed_encode_still_drops_the_encoders_and_their_memory():
    # Bug caught: not wrapping the encode loop in try/finally, so a mid-loop exception leaves
    # `encoders_loaded` True and the encoder weights resident — a later `ensure_set()` would then
    # load the compressed set right next to them, the exact case this module exists to prevent.
    events = []
    holder = _Holder(events)

    def _failing_encode(prompt):
        events.append(("encode", prompt))
        if prompt == "b":
            # Raise before binding a local to the weight: otherwise the in-flight exception's own
            # traceback keeps this frame (and the weight it names) alive through the `finally`.
            raise RuntimeError("encode failed")
        w = holder.encoders.weight
        return w[:1, :4] * 2, w[:1, :2] + 1

    holder.encode = _failing_encode
    cache = {}
    life = _lifecycle(holder, cache)
    with pytest.raises(RuntimeError, match="encode failed"):
        life.ensure_embeddings("a", "b")
    assert events[-1] == "unload_encoders"
    assert not life.encoders_loaded
    # "a" was encoded and evaluated before "b" raised, so it is fully cached (never half-cached:
    # the implementation evaluates a pair before caching it); "b" raised before returning a pair,
    # so it was never cached at all.
    assert set(cache) == {"a"}
    assert int(mx.get_active_memory()) <= life.retained_bound()


def test_pre_loaded_encoders_are_not_reloaded_but_are_still_dropped_at_the_end():
    # Bug caught: `ensure_embeddings` calling `load_encoders` again when `encoders_loaded` was
    # already True at construction (the model warmed the encoders itself before handing the
    # lifecycle its state), double-loading and double-counting; or the opposite bug of never
    # dropping them because "this call didn't load them".
    events = []
    holder = _Holder(events)
    holder.load_encoders()
    events.clear()
    life = _lifecycle(holder, encoders_loaded=True)
    life.ensure_embeddings("p")
    assert "load_encoders" not in events
    assert events[0] == ("encode", "p")
    assert events[-1] == "unload_encoders"
    assert life.counters.encoder_loads == 0


def test_an_eval_failure_drops_the_encoders_and_the_original_error_escapes(monkeypatch):
    # Bug caught: a `try/finally` whose `drop_encoders()` runs while the failing eval's traceback
    # still holds the lazy pair (and through it the 8 MiB encoder weight): `_reclaim` then raises
    # DFloatResourceError and the real RuntimeError is demoted to `__context__`.
    events = []
    holder = _Holder(events)

    def failing_eval(*arrays):
        raise RuntimeError("simulated eval failure")

    monkeypatch.setattr(lc, "_eval", failing_eval)
    life = _lifecycle(holder)
    with pytest.raises(RuntimeError, match="simulated eval failure") as info:
        life.ensure_embeddings("p")
    assert not isinstance(info.value, DFloatResourceError)
    assert not life.encoders_loaded
    assert "unload_encoders" in events
    assert int(mx.get_active_memory()) <= life.retained_bound()


def test_a_failure_path_reclaim_error_is_logged_not_raised_over_the_original(monkeypatch, caplog):
    # Bug caught: the failure path's `_reclaim` raising DFloatResourceError over the encode error,
    # so the caller sees a memory complaint instead of what actually broke.
    events = []
    holder = _Holder(events, leak_set=False)
    leaked = []

    def encode_and_leak(prompt):
        leaked.append(holder.encoders.weight)  # something else keeps the encoder weight alive
        raise ValueError("encode broke")

    holder.encode = encode_and_leak
    life = _lifecycle(holder)
    with (
        caplog.at_level("WARNING", logger="mlx_dfloat.mflux.flux1"),
        pytest.raises(ValueError, match="encode broke"),
    ):
        life.ensure_embeddings("p")
    assert "still active" in caplog.text
    assert not life.encoders_loaded


def test_the_retained_bound_is_read_at_each_drop_so_cached_embeddings_count():
    # Bug caught: the bound frozen at construction, so embeddings cached after it (or a VAE that
    # was evaluated later) read as a leak and a clean drop raises DFloatResourceError.
    holder = _Holder([])
    gc.collect()
    mx.clear_cache()
    extra = [0]
    base = int(mx.get_active_memory()) + MIB
    life = Lifecycle(
        load_encoders=holder.load_encoders,
        unload_encoders=holder.unload_encoders,
        encode=holder.encode,
        load_set=holder.load_set,
        unload_set=holder.unload_set,
        prompt_cache={},
        retained_bound=lambda: base + extra[0],
    )
    kept = mx.ones((2048, 2048), dtype=mx.bfloat16)  # 8 MiB that legitimately stays (embeddings)
    mx.eval(kept)
    extra[0] = 16 * MIB  # the caller's bound grew after construction
    life.ensure_set()
    life.drop_set()
    extra[0] = 0
    with pytest.raises(DFloatResourceError, match="still resident"):
        life.ensure_set()
    del kept


def test_ensure_set_refuses_to_load_a_second_copy_while_memory_is_still_active():
    # Bug caught: ensure_set loading the compressed set while a previous copy is still held
    # (a leaked provider, a failed drop), so two 15 GiB sets sit side by side.
    events = []
    holder = _Holder(events, leak_set=True)
    life = _lifecycle(holder)
    life.ensure_set()
    with pytest.raises(DFloatResourceError):
        life.drop_set()  # the leak: the previous set is still held
    events.clear()
    with pytest.raises(DFloatResourceError, match="a previous set is still resident"):
        life.ensure_set()
    assert events == []
    assert not life.set_resident
