import gc
import weakref

import mlx.core as mx
import pytest

from mlx_dfloat.errors import DFloatResourceError
from mlx_dfloat.mflux.flux1 import lifecycle as lc
from mlx_dfloat.mflux.flux1.lifecycle import Lifecycle

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


def _lifecycle(holder, cache=None, *, slack=MIB):
    gc.collect()
    mx.clear_cache()
    return Lifecycle(
        load_encoders=holder.load_encoders,
        unload_encoders=holder.unload_encoders,
        encode=holder.encode,
        load_set=holder.load_set,
        unload_set=holder.unload_set,
        prompt_cache={} if cache is None else cache,
        retained_bound_bytes=int(mx.get_active_memory()) + slack,
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
    assert int(mx.get_active_memory()) <= life.retained_bound_bytes  # the 8 MiB weight is gone


def test_an_encoder_object_is_unreachable_after_the_drop():
    # Bug caught: the lifecycle keeping its own reference to the encoders (the weakref stays alive).
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
