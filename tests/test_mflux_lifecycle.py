"""The lifecycle moved from flux1/ to mflux/ so every family shares it; FLUX.1's old path stays importable."""

import gc
import logging

import mlx.core as mx
import pytest
from tests._fake_counters import FakeCounters

from mlx_dfloat import _guard
from mlx_dfloat._watchdog import Watchdog
from mlx_dfloat.errors import DFloatWatchdogError
from mlx_dfloat.mflux import lifecycle as shared
from mlx_dfloat.mflux.flux1 import lifecycle as old


def test_the_old_flux1_path_re_exports_the_same_classes():
    # Bug caught: flux1/lifecycle.py left as a second copy (two classes: isinstance and counters diverge), or
    # dropped (DFloatFlux1's import breaks).
    assert old.Lifecycle is shared.Lifecycle
    assert old.LifecycleCounters is shared.LifecycleCounters


def test_it_logs_under_the_family_neutral_logger_name(caplog):
    # Bug caught: the shared lifecycle still logging as "mlx_dfloat.mflux.flux1" (a Z-Image warning filed under FLUX.1).
    def encode(prompt):
        raise ValueError("encode broke")

    leaked = []

    def load_encoders():
        leaked.append(
            mx.ones((2048, 2048), dtype=mx.bfloat16)
        )  # kept alive: the reclaim check must complain
        mx.eval(leaked[0])

    gc.collect()  # garbage an earlier test left would otherwise be freed inside the check and pass it
    mx.clear_cache()
    bound = int(mx.get_active_memory())
    life = shared.Lifecycle(
        load_encoders=load_encoders,
        unload_encoders=lambda: None,
        encode=encode,
        load_set=lambda: None,
        unload_set=lambda: None,
        prompt_cache={},
        retained_bound=lambda: bound,
    )
    with (
        caplog.at_level(logging.WARNING, logger="mlx_dfloat.mflux"),
        pytest.raises(ValueError, match="encode broke"),
    ):
        life.ensure_embeddings("p")
    assert [r.name for r in caplog.records] == ["mlx_dfloat.mflux"]


@pytest.fixture
def raise_mode(tmp_path):
    """A started raise-mode watchdog (stop line 9_000) over FakeCounters, and a call that trips it.

    Stopped at teardown; no test may see it end the "process".
    """
    fake, exits = FakeCounters(), []
    watchdog = Watchdog(
        tmp_path,
        ceiling=10_000,
        budget=None,
        interval=0.01,
        counters=fake,
        exit_process=exits.append,
        mode="raise",
        stop_margin=1_000,
    ).start()

    def trip():
        fake.raise_to(9_500)
        assert watchdog.tripped.wait(timeout=5)

    yield watchdog, trip
    watchdog.stop()
    assert exits == []


def _lifecycle(calls, *, load_encoders=None, encode=None, cache=None):
    """A Lifecycle whose callbacks log to ``calls``; a bound no drop can miss."""

    def default_encode(prompt):
        calls.append(("encode", prompt))
        return (mx.ones((2,)), mx.zeros((2,)))

    return shared.Lifecycle(
        load_encoders=load_encoders or (lambda: calls.append(("load_encoders",))),
        unload_encoders=lambda: calls.append(("unload_encoders",)),
        encode=encode or default_encode,
        load_set=lambda: calls.append(("load_set",)),
        unload_set=lambda: calls.append(("unload_set",)),
        prompt_cache={} if cache is None else cache,
        retained_bound=lambda: 1 << 62,
    )


def test_a_trip_during_encode_raises_and_drops_the_encoders(raise_mode, monkeypatch):
    # Bug caught: the checkpoint before the pair's evaluation (the long call that crosses the
    # line, so the next prompt is encoded past the trip) or after the cache store (the stopped
    # prompt's pair kept), or outside the try that drops the encoders.
    watchdog, trip = raise_mode
    calls, cache = [], {}
    real_eval = shared._eval

    def eval_crossing_the_line(*pair):
        real_eval(*pair)
        trip()

    monkeypatch.setattr(shared, "_eval", eval_crossing_the_line)
    life = _lifecycle(calls, cache=cache)
    with _guard.scope(watchdog), pytest.raises(DFloatWatchdogError) as caught:
        life.ensure_embeddings("a", "b")
    assert caught.value.site == "prompt_encoded"
    assert life.encoders_loaded is False
    assert "a" not in cache
    assert calls == [("load_encoders",), ("encode", "a"), ("unload_encoders",)]


def test_a_trip_during_the_encoder_load_raises_before_any_encode(raise_mode):
    # Bug caught: no checkpoint after the load (the prompt is encoded past the trip), or one
    # placed before the try, so the loaded encoders are never dropped.
    watchdog, trip = raise_mode
    calls = []

    def load_encoders():
        calls.append(("load_encoders",))
        trip()

    life = _lifecycle(calls, load_encoders=load_encoders)
    with _guard.scope(watchdog), pytest.raises(DFloatWatchdogError) as caught:
        life.ensure_embeddings("a")
    assert caught.value.site == "encoder_loaded"
    assert life.encoders_loaded is False
    assert calls == [("load_encoders",), ("unload_encoders",)]


def test_a_tripped_call_raises_before_the_set_load(raise_mode):
    # Bug caught: no checkpoint before the set load, so a call that is already over the stop line
    # reads the whole compressed set before stopping.
    watchdog, trip = raise_mode
    calls = []
    life = _lifecycle(calls)
    trip()
    with _guard.scope(watchdog), pytest.raises(DFloatWatchdogError) as caught:
        life.ensure_set()
    assert caught.value.site == "set_load_start"
    assert (calls, life.set_resident) == ([], False)


def test_a_drop_waits_for_the_gpu_before_it_collects_and_checks(monkeypatch):
    # Bug caught: `_reclaim` collecting and reading active memory with GPU work still in flight (no synchronize
    # first): a buffer whose free is pending in a command's completion handler is still counted active, so the check
    # reports a false "still active" and the bytes land in the cache after the clear.
    events = []
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: events.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear_cache"))
    monkeypatch.setattr(gc, "collect", lambda *a: events.append("collect") or 0)
    life = shared.Lifecycle(
        load_encoders=lambda: None,
        unload_encoders=lambda: None,
        encode=lambda prompt: (),
        load_set=lambda: None,
        unload_set=lambda: events.append("unload_set"),
        prompt_cache={},
        retained_bound=lambda: events.append("bound") or 1 << 62,
    )
    life.drop_set()
    assert events == ["unload_set", "synchronize", "collect", "clear_cache", "bound"]


def _record_drop(monkeypatch, events, *, footprints=(800, 800)):
    """Record every step a set drop takes: waits, collections, clears, the bound read, kicks, footprint reads, sleeps."""
    from mlx_dfloat.mflux import _drain

    remaining = list(footprints)
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: events.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear_cache"))
    monkeypatch.setattr(gc, "collect", lambda *a: events.append("collect") or 0)
    monkeypatch.setattr(_drain, "_kick", lambda: events.append("kick"))
    monkeypatch.setattr(_drain, "_sleep", lambda seconds: events.append("sleep"))

    def read():
        events.append("read")
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(shared, "_read_footprint", read, raising=False)
    return shared.Lifecycle(
        load_encoders=lambda: None,
        unload_encoders=lambda: events.append("unload_encoders"),
        encode=lambda prompt: (),
        load_set=lambda: None,
        unload_set=lambda: events.append("unload_set"),
        prompt_cache={},
        retained_bound=lambda: events.append("bound") or 1 << 62,
    )


class _ScopeOnly:
    """An active guard that never tripped: what an armed call's scope looks like to the lifecycle."""

    class tripped:  # noqa: N801
        @staticmethod
        def is_set():
            return False


def test_an_unarmed_drop_makes_exactly_the_calls_it_made_before(monkeypatch):
    # Bug caught: the drain run on an unarmed path (the command, watchdog=None, verify_image), which then pays a GPU
    # kick and up to 0.25 s of polling on every drop that no watchdog asked for. The list is the 0.2.2 sequence.
    events = []
    life = _record_drop(monkeypatch, events)
    life.drop_set()
    life.drop_encoders()
    assert events == [
        "unload_set",
        "synchronize",
        "collect",
        "clear_cache",
        "bound",
        "unload_encoders",
        "synchronize",
        "collect",
        "clear_cache",
        "bound",
    ]


def test_an_armed_drop_drains_after_the_check(monkeypatch):
    # Bug caught: no drain inside an armed call, so the released pages stay in the footprint and the next sample (the
    # set load right after the encoders' drop) trips on memory the call already gave back; or the drain run before the
    # active-memory check (a failed check would then have kicked for nothing).
    events = []
    life = _record_drop(monkeypatch, events, footprints=(900 << 20, 700 << 20, 700 << 20))
    with _guard.scope(_ScopeOnly()):
        life.drop_encoders()
    assert events == [
        "unload_encoders",
        "synchronize",
        "collect",
        "clear_cache",
        "bound",
        "synchronize",
        "clear_cache",
        "kick",
        "read",
        "sleep",
        "read",
        "sleep",
        "read",
        "synchronize",
        "clear_cache",
    ]
