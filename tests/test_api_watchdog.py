"""A model call through ``with_call_caps`` runs under a raise-mode memory watchdog by default.

mflux-free: a small fake model class carries what the decorator reads (``watchdog``, ``open_phase``, ``_model``, a
real ``Lifecycle`` over in-memory callbacks), and its ``generate_image`` runs two checkpoints around a hook. The
autouse ``_api_watchdog_fakes`` fixture (``tests/conftest.py``) puts the fake allocator in front of the watchdog,
fixes the host ceiling at 30_064_771_072 bytes (a 32 GiB Mac minus 4 GiB) and records any process exit, so every
expected number below is a literal. Hooks raise the fake and wait on the watchdog's ``tripped`` event with a 5 s
timeout; the only fixed waits are the "nothing tripped within 0.3 s" checks.
"""

import gc
import os
import platform
import re
import stat
import tempfile
import threading
import time
import weakref
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_dfloat import _guard, _watchdog
from mlx_dfloat.errors import DFloatUnsupportedError, DFloatWatchdogError
from mlx_dfloat.mflux import _drain, _pipeline, lifecycle
from mlx_dfloat.mflux._watchconfig import DEFAULT_WATCHDOG, WatchdogConfig
from mlx_dfloat.mflux.lifecycle import Lifecycle

# 32 GiB - 4 GiB, and that minus the default 1 GiB stop margin (27 GiB).
CEILING = 30_064_771_072
STOP_LINE = 28_991_029_248
THREAD = "mlx-dfloat-watchdog"

# Taken before any fixture patches it, for the default-location test.
_REAL_ARTIFACT_ROOT = _pipeline._artifact_root
# The three watchdog seams as shipped, taken before the autouse fixture replaces them.
_SHIPPED_SEAMS = (_pipeline._counters, _pipeline._default_ceiling, _pipeline._exit_process)


class _Interrupt(BaseException):
    """Stands in for a KeyboardInterrupt landing between two statements of the decorator."""


@pytest.fixture(autouse=True)
def aborts(tmp_path, monkeypatch):
    """Every abort record of this file lands under the test's own folder, never in the real temp dir."""
    root = tmp_path / "aborts"
    monkeypatch.setattr(_pipeline, "_artifact_root", lambda: root)
    return root


@pytest.fixture
def fakes(request):
    """The autouse ``_api_watchdog_fakes`` value: the fake allocator and the exit recorder of this test."""
    return request.getfixturevalue("_api_watchdog_fakes")


class _Held:
    """An object only the fake method's frame refers to."""


def _watchdog_threads():
    return {t for t in threading.enumerate() if t.name == THREAD and t.is_alive()}


class FakeModel:
    """What the decorator reads on a model class, around a body with two checkpoints and three hooks."""

    def __init__(
        self,
        *,
        watchdog=DEFAULT_WATCHDOG,
        hook=None,
        inner=None,
        tail=None,
        leak_set=False,
        load_hook=None,
        encode_hook=None,
    ):
        self.watchdog = watchdog
        self._model = "fake-model"
        self.open_phase = None
        self.hook, self.inner, self.tail = hook, inner, tail
        self.load_hook = load_hook
        self.encode_hook = encode_hook
        self.ran = []
        self.held = None
        self.bound = 2**62
        self._set = None
        self._leaked = []
        self._leak_set = leak_set
        self._lifecycle = Lifecycle(
            load_encoders=lambda: None,
            unload_encoders=lambda: None,
            encode=lambda prompt: (),
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache={},
            retained_bound=lambda: self.bound,
        )

    def _load_set(self):
        if self.load_hook is not None:  # a trip during the set load: the set is not resident yet
            self.load_hook(self)
            _guard.checkpoint("set_group")
        self._set = mx.zeros((1024,), dtype=mx.float32)
        mx.eval(self._set)

    def _unload_set(self):
        if self._leak_set:  # the drop leaves the set referenced, so the reclaim check fails
            self._leaked.append(self._set)
        self._set = None

    @_pipeline.with_call_caps
    def generate_image(
        self, seed, prompt, num_inference_steps=2, height=64, width=64, guidance=None
    ):
        held = _Held()
        self.held = weakref.ref(held)
        self.ran.append("body")
        self._lifecycle.ensure_set()
        self.open_phase = "denoise"
        _guard.checkpoint("block_start")
        if self.hook is not None:
            self.hook(self)
        if self.inner is not None:
            self.inner.generate_image(seed, prompt)
        _guard.checkpoint("block_end")
        self.open_phase = "vae"
        if self.tail is not None:
            self.tail(self)
        self.open_phase = None
        return "image"

    @_pipeline.with_call_caps
    def encode(self, prompt):
        self.ran.append("encode")
        self.open_phase = "encode"
        if self.encode_hook is not None:
            self.encode_hook(self)
        _guard.checkpoint("prompt_encoded")
        self.open_phase = None
        return ()


def overrun(counters, level):
    """A hook that raises the fake to ``level`` and waits for the active watchdog to trip (a no-op wait unarmed)."""

    def hook(model):
        watchdog = _guard.active()
        counters.raise_to(level)
        if watchdog is not None:
            model.ran.append(("tripped", watchdog.tripped.wait(5.0)))

    return hook


def _sampled_at(watchdog, level):
    """Poll (5 s) until the watchdog has sampled ``level``; True if it did."""
    deadline = time.monotonic() + 5.0
    while watchdog.peak_watched < level:
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_model_built_with_no_watchdog_argument_is_armed(fakes):
    # Bug caught: the default config off (the call runs unwatched and returns), or a stop margin other than 1 GiB.
    model = FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    err = excinfo.value
    assert err.record["ceiling"] == 30_064_771_072
    assert err.record["stop_line"] == 28_991_029_248
    assert (err.reason, err.phase, err.site) == ("memory", "denoise", "block_end")


def test_the_stop_line_itself_does_not_trip(fakes):
    # Bug caught: `>=` at the stop line in the armed call's watchdog, or a margin one byte off. The hook proves a
    # watchdog sampled the level before checking it did not trip.
    seen = []

    def hook(model):
        watchdog = _guard.active()
        fakes.counters.raise_to(STOP_LINE)
        if watchdog is None:
            seen.append(None)
            return
        sampled = _sampled_at(watchdog, STOP_LINE)
        seen.append((sampled, watchdog.tripped.wait(0.3)))

    assert FakeModel(hook=hook).generate_image(1, "p") == "image"
    assert seen == [(True, False)]


def test_watchdog_none_runs_unwatched(fakes):
    # Bug caught: watchdog=None still arming (a thread starts; at twice the ceiling it escalates and the fixture's
    # teardown fails on the recorded exit).
    before = _watchdog_threads()
    seen = []

    def hook(model):
        fakes.counters.raise_to(2 * CEILING)
        seen.append((_guard.active(), len(_watchdog_threads() - before)))
        time.sleep(0.2)  # several sample intervals: an armed watchdog would escalate here

    assert FakeModel(watchdog=None, hook=hook).generate_image(1, "p") == "image"
    assert seen == [(None, 0)]


def test_a_nested_call_does_not_arm_a_second_watchdog():
    # Bug caught: the decorator arming on every call, so a nested encode inside generate_image starts a second thread.
    before = _watchdog_threads()
    seen = []
    inner = FakeModel(hook=lambda model: seen.append(len(_watchdog_threads() - before)))
    assert FakeModel(inner=inner).generate_image(1, "p") == "image"
    assert seen == [1]


def test_a_nested_lower_ceiling_wins_for_the_inner_call_only(fakes):
    # Bug caught: the nested model's lower ceiling ignored (no trip at 20 GB - 1 GiB + 1), or left in force after the
    # inner call returns (the same level trips the outer).
    level = 20_000_000_000 - 1_073_741_824 + 1
    lower = WatchdogConfig(ceiling=20_000_000_000)
    inner = FakeModel(watchdog=lower, hook=overrun(fakes.counters, level))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(inner=inner).generate_image(1, "p")
    assert excinfo.value.record["ceiling"] == 20_000_000_000
    assert excinfo.value.record["stop_line"] == 18_926_258_176

    fakes.counters.raise_to(0)
    seen = []

    def after_inner(model):
        watchdog = _guard.active()
        fakes.counters.raise_to(level)
        seen.append((_sampled_at(watchdog, level), watchdog.tripped.wait(0.3)))

    outer = FakeModel(inner=FakeModel(watchdog=lower), tail=after_inner)
    assert outer.generate_image(1, "p") == "image"
    assert seen == [(True, False)]


def test_an_outer_guard_from_another_call_keeps_its_own_ceiling(fakes):
    # Bug caught: a nested model built with watchdog=None stepping out of the outer call's guard (scope reset to
    # None), so an overrun inside it runs on unwatched.
    inner = FakeModel(watchdog=None, hook=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(inner=inner).generate_image(1, "p")
    assert excinfo.value.record["ceiling"] == 30_064_771_072


def test_a_ceiling_above_the_host_ceiling_is_refused_before_anything_runs():
    # Bug caught: a ceiling above the host's accepted (the watchdog then guards a line the machine cannot reach), or
    # refused only after the body ran.
    model = FakeModel(watchdog=WatchdogConfig(ceiling=30_064_771_073))
    with pytest.raises(DFloatUnsupportedError, match="30064771073"):
        model.generate_image(1, "p")
    assert model.ran == []
    assert model._lifecycle.counters.set_loads == 0


def test_the_abort_drops_a_resident_set_and_clears_frames(fakes):
    # Bug caught: the abort leaving the set resident (the next call reuses a half-dropped set), or the traceback
    # keeping the method's locals alive for as long as the caller holds the error.
    model = FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert model._lifecycle.set_resident is False
    assert excinfo.value.site == "block_end"  # the error is still held while the frames are checked
    assert model.held() is None


def test_a_reclaim_failure_is_attached_to_the_abort(fakes):
    # Bug caught: the DFloatResourceError from the drop's reclaim check replacing the abort, so the caller sees a
    # "still resident" error and never learns the watchdog stopped the call; or the reclaim failure lost.
    def hook(model):
        model.bound = 0
        overrun(fakes.counters, STOP_LINE + 1)(model)

    model = FakeModel(hook=hook, leak_set=True)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("DFloatResourceError" in note for note in notes)


def test_no_watchdog_thread_survives_the_call(fakes):
    # Bug caught: the watchdog not stopped on the way out (after an abort or a normal return), leaving a sampler that
    # can still end the process later.
    before = _watchdog_threads()
    during = []

    def hook(model):
        during.append(len(_watchdog_threads() - before))

    assert FakeModel(hook=hook).generate_image(1, "p") == "image"
    assert _watchdog_threads() - before == set()

    def tripping(model):
        hook(model)
        overrun(fakes.counters, STOP_LINE + 1)(model)

    with pytest.raises(DFloatWatchdogError):
        FakeModel(hook=tripping).generate_image(1, "p")
    assert _watchdog_threads() - before == set()
    assert during == [1, 1]


def test_a_trip_after_the_last_checkpoint_raises_at_call_end(fakes):
    # Bug caught: no checkpoint after the method returns, so a trip during the VAE decode (no checkpoint inside it)
    # returns an image from an overrun call; or the phase read when the error is raised instead of at the trip.
    model = FakeModel(tail=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert (excinfo.value.site, excinfo.value.phase) == ("call_end", "vae")


def test_the_artifact_lands_in_the_default_root_with_a_unique_name(fakes, aborts):
    # Bug caught: a fixed file name (the second abort overwrites the first), or the record written somewhere other
    # than the root when the config names no folder.
    errors = []
    for _ in range(2):
        fakes.counters.raise_to(0)
        with pytest.raises(DFloatWatchdogError) as excinfo:
            FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
        errors.append(excinfo.value)
    names = sorted(p.name for p in aborts.iterdir())
    assert len(names) == 2
    pattern = re.compile(rf"abort-\d{{8}}T\d{{12}}Z-{os.getpid()}\.json")
    assert all(pattern.fullmatch(name) for name in names), names
    assert sorted(err.artifact.name for err in errors) == names


def test_the_artifact_root_defaults_to_the_temp_dir():
    # Bug caught: the records written to the working directory or a fixed /tmp path instead of the system temp dir.
    assert _REAL_ARTIFACT_ROOT() == Path(tempfile.gettempdir()) / "mlx-dfloat"


def test_the_default_counter_reader_is_the_real_one(monkeypatch):
    # Bug caught: the armed watchdog not reading the counters through the seam (or the seam's default not the real
    # reader). The stop line is 1 MiB: a bare Python process with mlx.core imported measured a 17_334_920-byte
    # footprint on an M1 Max (2026-11-24), so any real reading is above it, and nowhere near the 28 GiB ceiling.
    monkeypatch.setattr(_pipeline, "_counters", _watchdog.read_counters)
    config = WatchdogConfig(ceiling=30_064_771_072, stop_margin=30_064_771_072 - 1024**2)

    def hook(model):
        model.ran.append(("tripped", _guard.active().tripped.wait(5.0)))

    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(watchdog=config, hook=hook).generate_image(1, "p")
    assert excinfo.value.record["verdict_memory"] > 1024**2


def test_default_ceiling_comes_from_the_injected_host_figure(fakes, monkeypatch):
    # Bug caught: the default ceiling read from the host directly instead of through the seam at call time.
    monkeypatch.setattr(_pipeline, "_default_ceiling", lambda: 17_179_869_184)
    model = FakeModel(hook=overrun(fakes.counters, 16_106_127_361))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert excinfo.value.record["ceiling"] == 17_179_869_184


def test_the_record_names_the_call(fakes):
    # Bug caught: an abort record that cannot say which model, method and arguments it stopped, or one that records
    # the prompt.
    model = FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(7, "a secret prompt", num_inference_steps=3, height=48, width=80)
    context = excinfo.value.record["context"]
    assert {k: context[k] for k in ("model", "class", "method", "seed", "height", "width")} == {
        "model": "fake-model",
        "class": "FakeModel",
        "method": "generate_image",
        "seed": 7,
        "height": 48,
        "width": 80,
    }
    assert (context["num_inference_steps"], context["guidance"]) == (3, None)
    assert context["versions"]["mlx"] == mx.__version__
    assert context["macos"] == platform.mac_ver()[0]
    assert "a secret prompt" not in repr(excinfo.value.record)


def test_the_seams_ship_as_the_real_reader_ceiling_and_exit():
    # Bug caught: a seam default left pointing at a fake or another function (the autouse fixture replaces all three
    # before every test, so nothing else would see it): a shipped call reading no real memory, a ceiling from the
    # wrong figure, or an exit other than the watchdog module's own.
    counters, ceiling, exit_process = _SHIPPED_SEAMS
    assert counters is _watchdog.read_counters
    assert ceiling is _watchdog.default_ceiling
    assert exit_process is None


def _recording_cleanup(monkeypatch, events):
    """Record, in ``events``, every synchronize, garbage collection, cache clear and kick the abort path makes."""
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: events.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear_cache"))
    monkeypatch.setattr(gc, "collect", lambda *a: events.append("collect") or 0)
    monkeypatch.setattr(_drain, "_kick", lambda: events.append("kick"))
    monkeypatch.setattr(_drain, "_sleep", lambda seconds: events.append("sleep"))
    # A footprint that does not fall: each drain polls its whole 0.25 s budget, 12 sleeps of 0.02 s (D26: a flat
    # reading before any fall does not end the poll), whatever the real footprint does.
    monkeypatch.setattr(lifecycle, "_read_footprint", lambda: 1, raising=False)


def test_an_abort_during_the_set_load_still_empties_the_buffer_pool(fakes, monkeypatch):
    # Bug caught: `_after_abort` returning before any cleanup when the set is not resident yet (a stop at a set_group
    # checkpoint), so the groups loaded so far stay in MLX's buffer pool after the error; or the pool emptied before
    # the GPU finished (no synchronize first), or without the collection that frees the dropped arrays.
    events = []

    def load_hook(model):
        overrun(fakes.counters, STOP_LINE + 1)(model)
        events.append("tripped")

    model = FakeModel(load_hook=load_hook)
    _recording_cleanup(monkeypatch, events)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert excinfo.value.site == "set_group"
    assert model._lifecycle.set_resident is False
    after = events[events.index("tripped") + 1 :]
    # The pool emptied, then the drain (W2 item 3: it replaces the bare kick and clear).
    assert after == [
        *["synchronize", "collect", "clear_cache"],
        *["synchronize", "clear_cache", "kick", *["sleep"] * 12, "synchronize", "clear_cache"],
    ]


def test_an_abort_with_the_set_resident_kicks_and_clears_after_the_drop(fakes, monkeypatch):
    # Bug caught: the kick and the second clear run before the set's drop (its freed buffers then reach the pool after
    # the last clear), or skipped when the set was resident.
    events = []

    def hook(model):
        overrun(fakes.counters, STOP_LINE + 1)(model)
        events.append("tripped")

    model = FakeModel(hook=hook)
    real_unload = model._unload_set

    def unload():
        events.append("unload_set")
        real_unload()

    model._lifecycle._unload_set = unload
    _recording_cleanup(monkeypatch, events)
    with pytest.raises(DFloatWatchdogError):
        model.generate_image(1, "p")
    after = events[events.index("tripped") + 1 :]
    assert after[: after.index("unload_set")] == ["synchronize", "collect", "clear_cache"]
    drain = ["synchronize", "clear_cache", "kick", *["sleep"] * 12, "synchronize", "clear_cache"]
    # The drop's own reclaim drains inside the armed call, then the abort drains once more (W2 item 3).
    assert after[after.index("unload_set") :] == [
        *["unload_set", "synchronize", "collect", "clear_cache"],
        *drain,
        *drain,
    ]


def test_an_interrupt_right_after_the_start_still_stops_the_watchdog(monkeypatch):
    # Bug caught: `watchdog.start()` outside the `try` whose `finally` stops it, so an interrupt landing between the
    # two leaves a live raise-mode sampler that can end the process later.
    before = _watchdog_threads()
    started = []
    real_start = _watchdog.Watchdog.start

    def start_then_interrupt(self):
        started.append(self)
        real_start(self)
        raise _Interrupt

    monkeypatch.setattr(_watchdog.Watchdog, "start", start_then_interrupt)
    try:
        with pytest.raises(_Interrupt):
            FakeModel().generate_image(1, "p")
        alive = _watchdog_threads() - before
    finally:
        for watchdog in started:
            watchdog.stop()
    assert alive == set()


def test_a_trip_between_the_last_checkpoint_and_the_stop_raises(fakes, aborts, monkeypatch):
    # Bug caught: the watchdog stopped only after the `call_end` checkpoint, so a trip in between writes an abort record
    # for a call that returns normally (a file on disk with no error to match it), or the late trip ignored.
    real_stop = _watchdog.Watchdog.stop
    late = []

    def trip_then_stop(self):
        if not late:
            late.append(True)
            fakes.counters.raise_to(STOP_LINE + 1)
            late.append(self.tripped.wait(5.0))
        real_stop(self)

    monkeypatch.setattr(_watchdog.Watchdog, "stop", trip_then_stop)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel().generate_image(1, "p")
    assert late == [True, True]
    assert excinfo.value.site == "call_end"
    assert [p.name for p in aborts.iterdir()] == [excinfo.value.artifact.name]


@pytest.mark.parametrize("margin", [30_064_771_072, 30_064_771_073])
def test_a_margin_not_below_the_default_ceiling_is_refused_before_anything_runs(margin):
    # Bug caught: a margin at or above the host ceiling accepted when the config names no ceiling (the stop line is
    # then 0 or below and every sample trips), or refused only after the body ran.
    model = FakeModel(watchdog=WatchdogConfig(stop_margin=margin))
    with pytest.raises(DFloatUnsupportedError, match=str(margin)):
        model.generate_image(1, "p")
    assert model.ran == []
    assert model._lifecycle.counters.set_loads == 0


def test_a_margin_one_below_the_default_ceiling_runs():
    # Bug caught: the refusal written as `margin + 1 >= ceiling`, refusing the margin that leaves a 1-byte stop line.
    model = FakeModel(watchdog=WatchdogConfig(stop_margin=30_064_771_071))
    assert model.generate_image(1, "p") == "image"


def test_a_direct_encode_call_is_armed_and_raises(fakes):
    # Bug caught: arming tied to `generate_image` (by name or by a flag only it sets), so a caller who encodes on its
    # own runs unwatched and the overrun returns.
    model = FakeModel(encode_hook=overrun(fakes.counters, STOP_LINE + 1))
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.encode("p")
    assert (excinfo.value.site, excinfo.value.phase) == ("prompt_encoded", "encode")
    assert excinfo.value.record["context"]["method"] == "encode"


def test_a_failing_cleanup_step_is_noted_and_the_abort_still_leaves(fakes, monkeypatch):
    # Bug caught: a cleanup step that raises (here the pool clear) replacing the typed error and skipping the steps
    # after it. Three steps clear the pool (the abort's own clear, the set drop's reclaim, the drain), so three notes.
    def hook(model):
        overrun(fakes.counters, STOP_LINE + 1)(model)

        def broken_clear():
            raise RuntimeError("pool clear failed")

        monkeypatch.setattr(mx, "clear_cache", broken_clear)

    model = FakeModel(hook=hook)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    notes = getattr(excinfo.value, "__notes__", [])
    assert sum("RuntimeError: pool clear failed" in note for note in notes) == 3
    assert model._lifecycle.set_resident is False
    assert model.held() is None


def _record_starts(monkeypatch):
    started = []
    real_start = _watchdog.Watchdog.start

    def start(self):
        started.append(self)
        return real_start(self)

    monkeypatch.setattr(_watchdog.Watchdog, "start", start)
    return started


@pytest.mark.parametrize("level", [STOP_LINE, STOP_LINE + 1, 2 * CEILING])
def test_a_call_that_starts_at_or_above_the_stop_line_raises_before_any_work(
    fakes, aborts, monkeypatch, level
):
    # Bug caught: no reading at arm time, so a process already over the line starts the sampler and the call (the
    # first sample, about 50 ms in, trips it; above the hard line it ends the process); or `>` instead of "at or
    # above" at the stop line itself. 2 * CEILING is above the hard line: the call raises and nothing exits.
    started = _record_starts(monkeypatch)
    fakes.counters.raise_to(level)
    model = FakeModel()
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    err = excinfo.value
    assert (err.site, err.reason, err.phase) == ("call_start", "memory", None)
    assert err.record["verdict_memory"] == level
    assert err.artifact.parent == aborts
    assert (started, model.ran, model._lifecycle.counters.set_loads) == ([], [], 0)
    assert fakes.exits.codes == []


def test_a_call_that_starts_one_byte_below_the_stop_line_runs(fakes, monkeypatch):
    # Bug caught: the arm-time comparison one byte too eager (or against the ceiling minus twice the margin).
    started = _record_starts(monkeypatch)
    fakes.counters.raise_to(STOP_LINE - 1)
    assert FakeModel().generate_image(1, "p") == "image"
    assert len(started) == 1


def test_another_error_after_a_trip_names_the_trip_and_cleans_up(fakes, aborts):
    # Bug caught: an error other than the typed one leaving a tripped call with no word of the trip or its record,
    # and with the set resident and the frames alive (no abort cleanup).
    def hook(model):
        overrun(fakes.counters, STOP_LINE + 1)(model)
        raise ValueError("model bug")

    model = FakeModel(hook=hook)
    with pytest.raises(ValueError, match="model bug") as excinfo:
        model.generate_image(1, "p")
    (record,) = aborts.iterdir()
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("(memory, phase denoise)" in n and f"record: {record}" in n for n in notes), notes
    assert model._lifecycle.set_resident is False
    assert model.held() is None


def test_another_error_with_no_trip_is_left_alone(fakes):
    # Bug caught: the trip note and the abort cleanup run for every error, so a plain failure drops the set and reads
    # as a watchdog stop.
    def hook(model):
        raise ValueError("model bug")

    model = FakeModel(hook=hook)
    with pytest.raises(ValueError, match="model bug") as excinfo:
        model.generate_image(1, "p")
    assert getattr(excinfo.value, "__notes__", []) == []
    assert model._lifecycle.set_resident is True


def test_the_default_folder_is_created_private(fakes, aborts):
    # Bug caught: the default folder made with the umask's mode (0o755), so other users can list the records.
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert stat.S_IMODE(aborts.stat().st_mode) == 0o700
    assert excinfo.value.artifact.parent == aborts


def test_a_symlinked_default_folder_gets_no_record(fakes, aborts, tmp_path):
    # Bug caught: a default folder that another user pre-placed as a symlink trusted, so the record lands wherever it
    # points; or the refusal costing the typed error.
    target = tmp_path / "elsewhere"
    target.mkdir()
    aborts.symlink_to(target)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert excinfo.value.artifact is None
    assert "is a symlink" in excinfo.value.record["artifact_error"]
    assert list(target.iterdir()) == []


def test_a_default_folder_owned_by_another_user_gets_no_record(fakes, aborts, monkeypatch):
    # Bug caught: the owner not checked, so a folder another user created (and can read) receives the record.
    aborts.mkdir(mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert excinfo.value.artifact is None
    assert "is owned by uid" in excinfo.value.record["artifact_error"]
    assert list(aborts.iterdir()) == []


def test_a_symlinked_folder_the_caller_names_is_still_written(fakes, tmp_path):
    # Bug caught: the default folder's checks applied to a folder the caller chose, refusing a symlink they made.
    target = tmp_path / "chosen"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    model = FakeModel(
        watchdog=WatchdogConfig(artifact_dir=link), hook=overrun(fakes.counters, STOP_LINE + 1)
    )
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert excinfo.value.artifact is not None
    assert [p.name for p in target.iterdir()] == [excinfo.value.artifact.name]


def test_a_default_folder_that_is_a_file_gets_no_record(fakes, aborts):
    # Bug caught: a non-folder at the default path passed on to the write, which then fails with a less useful error.
    aborts.write_text("not a folder")
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert excinfo.value.artifact is None
    assert "is not a folder" in excinfo.value.record["artifact_error"]
    assert aborts.read_text() == "not a folder"


def test_a_default_folder_that_cannot_be_made_still_raises_the_typed_error(
    fakes, tmp_path, monkeypatch
):
    # Bug caught: the folder check raising on the sampler thread (a missing parent), which kills the sampler before the
    # trip is set, so the call runs on unwatched.
    root = tmp_path / "missing" / "aborts"
    monkeypatch.setattr(_pipeline, "_artifact_root", lambda: root)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert excinfo.value.artifact is None
    assert excinfo.value.record["artifact_error"].startswith("FileNotFoundError")
    assert not root.parent.exists()


def test_an_entry_trip_drops_a_set_left_resident_so_the_next_call_runs(fakes, monkeypatch):
    # Bug caught: the call_start raise skipping the abort cleanup, so a set left resident by an earlier call keeps
    # memory over the line and every later call raises at entry for good.
    model = FakeModel()
    assert model.generate_image(1, "p") == "image"
    assert model._lifecycle.set_resident is True
    started = _record_starts(monkeypatch)
    fakes.counters.raise_to(STOP_LINE + 1)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(1, "p")
    assert excinfo.value.site == "call_start"
    assert (started, model.ran) == ([], ["body"])
    assert model._lifecycle.set_resident is False
    fakes.counters.raise_to(0)
    assert model.generate_image(1, "p") == "image"
    assert model._lifecycle.counters.set_loads == 2


def test_memory_that_falls_after_the_arm_time_drain_does_not_raise(fakes, aborts, monkeypatch):
    # Bug caught: the arm-time reading trips on pages an earlier unarmed call freed but the driver has not yet
    # released (they leave the footprint only after the next GPU submission), so a call raises at call_start for
    # memory that a drain gives back.
    started = _record_starts(monkeypatch)
    monkeypatch.setattr(_drain, "_kick", lambda: fakes.counters.raise_to(STOP_LINE - 1))
    monkeypatch.setattr(_drain, "_sleep", lambda seconds: None)
    fakes.counters.raise_to(STOP_LINE + 1)
    model = FakeModel()
    assert model.generate_image(1, "p") == "image"
    assert len(started) == 1
    assert not aborts.exists()


def test_a_call_below_the_stop_line_reads_once_at_arm_time_and_never_drains(fakes, monkeypatch):
    # Bug caught: the drain (0.25 s at most, plus a GPU wait) run on every call, or a second arm-time reading taken
    # when the first is already below the line.
    events = []
    real_sample = _watchdog.Watchdog._sample
    real_drain = _drain.drain
    real_start = _watchdog.Watchdog.start

    def sample(self):
        if threading.current_thread() is threading.main_thread():
            events.append("read")
        return real_sample(self)

    def drain(*args, **kwargs):
        events.append("drain")
        return real_drain(*args, **kwargs)

    def start(self):
        events.append("start")
        return real_start(self)

    monkeypatch.setattr(_watchdog.Watchdog, "_sample", sample)
    monkeypatch.setattr(_drain, "drain", drain)
    monkeypatch.setattr(_watchdog.Watchdog, "start", start)
    fakes.counters.raise_to(STOP_LINE - 1)
    assert FakeModel().generate_image(1, "p") == "image"
    assert events[: events.index("start") + 1] == ["read", "start"]


def test_a_watchdog_reassigned_to_false_after_construction_is_refused_with_the_typed_error(fakes):
    # Bug caught: the constructor's check not repeated at call time, so `model.watchdog = False` fails with an
    # AttributeError from deep inside the decorator instead of the package's refusal.
    model = FakeModel()
    model.watchdog = False
    with pytest.raises(DFloatUnsupportedError, match="watchdog must be a WatchdogConfig"):
        model.generate_image(1, "p")
    assert model.ran == []


def test_an_own_default_folder_open_to_everyone_is_made_private_and_written(fakes, aborts):
    # Bug caught: an existing default folder this user owns accepted at any mode, so an own 0o777 folder (made by an
    # earlier tool, or a loose umask) lets every user list and replace the records.
    aborts.mkdir()
    aborts.chmod(0o777)
    with pytest.raises(DFloatWatchdogError) as excinfo:
        FakeModel(hook=overrun(fakes.counters, STOP_LINE + 1)).generate_image(1, "p")
    assert stat.S_IMODE(aborts.stat().st_mode) == 0o700
    assert excinfo.value.artifact is not None
    assert excinfo.value.artifact.parent == aborts
    assert [p.name for p in aborts.iterdir()] == [excinfo.value.artifact.name]
