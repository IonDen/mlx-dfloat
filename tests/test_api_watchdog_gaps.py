"""Three things about the armed model call that the other watchdog tests leave open (mflux-free).

1. The autouse ``_api_watchdog_fakes`` fixture's exit recorder is in force: an escalation inside a test lands in the
   recorder and pytest goes on.
2. ``clear_call_frames(outer=...)``: a call made inside the caller's own ``except`` block clears its own frames but
   leaves the frames of the exception the caller is handling.
3. An explicit ``WatchdogConfig(artifact_dir=..., grace=..., wall_budget=...)`` reaches the armed watchdog.

A small decorated fake carries what ``with_call_caps`` reads (``watchdog``, ``open_phase``, ``_model``) and runs one
checkpoint after a hook. The fixture fixes the host ceiling at 30_064_771_072 bytes (a 32 GiB Mac minus 4 GiB), so
every number below is a literal. Waits poll with a 5 s timeout.
"""

import time
import weakref

import pytest

from mlx_dfloat import _guard
from mlx_dfloat.errors import DFloatWatchdogError
from mlx_dfloat.mflux import _pipeline
from mlx_dfloat.mflux._watchconfig import WatchdogConfig

# 32 GiB - 4 GiB, and that minus the default 1 GiB stop margin (27 GiB).
CEILING = 30_064_771_072
STOP_LINE = 28_991_029_248


@pytest.fixture(autouse=True)
def default_root(tmp_path, monkeypatch):
    """The folder a config with no ``artifact_dir`` writes to, under the test's own folder."""
    root = tmp_path / "default-root"
    monkeypatch.setattr(_pipeline, "_artifact_root", lambda: root)
    return root


@pytest.fixture
def fakes(request):
    """The autouse ``_api_watchdog_fakes`` value: the fake allocator and the exit recorder of this test."""
    return request.getfixturevalue("_api_watchdog_fakes")


class GapModel:
    """A model class as the decorator sees it: the hook runs in the denoise phase, then one checkpoint."""

    def __init__(self, watchdog, hook):
        self.watchdog = watchdog
        self._model = "gap-model"
        self.open_phase = None
        self.hook = hook

    @_pipeline.with_call_caps
    def generate_image(self, seed=1):
        self.open_phase = "denoise"
        self.hook(self)
        _guard.checkpoint("block_end")
        self.open_phase = None
        return "image"


def _poll(condition):
    deadline = time.monotonic() + 5.0
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _trip_and_wait(counters, level, seen):
    def hook(model):
        counters.raise_to(level)
        seen["tripped"] = _guard.active().tripped.wait(5.0)

    return hook


def test_an_escalation_during_a_test_lands_in_the_exit_recorder(fakes):
    # Bug caught: the fixture not putting its recorder in front of the armed watchdog's exit, so a call that reaches
    # the ceiling ends pytest with code 70 and no summary instead of recording the code.
    seen = {}

    def hook(model):
        fakes.counters.raise_to(CEILING)
        seen["exited"] = _poll(lambda: fakes.exits.codes != [])

    with pytest.raises(DFloatWatchdogError):
        GapModel(WatchdogConfig(), hook).generate_image()

    assert seen["exited"] is True
    assert fakes.exits.take() == [70]


def test_a_call_inside_the_callers_except_block_leaves_the_handled_exceptions_frames(fakes):
    # Bug caught: the decorator not taking the exception being handled at entry as the walk's stop (outer None), so
    # clearing the abort's chain also clears the caller's handled exception's frames (the abort's __context__).
    class Evidence:
        pass

    def failing_helper():
        evidence = Evidence()
        refs.append(weakref.ref(evidence))
        raise ValueError("the caller's own failure")

    refs = []
    seen = {}
    model = GapModel(WatchdogConfig(), _trip_and_wait(fakes.counters, STOP_LINE + 1, seen))
    try:
        failing_helper()
    except ValueError as exc:
        handled = exc
        with pytest.raises(DFloatWatchdogError) as excinfo:
            model.generate_image()

    assert excinfo.value.__context__ is handled
    assert seen["tripped"] is True
    assert refs[0]() is not None  # failing_helper's frame still holds its local


def test_an_explicit_artifact_dir_receives_the_record(tmp_path, fakes, default_root):
    # Bug caught: the decorator ignoring config.artifact_dir (always the default root).
    seen = {}
    mine = tmp_path / "mine"
    model = GapModel(
        WatchdogConfig(artifact_dir=mine), _trip_and_wait(fakes.counters, STOP_LINE + 1, seen)
    )

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image()

    assert seen["tripped"] is True
    assert excinfo.value.artifact is not None
    assert excinfo.value.artifact.parent == mine
    assert excinfo.value.artifact.is_file()
    assert not default_root.exists()


def test_an_explicit_wall_budget_trips_with_reason_wall(fakes):
    # Bug caught: the decorator not passing config.wall_budget to the watchdog (no wall limit at all), so the call
    # runs past its budget and returns.
    seen = {}

    def hook(model):
        seen["tripped"] = _guard.active().tripped.wait(5.0)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        GapModel(WatchdogConfig(wall_budget=0.2), hook).generate_image()

    assert seen["tripped"] is True
    assert excinfo.value.reason == "wall"
    assert excinfo.value.record["budget"] == 0.2


def test_an_explicit_grace_ends_the_process_when_no_checkpoint_comes_in_time(fakes):
    # Bug caught: the decorator not passing config.grace (the 10 s default instead), so a trip that no checkpoint
    # takes for 0.3 s is not escalated within the 5 s this test waits.
    seen = {}

    def hook(model):
        # A wall trip with memory at zero: grace still ends a wall trip, and the ceiling is far away.
        seen["tripped"] = _guard.active().tripped.wait(5.0)
        seen["exited"] = _poll(lambda: fakes.exits.codes != [])

    with pytest.raises(DFloatWatchdogError) as excinfo:
        GapModel(WatchdogConfig(wall_budget=0.2, grace=0.3), hook).generate_image()

    assert seen["tripped"] is True
    assert seen["exited"] is True
    assert excinfo.value.reason == "wall"
    assert fakes.exits.take() == [71]
