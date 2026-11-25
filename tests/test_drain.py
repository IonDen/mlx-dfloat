"""The drain run after an armed call frees memory: wait, clear, kick, poll the footprint, wait, clear.

MLX-free in effect: every step the drain takes is recorded through monkeypatched ``mx`` calls and the module's kick and
sleep, and the footprint comes from a scripted reader, so no test pays wall time and every count is a literal.
"""

import mlx.core as mx
import pytest

from mlx_dfloat.mflux import _drain

MiB = 1 << 20
GiB = 1 << 30


@pytest.fixture
def events(monkeypatch):
    """Record every synchronize, cache clear, kick and sleep the drain makes, in order."""
    log = []
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: log.append("synchronize"))
    monkeypatch.setattr(mx, "clear_cache", lambda: log.append("clear_cache"))
    monkeypatch.setattr(_drain, "_kick", lambda: log.append("kick"))
    monkeypatch.setattr(_drain, "_sleep", lambda seconds: log.append(("sleep", seconds)))
    return log


def _reader(values, log):
    """A footprint reader that returns ``values`` in turn (the last one forever) and logs each read."""
    remaining = list(values)

    def read():
        value = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        log.append(("read", value))
        return value

    return read


def _reads(log):
    return [e[1] for e in log if isinstance(e, tuple) and e[0] == "read"]


def _sleeps(log):
    return [e for e in log if isinstance(e, tuple) and e[0] == "sleep"]


def test_the_drain_kicks_before_it_polls_and_clears_again_after(events):
    # Bug caught: the poll started before the kick (the freed pages leave the footprint only after the next GPU
    # submission, so the poll sees no fall and ends at once), or no last synchronize and clear (the kick's own
    # buffers stay in the pool).
    _drain.drain(_reader([500 * MiB, 400 * MiB, 400 * MiB], events), budget_s=0.25)
    assert events == [
        "synchronize",
        "clear_cache",
        "kick",
        ("read", 500 * MiB),
        ("sleep", 0.02),
        ("read", 400 * MiB),
        ("sleep", 0.02),
        ("read", 400 * MiB),
        "synchronize",
        "clear_cache",
    ]


def test_the_poll_goes_on_while_the_footprint_falls_and_stops_when_it_does_not(events):
    # Bug caught: a poll that stops after the first fall (the drop is still under way), or one that ignores a reading
    # equal to the last and runs to the budget.
    _drain.drain(
        _reader([900 * MiB, 700 * MiB, 600 * MiB, 600 * MiB, 100 * MiB], events), budget_s=0.25
    )
    assert _reads(events) == [900 * MiB, 700 * MiB, 600 * MiB, 600 * MiB]


def test_a_fall_that_starts_late_is_still_waited_for(events):
    # Bug caught: a poll that ends at the first flat reading before any fall. On an idle GPU the freed pages leave the
    # footprint 80-700 ms after the kick, so that poll ends after one 20 ms sleep and drains nothing.
    values = [900 * MiB, 900 * MiB, 900 * MiB, 700 * MiB, 600 * MiB, 600 * MiB, 100 * MiB]
    _drain.drain(_reader(values, events), budget_s=0.25)
    assert _reads(events) == values[:-1]


def test_a_footprint_that_never_falls_is_polled_for_the_whole_budget(events):
    # Bug caught: a flat footprint ending the poll before the budget, so a fall that comes after 20 ms is never seen.
    _drain.drain(_reader([500 * MiB], events), budget_s=0.25)
    assert _sleeps(events) == [("sleep", 0.02)] * 12
    assert events[-2:] == ["synchronize", "clear_cache"]


def test_the_poll_stops_at_the_budget_while_the_footprint_still_falls(events):
    # Bug caught: a poll with no end while memory keeps falling, or one step past the budget. 0.25 s of 0.02 s steps
    # is 12 sleeps (a 13th would end at 0.26 s).
    _drain.drain(_reader([n * MiB for n in range(1000, 0, -1)], events), budget_s=0.25)
    assert _sleeps(events) == [("sleep", 0.02)] * 12
    assert events[-2:] == ["synchronize", "clear_cache"]


def test_a_dip_of_a_few_pages_does_not_end_the_poll(events):
    # Bug caught: the poll counting any lower reading as the fall, so footprint noise (a 352 KiB dip was seen on an
    # idle Mac) followed by a flat reading ends the poll at 40 ms, before the real release arrives.
    _drain.drain(_reader([9 * GiB, 9 * GiB - 360_448], events), budget_s=0.25)
    assert _sleeps(events) == [("sleep", 0.02)] * 12
    assert events[-2:] == ["synchronize", "clear_cache"]


def test_a_two_mib_fall_then_a_flat_reading_ends_the_poll(events):
    # Bug caught: a fall threshold above 2 MiB, or a latch that also counts the flat reading after the fall, so a real
    # release runs the poll to the budget.
    _drain.drain(_reader([9 * GiB, 9 * GiB - 2_097_152], events), budget_s=0.25)
    assert _reads(events) == [9 * GiB, 9 * GiB - 2_097_152, 9 * GiB - 2_097_152]
    assert _sleeps(events) == [("sleep", 0.02)] * 2


def test_a_one_mib_fall_in_smaller_steps_counts_as_the_fall(events):
    # Bug caught: the fall measured from the reading before instead of the first one, so a release that leaves the
    # footprint in steps under 1 MiB is never counted and the poll runs to the budget; or a fall of exactly 1 MiB not
    # counted.
    values = [9 * GiB, 9 * GiB - 524_288, 9 * GiB - 1_048_576]
    _drain.drain(_reader(values, events), budget_s=0.25)
    assert _reads(events) == [*values, 9 * GiB - 1_048_576]
    assert _sleeps(events) == [("sleep", 0.02)] * 3


def test_a_sleep_passed_in_replaces_the_module_one(events):
    # Bug caught: the injected sleep ignored, so a caller that needs no wall time still sleeps.
    slept = []
    _drain.drain(_reader([5 * MiB, 4 * MiB, 4 * MiB], events), budget_s=0.25, sleep=slept.append)
    assert slept == [0.02, 0.02]
    assert not any(isinstance(e, tuple) and e[0] == "sleep" for e in events)
