import pytest
from scripts._bench_common import per_dispatch_guard


@pytest.mark.parametrize("bytes_out", [24_000_000, 25_000_000])
def test_a_dispatch_projected_at_or_under_the_limit_passes(bytes_out):
    # Bug caught: `>=` in place of `>`, or a guard that trips on every call.
    per_dispatch_guard(bytes_out, 100e6)  # 0.24 s and exactly 0.25 s against a 0.25 s limit


def test_a_dispatch_projected_over_the_limit_raises_naming_seconds_and_limit():
    # Bug caught: a guard that never trips, letting one dispatch run past the GPU watchdog window.
    with pytest.raises(RuntimeError, match=r"0\.26.*0\.25"):
        per_dispatch_guard(26_000_000, 100e6)
