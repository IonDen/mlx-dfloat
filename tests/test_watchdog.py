from scripts._watchdog import verdict


def test_memory_verdict_uses_process_rss():
    # Bug caught: a watchdog that only counts MLX memory would miss a NumPy blow-up.
    assert verdict(rss=11, ceiling=10, elapsed=1.0, budget=100.0) == "memory"


def test_wall_verdict():
    assert verdict(rss=1, ceiling=10, elapsed=101.0, budget=100.0) == "wall"


def test_no_verdict_under_both_limits():
    assert verdict(rss=10, ceiling=10, elapsed=100.0, budget=100.0) is None
