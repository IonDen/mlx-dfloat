"""An in-memory stand-in for the watchdog's counter reader, so tests decide what it sees."""

import threading

import mlx_dfloat._watchdog as wd


class FakeCounters:
    """Settable counters; calling the object returns them as one ``Counters`` reading.

    A lock keeps a reading whole: the sampler thread never sees ``footprint`` from one
    ``raise_to`` and the MLX numbers from another.
    """

    def __init__(self, footprint=0, rss=0, mlx_active=0, mlx_cache=0):
        self._lock = threading.Lock()
        self.footprint = footprint
        self.rss = rss
        self.mlx_active = mlx_active
        self.mlx_cache = mlx_cache

    def __call__(self):
        with self._lock:
            return wd.Counters(
                footprint=self.footprint,
                rss=self.rss,
                mlx_active=self.mlx_active,
                mlx_cache=self.mlx_cache,
            )

    def raise_to(self, watched):
        """Set the watched number to ``watched``, carried by the footprint counter.

        MLX active + cache together hold half of it, so the footprint is the counter reported.
        """
        with self._lock:
            self.footprint = watched
            half = watched // 2
            self.mlx_active = half // 2
            self.mlx_cache = half - half // 2
