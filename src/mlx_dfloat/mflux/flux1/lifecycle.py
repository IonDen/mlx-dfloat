"""The encoder / compressed-set lifecycle: the text encoders and the compressed transformer are never resident together.

On a 32 GB Mac the T5 encoder (8.9 GiB) and the compressed FLUX.1 set (15.2 GiB) do not fit under the
fit rule, so a prompt is encoded first, its embeddings evaluated, the encoders dropped, and only then
the set loaded. A new prompt after the set is resident drops the set, reloads the encoders, encodes,
drops them and reloads the set. The object here holds no arrays and no modules: the callbacks own
them, so a drop releases them.
"""

import gc
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import mlx.core as mx

from mlx_dfloat.errors import DFloatResourceError

log = logging.getLogger("mlx_dfloat.mflux.flux1")
_eval = mx.eval  # looked up on the module, so a test can record the evaluation


@dataclass(slots=True)
class LifecycleCounters:
    """Reloads and their seconds, for the report."""

    encoder_loads: int = 0
    encoder_seconds: float = 0.0
    set_loads: int = 0
    set_seconds: float = 0.0
    forced_set_drops: int = 0

    def as_dict(self) -> dict[str, float | int]:
        """The counters as a JSON-ready dict."""
        return asdict(self)


class Lifecycle:
    """Drives the encoder and set callbacks so that the two are never resident at once."""

    def __init__(
        self,
        *,
        load_encoders: Callable[[], None],
        unload_encoders: Callable[[], None],
        encode: Callable[[str], tuple[mx.array, mx.array]],
        load_set: Callable[[], None],
        unload_set: Callable[[], None],
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        retained_bound_bytes: int,
        encoders_loaded: bool = False,
    ) -> None:
        """Bind the callbacks, the prompt cache mflux reads, and the active-memory bound a drop must reach."""
        self._load_encoders = load_encoders
        self._unload_encoders = unload_encoders
        self._encode = encode
        self._load_set = load_set
        self._unload_set = unload_set
        self._cache = prompt_cache
        self.retained_bound_bytes = retained_bound_bytes
        self.encoders_loaded = encoders_loaded
        self.set_resident = False
        self.counters = LifecycleCounters()

    def ensure_embeddings(self, *prompts: str) -> None:
        """Encode the prompts not yet cached: the set is dropped first, the encoders after.

        Each pair is evaluated before it is cached: a lazy pair would keep every encoder weight
        alive past the drop. The encode loop runs under a ``finally`` so a mid-loop failure still
        drops the encoders instead of leaving them resident next to a set loaded later.
        """
        missing = [p for p in prompts if p not in self._cache]
        if not missing:
            return
        if self.set_resident:
            log.info(
                "new prompt with the compressed set resident: dropping the set to load the text encoders"
            )
            self.counters.forced_set_drops += 1
            self.drop_set()
        if not self.encoders_loaded:
            start = time.perf_counter()
            self._load_encoders()
            self.encoders_loaded = True
            self.counters.encoder_loads += 1
            self.counters.encoder_seconds += time.perf_counter() - start
        try:
            for prompt in missing:
                pair = self._encode(prompt)
                _eval(*pair)
                self._cache[prompt] = pair
        finally:
            self.drop_encoders()

    def ensure_set(self) -> None:
        """Load and attach the compressed set unless it is resident."""
        if self.set_resident:
            return
        start = time.perf_counter()
        self._load_set()
        self.set_resident = True
        self.counters.set_loads += 1
        self.counters.set_seconds += time.perf_counter() - start

    def drop_set(self) -> None:
        """Detach and release the compressed set; assert the memory came back."""
        self._unload_set()
        self.set_resident = False
        self._reclaim("compressed set")

    def drop_encoders(self) -> None:
        """Release the text encoders; assert the memory came back."""
        self._unload_encoders()
        self.encoders_loaded = False
        self._reclaim("text encoders")

    def _reclaim(self, what: str) -> None:
        gc.collect()
        mx.clear_cache()
        active = int(mx.get_active_memory())
        if active > self.retained_bound_bytes:
            raise DFloatResourceError(
                f"after dropping the {what}, {active / 1024**3:.2f} GiB of MLX memory is still active "
                f"(bound {self.retained_bound_bytes / 1024**3:.2f} GiB): something else holds it"
            )
