"""FLUX.1 memory rules: the cache limit and the fit phases. Measured at schnell 1024² only; elsewhere predicted."""

from collections.abc import Mapping
from typing import Any

from mlx_dfloat.mflux.flux1.names import DOUBLE_PREFIX, SINGLE_PREFIX

ALLOWANCE_AT_REFERENCE = (
    1_500_000_000  # recycled activation volume measured at 1024², 256 text tokens
)
REFERENCE_TOKENS = 4096 + 256
ALLOWANCE_FLOOR = 500_000_000
MEASURED_LIMIT_AT_1024 = 2_500_000_000


def text_tokens(model_config: Any) -> int:
    """The model's text sequence length (256 for schnell, 512 for dev and Krea-dev)."""
    return int(model_config.max_sequence_length)


def activation_allowance(*, height: int, width: int, text_tokens: int) -> int:
    """Cache room for the activation buffers a block frees, scaled linearly with the token count."""
    tokens = height * width // 256 + text_tokens
    return max(ALLOWANCE_FLOOR, int(ALLOWANCE_AT_REFERENCE * tokens / REFERENCE_TOKENS))


def cache_limit_for(
    largest: Mapping[str, int],
    *,
    policy: str,
    height: int,
    width: int,
    text_tokens: int,
    override: int | None = None,
) -> int:
    """The MLX buffer-cache limit for one generate call: room for a decoded buffer of each kind next to the activations."""
    if override is not None:
        return override
    limit = (
        largest[DOUBLE_PREFIX]
        + largest[SINGLE_PREFIX]
        + activation_allowance(height=height, width=width, text_tokens=text_tokens)
    )
    if policy == "depth2":
        limit += max(largest.values())
    if height * width >= 1024 * 1024:
        limit = max(limit, MEASURED_LIMIT_AT_1024)
    return limit


def flux_phases(
    *,
    compressed_bytes: int,
    extras_bytes: int,
    largest: Mapping[str, int],
    policy: str,
    cache_limit: int,
    allowance: int,
    encoders_bytes: int,
    vae_bytes: int,
    vae_transient_bytes: int,
    overhead_bytes: int,
) -> dict[str, dict[str, int]]:
    """The three phases of a generation, each term once."""
    in_flight = max(largest.values()) * (2 if policy == "depth2" else 1)
    return {
        "encode": {
            "encoders": encoders_bytes,
            "activations": allowance,
            "overhead": overhead_bytes,
        },
        "denoise": {
            "compressed": compressed_bytes,
            "extras": extras_bytes,
            "decoded": in_flight,
            "cache": cache_limit,
            "overhead": overhead_bytes,
        },
        "vae": {
            "compressed": compressed_bytes,
            "extras": extras_bytes,
            "vae": vae_bytes,
            "transient": vae_transient_bytes,
            "overhead": overhead_bytes,
        },
    }
