"""FLUX.1 memory rules: the cache limit and the fit phases. Measured at schnell 1024² only; elsewhere predicted."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.memory import FitEstimate, fit_estimate
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
    denoise_activation_bytes: int = 0,
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
            "activations": denoise_activation_bytes,
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


# Measured on 2026-09-28 (schnell 1024², one process: encode, drop, set, one step, VAE decode; M1 Max 32 GB,
# macOS 27.0, mlx 0.32.2, mflux 0.20.0, git 4601a8b). OVERHEAD is the footprint that MLX's counters do not see
# after construction (Python, torch and transformers imports, the runtime); VAE_TRANSIENT is what the float32
# decode adds at 1024² on top of everything resident after the set load (the VAE phase peaked at 23.29 GiB with
# the set resident, over the 23.0 GiB fit rule, which is why a call may drop the set before decoding).
# Re-measure before changing either.
OVERHEAD_BYTES = 247_712_510  # 0.23 GiB: build-phase footprint minus MLX active and cache
VAE_TRANSIENT_BYTES = (
    8_392_982_528  # 7.82 GiB: VAE-phase footprint peak minus the footprint after the set load
)
# Measured on 2026-09-28 (the `mlx-dfloat generate` runs, schnell 1024², 4 steps, per-block, 2.5 GB cache limit;
# M1 Max 32 GB, mlx 0.32.2, mflux 0.20.0, git 2fe3b2c): the process footprint peaked at 19.94 GiB against a
# denoise estimate of 18.38 GiB without this term. The 1.56 GiB gap is the activation volume the step holds
# beyond the cache limit, at 4096 image + 256 text tokens; dev (512 text tokens) measured 20.10 GiB.
DENOISE_ACTIVATION_AT_REFERENCE = 1_675_000_000
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path


def vae_transient_bytes(*, height: int, width: int) -> int:
    """What the float32 VAE decode adds on top of what is resident: the measured 1024² value as a floor.

    At or below 1024² this is the measured value (smaller images were not measured lower, so the
    floor stays). Above 1024² it grows with the pixel count: a prediction, not a measurement.
    """
    return int(VAE_TRANSIENT_BYTES * max(1.0, (height * width) / (1024 * 1024)))


def denoise_activation_bytes(*, height: int, width: int, text_tokens: int) -> int:
    """The activation volume a denoise step holds beyond the cache limit, linear in the token count.

    Calibrated on the measured schnell 1024² peak (4096 image + 256 text tokens); every other size
    is a prediction.
    """
    tokens = height * width // 256 + text_tokens
    return int(DENOISE_ACTIVATION_AT_REFERENCE * tokens / REFERENCE_TOKENS)


@dataclass(frozen=True, slots=True, kw_only=True)
class FluxSizes:
    """Resident bytes of the components, from file sizes (exact) and the extras' tensor sizes."""

    compressed: int
    extras: int
    encoders: int
    vae: int


def safetensors_bytes(root: Path, *subdirs: str) -> int:
    """The size of every ``*.safetensors`` file directly under each ``root/subdir`` (missing subdirs count zero)."""
    return sum(
        p.stat().st_size
        for sub in subdirs
        for p in (root / sub).glob("*.safetensors")
        if p.is_file()
    )


def sizes_for(ckpt: DF11Checkpoint, base_root: Path) -> FluxSizes:
    """Sizes of a DF11 checkpoint (compressed set apart from its extras) and of a base's encoders and VAE."""
    total = sum(p.stat().st_size for p in ckpt.root.glob("*.safetensors") if p.is_file())
    extras = sum(info.nbytes for _path, info in ckpt.extras.values())
    return FluxSizes(
        compressed=total - extras,
        extras=extras,
        encoders=safetensors_bytes(base_root, "text_encoder", "text_encoder_2"),
        vae=safetensors_bytes(base_root, "vae"),
    )


def fit_for(
    *,
    sizes: FluxSizes,
    largest: Mapping[str, int],
    policy: str,
    cache_limit: int,
    allowance: int,
    budget: int,
    height: int,
    width: int,
    text_tokens: int,
    vae_with_set: bool = True,
) -> FitEstimate:
    """The phase estimate for one generate call against ``budget`` (a prediction, labelled as such by the caller).

    ``height`` and ``width`` scale the VAE transient (above 1024²) and, with ``text_tokens``, the
    denoise activation term. ``vae_with_set=False`` plans the VAE phase after the compressed set has
    been dropped (what a call does when the resident variant would not fit).
    """
    phases = flux_phases(
        compressed_bytes=sizes.compressed,
        extras_bytes=sizes.extras,
        largest=largest,
        policy=policy,
        cache_limit=cache_limit,
        allowance=allowance,
        encoders_bytes=sizes.encoders,
        vae_bytes=sizes.vae,
        vae_transient_bytes=vae_transient_bytes(height=height, width=width),
        overhead_bytes=OVERHEAD_BYTES,
        denoise_activation_bytes=denoise_activation_bytes(
            height=height, width=width, text_tokens=text_tokens
        ),
    )
    if not vae_with_set:
        phases["vae"].pop("compressed")
        phases["vae"].pop("extras")
    return fit_estimate(phases, budget_bytes=budget)
