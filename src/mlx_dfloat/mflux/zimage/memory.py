"""Z-Image memory rules: the cache limit and the fit phases. Measured at 1024² (Turbo and base) only; elsewhere predicted."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.memory import FitEstimate, decoded_bytes, fit_estimate, safetensors_bytes
from mlx_dfloat.mflux.zimage.names import NONBLOCK_GROUPS

# FLUX.1's measured rate (recycled activation volume 1.5e9 bytes at 1024² with 256 text tokens), not re-measured
# for Z-Image: the cache limit it feeds was validated only by the one-step calibration runs below.
ALLOWANCE_AT_REFERENCE = 1_500_000_000
ALLOWANCE_REFERENCE_TOKENS = 4096 + 256
REFERENCE_TOKENS = 4096 + 512  # the calibration runs: 1024² (4096 image tokens) + 512 text tokens
ALLOWANCE_FLOOR = 500_000_000
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path

# Measured on 2026-10-08 (Z-Image-Turbo guidance 0 and Z-Image base guidance 4, 1024², seed 42; one process per
# unit: build, encode, drop, set load, one step, VAE decode on the resident set; M1 Max 32 GB, macOS 27.0.1,
# mlx 0.32.2, mflux 0.20.0, git 3da18bb; one record per unit). Each constant is the larger of the two units, except
# VAE_TRANSIENT (see below). Re-measure before changing any.
# OVERHEAD: the footprint MLX's counters do not see after the build phase (turbo 510_030_936, base 470_479_960).
OVERHEAD_BYTES = 510_030_936  # 0.475 GiB
# VAE_TRANSIENT: calibrated on the two measured 1024² generate runs (2026-10-08, same versions, Z-Image base and
# Turbo; footprint peaks 14_115_020_240 and 14_074_273_664, both in the VAE phase) so the VAE phase reproduces the
# larger one: base peak minus the phase's other terms (8_369_046_600 + 6_088_832 + 19_660_800 + 167_666_902 +
# OVERHEAD_BYTES) = 5_042_526_170 (Turbo needs 4_996_540_342). It absorbs the float32 decoder's own activations
# and the footprint the OS has not yet taken back from the denoise phase when the decode starts: MLX's cache is
# cleared by then, yet at the VAE start the footprint sits about 2.2 GiB above MLX's active memory plus OVERHEAD
# (turbo 11_254_325_096 - 8_392_187_384 - 510_030_936; base 11_312_586_144 - 8_389_484_140 - 510_030_936). The
# one-step value it replaces (VAE-phase peak minus the footprint after denoise, 3_982_295_040) left that part out.
VAE_TRANSIENT_BYTES = 5_042_526_170  # 4.696 GiB
# DENOISE_ACTIVATION: denoise footprint peak minus (compressed + extras + non-block + in-flight + cache limit +
# overhead) of the same unit: turbo 12_781_477_784 - 11_583_577_874 = 1_197_899_910; base 13_061_987_888 -
# 11_538_787_646 = 1_523_200_242, at 4096 image + 512 text tokens.
DENOISE_ACTIVATION_AT_REFERENCE = 1_523_200_242  # 1.419 GiB


def text_tokens(model_config: Any) -> int:
    """The model's text sequence length (512 for Z-Image and Z-Image-Turbo)."""
    return int(model_config.max_sequence_length)


def activation_allowance(*, height: int, width: int, text_tokens: int) -> int:
    """Cache room for the activation buffers a block frees, scaled linearly with the token count."""
    tokens = height * width // 256 + text_tokens
    return max(ALLOWANCE_FLOOR, int(ALLOWANCE_AT_REFERENCE * tokens / ALLOWANCE_REFERENCE_TOKENS))


def cache_limit_for(
    largest: Mapping[str, int],
    *,
    policy: str,
    height: int,
    width: int,
    text_tokens: int,
    override: int | None = None,
) -> int:
    """The MLX buffer-cache limit for one generate call: the two largest decoded groups next to the activations.

    No 1024² floor: Z-Image's own working point (2.31 GB at 1024², measured in one-step runs) is what this
    formula gives.
    """
    if override is not None:
        return override
    two = sum(sorted(largest.values(), reverse=True)[:2])
    limit = two + activation_allowance(height=height, width=width, text_tokens=text_tokens)
    if policy == "depth2":
        limit += max(largest.values())
    return limit


def vae_transient_bytes(*, height: int, width: int) -> int:
    """What the float32 VAE decode adds on top of what is resident: the measured 1024² value as a floor.

    Above 1024² it grows with the pixel count: a prediction, not a measurement.
    """
    return int(VAE_TRANSIENT_BYTES * max(1.0, (height * width) / (1024 * 1024)))


def denoise_activation_bytes(*, height: int, width: int, text_tokens: int) -> int:
    """The activation volume a denoise step holds beyond the cache limit, linear in the token count.

    Calibrated on the measured 1024² peak (4096 image + 512 text tokens); every other size is a prediction.
    """
    tokens = height * width // 256 + text_tokens
    return int(DENOISE_ACTIVATION_AT_REFERENCE * tokens / REFERENCE_TOKENS)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZImageSizes:
    """Resident bytes of the components: file sizes (exact), the extras' tensor sizes, the decoded non-block groups."""

    compressed: int
    extras: int
    nonblock: int
    encoders: int
    vae: int


def sizes_for(ckpt: DF11Checkpoint, base_root: Path) -> ZImageSizes:
    """Sizes of a DF11 checkpoint (compressed set apart from its extras) and of a base's encoder and VAE."""
    total = sum(p.stat().st_size for p in ckpt.root.glob("*.safetensors") if p.is_file())
    extras = sum(info.nbytes for _path, info in ckpt.extras.values())
    return ZImageSizes(
        compressed=total - extras,
        extras=extras,
        nonblock=sum(decoded_bytes(ckpt.groups[g]) for g in NONBLOCK_GROUPS if g in ckpt.groups),
        encoders=safetensors_bytes(base_root, "text_encoder"),
        vae=safetensors_bytes(base_root, "vae"),
    )


def zimage_phases(
    *,
    compressed_bytes: int,
    extras_bytes: int,
    nonblock_bytes: int,
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
            "nonblock": nonblock_bytes,
            "decoded": in_flight,
            "cache": cache_limit,
            "activations": denoise_activation_bytes,
            "overhead": overhead_bytes,
        },
        "vae": {
            "compressed": compressed_bytes,
            "extras": extras_bytes,
            "nonblock": nonblock_bytes,
            "vae": vae_bytes,
            "transient": vae_transient_bytes,
            "overhead": overhead_bytes,
        },
    }


def fit_for(
    *,
    sizes: ZImageSizes,
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

    ``vae_with_set=False`` plans the VAE phase after the compressed set has been dropped: the compressed
    groups and the decoded non-block weight leave with it; the extras stay (the transformer's own BF16
    parameters, loaded once at construction).
    """
    phases = zimage_phases(
        compressed_bytes=sizes.compressed,
        extras_bytes=sizes.extras,
        nonblock_bytes=sizes.nonblock,
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
        for term in ("compressed", "nonblock"):
            phases["vae"].pop(term)
    return fit_estimate(phases, budget_bytes=budget)
