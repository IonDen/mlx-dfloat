"""Phase arithmetic shared by the mflux families: the cache limit, the activation terms and the fit phases.

A family supplies its measured numbers as a ``PhaseConstants`` record; every rule here is family-free. Estimates are
predictions: the constants are measured at one working point (1024²) and every other size is scaled from it.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.integrate.memory import FitEstimate, decoded_bytes, fit_estimate, safetensors_bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseConstants:
    """A family's measured memory constants, in bytes, and the token counts they were measured at.

    ``overhead_bytes`` is the footprint MLX's counters do not see; ``vae_transient_bytes`` what the VAE decode adds
    at 1024²; ``denoise_activation_at_reference`` the activation volume a denoise step holds beyond the cache limit
    at ``reference_tokens`` (image + text tokens). The allowance fields set the cache room for freed activation
    buffers: ``allowance_at_reference`` at ``allowance_reference_tokens``, never below ``allowance_floor``.
    ``max_measured_pixels`` is the largest image the constants were measured at.
    """

    overhead_bytes: int
    vae_transient_bytes: int
    denoise_activation_at_reference: int
    reference_tokens: int
    allowance_at_reference: int = 1_500_000_000
    allowance_reference_tokens: int = 4096 + 256
    allowance_floor: int = 500_000_000
    max_measured_pixels: int = 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilySizes:
    """Resident bytes of the components: file sizes (exact), the extras' tensor sizes, the decoded non-block groups.

    ``encoders`` is the text encoders' file size for FLUX.1 and Z-Image; FLUX.2 Klein replaces it with the bytes of
    the encoder layers its prompt embedding reads (``flux2.memory.encoder_bytes_used``), which is less than the file.
    """

    compressed: int
    extras: int
    nonblock: int
    encoders: int
    vae: int


def sizes_for(
    ckpt: DF11Checkpoint, base_root: Path, *, nonblock_groups: Iterable[str]
) -> FamilySizes:
    """Sizes of a DF11 checkpoint (compressed set apart from its extras) and of a base's text encoder and VAE.

    ``nonblock_groups`` names the groups decoded once and kept resident; those the checkpoint lacks count zero.
    """
    total = sum(p.stat().st_size for p in ckpt.root.glob("*.safetensors") if p.is_file())
    extras = sum(info.nbytes for _path, info in ckpt.extras.values())
    return FamilySizes(
        compressed=total - extras,
        extras=extras,
        nonblock=sum(decoded_bytes(ckpt.groups[g]) for g in nonblock_groups if g in ckpt.groups),
        encoders=safetensors_bytes(base_root, "text_encoder"),
        vae=safetensors_bytes(base_root, "vae"),
    )


def activation_allowance(c: PhaseConstants, *, height: int, width: int, text_tokens: int) -> int:
    """Cache room for the activation buffers a block frees, scaled linearly with the token count."""
    tokens = height * width // 256 + text_tokens
    return max(
        c.allowance_floor,
        int(c.allowance_at_reference * tokens / c.allowance_reference_tokens),
    )


def cache_limit_for(
    c: PhaseConstants,
    largest: Mapping[str, int],
    *,
    policy: str,
    height: int,
    width: int,
    text_tokens: int,
    override: int | None = None,
) -> int:
    """The MLX buffer-cache limit for one generate call: the two largest decoded groups next to the activations.

    ``depth2`` adds one more of the largest decoded group (the look-ahead buffer); ``override`` wins outright.
    """
    if override is not None:
        return override
    two = sum(sorted(largest.values(), reverse=True)[:2])
    limit = two + activation_allowance(c, height=height, width=width, text_tokens=text_tokens)
    if policy == "depth2":
        limit += max(largest.values())
    return limit


def vae_transient_bytes(c: PhaseConstants, *, height: int, width: int) -> int:
    """What the VAE decode adds on top of what is resident: the measured 1024² value as a floor.

    Above 1024² it grows with the pixel count: a prediction, not a measurement.
    """
    return int(c.vae_transient_bytes * max(1.0, (height * width) / (1024 * 1024)))


def denoise_activation_bytes(
    c: PhaseConstants, *, height: int, width: int, text_tokens: int
) -> int:
    """The activation volume a denoise step holds beyond the cache limit, linear in the token count.

    Calibrated at ``c.reference_tokens``; every other size is a prediction.
    """
    tokens = height * width // 256 + text_tokens
    return int(c.denoise_activation_at_reference * tokens / c.reference_tokens)


def family_phases(
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
    """The three phases of a generation (``encode``, ``denoise``, ``vae``), each term once."""
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
    c: PhaseConstants,
    *,
    sizes: FamilySizes,
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

    ``vae_with_set=False`` plans the VAE phase after the compressed set has been dropped (compressed, extras and
    the decoded non-block weights all leave with it).
    """
    phases = family_phases(
        compressed_bytes=sizes.compressed,
        extras_bytes=sizes.extras,
        nonblock_bytes=sizes.nonblock,
        largest=largest,
        policy=policy,
        cache_limit=cache_limit,
        allowance=allowance,
        encoders_bytes=sizes.encoders,
        vae_bytes=sizes.vae,
        vae_transient_bytes=vae_transient_bytes(c, height=height, width=width),
        overhead_bytes=c.overhead_bytes,
        denoise_activation_bytes=denoise_activation_bytes(
            c, height=height, width=width, text_tokens=text_tokens
        ),
    )
    if not vae_with_set:
        for term in ("compressed", "extras", "nonblock"):
            phases["vae"].pop(term)
    return fit_estimate(phases, budget_bytes=budget)
