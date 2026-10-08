"""FLUX.2 Klein memory rules: the measured constants per size, the text-token count and the encoder's resident bytes.

The phase arithmetic itself is the shared one (``mlx_dfloat.mflux._phases``). Klein differs in one input: its text
encoder runs only the layers its prompt embedding reads (see ``encoder_bytes_used``), so the encode phase counts
those layers, not the encoder's file size. Both sizes' constants are measured at 1024²; every other size is a
prediction.
"""

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlx_dfloat._safetensors import read_header
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants
from mlx_dfloat.mflux._phases import sizes_for as family_sizes_for
from mlx_dfloat.mflux.flux2.names import NONBLOCK_GROUPS

REFERENCE_TOKENS = 4096 + 512  # the calibration run: 1024² (4096 image tokens) + 512 text tokens
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path
# The hidden states Klein's prompt embedding stacks (mflux 0.20.0 ``Flux2Klein._encode_prompt_pair``).
TEXT_ENCODER_OUT_LAYERS: tuple[int, ...] = (9, 18, 27)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# Measured on 2026-10-08 (FLUX.2-klein-base-4B and FLUX.2-klein-base-9B, guidance 4 (two transformer calls per
# step), 1024², seed 42; one process per size: build, encode, drop, set load, one step, VAE decode on the resident
# set; M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0; one record per size). Each size keeps its own
# constants: the two VAE transients differ by 1.3 GiB (the 9B decode starts from a larger footprint, so less of the
# OS's lag shows as transient), so one shared value would be wrong for one of them. Re-measure before changing any.
#
# OVERHEAD: the footprint MLX's counters do not see after the build phase (footprint - MLX active - MLX cache):
# 4B 571_294_944 - 190_128_206 - 1_986; 9B 584_418_528 - 205_861_966 - 2_770.
OVERHEAD_4B, OVERHEAD_9B = 381_164_752, 378_553_792  # 0.355 / 0.353 GiB
# VAE_TRANSIENT: calibrated so the VAE phase reproduces the measured call peak (the VAE decode on the resident set):
# the peak minus the phase's other terms (compressed + extras + decoded non-block + VAE file + overhead).
# 4B: 13_621_449_840 - (5_239_059_784 + 22_035_456 + 368_050_176 + 168_120_878 + 381_164_752) = 7_443_018_794.
# 9B: 19_570_665_528 - (12_275_304_729 + 37_769_216 + 671_088_640 + 168_120_878 + 378_553_792) = 6_039_828_273.
# As for Z-Image, it absorbs the float32 decoder's activations and the footprint the OS has not yet taken back from
# the denoise phase when the decode starts. The one-step value (VAE-phase peak minus the footprint after denoise:
# 4B 5_024_366_592, 9B 3_803_234_304) leaves that part out and puts the 4B VAE phase 2.1 GiB under its measured
# peak. FLUX.1, whose VAE has the same architecture, measured 7.82 GiB.
VAE_TRANSIENT_4B, VAE_TRANSIENT_9B = 7_443_018_794, 6_039_828_273  # 6.932 / 5.625 GiB
# DENOISE_ACTIVATION: denoise footprint peak minus (compressed + extras + non-block + in-flight + cache limit +
# overhead), at 4096 image + 512 text tokens.
# 4B: 9_168_770_160 - (5_239_059_784 + 22_035_456 + 368_050_176 + 490_733_568 + 2_324_335_646 + 381_164_752).
# 9B: 17_534_117_944 - (12_275_304_729 + 37_769_216 + 671_088_640 + 872_415_232 + 2_896_858_142 + 378_553_792).
DENOISE_ACTIVATION_4B, DENOISE_ACTIVATION_9B = 343_390_778, 402_128_193  # 0.320 / 0.375 GiB

CONSTANTS: dict[str, PhaseConstants] = {
    "4b": PhaseConstants(
        overhead_bytes=OVERHEAD_4B,
        vae_transient_bytes=VAE_TRANSIENT_4B,
        denoise_activation_at_reference=DENOISE_ACTIVATION_4B,
        reference_tokens=REFERENCE_TOKENS,
        max_measured_pixels=MAX_MEASURED_PIXELS,
    ),
    "9b": PhaseConstants(
        overhead_bytes=OVERHEAD_9B,
        vae_transient_bytes=VAE_TRANSIENT_9B,
        denoise_activation_at_reference=DENOISE_ACTIVATION_9B,
        reference_tokens=REFERENCE_TOKENS,
        max_measured_pixels=MAX_MEASURED_PIXELS,
    ),
}


def text_tokens(model_config: Any) -> int:
    """The model's text sequence length (512 for every FLUX.2 Klein)."""
    return int(model_config.max_sequence_length)


def encoder_bytes_used(
    base_root: Path, *, out_layers: tuple[int, ...] = TEXT_ENCODER_OUT_LAYERS
) -> int:
    """The text encoder's bytes a Klein prompt encode makes resident, read from the safetensors headers.

    The encoder returns its embedding output as hidden state 0 and layer ``i``'s output as hidden state ``i + 1``, and
    Klein stacks hidden states ``out_layers`` only. Under lazy evaluation the layers after the deepest one asked for
    never run, so their weights are never loaded: for 36 layers and ``(9, 18, 27)`` that is layers 0-26. Every tensor
    outside the numbered layers (embeddings, final norm) counts in full, except ``lm_head.*``: mflux's encoder
    has no output head and its weight mapping names only ``model.*`` tensors, so an untied head (Qwen3-8B, the 9B
    encoder, ships one of 1.24 GB) is never loaded.

    Raises:
        DFloatFormatError: An encoder file's header cannot be read.
    """
    needed = max(out_layers)
    total = 0
    for path in sorted((base_root / "text_encoder").glob("*.safetensors")):
        for name, info in read_header(path).items():
            if name.startswith("lm_head."):
                continue
            layer = _LAYER.search(name)
            if layer is None or int(layer.group(1)) < needed:
                total += info.nbytes
    return total


def sizes_for(ckpt: DF11Checkpoint, base_root: Path) -> FamilySizes:
    """The family sizes of a Klein checkpoint and base, with the encoder counted by the layers Klein runs."""
    sizes = family_sizes_for(ckpt, base_root, nonblock_groups=NONBLOCK_GROUPS)
    return replace(sizes, encoders=encoder_bytes_used(base_root))
