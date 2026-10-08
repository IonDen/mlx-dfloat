"""Qwen-Image 2.1 memory rules: the constants, the text-token count and the text encoder's resident bytes.

The phase arithmetic itself is the shared one (``mlx_dfloat.mflux._phases``). Qwen-Image 2.1 differs in two inputs:
its text encoder's shards also hold a vision tower and an output head that mflux never loads (see
``encoder_bytes_used``), and its prompt length varies per call (up to the tokenizer's 2048 tokens), so the text
tokens are counted per prompt (``prompt_tokens``) rather than read from the model config.

The transformer has one block kind, so the cache limit holds one decoded block next to the activations (per-block
evaluation frees one block-sized buffer at a time), and the family carries its own activation allowance. The VAE
transient is mflux's VAE decode, which runs in float32 (mflux 0.20.0 ``qwen21_vae.py:46-48``: the float32 latent
mean and standard deviation promote the latents): re-measure it when an mflux upgrade changes that dtype. No Qwen-Image 2.1 per-block overhead scenario is benched.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from mlx_dfloat._safetensors import read_header
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants
from mlx_dfloat.mflux._phases import sizes_for as family_sizes_for
from mlx_dfloat.mflux.qwen21.names import NONBLOCK_GROUPS

# The only text-encoder tensors mflux maps (its Qwen-Image 2.1 weight mapping names model.language_model.* only).
ENCODER_PREFIX = "model.language_model."
# The tokenizer's max_length (mflux's weight definition; tokenization truncates at it, system prefix included).
TEXT_MAX_LENGTH = 2048
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path

# Measured on 2026-10-08 (the first one-step calibration run, at the old cache limit): Qwen-Image-2.1 DF11
# (mingyi456/Qwen-Image-2.1-DF11-ComfyUI @ 1b22a3a) over the Qwen/Qwen-Image-2.1 base (@ d26bb61), 1024², seed 42,
# guidance 4 with the negative prompt " " (two transformer calls per step); one process: build, encode, drop, set load,
# one step, VAE decode on the resident set; M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0. Bytes throughout.
# Re-measure before changing any. Each run's phase peaks: bench/results/calibration/qwen-image-2.1-1024.json.
#
# REFERENCE_TOKENS: the calibration point, 1024² (4096 image tokens) + the run's text tokens, 33 (the prompt's; the
# negative prompt's 9 are fewer, and the call is sized by its longest prompt).
REFERENCE_TOKENS = 4096 + 33
# OVERHEAD: the footprint MLX's counters do not see after the build phase (footprint - MLX active - MLX cache):
# 606_274_808 - 146_825_736 - 3_974.
OVERHEAD = 459_445_098  # 0.428 GiB
# VAE_TRANSIENT, by the overall-peak rule over every 1024² sample: the highest measured peak of the VAE decode on the
# resident set minus the VAE phase's other terms (compressed + extras + decoded non-block + VAE file + OVERHEAD):
# 22_788_580_400 - (9_587_103_379 + 137_388_032 + 134_217_728 + 1_350_989_512 + 459_445_098). Eleven samples on
# 2026-10-08 (the calibration run above, four calibration reruns of one step at two cache limits, three 40-step MEASURED
# runs, three 4-step identity runs with CFG; the highest is the second one-step rerun at the old cache limit)
# peaked at 21.76 to 22.79 GB in the VAE phase, with MLX's own VAE peak the same in every calibration run: about 1 GiB of
# run-to-run spread outside MLX's counters, as Z-Image's VAE phase shows too. A fit check must not under-predict, so
# the term covers the highest sample; the first run alone (22_139_643_024) would put the VAE phase 0.60 GiB under it.
# About 1.32x the FLUX.1 VAE's 7.82 GiB.
VAE_TRANSIENT = 11_119_436_651  # 10.356 GiB
# DENOISE_ACTIVATION: denoise footprint peak minus the denoise phase's other terms at the planner's cache limit
# (compressed + extras + non-block + in-flight + cache limit + OVERHEAD), at REFERENCE_TOKENS:
# 13_939_106_072 - (9_587_103_379 + 137_388_032 + 134_217_728 + 436_207_616 + 2_731_761_634 + 459_445_098). Measured
# 2026-10-08 (the first one-step calibration run at the new cache limit, in the cache-limit A/B: the same model, prompt,
# size, seed and guidance as above, one step at the 2_731_761_634 limit); the larger of the two runs at that limit
# (the other's denoise peak, 13_506_126_128, gives 20_002_641).
DENOISE_ACTIVATION = 452_982_585  # 0.422 GiB
# ALLOWANCE: the cache room for the activation buffers a block frees, at ALLOWANCE_TOKENS (1024² + the calibration prompt's
# 33 tokens). The block-cache probe (one synthetic block at 1024² + 33 tokens, the decoded buffer allocated then
# released per iteration) saw the buffer reused at a 2605 MiB limit and released at 1773 and 2189 MiB: the limit is set
# at 2_731_761_634 B, so the allowance is that minus one decoded block, 2_731_761_634 - 436_207_616. Set at the probe's
# reuse point: an A/B of real calibration steps at this limit and the old 1_859_346_402 (two runs each, 2026-10-08) showed
# no measurable difference in step time or denoise footprint.
ALLOWANCE = 2_295_554_018  # 2.138 GiB
ALLOWANCE_TOKENS = 4096 + 33
# ENCODE_ACTIVATION: what the prompt encode held beyond the language model's tensors by MLX's measure: the encode's
# MLX peak minus what was active at the phase's start (15_350_204_544 - 146_825_736 = 15_203_378_808) minus the language
# model (15_136_811_008). It sizes the encode phase's activations in place of the denoise allowance. By footprint the
# encode peaked 195_114_382 B above the phase this predicts (non-MLX memory the build-time overhead does not cover).
# One 33-token prompt: the term is not scaled with the prompt length.
ENCODE_ACTIVATION = 66_567_800
CONSTANTS: PhaseConstants = PhaseConstants(
    overhead_bytes=OVERHEAD,
    vae_transient_bytes=VAE_TRANSIENT,
    denoise_activation_at_reference=DENOISE_ACTIVATION,
    reference_tokens=REFERENCE_TOKENS,
    allowance_at_reference=ALLOWANCE,
    allowance_reference_tokens=ALLOWANCE_TOKENS,
    max_measured_pixels=MAX_MEASURED_PIXELS,
    encode_activation_bytes=ENCODE_ACTIVATION,
)
# What a prompt encode may hold beyond the estimate's encode phase before the model warns (the bound is the language
# model + ENCODE_ACTIVATION + this slack). The run's encode excess by footprint (peak - language model - overhead:
# 15_857_938_288 - 15_136_811_008 - 459_445_098 = 261_682_182, 249.6 MiB) rounded up to 256 MiB. By MLX's measure the
# run held exactly ENCODE_ACTIVATION, the whole slack under the bound; an encode that also loads the vision tower and
# lm_head (~2_397_528_480 more) is over it by 2_129_093_024 at every prompt length (the bound does not grow with it).
ENCODE_SLACK_BYTES = 256 * 1024**2
# The prompt length ENCODE_ACTIVATION was measured at (the calibration prompt's text tokens).
ENCODE_ACTIVATION_TOKENS = 33


def encode_peak_bound(encode_phase_bytes: int, constants: PhaseConstants = CONSTANTS) -> int:
    """What a prompt encode may hold by MLX's measure before the model warns.

    That is the estimate's encode phase without its overhead (the language model + ``ENCODE_ACTIVATION``) plus
    ``ENCODE_SLACK_BYTES``.
    """
    return encode_phase_bytes - constants.overhead_bytes + ENCODE_SLACK_BYTES


def encode_peak_warning(held: int, bound: int, *, text_tokens: int) -> str | None:
    """The warning for an encode that held ``held`` bytes against ``bound``, or None at or under it.

    Two causes fit an excess: mflux loading more of the encoder than its language model, or a prompt longer than the
    one ``ENCODE_ACTIVATION`` was measured on (the term is not scaled with the prompt length). The message names the
    call's longest prompt in tokens so the reader can tell which.
    """
    if held <= bound:
        return None
    return (
        f"the prompt encode held {held / 1024**3:.2f} GiB for a {text_tokens}-token prompt, over the "
        f"{bound / 1024**3:.2f} GiB the fit estimate allows: mflux may now load more of the encoder than its "
        f"language model, or this prompt holds more than the encode term, which was measured on one "
        f"{ENCODE_ACTIVATION_TOKENS}-token prompt and is not scaled with the prompt length; the estimate "
        "under-predicts the encode phase either way"
    )


def encoder_bytes_used(base_root: Path) -> int:
    """The text encoder's bytes a Qwen-Image 2.1 prompt encode makes resident, read from the safetensors headers.

    Only the language model's tensors (``model.language_model.*``) are loaded, and every one of its layers runs (the
    prompt embedding is the final hidden state). The vision tower and the output head in the same shards are never
    loaded, so they do not count.

    Raises:
        DFloatFormatError: An encoder file's header cannot be read.
    """
    return sum(
        info.nbytes
        for path in sorted((base_root / "text_encoder").glob("*.safetensors"))
        for name, info in read_header(path).items()
        if name.startswith(ENCODER_PREFIX)
    )


def prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """The text tokens mflux feeds the transformer for ``prompt`` (imports mflux).

    As mflux encodes it: a blank prompt becomes ``" "``, the templated prompt is tokenized (at most
    ``TEXT_MAX_LENGTH`` tokens, the system prefix included) and the system prefix is dropped.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
        Qwen21PromptEncoder,
    )

    text = prompt if prompt and prompt.strip() else " "
    total = min(int(tokenizer.tokenize(text).input_ids.shape[1]), TEXT_MAX_LENGTH)
    return total - int(Qwen21PromptEncoder._system_prefix_length(tokenizer))


def sizes_for(ckpt: DF11Checkpoint, base_root: Path) -> FamilySizes:
    """The family sizes of a Qwen-Image 2.1 checkpoint and base, the encoder counted by its language model."""
    sizes = family_sizes_for(ckpt, base_root, nonblock_groups=NONBLOCK_GROUPS)
    return replace(sizes, encoders=encoder_bytes_used(base_root))
