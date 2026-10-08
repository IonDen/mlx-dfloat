"""Krea 2 memory rules: the constants, the text-token count and the text encoder's resident bytes.

The phase arithmetic itself is the shared one (``mlx_dfloat.mflux._phases``). Krea 2 differs in three inputs. Its text
encoder's file also holds a vision tower that mflux never loads, and the prompt embedding taps hidden states up to the
output of layer 34, so the encoder's last layer and final norm never run (see ``encoder_bytes_used``). Its prompt
length varies per call (up to the tokenizer's 1024 tokens), and mflux strips the chat-template prefix before the
transformer sees the text, so the text tokens are counted per prompt after that strip (``prompt_tokens``). And
classifier-free guidance runs as two batch-1 transformer calls, so one set of constants serves both Krea 2 models and
a CFG step is sized by its longer prompt (``text_tokens_for``).

The seven non-block groups (the four text-fusion blocks, ``tmlp``, ``tproj``, ``txtmlp``) are decoded at the start of
each transformer call and released before block 0's decode, so the model plans with ``sizes_for(...,
nonblock_per_call=True)`` (no resident non-block bytes) and the denoise term carries their call-start decode. Each
call starts with MLX's cache emptied and drained, and so does block 0 (``mlx_dfloat.mflux.krea2.transformer``). At
1024² that keeps a guided Krea 2 Raw step's denoise peak at 21.14 GiB, and two Krea 2 Turbo steps at 21.15 GiB: a
step's later calls follow the same path as its first. The peak is block 0's own evaluation, with the cache far below
the planner's cache limit, so the cache term over-counts it and the measured activation term is 0 (the
size-independent floor carries it).

At 1024² on a 32 GB Mac the set is always dropped before the VAE decode. The transformer's residual stream runs float32
(the latents are float32), and so does the VAE decode.
"""

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlx_dfloat._safetensors import read_header
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants
from mlx_dfloat.mflux._phases import sizes_for as family_sizes_for
from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS

# The text-encoder prefixes mflux keeps (its weight definition strips either and drops every other tensor).
ENCODER_PREFIXES: tuple[str, ...] = ("model.language_model.", "language_model.")
# The tokenizer's max_length (mflux's weight definition; tokenization truncates at it, the template included).
TEXT_MAX_LENGTH = 1024
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path

# Measured on 2026-10-08 (M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0) at 1024², seed 42, the 30-token
# lighthouse prompt; one process per run: build, encode, drop the encoder, set load, the denoise steps at the planner's
# cache limit for the block-cache probe's allowance (4_500_000_000 B), the set dropped, the VAE decode. The constants
# rest on the runs of the shipped mechanism (per-call non-block decode, MLX's cache emptied and drained at each call's
# start and again before block 0's decode): Krea 2 Raw (mingyi456/Krea-2-Raw-DF11-ComfyUI @ 8320616 over
# krea/Krea-2-Raw @ 6b0ece7), one step at guidance 3.5 with mflux's " " negative (two batch-1 calls; "derisk-raw-2" and
# "confirm-raw-24711fb"), and Krea 2 Turbo (mingyi456/Krea-2-Turbo-DF11-ComfyUI @ 978da5f over krea/Krea-2-Turbo @
# 98e0fe1), two steps at guidance 1.0 ("derisk-turbo-2" and "confirm-turbo-24711fb"). The VAE term also counts the Raw
# run with the non-block groups resident ("derisk-raw-1"): its VAE phase ran with the set dropped, as in every
# mechanism. Runs of earlier per-call code are recorded as history and source nothing. The git fields in the records
# name development commits folded into the 0.2.0 release. Records: bench/results/calibration/krea-2-raw-1024.json and
# krea-2-1024.json (constants_from names the run of each term). Bytes throughout; re-measure before changing any.
#
# REFERENCE_TOKENS: the calibration point, 1024² (4096 image tokens) + the runs' text tokens, 30 (the prompt's; the
# negative " " is 6, and a step is sized by its longer call).
REFERENCE_TOKENS = 4096 + 30
# OVERHEAD: the footprint MLX's counters do not see after the build (footprint - MLX active - MLX cache), the largest
# of the four shipped runs: derisk-turbo-2, 529_171_680 - 4_805_792 - 4_974 (the others 522_673_362 to 523_787_474).
OVERHEAD = 524_360_914  # 0.488 GiB
# VAE_TRANSIENT, by the overall-peak rule over every 1024² VAE-phase sample of the shipped runs and the resident run
# (all decoded with the set dropped): the highest VAE footprint peak minus the dropped-set phase's other terms (extras
# + VAE file + OVERHEAD), from derisk-raw-1: 11_869_477_608 - (4_560_024 + 507_591_892 + 524_360_914). The shipped
# runs give 9_127_156_298 to 9_489_078_618; a fit check must not under-predict, so the term covers the highest sample
# (the superseded runs' VAE peaks are all below it too).
VAE_TRANSIENT = 10_832_964_778  # 10.089 GiB
# DENOISE_ACTIVATION, by the overall-peak rule over every shipped 1024² denoise sample (the four one-step runs, the two
# MEASURED rows, the 52-step card recipe and the identity check's df11 side): per run, the denoise footprint peak minus
# the phase's other terms with that run's own sizes at the planner's limit (compressed + extras + non-block 0 + one
# block 868_220_928 + cache limit 4_500_000_000 + OVERHEAD). Every residual is negative; the highest is the Turbo
# MEASURED row's, 22_736_402_568 - 23_437_217_164 = -700_814_596. With the cache emptied at each call's start and before
# block 0, the peak is block 0's and the cache never holds the cache-limit term's full 4.5e9 B there. The term is
# floored at 0, so the activation is the floor below at every size, and the 1024² estimate sits 1.51e9 to 1.54e9 B
# over every shipped peak.
DENOISE_ACTIVATION = 0
# DENOISE_ACTIVATION_FLOOR: a size-independent lower bound on the denoise term. The residual stream is float32 and the
# block weights BF16, so each block's MLP makes MLX copy mlp.gate and mlp.up to float32 whatever the image size: 2 x
# 24576 x 4096 x 4, counted once, not per token (ERNIE-Image's rule for its batch-2 copies).
DENOISE_ACTIVATION_FLOOR = 805_306_368  # 0.75 GiB
# ALLOWANCE: the cache room for the activation buffers a block frees, at ALLOWANCE_TOKENS. The block-cache probe (one
# synthetic block at 1024² + 30 tokens, its decoded buffer allocated then released per iteration) saw the buffer reused
# at a 4.5e9 B limit; the allowance is that limit minus one decoded block, 4_500_000_000 - 868_220_928. The shipped
# runs' denoise peaks at this limit sat 1.40e9 to 1.42e9 B under the gate line (the fit budget minus 0.5 GiB).
ALLOWANCE = 3_631_779_072  # 3.382 GiB
ALLOWANCE_TOKENS = 4096 + 30
# ENCODE_ACTIVATION: what a prompt encode held beyond the language model's evaluated tensors by footprint (encode
# footprint peak - encoder_bytes_used - OVERHEAD), the largest of the four shipped runs: confirm-raw-24711fb,
# 8_598_264_624 - 7_843_074_560 - 524_360_914 (the others 163_884_078 to 228_387_886). Taken by footprint so the encode
# phase never under-predicts a calibration run (by MLX's measure the runs held 64_449_856 to 110_119_296 beyond the
# language model). mflux encodes a guided call's two prompts one after the other, so the term is per prompt, at 30
# tokens, unscaled.
ENCODE_ACTIVATION = 230_829_150
CONSTANTS: PhaseConstants = PhaseConstants(
    overhead_bytes=OVERHEAD,
    vae_transient_bytes=VAE_TRANSIENT,
    denoise_activation_at_reference=DENOISE_ACTIVATION,
    reference_tokens=REFERENCE_TOKENS,
    allowance_at_reference=ALLOWANCE,
    allowance_reference_tokens=ALLOWANCE_TOKENS,
    max_measured_pixels=MAX_MEASURED_PIXELS,
    encode_activation_bytes=ENCODE_ACTIVATION,
    denoise_activation_floor_bytes=DENOISE_ACTIVATION_FLOOR,
)
# What a prompt encode may hold by MLX's measure beyond the estimate's encode phase before the model warns (the bound
# is the evaluated language model + ENCODE_ACTIVATION + this slack, 8_342_339_166 B). 256 MiB: every recorded encode
# held 389_145_310 to 474_185_294 B under the bound, and an encode that also loads the vision tower (830_695_424 B more)
# is over it by 441_550_114 at every prompt length.
ENCODE_SLACK_BYTES = 256 * 1024**2
# The prompt length ENCODE_ACTIVATION was measured at (the calibration prompt's text tokens).
ENCODE_ACTIVATION_TOKENS = 30
_LAYER = re.compile(r"^(?:model\.)?language_model\.layers\.(\d+)\.")


def encode_peak_bound(encode_phase_bytes: int, constants: PhaseConstants = CONSTANTS) -> int:
    """What a prompt encode may hold by MLX's measure before the model warns.

    That is the estimate's encode phase without its overhead (the evaluated language model and the encode
    activations) plus ``ENCODE_SLACK_BYTES``.
    """
    return encode_phase_bytes - constants.overhead_bytes + ENCODE_SLACK_BYTES


def encode_peak_warning(held: int, bound: int, *, text_tokens: int) -> str | None:
    """The warning for an encode that held ``held`` bytes against ``bound``, or None at or under it.

    Two causes fit an excess: mflux loading the vision tower stored in the encoder's file, or a prompt longer than the
    one the encode term was measured on (the term is not scaled with the prompt length). The message names the call's
    longest prompt in tokens so the reader can tell which.
    """
    if held <= bound:
        return None
    return (
        f"the prompt encode held {held / 1024**3:.2f} GiB for a {text_tokens}-token prompt, over the "
        f"{bound / 1024**3:.2f} GiB the fit estimate allows: mflux may now load the vision tower stored in the text "
        f"encoder's file, or this prompt holds more than the encode term, which was measured on one "
        f"{ENCODE_ACTIVATION_TOKENS}-token prompt and is not scaled with the prompt length; the estimate "
        "under-predicts the encode phase either way"
    )


def encoder_bytes_used(base_root: Path) -> int:
    """The text encoder's bytes a Krea 2 prompt encode makes resident, read from the safetensors headers.

    Only the language model's tensors are loaded; the vision tower in the same file never is. The prompt embedding
    stacks hidden states up to the output of the second-to-last layer, so under lazy evaluation the last layer (and
    the final norm after it) never run: of ``n`` numbered layers the first ``n - 1`` count (35 of the published 36).
    Every other language-model tensor (the embeddings, the final norm) counts in full.

    Raises:
        DFloatFormatError: An encoder file's header cannot be read.
    """
    tensors = [
        (name, info.nbytes)
        for path in sorted((base_root / "text_encoder").glob("*.safetensors"))
        for name, info in read_header(path).items()
        if name.startswith(ENCODER_PREFIXES)
    ]
    indices = [int(m.group(1)) for name, _n in tensors if (m := _LAYER.match(name)) is not None]
    last = max(indices, default=-1)
    total = 0
    for name, nbytes in tensors:
        layer = _LAYER.match(name)
        if layer is None or int(layer.group(1)) < last:
            total += nbytes
    return total


def prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """The text tokens mflux feeds the transformer for ``prompt``: its ids after the chat-template prefix.

    The strip is the one ``Krea2TextEncoder.get_prompt_embeds`` applies (imports mflux).
    """
    require_mflux()
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    ids = tokenizer.tokenize(prompt).input_ids
    return int(ids.shape[1]) - int(Krea2TextEncoder._template_end(ids))


def text_tokens_for(tokenizer: Any, prompts: Sequence[str]) -> int:
    """The text tokens a step is sized by: the longest of its calls' prompts (a CFG step runs one call per prompt)."""
    return max(prompt_tokens(tokenizer, p) for p in prompts)


def sizes_for(
    ckpt: DF11Checkpoint, base_root: Path, *, nonblock_per_call: bool = False
) -> FamilySizes:
    """The family sizes of a Krea 2 checkpoint and base, the encoder counted by its evaluated language model.

    With ``nonblock_per_call`` the non-block groups are decoded per transformer call rather than kept decoded with the
    set, so ``nonblock`` is 0; the call-start transient belongs to the denoise activation term.
    """
    sizes = family_sizes_for(ckpt, base_root, nonblock_groups=NONBLOCK_GROUPS)
    sizes = replace(sizes, encoders=encoder_bytes_used(base_root))
    return replace(sizes, nonblock=0) if nonblock_per_call else sizes
