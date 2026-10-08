"""ERNIE-Image memory rules: the constants per CFG batch, the text-token count and the text encoder's resident bytes.

The phase arithmetic itself is the shared one (``mlx_dfloat.mflux._phases``). ERNIE-Image differs in three inputs. Its
text encoder's file also holds a vision tower and a projector that mflux never loads (see ``encoder_bytes_used``).
Its prompt length varies per call (up to the tokenizer's 2048 tokens), so the text tokens are counted per prompt
(``prompt_tokens``). And classifier-free guidance runs as one batch-2 transformer call, which doubles the activations
but not the decoded weights, so the constants are keyed by the call's batch size (1 or 2), not by the model.

The transformer has one block kind, so the cache limit holds one decoded block next to the activations, and each batch
size carries its own activation allowance. One encode term serves both batch sizes: the larger of the two measured
encodes at 26 tokens, one prompt (batch 1) and two (batch 2). The VAE transient is mflux's Flux2VAE decode; re-measure it when an mflux
upgrade changes the VAE's dtype.
"""

import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_dfloat._safetensors import read_header
from mlx_dfloat.format import DF11Checkpoint
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants
from mlx_dfloat.mflux._phases import sizes_for as family_sizes_for
from mlx_dfloat.mflux.ernie.names import NONBLOCK_GROUPS

# The only text-encoder tensors mflux loads (its weight definition's prefix filter for the text encoder).
ENCODER_PREFIX = "language_model.model."
# The tokenizer's max_length (mflux's weight definition; tokenization truncates at it).
TEXT_MAX_LENGTH = 2048
MAX_MEASURED_PIXELS = 1024 * 1024  # no run above 1024² on this path

# Measured on 2026-10-08 (git a771e58, M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0): two one-step calibration runs at
# 1024², seed 42, the 26-token lighthouse prompt; one process each: build, encode, drop, set load, one step at the
# planner's cache limit for the block-cache probe's allowance, VAE decode on the resident set. ERNIE-Image-Turbo
# (mingyi456/ERNIE-Image-Turbo-DF11 @ 27f84b4 over baidu/ERNIE-Image-Turbo @ bc68c81) at guidance 1.0: batch 1, run
# "derisk-turbo-1" in bench/results/calibration/ernie-image-turbo-1024.json. ERNIE-Image (mingyi456/ERNIE-Image-DF11
# @ c2dd30a over baidu/ERNIE-Image @ 5346b31) at guidance 4.0 without a negative prompt (mflux's " " then the prompt):
# batch 2, run "derisk-base-1" in bench/results/calibration/ernie-image-1024.json. Bytes throughout; re-measure before
# changing any.
#
# REFERENCE_TOKENS: the calibration point, 1024² (4096 image tokens) + the run's text tokens, 26 (the prompt's; the
# negative " " is 2, and a call is sized by its longest prompt).
REFERENCE_TOKENS = 4096 + 26
# OVERHEAD: the footprint MLX's counters do not see after the build (footprint - MLX active - MLX cache), the larger of
# the two runs: Turbo 644_400_592 - 27_961_608 - 4_806 = 616_434_178 (the base's 548_194_818). One value for both
# batches: the build is the same.
OVERHEAD = 616_434_178  # 0.574 GiB
# VAE_TRANSIENT, by the overall-peak rule over every 1024² sample (the two calibration runs so far): the highest measured
# VAE-phase peak on the resident set minus the VAE phase's other terms (compressed + extras + non-block + VAE file +
# OVERHEAD), from the Turbo run: 19_163_080_232 - (10_894_810_183 + 27_961_600 + 335_544_320 + 168_120_878 +
# 616_434_178). The base run's peak (19_127_199_104, with its own sizes) would give a term 35_821_552 B lower. A fit
# check must not under-predict, so the term covers the highest sample; later 1024² samples join the rule. One value
# for both batches: the VAE decode does not see the CFG batch. About 0.96x FLUX.2 Klein 4B's for the same Flux2VAE.
VAE_TRANSIENT = 7_120_209_073  # 6.631 GiB
# DENOISE_ACTIVATION per batch, by the overall-peak rule over every 1024² denoise-phase sample: the highest denoise
# footprint peak minus the denoise phase's other terms at the planner's cache limit (compressed + extras + non-block +
# in-flight block 436_207_616 + cache limit + OVERHEAD), at REFERENCE_TOKENS. Batch 1 (limit 2_600_000_000): the second
# CAPPED 24 GB run of ERNIE-Image-Turbo (label "capped-24-gb-2" in ernie-image-turbo-1024.json; git f4cd325), whose
# watched peak was in the denoise phase: 15_446_156_600 - (10_894_810_183 + 27_961_600 + 335_544_320 + 436_207_616 +
# 2_600_000_000 + 616_434_178) = 535_198_703. The first identical run ("capped-24-gb-1", git 5e2e7c2) peaked at
# 15_069_554_288 (term 158_596_391), 376_602_312 B (0.35 GiB) lower, and the calibration run's 14_974_035_520 gave
# 63_077_623: the denoise phase varies that much between identical runs. A fit check must not under-predict, so the
# term covers the highest sample. Batch 2 (base, limit 3_600_000_000), its calibration
# run, the only denoise-phase sample: 16_361_874_888 - (10_894_750_607 + 27_961_600 + 335_544_320 + 436_207_616 +
# 3_600_000_000 + 616_434_178) = 450_976_567.
DENOISE_ACTIVATION: dict[int, int] = {1: 535_198_703, 2: 450_976_567}
# ALLOWANCE per batch: the cache room for the activation buffers a block frees, at REFERENCE_TOKENS. The block-cache
# probe (one synthetic block at 1024² + 26 tokens, its decoded buffer allocated then released per iteration) saw the
# buffer reused from the second iteration at limits of 2.6e9 B (batch 1) and 3.6e9 B (batch 2, the smallest tried), with
# the steady cache drop of a clean reuse; the allowance is that limit minus one decoded block (436_207_616).
ALLOWANCE: dict[int, int] = {1: 2_163_792_384, 2: 3_163_792_384}
# DENOISE_ACTIVATION_FLOOR per batch: a size-independent lower bound on the denoise term. ERNIE-Image's hidden state is
# float32 (mflux computes the timestep conditioning in float32), and a float32 input times a BF16 weight makes MLX copy
# the weight to float32 for the matmul: the MLP's gate_proj and up_proj together, 2 x 12288 x 4096 x 4 = 402_653_184 B,
# whatever the image size. Batch 2 only: its 1024² term (450_976_567) is above the floor, so the calibration point is
# unchanged. Batch 1 carries none until a 512² run measures whether a one-prompt step holds the copies beyond the cache
# term (its 1024² term, 535_198_703, is above the floor anyway).
DENOISE_ACTIVATION_FLOOR: dict[int, int | None] = {1: None, 2: 402_653_184}
# ENCODE_ACTIVATION: what a prompt encode held beyond the language model's evaluated tensors by footprint (encode
# footprint peak - encoder_bytes_used - OVERHEAD), the larger of the two runs: Turbo, 7_335_614_536 - 6_625_216_512 -
# 616_434_178 = 93_963_846 (the base: 15_943_190). Taken by footprint so the encode phase never under-predicts a
# calibration run (by MLX's measure the holds were 4_188_932 and 29_933_258). The encoder's last layer never runs (mflux
# returns the second-to-last hidden state): counted with every language-model tensor (6_858_012_672) the term came out
# negative. Measured at 26 tokens with one prompt (batch 1) and two (batch 2, where both encoder passes are one lazy
# graph evaluated once); the larger serves both batches, unscaled for longer prompts.
ENCODE_ACTIVATION = 93_963_846


def _constants(batch: int) -> PhaseConstants:
    return PhaseConstants(
        overhead_bytes=OVERHEAD,
        vae_transient_bytes=VAE_TRANSIENT,
        denoise_activation_at_reference=DENOISE_ACTIVATION[batch],
        reference_tokens=REFERENCE_TOKENS,
        allowance_at_reference=ALLOWANCE[batch],
        allowance_reference_tokens=REFERENCE_TOKENS,
        max_measured_pixels=MAX_MEASURED_PIXELS,
        encode_activation_bytes=ENCODE_ACTIVATION,
        denoise_activation_floor_bytes=DENOISE_ACTIVATION_FLOOR[batch],
    )


CONSTANTS: dict[int, PhaseConstants] = {1: _constants(1), 2: _constants(2)}
# What a prompt encode may hold by MLX's measure beyond the estimate's encode phase before the model warns (the bound
# is the evaluated language model + ENCODE_ACTIVATION + this slack, 6_987_615_814 B). 256 MiB: the runs held 358_210_370
# (Turbo) and 332_466_044 (base) under the bound, and an encode that also loads the vision tower and projector
# (840_167_424 B more) is over it by 507_701_380 at every prompt length.
ENCODE_SLACK_BYTES = 256 * 1024**2
# The prompt length ENCODE_ACTIVATION was measured at (the calibration prompt's text tokens).
ENCODE_ACTIVATION_TOKENS = 26
_LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def encode_peak_bound(encode_phase_bytes: int, constants: PhaseConstants) -> int:
    """What a prompt encode may hold by MLX's measure before the model warns.

    That is the estimate's encode phase without its overhead (the language model and the encode activations) plus
    ``ENCODE_SLACK_BYTES``. ``constants`` are the call's (``CONSTANTS[batch]``).
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
    """The text encoder's bytes an ERNIE-Image prompt encode makes resident, read from the safetensors headers.

    Only the language model's tensors (``language_model.model.*``) are loaded; the vision tower and the projector in
    the same file are never loaded. The encoder returns its second-to-last hidden state, so under lazy evaluation its
    last layer never runs and its weights are never read: of ``n`` numbered layers the first ``n - 1`` count (25 of the
    published 26). Every language-model tensor outside the numbered layers (embeddings, final norm) counts in full.

    Raises:
        DFloatFormatError: An encoder file's header cannot be read.
    """
    tensors = [
        (name, info.nbytes)
        for path in sorted((base_root / "text_encoder").glob("*.safetensors"))
        for name, info in read_header(path).items()
        if name.startswith(ENCODER_PREFIX)
    ]
    indices = [int(m.group(1)) for name, _n in tensors if (m := _LAYER.match(name)) is not None]
    evaluated = max(indices, default=-1)  # layers 0 .. n - 2 run; the last one, n - 1, does not
    total = 0
    for name, nbytes in tensors:
        layer = _LAYER.match(name)
        if layer is None or int(layer.group(1)) < evaluated:
            total += nbytes
    return total


def prompt_tokens(tokenizer: Any, prompt: str) -> int:
    """The text tokens mflux feeds the transformer for ``prompt``: its attention mask's ones, at most 2048."""
    return min(int(mx.sum(tokenizer.tokenize(prompt).attention_mask[0])), TEXT_MAX_LENGTH)


def text_tokens_for(tokenizer: Any, prompts: Sequence[str]) -> int:
    """The text tokens of one call: mflux pads every prompt of a CFG batch to the longest."""
    return max(prompt_tokens(tokenizer, p) for p in prompts)


def sizes_for(ckpt: DF11Checkpoint, base_root: Path) -> FamilySizes:
    """The family sizes of an ERNIE-Image checkpoint and base, the encoder counted by its language model."""
    sizes = family_sizes_for(ckpt, base_root, nonblock_groups=NONBLOCK_GROUPS)
    return replace(sizes, encoders=encoder_bytes_used(base_root))
