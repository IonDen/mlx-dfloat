"""FLUX.2 Klein's memory constants and its encoder sizing, checked against the 1024² de-risk measurements.

Measured inputs (2026-10-08, base 4B and base 9B, guidance 4, 1024², seed 42; one process per size: build, encode,
drop, set load, one step, VAE decode on the resident set; M1 Max 32 GB, mlx 0.32.2, mflux 0.20.0):
4B footprint peaks encode 7_689_574_400, denoise 9_168_770_160, vae 13_621_449_840 (the call's peak); file sizes
compressed 5_239_059_784, extras 22_035_456, decoded non-block 368_050_176, VAE 168_120_878; cache limit 2_324_335_646.
9B footprint peaks encode 12_849_669_864, denoise 17_534_117_944, vae 19_570_665_528 (the call's peak); compressed
12_275_304_729, extras 37_769_216, decoded non-block 671_088_640, VAE 168_120_878; cache limit 2_896_858_142.
"""

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_bf16_original, write_checkpoint

from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux._phases import FamilySizes, activation_allowance, fit_for
from mlx_dfloat.mflux.flux2 import memory as kmem

GIB = 1024**3
KLEIN_4B_LARGEST = {"transformer_blocks": 490_733_568, "single_transformer_blocks": 245_366_784}
# Base 4B text encoder, read from its two shard headers (black-forest-labs/FLUX.2-klein-base-4B @ a3b4f48):
# embed_tokens 777_912_320 + final norm 5_120 + 36 layers x 201_861_632 = 8_044_936_192 tensor bytes (file 8_044_981_992).
# Klein stacks hidden states 9, 18 and 27 (mflux 0.20.0 flux2_klein.py:159); hidden_states[0] is the embedding
# output (qwen3_text_encoder.py:93-101), so index 27 is layers[26]'s output: layers 0-26 run, 27-35 never do.
# 777_912_320 + 5_120 + 27 x 201_861_632 = 6_228_181_504.
KLEIN_4B_ENCODER_USED = 6_228_181_504
MEASURED_4B = {"encode": 7_689_574_400, "denoise": 9_168_770_160, "vae": 13_621_449_840}
KLEIN_9B_LARGEST = {"transformer_blocks": 872_415_232, "single_transformer_blocks": 436_207_616}
# Base 9B text encoder (black-forest-labs/FLUX.2-klein-base-9B @ 3277332), from its headers: embed_tokens
# 1_244_659_712 + final norm 8_192 + 36 layers x 385_892_864 + an untied lm_head 1_244_659_712 (file 16_381_516_824).
# mflux's encoder has no lm_head (flux2_weight_mapping.py:496-533 maps model.* only), so it is never loaded:
# 1_244_659_712 + 8_192 + 27 x 385_892_864 = 11_663_775_232.
KLEIN_9B_ENCODER_USED = 11_663_775_232
MEASURED_9B = {"encode": 12_849_669_864, "denoise": 17_534_117_944, "vae": 19_570_665_528}


def _write_encoder(root, *, layers=36, per_layer_rows=4):
    """A Qwen3-shaped text encoder: embeddings (8, 4), a final norm (4,), ``layers`` layers of two (rows, 4) matrices
    and an untied lm_head (8, 4).

    bf16: embeddings 64 B, norm 8 B, each layer 2 x rows x 4 x 2 B (rows 4: 64 B), lm_head 64 B. Two shards and an
    index.
    """
    tensors = {
        "model.embed_tokens.weight": np.zeros((8, 4), np.uint16),
        "model.norm.weight": np.zeros((4,), np.uint16),
        "lm_head.weight": np.zeros((8, 4), np.uint16),
    }
    for i in range(layers):
        for proj in ("self_attn.q_proj", "mlp.down_proj"):
            tensors[f"model.layers.{i}.{proj}.weight"] = np.zeros((per_layer_rows, 4), np.uint16)
    write_bf16_original(root / "text_encoder", tensors)


def test_the_encoder_counts_only_the_layers_klein_evaluates(tmp_path):
    # Bug caught: the encode phase sized by the encoder's whole file (all 36 layers and the head: 2_440), by 28 layers
    # (hidden_states[27] read as layers[27]'s output: 1_864), or with the untied lm_head counted (mflux never loads it:
    # 1_864 + 64 - 64 = 1_864). Klein needs the embeddings, the norm and layers 0-26: 64 + 8 + 27 x 64.
    _write_encoder(tmp_path)
    assert kmem.encoder_bytes_used(tmp_path) == 64 + 8 + 27 * 64


def test_the_layers_needed_follow_the_deepest_hidden_state_asked_for(tmp_path):
    # Bug caught: the cut hard-coded at 27 layers instead of read from the hidden states asked for (out_layers).
    _write_encoder(tmp_path)
    assert kmem.encoder_bytes_used(tmp_path, out_layers=(3, 5)) == 64 + 8 + 5 * 64


def test_sizes_for_replaces_the_encoder_file_size_with_the_bytes_used(tmp_path):
    # Bug caught: Klein's sizes still carrying the encoder's file size (the encode phase over-predicted by the 9 unused
    # layers), or the DF11 / VAE terms lost on the way.
    rng = np.random.default_rng(0)
    write_checkpoint(
        tmp_path / "df11",
        groups={
            "transformer_blocks.0": [random_bf16(rng, (8, 4))],
            "context_embedder": [random_bf16(rng, (4, 6))],
        },
        patterns={r"transformer_blocks\.\d+": ("attn.to_q",), "context_embedder": ()},
        extras={"x_embedder.weight": np.zeros((3, 5), dtype=np.uint16)},
        single_file=True,
    )
    _write_encoder(tmp_path / "base")
    (tmp_path / "base" / "vae").mkdir()
    (tmp_path / "base" / "vae" / "vae.safetensors").write_bytes(b"\0" * 100)

    ckpt = open_checkpoint(tmp_path / "df11")
    sizes = kmem.sizes_for(ckpt, tmp_path / "base")
    total = (tmp_path / "df11" / "model.safetensors").stat().st_size
    assert sizes.encoders == 64 + 8 + 27 * 64
    assert sizes.extras == 30  # 3 x 5 BF16
    assert sizes.compressed == total - 30
    assert sizes.nonblock == 48  # context_embedder decoded: 4 x 6 BF16
    assert sizes.vae == 100


def _fit(size, *, compressed, extras, nonblock, encoders, largest, cache_limit):
    c = kmem.CONSTANTS[size]
    sizes = FamilySizes(
        compressed=compressed, extras=extras, nonblock=nonblock, encoders=encoders, vae=168_120_878
    )
    return fit_for(
        c,
        sizes=sizes,
        largest=largest,
        policy="per-block",
        cache_limit=cache_limit,
        allowance=activation_allowance(c, height=1024, width=1024, text_tokens=512),
        budget=24_653_119_488,
        height=1024,
        width=1024,
        text_tokens=512,
    )


def _fit_4b(encoders=KLEIN_4B_ENCODER_USED, compressed=5_239_059_784):
    return _fit(
        "4b",
        compressed=compressed,
        extras=22_035_456,
        nonblock=368_050_176,
        encoders=encoders,
        largest=KLEIN_4B_LARGEST,
        cache_limit=2_324_335_646,
    )


def _fit_9b(encoders=KLEIN_9B_ENCODER_USED, compressed=12_275_304_729):
    return _fit(
        "9b",
        compressed=compressed,
        extras=37_769_216,
        nonblock=671_088_640,
        encoders=encoders,
        largest=KLEIN_9B_LARGEST,
        cache_limit=2_896_858_142,
    )


# The constants are calibrated on the de-risk runs (MEASURED_4B / MEASURED_9B), so there the estimate equals the
# measurement by construction: the in-sample checks are exact. Whether the model predicts is decided by the runs below,
# none of them used for calibration.


def test_the_4b_estimate_at_1024_equals_its_calibration_run_phase_for_phase():
    # Bug caught: a term dropped from a phase or counted twice, or the VAE transient taken as the one-step value (VAE
    # phase peak minus the footprint after denoise, 5_024_366_592), which puts the VAE phase at 11_202_797_638. The
    # call's peak is the VAE decode on the resident set.
    fit = _fit_4b()
    assert fit.peak_phase == "vae"
    assert fit.phases["vae"] == MEASURED_4B["vae"]
    assert fit.phases["denoise"] == MEASURED_4B["denoise"]


def test_the_9b_estimate_at_1024_equals_its_calibration_run_phase_for_phase():
    # Bug caught: the 9B planned with the 4B constants, a term dropped, or the one-step transient (3_803_234_304).
    fit = _fit_9b()
    assert fit.peak_phase == "vae"
    assert fit.phases["vae"] == MEASURED_9B["vae"]
    assert fit.phases["denoise"] == MEASURED_9B["denoise"]


# Watched (footprint) peaks of 1024² runs not used for calibration (2026-10-08, M1 Max 32 GB, mlx 0.32.2, mflux
# 0.20.0, seed 42, per-block, the set kept through the VAE decode): the image identity run's df11 side and the four
# `mlx-dfloat generate --tier 32` rows. A distilled checkpoint's compressed size is its own (a few KB above the base
# one's). Predicted - measured: +0.45, -0.09, +0.81, +0.63, -0.18 GiB.
OUT_OF_SAMPLE = [
    pytest.param(
        "4b", 5_239_059_784, 13_142_463_552, id="identity-df11-base-4b-4-steps-guidance-4"
    ),
    pytest.param("4b", 5_239_059_784, 13_721_457_680, id="measured-base-4b-50-steps-guidance-4"),
    pytest.param(
        "4b", 5_239_066_183, 12_752_884_632, id="measured-distilled-4b-4-steps-guidance-1"
    ),
    pytest.param("9b", 12_275_304_729, 18_895_939_424, id="measured-base-9b-50-steps-guidance-4"),
    pytest.param(
        "9b", 12_275_371_955, 19_761_342_400, id="measured-distilled-9b-4-steps-guidance-1"
    ),
]


@pytest.mark.parametrize(("size", "compressed", "measured"), OUT_OF_SAMPLE)
def test_the_estimate_predicts_runs_it_was_not_calibrated_on_within_a_gib(
    size, compressed, measured
):
    # Bug caught: a calibration that only reproduces its own run (an overfit constant, the one-step VAE transient, a
    # phase that ignores step count or guidance): the peak of an independent run more than 1 GiB from the prediction,
    # in either direction, or peaking in another phase.
    fit = (_fit_4b if size == "4b" else _fit_9b)(compressed=compressed)
    assert fit.peak_phase == "vae"
    assert abs(fit.peak_bytes - measured) <= 1 * GIB


def test_the_encode_phase_is_within_a_gib_of_the_measured_encode_for_both_sizes():
    # Bug caught: the encode phase sized by the encoder file. 4B: 8_044_981_992 B puts it 2.3 GiB above the measured
    # 7_689_574_400; 9B: 16_381_516_824 B puts it 5.1 GiB above 12_849_669_864 (and the 9B's never-loaded lm_head
    # alone, 1_244_659_712 B, 1.9 GiB above). With the layers Klein runs both are within 0.75 GiB. Not calibrated: the
    # encoder bytes come from the file headers.
    assert abs(_fit_4b().phases["encode"] - MEASURED_4B["encode"]) <= 1 * GIB
    assert abs(_fit_9b().phases["encode"] - MEASURED_9B["encode"]) <= 1 * GIB
    with_head = _fit_9b(encoders=KLEIN_9B_ENCODER_USED + 1_244_659_712)
    assert (
        with_head.phases["encode"] - MEASURED_9B["encode"] > 1 * GIB
    )  # the control: the bug shows here


def test_the_9b_activation_is_above_the_4b_one():
    # Bug caught: one value shared by both sizes, or the constants swapped (the 4096-wide 9B holds more activations
    # than the 3072-wide 4B: 402_128_193 against 343_390_778 measured).
    assert (
        kmem.CONSTANTS["9b"].denoise_activation_at_reference
        > kmem.CONSTANTS["4b"].denoise_activation_at_reference
    )


@pytest.mark.mflux
@pytest.mark.parametrize(
    "name", ["flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b"]
)
def test_text_tokens_is_512_for_every_klein(name):
    # Bug caught: a hard-coded token count drifting from mflux's (model_config.py:425, 447, 505, 534 say 512).
    from mflux.models.common.config.model_config import ModelConfig

    assert kmem.text_tokens(ModelConfig.from_name(model_name=name, base_model=None)) == 512
