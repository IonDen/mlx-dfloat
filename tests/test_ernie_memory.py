"""ERNIE-Image's memory rules: the encoder's resident bytes, the text-token count, the sizes, the measured constants.

Measured inputs (2026-10-08, git a771e58, M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0): two one-step calibration
runs at 1024², seed 42, the 26-token lighthouse prompt, one process each (build, encode, drop, set load, one step at the
planner's cache limit for the block-cache probe's allowance, VAE decode on the resident set): ERNIE-Image-Turbo at
guidance 1.0 (batch 1) and ERNIE-Image at guidance 4.0 (mflux's " " negative then the prompt: batch 2). Their phase
peaks, sizes, encode holds and post-build counters are committed in
``bench/results/calibration/ernie-image-turbo-1024.json`` and ``bench/results/calibration/ernie-image-1024.json``
(``constants_from`` names the run each term came from), and the tests below read them from there.
"""

import json
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.bench.capped import tier_limits
from mlx_dfloat.errors import DFloatResourceError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate import memory as imem
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants, activation_allowance, fit_for
from mlx_dfloat.mflux._pipeline import plan_call_for
from mlx_dfloat.mflux.ernie import memory as emem

# The calibration prompt's token count (mflux 0.20.0 TokenizerLoader on both base snapshots, 2026-10-08).
DERISK_TOKENS = 26


def _bf16(*shape):
    return mx.zeros(shape, dtype=mx.bfloat16)


def _write_encoder(root, *, layers=3):
    """An ERNIE text encoder file the way the base stores it: the language model under ``language_model.model.``
    (``layers`` numbered layers), the vision tower and the projector beside it (the real file's prefixes, 2026-10-08).

    bf16 bytes: embed_tokens 10 x 4 = 80, one up_proj 4 x 4 = 32 per layer, the final norm (4,) = 8, the vision tensor
    8 x 4 = 64, the projector 6 x 4 = 48.
    """
    enc = root / "text_encoder"
    enc.mkdir(parents=True)
    tensors = {
        "language_model.model.embed_tokens.weight": _bf16(10, 4),
        "language_model.model.norm.weight": _bf16(4),
        "vision_tower.transformer.layers.0.attention.q_proj.weight": _bf16(8, 4),
        "multi_modal_projector.linear_1.weight": _bf16(6, 4),
    }
    for i in range(layers):
        tensors[f"language_model.model.layers.{i}.mlp.up_proj.weight"] = _bf16(4, 4)
    mx.save_safetensors(str(enc / "model.safetensors"), tensors)


def test_encoder_bytes_count_the_language_model_layers_mflux_evaluates(tmp_path):
    # Bug caught: the encoder's file size counted (as for FLUX.1 and Z-Image), the vision tower or the projector
    # counted (mflux's prefix filter loads language_model.model.* only, ernie_weight_definition.py:31), or the last
    # layer counted: mflux returns the second-to-last hidden state (text_encoder.py:217-221), so under lazy evaluation
    # the last layer never runs and its weights are never read. Embeddings 80 + norm 8 + layers 0-1 2 x 32 = 152.
    _write_encoder(tmp_path)
    assert emem.encoder_bytes_used(tmp_path) == 152


def test_the_encoder_counts_25_of_26_layers(tmp_path):
    # Bug caught: the cut hard-coded or off by one against the published 26-layer encoder (25 evaluated, layers
    # 0-24): 80 + 8 + 25 x 32 = 888 (26 layers would give 920, 24 give 856).
    _write_encoder(tmp_path, layers=26)
    assert emem.encoder_bytes_used(tmp_path) == 888


def test_sizes_for_replaces_the_encoder_file_size_with_the_evaluated_language_model_bytes(tmp_path):
    # Bug caught: the sizes still carrying the file's size (vision tower included: the encode phase over-predicted),
    # or the DF11 / non-block / VAE terms lost when the encoder term is replaced, or a non-block group not counted.
    rng = np.random.default_rng(0)
    write_checkpoint(
        tmp_path / "df11",
        groups={
            "layers.0": [random_bf16(rng, (8, 4))],
            "adaLN_modulation.1": [random_bf16(rng, (16, 4))],
            "final_norm.linear": [random_bf16(rng, (8, 4))],
        },
        patterns={
            r"layers\.\d+": ("self_attention.to_q",),
            r"adaLN_modulation\.1": (),
            r"final_norm\.linear": (),
        },
        extras={"text_proj.weight": np.zeros((3, 5), dtype=np.uint16)},
        single_file=True,
    )
    _write_encoder(tmp_path / "base")
    (tmp_path / "base" / "vae").mkdir()
    (tmp_path / "base" / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    sizes = emem.sizes_for(open_checkpoint(tmp_path / "df11"), tmp_path / "base")
    total = (tmp_path / "df11" / "model.safetensors").stat().st_size
    assert sizes.encoders == 152
    assert sizes.extras == 30  # 3 x 5 BF16
    assert sizes.compressed == total - 30
    assert sizes.nonblock == 128 + 64  # adaLN_modulation.1 16 x 4 + final_norm.linear 8 x 4, BF16
    assert sizes.vae == 100


class _Tokens:
    def __init__(self, n):
        self.attention_mask = mx.ones((1, n), dtype=mx.int32)
        self.input_ids = mx.ones((1, n), dtype=mx.int32)


class _Tokenizer:
    """mflux's tokenizer protocol as ErniePromptEncoder uses it: tokenize(prompt) -> ids and an attention mask."""

    def __init__(self, lengths):
        self.lengths = lengths

    def tokenize(self, prompt):
        return _Tokens(self.lengths[prompt])


def test_prompt_tokens_count_the_attention_mask_and_cap_at_2048():
    # Bug caught: the cap missing or off by one, or the padded length counted instead of the mask (mflux's num_valid,
    # prompt_encoder.py:15-17). The cap is the tokenizer's max_length, 2048 (ernie_weight_definition.py:43).
    tok = _Tokenizer({"abc": 3, "u": 2047, "at": 2048, "o": 2049, "big": 3000})
    assert [emem.prompt_tokens(tok, p) for p in ("abc", "u", "at", "o", "big")] == [
        3,
        2047,
        2048,
        2048,
        2048,
    ]


def test_padding_is_not_counted():
    # Bug caught: the padded input length counted (a CFG batch pads every prompt to the longest: the short prompt's
    # mask has zeros).
    class Padded:
        attention_mask = mx.array([[1, 1, 0, 0, 0]], dtype=mx.int32)
        input_ids = mx.ones((1, 5), dtype=mx.int32)

    tok = type("T", (), {"tokenize": lambda self, p: Padded()})()
    assert emem.prompt_tokens(tok, "x") == 2


def test_a_cfg_call_is_sized_by_its_longest_prompt():
    # Bug caught: the positive prompt only (the negative can be the longer one; build_text_batch pads every prompt to
    # the longest, prompt_encoder.py:36).
    tok = _Tokenizer({" ": 2, "long": 40})
    assert emem.text_tokens_for(tok, (" ", "long")) == 40
    assert emem.text_tokens_for(tok, ("long", " ")) == 40


def test_the_encode_bound_is_the_phase_without_overhead_plus_the_slack():
    # Bug caught: the overhead kept in the bound, or the slack missing. 10e9 - 0.4e9 + 268_435_456, by hand.
    c = PhaseConstants(
        overhead_bytes=400_000_000,
        vae_transient_bytes=0,
        denoise_activation_at_reference=0,
        reference_tokens=1,
    )
    assert emem.encode_peak_bound(10_000_000_000, c) == 9_868_435_456


def test_the_encode_warning_is_silent_at_the_bound_and_names_the_prompt_length_over_it():
    # Bug caught: `>=` for `>` at the bound (a normal encode warning), or a warning that blames mflux's loading alone
    # when the prompt is longer than the one the encode term was measured on (26 tokens).
    bound = 7_000_000_000
    assert emem.encode_peak_warning(bound, bound, text_tokens=2048) is None
    message = emem.encode_peak_warning(bound + 1, bound, text_tokens=2048)
    assert message is not None
    assert "for a 2048-token prompt" in message
    assert f"measured on one {DERISK_TOKENS}-token prompt" in message
    assert "vision tower" in message


# --- the measured constants, against the committed calibration records -----------------------------------------

GIB = 1024**3
REPO = Path(__file__).resolve().parents[1]
RECORDS = {
    1: json.loads((REPO / "bench/results/calibration/ernie-image-turbo-1024.json").read_text()),
    2: json.loads((REPO / "bench/results/calibration/ernie-image-1024.json").read_text()),
}
RUNS = {b: {run["label"]: run for run in rec["runs"]} for b, rec in RECORDS.items()}
# The phase peaks each batch's constants were derived from (constants_from names the run per phase).
MEASURED = {
    b: {
        phase: RUNS[b][label]["footprint_peaks"][phase]
        for phase, label in rec["constants_from"].items()
    }
    for b, rec in RECORDS.items()
}
SIZES = {
    b: FamilySizes(**RUNS[b][rec["constants_from"]["denoise"]]["sizes"])
    for b, rec in RECORDS.items()
}
CACHE_LIMITS = {
    b: RUNS[b][rec["constants_from"]["denoise"]]["cache_limit"] for b, rec in RECORDS.items()
}
# Every 1024² VAE-phase sample of either model: the VAE term is one value for both batches (overall-peak rule).
VAE_SAMPLES = {
    f"{b}:{label}": run["footprint_peaks"]["vae"]
    for b in RUNS
    for label, run in RUNS[b].items()
    if "vae" in run["footprint_peaks"]
}
LARGEST = {
    "layers": 436_207_616
}  # one block kind; 36 groups of 218_103_808 elements (4 x 4096² + 3 x 4096 x 12288)
TEXT_TOKENS = 26  # the calibration prompt's (the base's negative " " is 2)
# The block-cache probe's allowances at 4096 + 26 tokens (the smallest limit at which each batch's decoded buffer is
# reused: 2.6e9 and 3.6e9 B, minus one decoded block).
ALLOWANCE = {1: 2_163_792_384, 2: 3_163_792_384}
# This Mac (M1 Max 32 GB): mx.device_info()["max_recommended_working_set_size"] and the RAM.
RECOMMENDED_32 = 26_800_603_136
RAM_32 = 34_359_738_368


def _fit(batch, *, text_tokens=TEXT_TOKENS, budget=24_653_119_488, vae_with_set=True):
    c = emem.CONSTANTS[batch]
    return fit_for(
        c,
        sizes=SIZES[batch],
        largest=LARGEST,
        policy="per-block",
        cache_limit=CACHE_LIMITS[batch],
        allowance=activation_allowance(c, height=1024, width=1024, text_tokens=text_tokens),
        budget=budget,
        height=1024,
        width=1024,
        text_tokens=text_tokens,
        vae_with_set=vae_with_set,
    )


def _plan(batch, budget, *, fit_check=True, cache_limit_override=None, policy="per-block"):
    return plan_call_for(
        constants=emem.CONSTANTS[batch],
        sizes=SIZES[batch],
        largest=LARGEST,
        policy=policy,
        cache_limit_override=cache_limit_override,
        fit_check=fit_check,
        budget=budget,
        height=1024,
        width=1024,
        text_tokens=TEXT_TOKENS,
        log=logging.getLogger("test"),
    )


# The constants are calibrated on the two calibration runs and, for batch 1's denoise term, the CAPPED 24 GB run (the
# overall-peak rule), so at their working point the estimate equals the measurement by construction: the in-sample
# checks are exact. No ERNIE-Image run away from 1024² and 26 tokens exists yet.


def test_the_calibration_records_hold_the_runs_the_constants_name():
    # Bug caught: a constant re-derived from a run that is not recorded, or a record of the wrong model or snapshot:
    # each batch's denoise term from its own run, the shared encode and VAE terms each from exactly one record.
    assert (RECORDS[1]["batch"], RECORDS[2]["batch"]) == (1, 2)
    assert (
        RECORDS[1]["df11"]
        == "mingyi456/ERNIE-Image-Turbo-DF11@27f84b44a3b78fcfaaadbaeeeae7cc7f7d75153b"
    )
    assert RECORDS[1]["base"] == "baidu/ERNIE-Image-Turbo@bc68c81e2a1730a394d5fc9fae70713dee940140"
    assert (
        RECORDS[2]["df11"] == "mingyi456/ERNIE-Image-DF11@c2dd30ad7dd5a928df2309581b282337f7cdb41f"
    )
    assert RECORDS[2]["base"] == "baidu/ERNIE-Image@5346b31d68c9c23758ba56ef8be5e9dc174c7f99"
    for b, rec in RECORDS.items():
        assert "denoise" in rec["constants_from"]
        assert set(rec["constants_from"].values()) <= set(RUNS[b]), b
        assert rec["text_tokens"] == TEXT_TOKENS
    for phase in ("encode", "vae"):
        assert sum(phase in rec["constants_from"] for rec in RECORDS.values()) == 1, phase


def test_the_record_sizes_count_the_evaluated_language_model():
    # Bug caught: the records carrying every language-model tensor (6_858_012_672 B, the published header's sum) while the
    # code counts the evaluated layers (the encode term would come out negative: the runs held about 6.63e9 B), or the
    # other way round. 6_858_012_672 - the last layer's 232_796_160 = 6_625_216_512.
    assert {b: s.encoders for b, s in SIZES.items()} == {1: 6_625_216_512, 2: 6_625_216_512}


def test_the_overhead_is_the_larger_post_build_gap_of_the_two_runs():
    # Bug caught: the overhead taken from one run (the base's 548_194_818 would put every phase of the Turbo run
    # 68_239_360 B low), or computed another way than footprint - MLX active - MLX cache after the build.
    gaps = {
        label: run["after_build"]["footprint"]
        - run["after_build"]["mlx_active"]
        - run["after_build"]["mlx_cache"]
        for b in RUNS
        for label, run in RUNS[b].items()
        if "after_build"
        in run  # the one-step calibration runs; the later samples record a VAE peak only
    }
    assert gaps == {"derisk-turbo-1": 616_434_178, "derisk-base-1": 548_194_818}
    assert RECORDS[1]["overhead_from"] == "derisk-turbo-1"
    assert {b: c.overhead_bytes for b, c in emem.CONSTANTS.items()} == {
        1: 616_434_178,
        2: 616_434_178,
    }


@pytest.mark.parametrize("batch", [1, 2])
def test_the_estimate_at_1024_equals_each_batchs_calibration_run_phase_for_phase(batch):
    # Bug caught: a term dropped from the denoise phase or counted twice, REFERENCE_TOKENS left at another prompt's
    # count (the activation scaled off the calibration point), the batch-1 activation used for batch 2 (the base's
    # denoise 387_898_944 B low), or the denoise term derived at another cache limit than the planner's.
    # The encode phase (one term for both batches): the evaluated language model 6_625_216_512 + the encode term
    # 93_963_846 + overhead 616_434_178 = 7_335_614_536, the Turbo run's encode footprint (the term is taken by
    # footprint, so the encode phase never under-predicts a calibration run; MLX's own measure gave 29_933_258).
    fit = _fit(batch)
    assert fit.peak_phase == "vae"
    assert fit.phases["denoise"] == MEASURED[batch]["denoise"]
    assert fit.phases["encode"] == 7_335_614_536 == MEASURED[1]["encode"]
    assert CACHE_LIMITS[batch] == 436_207_616 + ALLOWANCE[batch]


def test_the_vae_term_covers_every_1024_sample_and_equals_the_highest():
    # Bug caught: the VAE transient derived from one run instead of the highest sample (the base's calibration sample, with
    # its own sizes, is 35_821_552 B lower and puts the Turbo run's VAE phase that much low), or a sample dropped from a
    # record. Five samples: the two calibration runs, the two MEASURED rows and the base identity check's df11 side; the
    # highest is still the Turbo calibration run. One term serves both models, whose sets differ by 59_576 B, so the
    # estimate equals the sample only for the model whose record holds it (Turbo).
    assert len(VAE_SAMPLES) == 5
    for b in RUNS:
        fit = _fit(b)
        vae_peaks = [
            r["footprint_peaks"]["vae"] for r in RUNS[b].values() if "vae" in r["footprint_peaks"]
        ]
        assert all(peak <= fit.phases["vae"] for peak in vae_peaks), b
    source = next(b for b, rec in RECORDS.items() if "vae" in rec["constants_from"])
    assert source == 1
    assert _fit(1).phases["vae"] == MEASURED[1]["vae"] == max(VAE_SAMPLES.values())
    denoise_only = [r for r in RUNS[1].values() if set(r["footprint_peaks"]) == {"denoise"}]
    assert len(denoise_only) == 2  # the two CAPPED 24 GB runs: denoise samples, not VAE ones
    assert _fit(2).phases["vae"] == 19_163_020_656


@pytest.mark.parametrize(("batch", "limit"), [(1, 2_600_000_000), (2, 3_600_000_000)])
def test_the_planners_1024_cache_limit_is_one_block_plus_the_batchs_allowance(batch, limit):
    # Bug caught: the shared 1.5e9 allowance, two blocks budgeted for a one-kind family, or the batch-1 allowance at
    # CFG (the probe released the decoded buffer below 3.6e9 at batch 2). 436_207_616 + 2_163_792_384 / 3_163_792_384.
    assert _plan(batch, 24_653_119_488).cache_limit == limit


@pytest.mark.parametrize(("batch", "allowance"), [(1, 2_163_792_384), (2, 3_163_792_384)])
def test_a_cache_limit_override_under_the_one_kind_minimum_warns_for_both_policies(
    batch, allowance, caplog
):
    # Bug caught: the depth2 minimum missing its look-ahead group, or the per-block minimum without the allowance.
    per_block = 436_207_616 + allowance
    depth2 = 872_415_232 + allowance
    with caplog.at_level("WARNING", logger="test"):
        _plan(batch, 24_653_119_488, cache_limit_override=per_block)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="test"):
        _plan(batch, 24_653_119_488, cache_limit_override=per_block - 1)
    assert f"below the derived minimum {per_block}" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="test"):
        assert _plan(batch, 24_653_119_488, policy="depth2").cache_limit == depth2
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="test"):
        _plan(batch, 24_653_119_488, policy="depth2", cache_limit_override=per_block)
    assert f"below the derived minimum {depth2}" in caplog.text
    assert "twice, for depth2's look-ahead" in caplog.text


@pytest.mark.parametrize(("batch", "vae"), [(1, 19_163_080_232), (2, 19_163_020_656)])
def test_a_32_gb_mac_keeps_the_set_through_the_vae_decode(monkeypatch, batch, vae):
    # The shared rule drops the set before the VAE decode iff the VAE phase with the set is over the budget: the
    # recommended working set minus the 2 GiB reserve, 24_653_119_488 B here. The VAE phase (17.85 GiB for either batch) is
    # 5_490_039_256 / 5_490_098_832 B (5.11 GiB) under it. Bug caught: the decision inverted, or a margin added on top of the
    # reserve.
    monkeypatch.setattr(
        imem.mx, "device_info", lambda: {"max_recommended_working_set_size": RECOMMENDED_32}
    )
    budget = imem.budget_bytes()
    assert budget == 24_653_119_488
    plan = _plan(batch, budget)
    assert plan.drop_set_before_vae is False
    assert (plan.estimate.peak_phase, plan.estimate.peak_bytes) == ("vae", vae)
    # The boundary: one byte less of budget than the VAE phase drops the set (`>`, not `>=`, at the budget).
    assert _plan(batch, vae, fit_check=False).drop_set_before_vae is False
    assert _plan(batch, vae - 1, fit_check=False).drop_set_before_vae is True


# --- the CAPPED tiers at 1024², the 26-token prompt, each model's default call (Turbo batch 1, base batch 2) ----------
# The fit budget a real Mac of that size applies: its recommended working set (2/3 of its RAM) minus the 2 GiB reserve.
# 24 GB: 15_032_385_536 B (14.00 GiB); 16 GB: 9_305_762_474 B (8.67 GiB). The VAE phase with the set (19.16e9 B) is
# over both, so the set is dropped before the decode; the dropped-set VAE phase keeps the extras (they stay on the
# transformer): extras 27_961_600 + VAE file 168_120_878 + transient 7_120_209_073 + overhead 616_434_178 =
# 7_932_725_729 B. The encode phase is 7_335_614_536 B. The denoise phase decides:
# - Turbo (batch 1), denoise 15_446_156_600 B (14.39 GiB, the higher of two CAPPED 24 GB runs' peaks, by the
#   overall-peak rule): over the 24 GB budget by 413_771_064 B (0.39 GiB) and over the 16 GB budget: refused in the
#   denoise phase at both.
# - Base (batch 2), denoise 16_361_874_888 B (15.24 GiB): over both budgets (24 GB by 1_329_489_352 B): refused in the
#   denoise phase.


def _tier_budget(tier):
    return tier_limits(
        tier, host_ram_bytes=RAM_32, host_recommended_bytes=RECOMMENDED_32
    ).fit_budget_bytes


@pytest.mark.parametrize(
    ("batch", "size", "expected"),
    [(2, 1024, 450_976_567), (2, 512, 402_653_184), (1, 512, 136_331_547)],
)
def test_the_batch_2_denoise_term_is_floored_by_the_float32_weight_copies(batch, size, expected):
    # Bug caught: a small CFG call predicted below the float32 copies a batch-2 step holds whatever the image size
    # (a float32 hidden state times a BF16 weight makes MLX copy the weight to float32: gate_proj and up_proj, 2 x
    # 12288 x 4096 x 4 = 402_653_184 B), or the floor changing the 1024² term it was calibrated on. 512² + 26 tokens:
    # 450_976_567 x 1050 / 4122 = 114_877_582 scaled, floored; batch 1 carries no floor (not measured below 1024²):
    # 535_198_703 x 1050 / 4122 = 136_331_547.
    from mlx_dfloat.mflux._phases import denoise_activation_bytes

    c = emem.CONSTANTS[batch]
    assert denoise_activation_bytes(c, height=size, width=size, text_tokens=TEXT_TOKENS) == expected


def test_the_tier_budgets_are_a_real_macs():
    # Bug caught: a tier judged against its watchdog ceiling or the host's budget instead of the fit budget.
    assert (_tier_budget(16), _tier_budget(24)) == (9_305_762_474, 15_032_385_536)


def test_turbo_at_24_gb_is_refused_0_39_gib_over_the_budget():
    # Bug caught: the batch-1 denoise term taken from one run instead of the highest denoise sample (the calibration
    # run's passed the call 58 MB under the budget; the first CAPPED run's, 37 MB over it; the second identical run
    # peaked 0.35 GiB higher still), or the dropped-set VAE phase without the extras.
    forced = _plan(1, _tier_budget(24), fit_check=False)
    assert forced.estimate.peak_bytes - _tier_budget(24) == 413_771_064


@pytest.mark.parametrize(("batch", "tier"), [(1, 16), (1, 24), (2, 16), (2, 24)])
def test_every_capped_call_is_refused_in_the_denoise_phase(batch, tier):
    # Bug caught: the refusal naming the VAE phase (the set drop not applied), or a call that cannot hold the
    # compressed set and one step planned to fit.
    budget = _tier_budget(tier)
    with pytest.raises(DFloatResourceError, match="in the denoise phase"):
        _plan(batch, budget)
    forced = _plan(batch, budget, fit_check=False)
    assert forced.drop_set_before_vae is True
    assert forced.estimate.peak_phase == "denoise"
    assert forced.estimate.peak_bytes == {1: 15_446_156_600, 2: 16_361_874_888}[batch]
    assert forced.estimate.phases["vae"] == 7_932_725_729


# What the prompt encode may hold before the model warns (model._check_encode_peak): MLX's encode peak minus what was
# active when the phase began, against the encode phase without its overhead (evaluated language model + encode term)
# + the slack. The encode term is per prompt, so the bound does not depend on the batch or on the prompt length.
NOT_LOADED = (
    806_610_944 + 33_556_480
)  # the vision tower and the projector in the encoder's file (its published header)


def _encode_bound(batch, text_tokens=TEXT_TOKENS):
    return emem.encode_peak_bound(
        _fit(batch, text_tokens=text_tokens).phases["encode"], emem.CONSTANTS[batch]
    )


@pytest.mark.parametrize(("batch", "spare"), [(1, 358_210_370), (2, 332_466_044)])
def test_the_encode_bound_holds_each_measured_encode_with_its_spare(batch, spare):
    # Bug caught: a bound under what a normal encode holds (every call would warn). The bound is the evaluated
    # language model 6_625_216_512 + the footprint encode term 93_963_846 + the 256 MiB slack = 6_987_615_814; the
    # runs held 6_629_405_444 (Turbo) and 6_655_149_770 (base) by MLX's measure.
    run = RUNS[batch][next(iter(RUNS[batch]))]
    assert _encode_bound(batch) - run["mlx_encode_held"] == spare


@pytest.mark.parametrize("text_tokens", [1, TEXT_TOKENS, emem.TEXT_MAX_LENGTH])
def test_an_encode_that_also_loads_the_vision_tower_is_over_the_bound_at_any_prompt(text_tokens):
    # Bug caught: the encode bound carrying the denoise allowance (it grows with the prompt), or a slack wide enough
    # that an mflux change loading the vision tower and projector (+840_167_424 B) goes unnoticed. Over by
    # 6_655_149_770 + 840_167_424 - 6_987_615_814 = 507_701_380 at every prompt length, for the base run's hold.
    held = RUNS[2]["derisk-base-1"]["mlx_encode_held"]
    for batch in (1, 2):
        assert held + NOT_LOADED - _encode_bound(batch, text_tokens) == 507_701_380


# The later 1024² samples in the calibration records are the committed runs' own watched peaks (each reached in the VAE
# phase with the set resident), never hand-copied: the record of each run, and the calibration label it is filed under.
OUT_OF_SAMPLE = [
    ("bench/results/tiers/ernie-image-turbo-1024.json", 1, "measured-8-steps"),
    ("bench/results/tiers/ernie-image-1024.json", 2, "measured-50-steps"),
    ("bench/results/identity/ernie-image-1024/df11-result.json", 2, "identity-4-steps-cfg"),
]


def _record_run(path):
    """A committed record's report, its height and width, and its watched (footprint) peak."""
    record = json.loads((REPO / path).read_text())
    if (
        "report" in record
    ):  # verify_image's df11 side: the model's report under "report", the size in the key
        size = record["key"]["size"]
        return record["report"], size, size, record["footprint_peak_bytes"]
    return record, record["height"], record["width"], record["watched_peak_bytes"]


@pytest.mark.parametrize(("path", "batch", "label"), OUT_OF_SAMPLE)
def test_the_later_vae_samples_are_the_committed_runs_watched_peaks(path, batch, label):
    # Bug caught: a calibration sample that is not the committed run's peak (a typo, or a re-run whose record was not
    # carried over), so the overall-peak rule would be checked against a number no run measured.
    _report, _h, _w, watched = _record_run(path)
    assert RUNS[batch][label]["footprint_peaks"] == {"vae": watched}


# The committed 1024² records beyond the one-step calibration runs, each re-estimated with the current constants from
# its own report (its sizes, the cache limit in force, its text tokens, its policy, its CFG batch and its VAE
# strategy), so a re-run that changes the record is checked against the code as it stands. The MEASURED rows and the
# identity df11 side peak in the VAE decode with the set resident, whose 1024² estimate is the calibration value
# itself: they test that the step count and the CFG batch do not grow the 1024² peak. The Turbo 24 GB CAPPED row runs
# with the set dropped before the VAE and peaks in the denoise phase; the batch-1 denoise term is the highest of its
# runs' peaks (the overall-peak rule: two identical runs peaked 0.35 GiB apart), so a re-run of that row is checked
# against the estimate the earlier runs set. The check is
# one-sided: a phase varies by up to about 1 GiB run to run,
# and a run under the estimate is not the bug.
STEPS_AND_CFG_RECORDS = [
    pytest.param(
        "bench/results/tiers/ernie-image-turbo-1024.json", "vae", id="turbo-measured-8-steps"
    ),
    pytest.param(
        "bench/results/tiers/ernie-image-1024.json", "vae", id="base-measured-50-steps-cfg"
    ),
    pytest.param(
        "bench/results/identity/ernie-image-1024/df11-result.json",
        "vae",
        id="base-identity-df11-4-steps-cfg",
    ),
    pytest.param(
        "bench/results/tiers/ernie-image-turbo-1024-tier24.json", "denoise", id="turbo-capped-24-gb"
    ),
]


@pytest.mark.parametrize(("path", "phase"), STEPS_AND_CFG_RECORDS)
def test_a_committed_1024_record_peaks_within_the_current_estimate(path, phase):
    # Bug caught: a 1024² peak that grows with the step count or the CFG batch while the estimate stays at the one-step
    # value (a recorded run more than 0.25 GiB over the current prediction), the estimate peaking in another phase, or
    # a record checked against the wrong batch's constants. 1 GiB less of VAE transient turns the Turbo MEASURED row
    # and the identity side red (0.50 and 0.42 GiB over the lowered estimate); the base MEASURED row stays green (it sits
    # 1.05 GiB under the estimate, 0.05 GiB under the lowered one; the in-sample test pins the term) and so does the
    # CAPPED row (its peak is the denoise phase).
    report, height, width, watched = _record_run(path)
    c = emem.CONSTANTS[report["cfg_batch"]]
    tokens = report["text_tokens"]
    fit = fit_for(
        c,
        sizes=FamilySizes(**report["sizes"]),
        largest=LARGEST,
        policy=report["eval_policy"],
        cache_limit=report["cache_limit_in_force"],
        allowance=activation_allowance(c, height=height, width=width, text_tokens=tokens),
        budget=report["fit"]["budget_bytes"],
        height=height,
        width=width,
        text_tokens=tokens,
        vae_with_set=not report["drop_set_before_vae"],
    )
    assert fit.peak_phase == phase
    assert watched <= fit.peak_bytes + GIB // 4
