"""Qwen-Image 2.1's memory rules: the encoder's resident bytes, the text-token count, the sizes, the measured constants.

Measured inputs (2026-10-08, the first one-step de-risk run: Qwen-Image-2.1 DF11
(mingyi456/Qwen-Image-2.1-DF11-ComfyUI @ 1b22a3a) over the Qwen/Qwen-Image-2.1 base (@ d26bb61), 1024², seed 42,
guidance 4 with the negative prompt " " (two transformer calls per step), text tokens 33 (prompt) and 9 (negative);
one process: build, encode, drop, set load, one step, VAE decode on the resident set; M1 Max 32 GB, macOS 27.0.1,
mlx 0.32.2, mflux 0.20.0). Footprint peaks: encode 15_857_938_288, set load 10_347_487_632, denoise 13_526_704_528,
vae 22_139_643_024 (the call's peak). Sizes: compressed 9_587_103_379, extras 137_388_032, decoded non-block
(``modulation.1``) 134_217_728, the encoder's language model 15_136_811_008 (shards 17_534_339_488), VAE
1_350_989_512; largest decoded block 436_207_616; cache limit 1_859_346_402 (the run's; the planner now derives
2_731_761_634, see ``CACHE_LIMIT``). MLX's encode peak 15_350_204_544 with 146_825_736 active after the build: the
encode held 66_567_800 over the language model. The committed MEASURED record
``bench/results/tiers/qwen-image-2.1-1024.json`` carries the same five sizes (``sizes``), the 33 text tokens, the
2_731_761_634 cache limit and this Mac's working set and RAM (``limits.tier``).

The denoise term is re-derived at the planner's limit (2026-10-08, the first one-step de-risk run at that limit, in
the cache-limit A/B; same model, prompt, size, seed and guidance): denoise footprint peak 13_939_106_072 at the
2_731_761_634 limit. The VAE term covers the highest of nine 1024² VAE-phase footprint peaks (21.76-22.79 GB; MLX's VAE
peak was the same in every de-risk run): 22_788_580_400, from the second one-step run of the A/B at the old limit.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.bench.capped import tier_limits
from mlx_dfloat.errors import DFloatResourceError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate import memory as imem
from mlx_dfloat.mflux._phases import FamilySizes, activation_allowance, fit_for
from mlx_dfloat.mflux._pipeline import plan_call_for
from mlx_dfloat.mflux.qwen21 import memory as qmem

GIB = 1024**3
SIZES = FamilySizes(
    compressed=9_587_103_379,
    extras=137_388_032,
    nonblock=134_217_728,
    encoders=15_136_811_008,
    vae=1_350_989_512,
)
LARGEST = {"transformer_blocks": 436_207_616}  # one block kind; 32 groups of 218_103_808 elements
# The planner's 1024² limit: the largest decoded block 436_207_616 + Qwen's own allowance 2_295_554_018 at 4096 + 33
# tokens (the block-cache probe: a decoded buffer is reused at a 2605 MiB limit, released at 1773 and 2189 MiB).
CACHE_LIMIT = 2_731_761_634
# The de-risk run's limit: the block + the shared 1.5e9 allowance at 4096 + 256 tokens, rescaled to 4096 + 33.
CALIBRATION_CACHE_LIMIT = 1_859_346_402
TEXT_TOKENS = 33  # the de-risk prompt's (the negative " " is 9)
# The denoise peak is the first one-step A/B run's at CACHE_LIMIT; the VAE peak the highest of the nine 1024² samples
# (the second A/B run at the old limit); the encode peak the first de-risk run's.
MEASURED = {"encode": 15_857_938_288, "denoise": 13_939_106_072, "vae": 22_788_580_400}
# This Mac (M1 Max 32 GB): mx.device_info()["max_recommended_working_set_size"] and the RAM, from the de-risk record.
RECOMMENDED_32 = 26_800_603_136
RAM_32 = 34_359_738_368


def _bf16(*shape):
    return mx.zeros(shape, dtype=mx.bfloat16)


def _write_encoder(root):
    """A Qwen3-VL-shaped text encoder in two shards, the way the base stores it (``model.language_model.*``, the
    vision tower ``model.visual.*`` and an untied ``lm_head``).

    bf16 bytes: embed_tokens 10 x 4 = 80, up_proj 4 x 4 = 32, the vision tensor 8 x 4 = 64, lm_head 10 x 4 = 80.
    """
    enc = root / "text_encoder"
    enc.mkdir(parents=True)
    mx.save_safetensors(
        str(enc / "model-00001-of-00002.safetensors"),
        {
            "model.language_model.embed_tokens.weight": _bf16(10, 4),
            "model.visual.blocks.0.attn.qkv.weight": _bf16(8, 4),
        },
    )
    mx.save_safetensors(
        str(enc / "model-00002-of-00002.safetensors"),
        {
            "model.language_model.layers.0.mlp.up_proj.weight": _bf16(4, 4),
            "lm_head.weight": _bf16(10, 4),
        },
    )
    (enc / "model.safetensors.index.json").write_text("{}")


def test_encoder_bytes_count_the_language_model_only(tmp_path):
    # Bug caught: the encoder's file size counted (as for FLUX.1), or the vision tower or lm_head counted: mflux maps
    # model.language_model.* only (qwen21_weight_mapping.py:56-88), so the other tensors are never loaded. 80 + 32.
    _write_encoder(tmp_path)
    assert qmem.encoder_bytes_used(tmp_path) == 112


def test_sizes_for_replaces_the_encoder_file_size_with_the_language_model_bytes(tmp_path):
    # Bug caught: the sizes still carrying the shards' file size (vision tower and lm_head included: the encode phase
    # over-predicted), or the DF11 / non-block / VAE terms lost when the encoder term is replaced.
    rng = np.random.default_rng(0)
    write_checkpoint(
        tmp_path / "df11",
        groups={
            "transformer_blocks.0": [random_bf16(rng, (8, 4))],
            "modulation.1": [random_bf16(rng, (16, 4))],
        },
        patterns={r"transformer_blocks\.\d+": ("attn.to_q",), r"modulation\.1": ()},
        extras={"img_in.weight": np.zeros((3, 5), dtype=np.uint16)},
        single_file=True,
    )
    _write_encoder(tmp_path / "base")
    (tmp_path / "base" / "vae").mkdir()
    (tmp_path / "base" / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    sizes = qmem.sizes_for(open_checkpoint(tmp_path / "df11"), tmp_path / "base")
    total = (tmp_path / "df11" / "model.safetensors").stat().st_size
    assert sizes.encoders == 112
    assert sizes.extras == 30  # 3 x 5 BF16
    assert sizes.compressed == total - 30
    assert sizes.nonblock == 128  # modulation.1 decoded: 16 x 4 BF16
    assert sizes.vae == 100


class _StubTokenizer:
    """``tokenize(p)`` gives ``len(p) + 5`` ids (a 5-token system prefix around the prompt, no truncation); the
    wrapped tokenizer encodes the system prefix to 5 ids, as ``Qwen21PromptEncoder._system_prefix_length`` asks it."""

    def __init__(self):
        self.tokenizer = lambda text, add_special_tokens=True: {"input_ids": [0] * 5}

    def tokenize(self, prompt):
        return SimpleNamespace(input_ids=mx.zeros((1, len(prompt) + 5), dtype=mx.int32))


@pytest.mark.mflux
@pytest.mark.parametrize(
    ("prompt", "tokens"),
    [
        ("abc", 3),
        ("", 1),  # mflux encodes a blank prompt as " " (qwen21_prompt_encoder.py:24-25)
        ("   ", 1),
        ("x" * 2042, 2042),  # 2047 ids in all: under the cap
        ("x" * 2043, 2043),  # 2048: on it
        (
            "x" * 2044,
            2043,
        ),  # 2049 would be cut to 2048 (max_length 2048, truncation, tokenizer.py:98-103)
    ],
    ids=["abc", "empty", "blank", "under-cap", "on-cap", "over-cap"],
)
def test_prompt_tokens_drop_the_system_prefix_count_a_blank_as_a_space_and_cap_at_2048(
    prompt, tokens
):
    # Bug caught: the system prefix counted (mflux drops it: qwen21_prompt_encoder.py:28-31), "" counted as 0 tokens
    # (the allowance computed for a prompt mflux does not encode), or a long prompt counted past the tokenizer's
    # 2048-token cut (`>` / `>=` at the cap, or no cap: the activation term over-predicted).
    assert qmem.prompt_tokens(_StubTokenizer(), prompt) == tokens


@pytest.mark.mflux
def test_the_token_cap_is_mfluxs_tokenizer_max_length():
    # Bug caught: an mflux bump changing the tokenizer's max_length (qwen21_weight_definition.py:45) while the cap
    # here keeps 2048.
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    (definition,) = Qwen21WeightDefinition.get_tokenizers()
    assert definition.max_length == 2048
    assert qmem.TEXT_MAX_LENGTH == 2048


# The de-risk prompt (the run record's env.sh, $PROMPT): 33 text tokens by the real tokenizer.
DERISK_PROMPT = (
    "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the "
    "water"
)


@pytest.mark.mflux
@pytest.mark.network
@pytest.mark.parametrize(
    ("prompt", "tokens"),
    [(" ", 9), ("", 9), (DERISK_PROMPT, 33)],
    ids=["space", "empty", "derisk-prompt"],
)
def test_the_real_tokenizer_counts_the_text_tokens_the_calibration_used(prompt, tokens):
    # Bug caught: prompt_tokens disagreeing with the real tokenizer where the constants were calibrated (33 for the
    # de-risk prompt, 9 for its negative " "): the template or the system prefix miscounted, or "" not encoded as " "
    # (the stub tests cannot see either). Needs the base's processor/ files (a few MB, CPU only).
    base = os.environ.get("MLX_DFLOAT_QWEN21_BASE")
    if not base:
        pytest.skip("MLX_DFLOAT_QWEN21_BASE is not set (a Qwen-Image 2.1 snapshot with processor/)")
    from mlx_dfloat.mflux.qwen21.init import load_tokenizers

    tokenizer = load_tokenizers(Path(base))["qwen21"]
    assert qmem.prompt_tokens(tokenizer, prompt) == tokens


def _fit(*, text_tokens=TEXT_TOKENS, budget=24_653_119_488):
    c = qmem.CONSTANTS
    return fit_for(
        c,
        sizes=SIZES,
        largest=LARGEST,
        policy="per-block",
        cache_limit=CACHE_LIMIT,
        allowance=activation_allowance(c, height=1024, width=1024, text_tokens=text_tokens),
        budget=budget,
        height=1024,
        width=1024,
        text_tokens=text_tokens,
    )


def _plan(budget, *, fit_check=True, cache_limit_override=None):
    return plan_call_for(
        constants=qmem.CONSTANTS,
        sizes=SIZES,
        largest=LARGEST,
        policy="per-block",
        cache_limit_override=cache_limit_override,
        fit_check=fit_check,
        budget=budget,
        height=1024,
        width=1024,
        text_tokens=TEXT_TOKENS,
        log=logging.getLogger("test"),
    )


# The constants are calibrated on the de-risk run, so at its working point the estimate equals the measurement by
# construction: the in-sample checks are exact. Whether the model predicts is decided by runs not used for calibration
# (OUT_OF_SAMPLE below).


def test_the_estimate_at_1024_equals_the_calibration_run_phase_for_phase():
    # Bug caught: a term dropped from a phase or counted twice; the VAE transient taken from one run instead of the
    # highest sample (the first de-risk's 22_139_643_024 puts the VAE phase 648_937_376 B under it);
    # REFERENCE_TOKENS left at another prompt's count (the activation scaled off the calibration point, e.g. 4096 + 512
    # puts denoise 94_905_646 B low); or a borrowed family's constants (Klein 9B's put the VAE phase 4.2 GiB low).
    # The encode phase: the language model 15_136_811_008 + the measured encode term 66_567_800 + overhead
    # 459_445_098 (by footprint the run's encode peaked 195_114_382 B higher: memory MLX does not count, which the
    # overhead measured after the build does not cover).
    # The denoise phase is calibrated at the planner's own limit (CACHE_LIMIT), so a cache-limit term taken from
    # another limit (the de-risk's 1_859_346_402 puts it 872_415_232 B low) also shows here.
    fit = _fit()
    assert fit.peak_phase == "vae"
    assert fit.phases["vae"] == MEASURED["vae"]
    assert fit.phases["encode"] == 15_662_823_906
    assert fit.phases["denoise"] == MEASURED["denoise"]


def test_the_planners_1024_cache_limit_is_one_block_plus_qwens_own_allowance():
    # Bug caught: Qwen planned on the shared 1.5e9 allowance (the de-risk's 1_859_346_402: the probe released the
    # decoded buffer at that limit, so every block's output was allocated fresh), two blocks budgeted for the one
    # block kind (3_167_969_250), or the allowance's reference tokens left at the shared 4096 + 256 (2_659_262_210).
    assert _plan(24_653_119_488).cache_limit == CACHE_LIMIT


def test_a_cache_limit_override_under_qwens_derived_minimum_warns(caplog):
    # Bug caught: an override of one block or more passing silently for this one-kind family (the de-risk's own
    # 1_859_346_402 would have): the minimum is the block + the allowance, the planner's own limit.
    with caplog.at_level("WARNING", logger="test"):
        _plan(24_653_119_488, cache_limit_override=CACHE_LIMIT)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="test"):
        _plan(24_653_119_488, cache_limit_override=CALIBRATION_CACHE_LIMIT)
    assert "below the derived minimum 2731761634" in caplog.text


def test_a_32_gb_mac_keeps_the_set_through_the_vae_decode(monkeypatch):
    # The shared rule (_pipeline.plan_call_for) drops the set before the VAE decode iff the VAE phase with the set is
    # over the budget, with no margin beyond the budget's own: the recommended working set minus a 2 GiB reserve
    # (integrate.memory.budget_bytes), 26_800_603_136 - 2_147_483_648 = 24_653_119_488 B here. The measured VAE peak,
    # 22_788_580_400 B (the estimate's by calibration), is 1_864_539_088 B (1.74 GiB) under it: the set stays resident,
    # and a 32 GB Mac pays no set reload per call. Bug caught: the decision inverted, a margin added on top of the
    # reserve that flips it (more than 1.74 GiB), or the budget read without the reserve.
    monkeypatch.setattr(
        imem.mx, "device_info", lambda: {"max_recommended_working_set_size": RECOMMENDED_32}
    )
    budget = imem.budget_bytes()
    assert budget == 24_653_119_488
    plan = _plan(budget)
    assert plan.drop_set_before_vae is False
    assert plan.estimate.peak_phase == "vae"
    assert plan.estimate.peak_bytes == MEASURED["vae"]
    # The boundary: one byte less of budget than the VAE phase drops the set (`>`, not `>=`, at the budget).
    assert _plan(MEASURED["vae"]).drop_set_before_vae is False
    assert _plan(MEASURED["vae"] - 1).drop_set_before_vae is True


@pytest.mark.parametrize("tier", [16, 24])
def test_a_16_or_24_gb_tier_is_refused_in_the_encode_phase(tier):
    # The prediction for the CAPPED tiers at 1024² (the de-risk prompt, 33 tokens): the VAE phase with the set
    # (22_788_580_400 B) is over either tier's budget, so the set is dropped before the decode: the VAE phase is then
    # the extras (they stay on the transformer) + the VAE + its transient + overhead, 137_388_032 + 1_350_989_512 +
    # 11_119_436_651 + 459_445_098 = 13_067_259_293 B; the encode phase (the language model 15_136_811_008 + the encode term 66_567_800 + overhead 459_445_098 =
    # 15_662_823_906 B, 14.59 GiB) is then the peak and over both fit budgets. The fit budget is what a real Mac of
    # that size applies (generate passes limits.fit_budget_bytes): the tier's recommended working set (2/3 of its RAM)
    # minus the fit rule's 2 GiB reserve. 24 GB: 17_179_869_184 - 2_147_483_648 = 15_032_385_536 B (14.00 GiB), over by
    # 630_438_370 B; 16 GB: 11_453_246_122 - 2_147_483_648 = 9_305_762_474 B (8.67 GiB). The denoise phase
    # (13_939_106_072 B, 12.98 GiB) is under the 24 GB fit budget. Bug caught: the encode phase sized by less than the
    # language model (a call that cannot hold the encoder planned to fit), or the drop not applied (the refusal would
    # name the VAE phase).
    budget = tier_limits(
        tier, host_ram_bytes=RAM_32, host_recommended_bytes=RECOMMENDED_32
    ).fit_budget_bytes
    assert budget == {16: 9_305_762_474, 24: 15_032_385_536}[tier]
    with pytest.raises(DFloatResourceError, match="in the encode phase"):
        _plan(budget)
    forced = _plan(budget, fit_check=False)
    assert forced.drop_set_before_vae is True
    assert forced.estimate.peak_phase == "encode"
    assert forced.estimate.peak_bytes == 15_662_823_906
    assert forced.estimate.phases["vae"] == 13_067_259_293


# What the prompt encode may hold before the model warns (model._check_encode_peak): MLX's encode peak minus what was
# active when the phase began, against the encode phase without its overhead (language model + encode term) + the slack.
ENCODE_HELD = 15_350_204_544 - 146_825_736  # de-risk: MLX encode peak - MLX active after the build
# The tensors in the encoder's shards mflux does not load (the vision tower and the lm_head), counted as the shards'
# file size minus the language model's tensor bytes (the headers' few KB included: an upper bound by those bytes).
NOT_LOADED = 17_534_339_488 - 15_136_811_008


def _encode_bound(text_tokens):
    return qmem.encode_peak_bound(_fit(text_tokens=text_tokens).phases["encode"])


def test_the_encode_warning_bound_holds_the_measured_encode_with_the_slack_to_spare():
    # Bug caught: a bound under what a normal encode holds (every call would warn): the de-risk encode held
    # 66_567_800 B over the language model, exactly the encode term, so the whole 256 MiB slack is left.
    assert _encode_bound(TEXT_TOKENS) - ENCODE_HELD == 268_435_456


@pytest.mark.parametrize("text_tokens", [9, TEXT_TOKENS, qmem.TEXT_MAX_LENGTH])
def test_an_encode_that_also_loads_the_vision_tower_and_lm_head_is_over_the_bound_by_2_gb_at_any_prompt(
    text_tokens,
):
    # Bug caught: the encode bound carrying the denoise allowance again (it grows with the prompt: at 2048 tokens the
    # vision-loaded encode was over it by only 78_013_766 B), or a slack wide enough that an mflux change loading the
    # vision tower and lm_head (+2_397_528_480 B) goes unnoticed. The excess is the same at every prompt length:
    # 2_397_528_480 - 268_435_456.
    assert ENCODE_HELD + NOT_LOADED - _encode_bound(text_tokens) == 2_129_093_024


def test_the_encode_warning_is_silent_at_the_bound_and_names_the_prompt_length_over_it():
    # Bug caught: `>=` for `>` at the bound (a normal encode warning), or a warning that blames mflux's loading alone
    # when the prompt is longer than the one the encode term was measured on (33 tokens; a 2048-token negative prompt
    # may hold more activations, which the term does not scale for).
    bound = _encode_bound(TEXT_TOKENS)
    assert qmem.encode_peak_warning(bound, bound, text_tokens=2048) is None
    message = qmem.encode_peak_warning(bound + 1, bound, text_tokens=2048)
    assert message is not None
    assert "for a 2048-token prompt" in message
    assert "measured on one 33-token prompt" in message
    assert "more of the encoder than its language model" in message


# Watched (footprint) peaks of two 1024² runs with more steps than the one-step calibration runs, both on the 33-token
# de-risk prompt, at the planner's cache limit (2026-10-08): the MEASURED row (`mlx-dfloat generate
# --tier 32`, 40 steps, no CFG) and the identity check's df11 side (4 steps, guidance 4 with CFG); the values are the
# `footprint_peak_bytes` of the committed `bench/results/tiers/qwen-image-2.1-1024.json` and
# `bench/results/identity/qwen-image-2.1-1024/df11-result.json`. Both peak in the VAE
# decode with the set resident, whose estimate at 1024² is the calibration value itself (the VAE transient is floored
# there), so they test that the step count and guidance do not grow the 1024² peak, not any scaling term: no run at
# another size or prompt length is measured yet. Both are among the nine samples the VAE term covers, so the check is
# one-sided: the VAE phase varies by about 1 GiB run to run, and a run under the estimate is not the bug.
STEPS_AND_CFG_RUNS = [
    pytest.param(21_832_147_968, id="measured-40-steps"),
    pytest.param(22_065_292_384, id="identity-df11-4-steps-cfg"),
]


@pytest.mark.parametrize("measured", STEPS_AND_CFG_RUNS)
def test_the_1024_peak_does_not_grow_with_steps_or_guidance_beyond_the_calibrated_vae_phase(
    measured,
):
    # Bug caught: a 1024² peak that grows with the step count or with classifier-free guidance (a second transformer
    # call per step holding its activations into the VAE decode) while the estimate stays at the one-step value: a run
    # more than 0.25 GiB over the prediction, or the estimate peaking in another phase. Also red when the VAE term
    # under-predicts these runs: a one-step transient (9_860_776_704) puts the identity run 0.50 GiB over.
    fit = _fit()
    assert fit.peak_phase == "vae"
    assert measured <= fit.peak_bytes + GIB // 4
