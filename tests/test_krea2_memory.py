"""Krea 2's memory rules: the encoder's resident bytes, the text-token count, the sizes, the encode warning, and the
constants measured at 1024² against the committed calibration records."""

import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_dfloat.bench.capped import tier_limits
from mlx_dfloat.errors import DFloatResourceError
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants, activation_allowance, fit_for
from mlx_dfloat.mflux._pipeline import plan_call_for
from mlx_dfloat.mflux.krea2 import memory as kmem

# The calibration prompt's text tokens (mflux 0.20.0 TokenizerLoader on both base snapshots, 2026-10-08: 64 ids, the
# chat-template prefix ends at 34).
DERISK_TOKENS = 30


def _bf16(*shape):
    return mx.zeros(shape, dtype=mx.bfloat16)


def _write_encoder(root, *, prefix="language_model."):
    """A Krea 2 text encoder file the way the base stores it: the language model under ``language_model.`` (two
    numbered layers) and the vision tower beside it under ``visual.`` (the real file's prefixes, 2026-10-08 header).

    bf16 bytes: embed_tokens 10 x 4 = 80, one up_proj 4 x 4 = 32 per layer, the final norm (4,) = 8, the vision tensor
    8 x 4 = 64.
    """
    enc = root / "text_encoder"
    enc.mkdir(parents=True)
    mx.save_safetensors(
        str(enc / "model.safetensors"),
        {
            f"{prefix}embed_tokens.weight": _bf16(10, 4),
            f"{prefix}layers.0.mlp.up_proj.weight": _bf16(4, 4),
            f"{prefix}layers.1.mlp.up_proj.weight": _bf16(4, 4),
            f"{prefix}norm.weight": _bf16(4),
            "visual.blocks.0.attn.qkv.weight": _bf16(8, 4),
        },
    )


@pytest.mark.parametrize("prefix", ["language_model.", "model.language_model."])
def test_encoder_bytes_count_the_evaluated_language_model_only(tmp_path, prefix):
    # Bug caught: the vision tower counted (+64 here, +0.77 GiB on the real file), or the last layer counted although
    # the deepest tap (hidden state 35 = the output of layer 34) means layer 35 never runs (+32 here, +0.19 GiB real).
    # Both of mflux's prefixes (krea2_weight_definition.py:14) are the language model. By hand: 80 + 32 + 8 = 120.
    _write_encoder(tmp_path, prefix=prefix)
    assert kmem.encoder_bytes_used(tmp_path) == 120
    # On the published header the same rule gives 8_044_936_192 - layer 35's 201_861_632 = 7_843_074_560 B, the
    # encoder bytes the calibration records carry (test_the_record_sizes_charge_no_resident_nonblock_in_the_shipped_mode).


class _Tokenizer:
    """mflux's tokenizer protocol as the prompt encoder uses it: ``tokenize(prompt).input_ids`` of shape (1, L)."""

    def __init__(self, ids=None, lengths=None):
        self.ids = ids
        self.lengths = dict(lengths or {})

    def tokenize(self, prompt):
        ids = self.ids if self.ids is not None else [list(range(1, self.lengths[prompt] + 1))]
        return SimpleNamespace(
            input_ids=mx.array(ids, dtype=mx.int32), attention_mask=mx.ones((1, len(ids[0])))
        )


@pytest.mark.mflux
def test_prompt_tokens_are_the_ids_after_the_template_prefix():
    # Bug caught: the template prefix counted (the plan sized by about 35 extra tokens) or the attention mask counted
    # instead. By hand from text_encoder.py:117-127: the second <|im_start|> (151644) is at 3, followed by "user"
    # (872) and a newline (198), so the prefix ends at 6 and 3 ids remain; without the template nothing is stripped.
    templated = _Tokenizer(ids=[[151644, 9, 10, 151644, 872, 198, 5, 6, 7]])
    assert kmem.prompt_tokens(templated, "p") == 3
    assert kmem.prompt_tokens(_Tokenizer(ids=[[1, 2, 3]]), "p") == 3


@pytest.mark.mflux
def test_a_cfg_step_is_sized_by_its_longer_prompt():
    # Bug caught: a CFG step sized by its first prompt (the negative " " first would give 6) or by the sum; the two
    # calls run one after the other, so the step's activations are the longer call's.
    tokenizer = _Tokenizer(lengths={" ": 6, "long": 40})
    assert kmem.text_tokens_for(tokenizer, [" ", "long"]) == 40
    assert kmem.text_tokens_for(tokenizer, ["long", " "]) == 40


def test_the_encode_bound_is_the_phase_without_overhead_plus_the_slack():
    # Bug caught: the overhead kept in the bound, or the slack missing. 10e9 - 0.4e9 + 268_435_456, by hand.
    c = PhaseConstants(
        overhead_bytes=400_000_000,
        vae_transient_bytes=0,
        denoise_activation_at_reference=0,
        reference_tokens=1,
    )
    assert kmem.encode_peak_bound(10_000_000_000, c) == 9_868_435_456


def test_the_encode_warning_is_silent_at_the_bound_and_names_the_prompt_length_over_it():
    # Bug caught: `>=` for `>` at the bound (a normal encode warning), or a warning that blames mflux's loading alone
    # when the prompt is longer than the one the encode term was measured on (30 tokens).
    bound = 8_000_000_000
    assert kmem.encode_peak_warning(bound, bound, text_tokens=1024) is None
    message = kmem.encode_peak_warning(bound + 1, bound, text_tokens=1024)
    assert message is not None
    assert "for a 1024-token prompt" in message
    assert f"measured on one {DERISK_TOKENS}-token prompt" in message
    assert "the vision tower stored in the text encoder's file" in message


def test_the_sizes_of_the_published_raw_file_are_the_arithmetic_tables(tmp_path):
    # Bug caught: a non-block group left out of the resident set (the four text-fusion blocks are 0.64 GiB of the
    # 1.234), a block counted as non-block, the extras counted in the compressed set, or the encoder counted by file
    # size. Literals from the published header's matrix shapes: non-block 4 x 171_704_320 + 78_643_200 + 452_984_832 +
    # 106_954_752 = 1_325_400_064; extras 4_560_024 (header sum); compressed 17_545_490_584 - 4_560_024.
    from mlx_dfloat._layouts import KREA_2_RAW_COMFYUI
    from mlx_dfloat.format import open_checkpoint

    fixture = Path(__file__).parent / "fixtures" / "krea-2-raw-df11-header.bin"
    df11 = tmp_path / "df11"
    df11.mkdir()
    with (df11 / "model.safetensors").open(
        "wb"
    ) as fh:  # sparse: the header, then zeros to the file size
        fh.write(fixture.read_bytes())
        fh.truncate(17_545_490_584)
    zero = hashlib.sha256(bytes(4096)).hexdigest()  # the sparse file reads as zeros at every probe
    layout = dataclasses.replace(
        KREA_2_RAW_COMFYUI,
        probes=tuple(dataclasses.replace(p, sha256=zero) for p in KREA_2_RAW_COMFYUI.probes),
    )
    ckpt = open_checkpoint(df11, layouts=(layout,))
    base = tmp_path / "base"
    _write_encoder(base)
    (base / "vae").mkdir()
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 1000)
    sizes = kmem.sizes_for(ckpt, base)
    assert sizes.nonblock == 1_325_400_064
    assert sizes.extras == 4_560_024
    assert sizes.compressed == 17_540_930_560
    assert sizes.encoders == 120
    assert sizes.vae == 1000
    # Per-call decode (the shipped mode): no non-block weights stay decoded with the set. Bug caught:
    # the per-call path still charging 1.234 GiB, so the fit check refuses calls the measured runs passed.
    per_call = kmem.sizes_for(ckpt, base, nonblock_per_call=True)
    assert per_call.nonblock == 0
    assert (per_call.compressed, per_call.extras, per_call.encoders) == (
        17_540_930_560,
        4_560_024,
        120,
    )


# --- the measured constants, against the committed calibration records -----------------------------------------
# Measured 2026-10-08 (M1 Max 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0), the 30-token lighthouse prompt, 1024²,
# seed 42, at the planner's limit for the block-cache probe's allowance (4_500_000_000 B), the set dropped before the
# VAE decode. The shipped mechanism (per-call non-block decode, MLX's cache emptied and drained at each call's start
# and before block 0, git c555f50): Krea 2 Raw, one step at guidance 3.5 with mflux's " " negative (two batch-1 calls,
# derisk-raw-2), and Krea 2 Turbo, two steps at guidance 1.0 (derisk-turbo-2). Also recorded: Raw with the non-block
# groups resident (derisk-raw-1, git d1aa9df; a VAE sample), and four runs of superseded per-call code (history-*),
# never a constant's source. The tests read them from the records.

GIB = 1024**3
REPO = Path(__file__).resolve().parents[1]
RECORDS = {
    "krea-2-raw": json.loads((REPO / "bench/results/calibration/krea-2-raw-1024.json").read_text()),
    "krea-2": json.loads((REPO / "bench/results/calibration/krea-2-1024.json").read_text()),
}
RUNS = {m: {run["label"]: run for run in rec["runs"]} for m, rec in RECORDS.items()}
ALL_RUNS = {label: run for runs in RUNS.values() for label, run in runs.items()}
# The samples the constants rest on: the shipped mechanism's runs (every term), plus the resident run for the VAE term
# (its VAE phase ran with the set dropped, like every mechanism's).
SHIPPED = {label: run for label, run in ALL_RUNS.items() if run["mechanism"].startswith("shipped")}
# The one-process calibration runs: they alone record every phase's peak, the post-build counters and the encode hold.
DERISK = {label: run for label, run in ALL_RUNS.items() if "after_build" in run}
SHIPPED_DERISK = {label: run for label, run in SHIPPED.items() if label in DERISK}
VAE_RUNS = {**SHIPPED, "derisk-raw-1": ALL_RUNS["derisk-raw-1"]}
# Which run each term came from (one set of constants for both models, each phase named exactly once).
SOURCE = {
    phase: label for rec in RECORDS.values() for phase, label in rec["constants_from"].items()
}
# The planner's sizes: the shipped mode (per-call), so no decoded non-block bytes stay with the set.
SIZES = {
    m: FamilySizes(**RUNS[m][f"derisk-{'raw' if m == 'krea-2-raw' else 'turbo'}-2"]["sizes"])
    for m in RECORDS
}
LARGEST = {
    "blocks": 868_220_928
}  # one block kind: 434_110_464 elements (the header's eight block matrix shapes)
TEXT_TOKENS = 30  # the calibration prompt's (the negative " " is 6)
ALLOWANCE = (
    3_631_779_072  # the block-cache probe's, at 4096 + 30 tokens (limit 4_500_000_000 - one block)
)
BUDGET_32 = 24_653_119_488  # this Mac's fit budget: the recommended working set 26_800_603_136 - the 2 GiB reserve
RECOMMENDED_32 = 26_800_603_136
RAM_32 = 34_359_738_368


def _plan(
    model, budget=BUDGET_32, *, fit_check=True, cache_limit_override=None, policy="per-block"
):
    return plan_call_for(
        constants=kmem.CONSTANTS,
        sizes=SIZES[model],
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


def _fit(model, sizes=None, *, budget=BUDGET_32, vae_with_set=False):
    c = kmem.CONSTANTS
    return fit_for(
        c,
        sizes=SIZES[model] if sizes is None else sizes,
        largest=LARGEST,
        policy="per-block",
        cache_limit=868_220_928 + ALLOWANCE,
        allowance=activation_allowance(c, height=1024, width=1024, text_tokens=TEXT_TOKENS),
        budget=budget,
        height=1024,
        width=1024,
        text_tokens=TEXT_TOKENS,
        vae_with_set=vae_with_set,
    )


def test_the_calibration_records_name_the_pinned_repos_and_hold_the_runs_the_constants_name():
    # Bug caught: a constant re-derived from a run that is not recorded, a record of the wrong model or snapshot, a
    # phase sourced twice (or not at all), or the calls per step of the CFG run mislabelled.
    assert (
        RECORDS["krea-2-raw"]["df11"]
        == "mingyi456/Krea-2-Raw-DF11-ComfyUI@8320616b25ac9340a830a7fb21f1b0237e160e66"
    )
    assert (
        RECORDS["krea-2-raw"]["base"] == "krea/Krea-2-Raw@6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
    )
    assert (
        RECORDS["krea-2"]["df11"]
        == "mingyi456/Krea-2-Turbo-DF11-ComfyUI@978da5fb7647bd222d33125993abd8fdc2840cfc"
    )
    assert RECORDS["krea-2"]["base"] == "krea/Krea-2-Turbo@98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
    assert (RECORDS["krea-2-raw"]["calls_per_step"], RECORDS["krea-2"]["calls_per_step"]) == (2, 1)
    phases = [p for rec in RECORDS.values() for p in rec["constants_from"]]
    assert sorted(phases) == ["denoise", "encode", "vae"]
    for m, rec in RECORDS.items():
        assert set(rec["constants_from"].values()) <= set(RUNS[m]), m
        assert rec["text_tokens"] == TEXT_TOKENS
    assert RECORDS["krea-2"]["overhead_from"] == "derisk-turbo-2"
    # Every constant comes from a run of the shipped code, or (the VAE term) from the resident run.
    assert set(SOURCE.values()) | {RECORDS["krea-2"]["overhead_from"]} <= set(VAE_RUNS)
    assert {
        label: (run["git"][:7], run["mechanism"].split(":")[0]) for label, run in ALL_RUNS.items()
    } == {
        "derisk-raw-2": ("c555f50", "shipped"),
        "confirm-raw-24711fb": ("24711fb", "shipped"),
        "derisk-raw-1": ("d1aa9df", "resident"),
        "history-v1-raw": ("59c961b", "superseded (history)"),
        "history-mm1-raw": ("bedba36", "superseded (history)"),
        "measured-25-steps": ("f0f348f", "shipped"),
        "card-52-steps-cfg": ("f0f348f", "shipped"),
        "identity-4-steps-cfg": ("e3a508e", "shipped"),
        "derisk-turbo-2": ("c555f50", "shipped"),
        "confirm-turbo-24711fb": ("24711fb", "shipped"),
        "measured-8-steps": ("f0f348f", "shipped"),
        "history-v1-turbo": ("81e79f3", "superseded (history)"),
        "history-mm1-turbo": ("bedba36", "superseded (history)"),
    }
    assert (SHIPPED["derisk-raw-2"]["steps"], SHIPPED["derisk-turbo-2"]["steps"]) == (1, 2)


def test_the_record_sizes_charge_no_resident_nonblock_in_the_shipped_mode():
    # Bug caught: the per-call runs recorded with the 1.234 GiB of resident non-block weights (the estimate would
    # charge them twice: once as nonblock, once inside the measured denoise term), or the encoder counted by file.
    assert {label: run["sizes"]["nonblock"] for label, run in ALL_RUNS.items()} == {
        "derisk-raw-1": 1_325_400_064,
        "derisk-raw-2": 0,
        "derisk-turbo-2": 0,
        "history-v1-raw": 0,
        "history-mm1-raw": 0,
        "history-v1-turbo": 0,
        "history-mm1-turbo": 0,
        "measured-8-steps": 0,
        "measured-25-steps": 0,
        "confirm-raw-24711fb": 0,
        "confirm-turbo-24711fb": 0,
        "card-52-steps-cfg": 0,
        "identity-4-steps-cfg": 0,
    }
    assert {s.encoders for s in SIZES.values()} == {8_044_936_192 - 201_861_632}  # minus layer 35


def test_the_overhead_is_the_larger_post_build_gap_of_the_shipped_runs():
    # Bug caught: the overhead taken from one run (Raw's 522_673_362 would put the Turbo run's phases 1_687_552 B low),
    # from a superseded run (the first per-call Turbo run's 529_374_466), or computed otherwise than footprint - MLX
    # active - MLX cache after the build.
    gaps = {
        label: run["after_build"]["footprint"]
        - run["after_build"]["mlx_active"]
        - run["after_build"]["mlx_cache"]
        for label, run in SHIPPED_DERISK.items()
    }
    assert gaps == {
        "derisk-raw-2": 522_673_362,
        "confirm-raw-24711fb": 523_590_866,
        "derisk-turbo-2": 524_360_914,
        "confirm-turbo-24711fb": 523_787_474,
    }
    assert kmem.CONSTANTS.overhead_bytes == 524_360_914 == max(gaps.values())


def test_the_estimate_at_1024_equals_the_calibration_runs_phases():
    # Bug caught: a term dropped from a phase or counted twice, REFERENCE_TOKENS left at another prompt's count, the
    # resident non-block bytes charged in the per-call mode, or a term derived at another cache limit than the
    # planner's. In-sample, so exact for encode and VAE:
    # - encode: the evaluated language model 7_843_074_560 + the footprint encode term 230_829_150 + overhead
    #   524_360_914 = 8_598_264_624, confirm-raw-24711fb's encode peak (the highest of the shipped runs);
    # - VAE (set dropped): extras 4_560_024 + VAE file 507_591_892 + transient 10_832_964_778 + overhead 524_360_914 =
    #   11_869_477_608, derisk-raw-1's VAE peak;
    # - denoise (Raw): compressed 17_540_930_560 + extras 4_560_024 + nonblock 0 + one block 868_220_928 + cache limit
    #   4_500_000_000 + activation 805_306_368 (the floor: the measured term is 0) + overhead 524_360_914 =
    #   24_243_378_794, 1_541_677_514 B over derisk-raw-2's 22_701_701_280 (the drained cache never fills to the
    #   cache-limit term at the peak).
    fit = _fit("krea-2-raw")
    assert (
        fit.phases["encode"]
        == 8_598_264_624
        == ALL_RUNS[SOURCE["encode"]]["footprint_peaks"]["encode"]
        == max(run["footprint_peaks"]["encode"] for run in SHIPPED_DERISK.values())
    )
    assert fit.phases["vae"] == 11_869_477_608 == ALL_RUNS[SOURCE["vae"]]["footprint_peaks"]["vae"]
    assert fit.phases["denoise"] == 24_243_378_794
    assert (
        fit.phases["denoise"] - SHIPPED["derisk-raw-2"]["footprint_peaks"]["denoise"]
        == 1_541_677_514
    )
    assert (SOURCE["encode"], SOURCE["denoise"], SOURCE["vae"]) == (
        "confirm-raw-24711fb",
        "measured-8-steps",
        "derisk-raw-1",
    )


def test_the_denoise_term_is_the_highest_shipped_sample_floored_at_0():
    # Bug caught: the term taken from a superseded run (the second per-call Raw run's denoise sat at the
    # memory limit, 24_135_923_728 B: a term of 698_850_888 B that the shipped code no longer needs), from one shipped run
    # instead of the highest (Raw's residual -736_371_146 vs Turbo's -726_734_444), or left negative. Per shipped run,
    # the denoise peak minus the phase's other terms at that run's own sizes: the highest is Turbo's, below 0, so the
    # term is 0 and the float32-copy floor alone carries the activation.
    residual = {
        label: run["footprint_peaks"]["denoise"]
        - (
            run["sizes"]["compressed"]
            + run["sizes"]["extras"]
            + run["sizes"]["nonblock"]
            + 868_220_928
            + run["cache_limit"]
            + kmem.CONSTANTS.overhead_bytes
        )
        for label, run in SHIPPED_DERISK.items()
    }
    assert residual == {
        "derisk-raw-2": -736_371_146,
        "confirm-raw-24711fb": -733_766_258,
        "derisk-turbo-2": -726_734_444,
        "confirm-turbo-24711fb": -733_730_340,
    }
    assert kmem.CONSTANTS.denoise_activation_at_reference == max(0, *residual.values()) == 0
    for label, run in SHIPPED.items():
        assert (
            run["footprint_peaks"]["denoise"]
            <= _fit("krea-2-raw", FamilySizes(**run["sizes"])).phases["denoise"]
        ), label


def test_the_vae_term_covers_every_1024_sample_and_equals_the_highest():
    # Bug caught: the VAE transient derived from one run instead of the highest sample (the shipped runs alone would
    # put the resident run's decode 1_646_167_480 B over its estimate). Every run decoded with the set dropped, so each
    # sample, the superseded ones too, is checked against the dropped-set VAE phase with the run's own sizes.
    peaks = {label: run["footprint_peaks"]["vae"] for label, run in DERISK.items()}
    for label, run in DERISK.items():
        assert peaks[label] <= _fit("krea-2-raw", FamilySizes(**run["sizes"])).phases["vae"], label
    assert _fit("krea-2-raw").phases["vae"] == max(peaks.values()) == peaks["derisk-raw-1"]
    assert max(run["footprint_peaks"]["vae"] for run in SHIPPED_DERISK.values()) == 10_525_591_448
    # The later runs record one watched peak, reached in the denoise phase: no VAE-phase sample beyond these.
    assert all(
        "vae" not in run["footprint_peaks"]
        for label, run in ALL_RUNS.items()
        if label not in DERISK
    )


def test_the_planners_1024_cache_limit_is_one_block_plus_the_allowance():
    # Bug caught: the shared 1.5e9 allowance, two blocks budgeted for a one-kind family, or the allowance scaled from
    # another reference token count. 868_220_928 + 3_631_779_072.
    assert _plan("krea-2-raw").cache_limit == 4_500_000_000
    assert (kmem.CONSTANTS.allowance_at_reference, kmem.CONSTANTS.allowance_reference_tokens) == (
        3_631_779_072,
        4096 + 30,
    )


def test_a_cache_limit_override_under_the_one_kind_minimum_warns_for_both_policies(caplog):
    # Bug caught: the depth2 minimum missing its look-ahead group, or the per-block minimum without the
    # allowance. Per-block 868_220_928 + 3_631_779_072; depth2 1_736_441_856 + 3_631_779_072.
    per_block, depth2 = 4_500_000_000, 5_368_220_928
    with caplog.at_level("WARNING", logger="test"):
        _plan("krea-2-raw", cache_limit_override=per_block)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="test"):
        _plan("krea-2-raw", cache_limit_override=per_block - 1)
    assert f"below the derived minimum {per_block}" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="test"):
        assert _plan("krea-2-raw", policy="depth2", fit_check=False).cache_limit == depth2
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="test"):
        _plan("krea-2-raw", policy="depth2", cache_limit_override=per_block, fit_check=False)
    assert f"below the derived minimum {depth2}" in caplog.text


@pytest.mark.parametrize(
    ("model", "denoise"), [("krea-2-raw", 24_243_378_794), ("krea-2", 24_242_523_532)]
)
def test_a_32_gb_mac_drops_the_set_before_the_vae_at_1024_and_fits_in_the_denoise_phase(
    model, denoise
):
    # Bug caught: the set-drop decision inverted (a decode on the resident set runs into the watchdog), or the 32 GB
    # verdict lost. With the set the VAE phase would be compressed + extras + VAE file + transient + overhead: Raw
    # 17_540_930_560 + 4_560_024 + 507_591_892 + 10_832_964_778 + 524_360_914 = 29_410_408_168 B, Turbo 855_262 B less
    # (its compressed set is 17_540_075_298), both over the 24_653_119_488 budget. Without it the call peaks in the
    # denoise phase (the activation at the 805_306_368 floor): Raw 24_243_378_794 (409_740_694 B under the budget),
    # Turbo 24_242_523_532 (410_595_956 B under).
    plan = _plan(model)
    assert plan.drop_set_before_vae is True
    assert (plan.estimate.peak_phase, plan.estimate.peak_bytes) == ("denoise", denoise)
    assert (
        _fit(model, vae_with_set=True).phases["vae"]
        == {"krea-2-raw": 29_410_408_168, "krea-2": 29_409_552_906}[model]
    )
    assert BUDGET_32 - denoise == {"krea-2-raw": 409_740_694, "krea-2": 410_595_956}[model]


@pytest.mark.parametrize(
    ("size", "expected"), [(512, 805_306_368), (1024, 805_306_368), (2048, 805_306_368)]
)
def test_the_denoise_term_is_floored_by_the_float32_mlp_weight_copies(size, expected):
    # Bug caught: a call predicted below the float32 copies every step holds whatever the image size (the float32
    # residual stream times a BF16 weight makes MLX copy the weight to float32: mlp.gate and mlp.up, 2 x 24576 x 4096
    # x 4 = 805_306_368 B). The measured term is 0 (the shipped runs' highest residual is negative), so the floor is
    # the whole activation at every size, 2048² included; whether it scales above 1024² is unmeasured (sizes above
    # 1024² are refused without --no-fit-check).
    from mlx_dfloat.mflux._phases import denoise_activation_bytes

    assert kmem.CONSTANTS.denoise_activation_floor_bytes == 805_306_368 == 2 * 24576 * 4096 * 4
    got = denoise_activation_bytes(kmem.CONSTANTS, height=size, width=size, text_tokens=30)
    assert got == expected


def _tier_budget(tier):
    return tier_limits(
        tier, host_ram_bytes=RAM_32, host_recommended_bytes=RECOMMENDED_32
    ).fit_budget_bytes


def test_the_tier_budgets_are_a_real_macs():
    # Bug caught: a tier judged against its watchdog ceiling or the host's budget instead of the fit budget.
    assert (_tier_budget(16), _tier_budget(24)) == (9_305_762_474, 15_032_385_536)


# --- CAPPED predictions: each model's default call at 1024² (Raw 25 steps at 1.0, Turbo 8 at 1.0: one call, the
# 30-token prompt), against a 24 GB Mac's fit budget (15_032_385_536 B) and a 16 GB Mac's (9_305_762_474 B). The
# compressed set alone (17_540_930_560 B, 16.34 GiB) is over both, so the denoise phase decides: refused there at both
# tiers. Forced (fit_check off) the set is dropped before the decode, and the dropped-set VAE phase keeps the extras
# (they stay on the transformer): 4_560_024 + 507_591_892 + 10_832_964_778 + 524_360_914 = 11_869_477_608 B.


@pytest.mark.parametrize("model", ["krea-2-raw", "krea-2"])
@pytest.mark.parametrize("tier", [16, 24])
def test_every_capped_default_call_is_refused_in_the_denoise_phase(model, tier):
    # Bug caught: the refusal naming the VAE phase (the set drop not applied), a call that cannot hold the compressed
    # set planned to fit, or the dropped-set VAE phase without the extras.
    budget = _tier_budget(tier)
    with pytest.raises(DFloatResourceError, match="in the denoise phase"):
        _plan(model, budget)
    forced = _plan(model, budget, fit_check=False)
    assert forced.drop_set_before_vae is True
    assert forced.estimate.peak_phase == "denoise"
    assert (
        forced.estimate.peak_bytes
        == {"krea-2-raw": 24_243_378_794, "krea-2": 24_242_523_532}[model]
    )
    assert forced.estimate.phases["vae"] == 11_869_477_608


# What the prompt encode may hold before the model warns (model._check_encode_peak): MLX's encode peak minus what was
# active when the phase began, against the encode phase without its overhead (evaluated language model + encode
# term) + the slack: 7_843_074_560 + 230_829_150 + 268_435_456 = 8_342_339_166 B.
ENCODE_BOUND = 8_342_339_166
VISION_TOWER = 830_695_424  # the visual.* tensors in the encoder's file (its published header)


def test_the_encode_bound_holds_each_measured_encode_with_its_spare():
    # Bug caught: a bound under what a normal encode holds (every call would warn).
    bound = kmem.encode_peak_bound(_fit("krea-2-raw").phases["encode"], kmem.CONSTANTS)
    assert bound == ENCODE_BOUND
    # Every recorded encode, the superseded runs' too: the encode path did not change between them.
    spare = {label: bound - run["mlx_encode_held"] for label, run in DERISK.items()}
    assert spare == {
        "derisk-raw-2": 396_039_902,
        "confirm-raw-24711fb": 389_145_310,
        "derisk-raw-1": 399_350_774,
        "history-v1-raw": 402_987_390,
        "history-mm1-raw": 396_047_070,
        "derisk-turbo-2": 434_814_750,
        "confirm-turbo-24711fb": 405_439_478,
        "history-v1-turbo": 399_323_126,
        "history-mm1-turbo": 474_185_294,
    }


@pytest.mark.parametrize("text_tokens", [1, TEXT_TOKENS, 1024])
def test_an_encode_that_also_loads_the_vision_tower_is_over_the_bound_at_any_prompt(text_tokens):
    # Bug caught: the encode bound carrying the denoise allowance (it grows with the prompt), or a slack wide enough
    # that an mflux change loading the vision tower goes unnoticed. Over by 7_953_193_856 + 830_695_424 - 8_342_339_166
    # = 441_550_114 B at every prompt length, for the largest measured hold.
    c = kmem.CONSTANTS
    fit = fit_for(
        c,
        sizes=SIZES["krea-2-raw"],
        largest=LARGEST,
        policy="per-block",
        cache_limit=4_500_000_000,
        allowance=activation_allowance(c, height=1024, width=1024, text_tokens=text_tokens),
        budget=BUDGET_32,
        height=1024,
        width=1024,
        text_tokens=text_tokens,
    )
    bound = kmem.encode_peak_bound(fit.phases["encode"], c)
    held = max(run["mlx_encode_held"] for run in DERISK.values()) + VISION_TOWER
    assert held - bound == 441_550_114
    assert kmem.encode_peak_warning(held, bound, text_tokens=text_tokens) is not None


# --- the committed 1024² records of the shipped code, beyond the calibration runs -------------------------------------
# Each is a run's own report (mlx-dfloat generate --report, or verify_image's df11 side), never hand-copied; the
# calibration records file them under the label given here. Their watched peak is reached in the denoise phase (every
# other phase's boundary samples stay at or under 18.1e9 B; the VAE decode runs with the set dropped).
OUT_OF_SAMPLE = [
    ("bench/results/tiers/krea-2-1024.json", "krea-2", "measured-8-steps"),
    ("bench/results/tiers/krea-2-raw-1024.json", "krea-2-raw", "measured-25-steps"),
    (
        "bench/results/recipes/krea-2-raw-1024-52steps-guidance3.5.json",
        "krea-2-raw",
        "card-52-steps-cfg",
    ),
    (
        "bench/results/identity/krea-2-raw-1024/df11-result.json",
        "krea-2-raw",
        "identity-4-steps-cfg",
    ),
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


@pytest.mark.parametrize(("path", "model", "label"), OUT_OF_SAMPLE)
def test_the_later_denoise_samples_are_the_committed_runs_watched_peaks(path, model, label):
    # Bug caught: a calibration sample that is not the committed run's peak (a typo, or a re-run whose record was not
    # carried over), or a sample filed under the other model or the wrong mechanism.
    _report, _h, _w, watched = _record_run(path)
    run = RUNS[model][label]
    assert run["footprint_peaks"] == {"denoise": watched}
    assert run["mechanism"].startswith("shipped")


def test_the_denoise_term_stays_0_over_every_shipped_sample():
    # Bug caught: the term re-derived from the calibration runs only, so a longer run whose peak rose above its
    # estimate would not move it. Every shipped sample's residual (peak minus the phase's other terms) is still negative;
    # the highest is the Turbo MEASURED row's, 22_736_402_568 - 23_437_217_164 = -700_814_596.
    def residual(run):
        s = run["sizes"]
        return run["footprint_peaks"]["denoise"] - (
            s["compressed"]
            + s["extras"]
            + s["nonblock"]
            + 868_220_928
            + run["cache_limit"]
            + kmem.CONSTANTS.overhead_bytes
        )

    residuals = {label: residual(run) for label, run in SHIPPED.items()}
    assert len(residuals) == 8
    assert max(residuals.values()) == residuals["measured-8-steps"] == -700_814_596
    assert kmem.CONSTANTS.denoise_activation_at_reference == 0


# The committed records re-estimated with the current constants from their own reports (sizes, cache limit in force,
# text tokens, policy, VAE strategy). The check is one-sided (a run under the estimate is not the bug); the margins are
# pinned too, because the estimate sits about 1.53 GB over every shipped peak (the drained cache never fills to the
# cache-limit term at block 0), so a 0.25 GiB tolerance alone would let a 1.7 GB rise pass unnoticed.
RECORD_MARGINS = [
    pytest.param(
        "bench/results/tiers/krea-2-1024.json", 1_506_120_964, id="turbo-measured-8-steps"
    ),
    pytest.param(
        "bench/results/tiers/krea-2-raw-1024.json", 1_529_635_490, id="raw-measured-25-steps"
    ),
    pytest.param(
        "bench/results/recipes/krea-2-raw-1024-52steps-guidance3.5.json",
        1_528_865_130,
        id="raw-card-52-steps-cfg",
    ),
    pytest.param(
        "bench/results/identity/krea-2-raw-1024/df11-result.json",
        1_531_863_714,
        id="raw-identity-df11-4-steps-cfg",
    ),
]


@pytest.mark.parametrize(("path", "margin"), RECORD_MARGINS)
def test_a_committed_1024_record_peaks_within_the_current_estimate(path, margin):
    # Bug caught: a 1024² peak that grows with the step count or with CFG while the estimate stays at the one-step
    # value, the estimate peaking in another phase, or a record re-estimated from other sizes than its own. The
    # estimate is the report's own prediction (Turbo 24_242_523_532, Raw 24_243_378_794) minus the watched peak.
    report, height, width, watched = _record_run(path)
    c = kmem.CONSTANTS
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
    assert fit.peak_phase == "denoise"
    assert fit.peak_bytes == report["fit"]["peak_bytes"]
    assert watched <= fit.peak_bytes + GIB // 4
    assert fit.peak_bytes - watched == margin
