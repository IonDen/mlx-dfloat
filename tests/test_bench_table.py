"""README fragments rendered from result data, and the marker splice."""

import json
from pathlib import Path

import pytest

from mlx_dfloat.bench.capped import GIB
from mlx_dfloat.bench.results import Summary
from mlx_dfloat.bench.table import (
    ProofRecord,
    TierRow,
    caption,
    proof_from_files,
    render_overhead_block,
    render_proof_paragraph,
    render_tier_table,
    splice,
    tier_row_from_abort_artifact,
    tier_row_from_generate_report,
)
from mlx_dfloat.errors import DFloatFormatError


def _row(**over):
    base = {
        "mac_gb": 32,
        "ceiling_bytes": int(22.96 * GIB),
        "model": "schnell",
        "df11_bytes": 16_195_141_095,
        "watched_peak_bytes": int(19.94 * GIB),
        "footprint_peak_bytes": int(19.94 * GIB),
        "mlx_peak_bytes": int(17.4 * GIB),
        "label": "MEASURED",
        "status": "target",
        "limits_note": "host caps",
        "source": "bench/results/tiers/schnell-1024.json",
    }
    return TierRow(**{**base, **over})


def test_tier_table_renders_gib_labels_and_the_source_file():
    # Red when: render_tier_table truncates instead of rounding GiB, drops the source column, or
    # names the MLX column without what it counts; or the second column is called the fit budget,
    # which on a CAPPED row it is not (working set - 1.5 GiB is the watchdog ceiling there; the
    # fit budget takes 2 GiB).
    # By hand: 16_195_141_095 / 2**30 = 15.0829 -> "15.08 GiB"; int(22.96 * GIB) / 2**30 rounds to 22.96
    # (a truncating renderer would print 22.95).
    text = render_tier_table([_row()])
    assert text.startswith(
        "| Mac | Working set − reserve | Model | DF11 size | Peak (watched) | Peak footprint "  # noqa: RUF001
        "| Peak MLX (sampled active + cache, or exact phase peak) | Label | Status | Limits | Result |"
    )
    assert (
        "| 32 GB | 22.96 GiB | FLUX.1-schnell | 15.08 GiB | 19.94 GiB | 19.94 GiB | 17.40 GiB | MEASURED | target | host caps | `bench/results/tiers/schnell-1024.json` |"
        in text
    )


def _summary(**over):
    base = {
        "scenario_hash": "a" * 64,
        "rounds_seen": 3,
        "conditions": {
            "df11": {
                "median": 18.6,
                "spread": 0.02,
                "n": 15,
                "step_watched_peak": 1,
                "footprint_peak": 1,
                "mlx_peak": 1,
            }
        },
        "overhead": {"per-block": 0.085, "depth2": 0.061},
        "eval_cost_s": 0.4,
        "q8_ratio": 1.9,
        "paired_rounds": {},
        "pair_n": {},
    }
    return Summary(**{**base, **over})


def test_overhead_block_prints_each_scenario_with_paired_signed_percentages():
    # Bug caught: swapped per-block / depth-2 labels (the substring pins the pairing), a missing
    # sign, or the q8 ratio rendered as a percentage.
    text = render_overhead_block(
        {"flux1-dev-1024": _summary()},
        caption="M1 Max",
        reproducers={"flux1-dev-1024": "uv run x"},
        cache_limit_note="2.5 GB",
    )
    assert "per-block evaluation: +8.5 % (depth-2: +6.1 %)" in text
    assert "eval policy cost 0.40 s/step" in text
    assert "1.90×" in text  # noqa: RUF001
    # Bug caught: the q8 condition described as mflux "as shipped" (mflux's own generate sets no
    # cache limit; the bench ran q8 under the scenario's, like every other condition).
    assert "over the mflux q8 step (one eval per step, same cache limit): 1.90×" in text  # noqa: RUF001
    assert "as shipped" not in text
    assert "uv run x" in text
    assert "M1 Max" in text
    assert "2.5 GB" in text


def test_overhead_block_says_so_when_a_pair_is_missing():
    text = render_overhead_block(
        {"x": _summary(overhead={}, q8_ratio=None, eval_cost_s=None)},
        caption="c",
        reproducers={"x": "r"},
        cache_limit_note="n",
    )
    # per-block, depth-2, the eval policy cost and the q8 ratio: each says so on its own
    assert text.count("not measured") == 4


def test_splice_replaces_only_the_marked_block():
    # Red when: splice replaces the whole text or eats the marker lines.
    doc = "a\n<!-- bench:x -->\nold\n<!-- /bench:x -->\nb\n"
    assert splice(doc, "x", "new\n") == "a\n<!-- bench:x -->\nnew\n<!-- /bench:x -->\nb\n"


@pytest.mark.parametrize(
    "doc",
    [
        "a\nb\n",
        "<!-- bench:x -->\n<!-- /bench:x -->\n<!-- bench:x -->\n<!-- /bench:x -->\n",
        "<!-- bench:x -->\nno end\n",
    ],
)
def test_splice_refuses_missing_or_duplicated_markers(doc):
    # A README without exactly one marker pair must not be silently rewritten.
    with pytest.raises(DFloatFormatError, match="bench:x"):
        splice(doc, "x", "new\n")


def test_caption_names_chip_ram_os_versions_git_and_date():
    prov = {
        "device_info": {"device_name": "Apple M1 Max", "memory_size": 32 * GIB},
        "macos": "27.0",
        "mlx": "0.32.2",
        "mflux": "0.20.0",
        "git": "abc1234def",
    }
    assert (
        caption(prov, date="2026-09-30")
        == "Apple M1 Max, 32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, git abc1234, 2026-09-30"
    )


REPORT = {
    "model": "schnell",
    "label": "MEASURED",
    "limits": {
        "tier": {"tier_gb": 32, "ceiling_bytes": int(22.96 * GIB)},
        "effective": {},
        "applied": "host-caps",
    },
    "sizes": {"compressed": 16_081_241_447, "extras": 113_899_648},
    "watched_peak_bytes": int(19.94 * GIB),
    "footprint_peak_bytes": int(19.94 * GIB),
    "mlx_peak_bytes": int(19.19 * GIB),
    "peaks": {
        "label": "sampled at phase boundaries",
        "encode": {"mlx_peak": 1},
        "denoise": {"mlx_peak": int(17.4 * GIB)},
    },
}


def test_tier_row_from_generate_report_reads_the_ceiling_sizes_and_peaks():
    # Bug caught: the budget column read from fit.budget_bytes (the raw recommended set, no reserve),
    # the DF11 size excluding the extras, or the MLX peak taken from a phase's active-only
    # mx.get_peak_memory (17.4 GiB here) although the watchdog's active + cache peak is larger (19.19 GiB).
    row = tier_row_from_generate_report(REPORT, source="bench/results/tiers/schnell-1024.json")
    assert row.ceiling_bytes == int(22.96 * GIB)
    assert row.df11_bytes == 16_081_241_447 + 113_899_648
    assert row.mlx_peak_bytes == int(19.19 * GIB)
    assert row.status == "target"
    assert row.limits_note == "host caps"


def test_the_mlx_column_takes_an_exact_phase_peak_above_the_sampled_one():
    # Bug caught: a short MLX spike between two 0.05 s watchdog samples (the Z-Image VAE decode: sampled 12.20
    # GiB, MLX's own per-phase peak 12.65 GiB in z-image-1024.json) hidden, the column then understating the run.
    report = {
        **REPORT,
        "mlx_peak_bytes": 13_099_464_594,
        "peaks": {
            "label": "sampled at phase boundaries",
            "denoise": {"mlx_peak": 10_637_496_516},
            "vae": {"mlx_peak": 13_578_549_138},
        },
    }
    assert tier_row_from_generate_report(report, source="s").mlx_peak_bytes == 13_578_549_138
    no_peaks = {k: v for k, v in report.items() if k != "peaks"}
    assert tier_row_from_generate_report(no_peaks, source="s").mlx_peak_bytes == 13_099_464_594


def test_an_abort_row_reads_as_a_lower_bound_with_the_time_it_ran():
    # Bug caught: a stopped run's peaks printed like a finished run's (9.26 GiB was where the watchdog stopped
    # it, not where the run would have peaked), or without how long it ran. Literals from
    # the first 16 GB Turbo run (superseded by a re-run at the same path): 9_947_618_600 B = 9.26 GiB, 5.57 s, MLX 8.81 GiB.
    artifact = {
        "reason": "memory",
        "elapsed": 5.5735681660007685,
        "ceiling": 9_842_633_386,
        "peak_watched": 9_947_618_600,
        "peak_footprint": 9_947_618_600,
        "peak_mlx": 9_456_147_030,
        "context": {"model": "z-image-turbo", "tier_gb": 16, "label": "CAPPED"},
    }
    text = render_tier_table([tier_row_from_abort_artifact(artifact, source="a.json")])
    assert (
        "| 16 GB | 9.17 GiB | Z-Image-Turbo | not recorded | at least 9.26 GiB (stopped after 5.6 s) "
        "| at least 9.26 GiB | at least 8.81 GiB | CAPPED | stopped by the watchdog |"
    ) in text


def test_tier_row_reads_a_report_the_generate_command_actually_wrote():
    # Bug caught: the consumer reading a field the producer never writes, so every real report fails
    # with a missing-field error (a hand-built REPORT would hide it).
    path = Path(__file__).parent / "fixtures" / "generate_report_schnell_1024.json"
    report = json.loads(path.read_text())
    row = tier_row_from_generate_report(report, source="schnell-1024.json")
    assert row.watched_peak_bytes == 21266019216
    assert row.footprint_peak_bytes == 21266019216
    # the larger of the report's own mlx_peak_bytes (the watchdog's active + cache peak) and the
    # largest phase's active-only mlx_peak (18725825760, denoise's): the watchdog's here
    assert row.mlx_peak_bytes == 20488964670
    assert row.ceiling_bytes == 24653119488
    assert row.label == "MEASURED"
    assert row.status == "target"
    assert row.limits_note == "host caps"
    assert row.df11_bytes == 16195141095 + 113899648


def test_tier_row_marks_over_budget_and_refuses_a_proof_report():
    over = tier_row_from_generate_report(
        {**REPORT, "watched_peak_bytes": int(23.5 * GIB)}, source="s"
    )
    assert over.status == "over"
    with pytest.raises(DFloatFormatError, match="PROOF"):
        tier_row_from_generate_report({**REPORT, "label": "PROOF"}, source="s")


REPO = Path(__file__).resolve().parents[1]
# A real pass report: what `mlx-dfloat generate --memory-ceiling ... --report` wrote.
PASS_REPORT = json.loads((REPO / "bench/results/harness-proof/pass-512.json").read_text())
PROOF_CAP = PASS_REPORT["memory_ceiling_bytes"]


# The limits `generate` reads back after installing the host caps (20 GiB wired, 22 GiB memory) and
# puts in the run context it hands the watchdog; the cache value is illustrative.
HOST_LIMITS = {"memory": 22 * GIB, "cache": 24_653_119_488, "wired": 20 * GIB}


def _abort_artifact(tmp_path, monkeypatch, *argv, ceiling=PROOF_CAP, peak=None):
    """An abort artifact written by the real producers: generate's run context, the watchdog's _fire."""
    import mlx_dfloat._watchdog as wd
    from mlx_dfloat.mflux import generate as gen

    args = gen.build_parser().parse_args(["--prompt", "p", *argv])
    monkeypatch.setattr(wd, "_exit", lambda code: None)
    watchdog = wd.Watchdog(
        tmp_path,
        ceiling=ceiling,
        budget=60.0,
        context=gen._run_context(args, tier_gb=32, label="MEASURED", limits=HOST_LIMITS),
    )
    watchdog.peak_watched = peak if peak is not None else ceiling + 1
    watchdog._fire(
        "memory",
        {
            "footprint": ceiling + 1,
            "rss": 1,
            "mlx_active": 1,
            "mlx_cache": 0,
            "elapsed": 1.0,
            "verdict_memory": ceiling + 1,
            "verdict_counter": "footprint",
        },
    )
    artifact: dict[str, object] = json.loads((tmp_path / "abort.json").read_text())
    return artifact


def test_proof_record_and_paragraph(tmp_path, monkeypatch):
    # Red when: proof_from_files takes the cap from the artifact's peak, or the paragraph omits the cap or either size.
    abort = _abort_artifact(tmp_path, monkeypatch, "--height", "1024", "--width", "1024")
    proof = proof_from_files(PASS_REPORT, abort, source_dir="bench/results/harness-proof")
    assert isinstance(proof, ProofRecord)
    assert proof.cap_bytes == 20_669_530_112
    assert proof.pass_size == 512
    assert proof.abort_size == 1024
    text = render_proof_paragraph(proof)
    for needle in ("19.25 GiB", "512", "1024", "memory"):
        assert needle in text


def test_proof_reads_the_abort_size_from_the_artifacts_own_record(tmp_path, monkeypatch):
    # Bug caught: the abort size defaulted (it used to print 1024 whatever the aborted run was).
    abort = _abort_artifact(tmp_path, monkeypatch, "--height", "768", "--width", "768")
    proof = proof_from_files(PASS_REPORT, abort, source_dir="d")
    assert proof.abort_size == 768


def test_proof_refuses_an_artifact_without_the_runs_size(tmp_path, monkeypatch):
    # Bug caught: an artifact written by a watchdog that knew nothing of the run (no context)
    # rendered with an assumed size instead of refused.
    abort = _abort_artifact(tmp_path, monkeypatch)
    del abort["context"]
    with pytest.raises(DFloatFormatError, match="height"):
        proof_from_files(PASS_REPORT, abort, source_dir="d")


def test_proof_refuses_a_non_square_aborted_run(tmp_path, monkeypatch):
    # Bug caught: the abort size read from the height alone, so a 1024x512 run prints as 1024².
    abort = _abort_artifact(tmp_path, monkeypatch, "--height", "1024", "--width", "512")
    with pytest.raises(DFloatFormatError, match="square"):
        proof_from_files(PASS_REPORT, abort, source_dir="d")


def test_proof_refuses_an_aborted_run_of_another_model(tmp_path, monkeypatch):
    # Bug caught: a dev abort paired with a schnell pass, so the proof compares two recipes.
    abort = _abort_artifact(tmp_path, monkeypatch, "--model", "dev")
    with pytest.raises(DFloatFormatError, match="model"):
        proof_from_files(PASS_REPORT, abort, source_dir="d")


def test_proof_compares_the_seed_only_when_both_records_carry_one(tmp_path, monkeypatch):
    # Bug caught: a seed mismatch accepted when both sides name a seed, or a pass report without a
    # seed (generate's report has none) refused.
    abort = _abort_artifact(tmp_path, monkeypatch, "--seed", "42")
    assert proof_from_files(PASS_REPORT, abort, source_dir="d").abort_size == 1024
    with pytest.raises(DFloatFormatError, match="seed"):
        proof_from_files({**PASS_REPORT, "seed": 7}, abort, source_dir="d")
    assert proof_from_files({**PASS_REPORT, "seed": 42}, abort, source_dir="d").pass_size == 512


def test_proof_refuses_a_mismatched_cap_a_non_proof_label_or_a_non_square_size(
    tmp_path, monkeypatch
):
    # Red when: proof_from_files stops comparing the pass report's cap with the artifact's ceiling.
    abort = _abort_artifact(tmp_path, monkeypatch)
    with pytest.raises(DFloatFormatError, match="cap"):
        proof_from_files(PASS_REPORT, {**abort, "ceiling": 99}, source_dir="d")
    with pytest.raises(DFloatFormatError, match="PROOF"):
        proof_from_files({**PASS_REPORT, "label": "MEASURED"}, abort, source_dir="d")
    with pytest.raises(DFloatFormatError, match="square"):
        proof_from_files({**PASS_REPORT, "width": 256}, abort, source_dir="d")


def test_tier_row_names_a_missing_field():
    # Red when: a missing key surfaces as a bare KeyError instead of DFloatFormatError.
    bad = {k: v for k, v in REPORT.items() if k != "sizes"}
    with pytest.raises(DFloatFormatError, match="sizes"):
        tier_row_from_generate_report(bad, source="s")


def test_tier_row_needs_the_mlx_peak_of_the_report():
    # Red when: a report without the watchdog's MLX peak falls back to a per-phase number silently.
    bad = {k: v for k, v in REPORT.items() if k != "mlx_peak_bytes"}
    with pytest.raises(DFloatFormatError, match="mlx_peak_bytes"):
        tier_row_from_generate_report(bad, source="s")


def test_tier_row_tier_caps_note():
    # Red when: limits_note ignores limits["applied"], or still says a CAPPED row ran under MLX's
    # defaults when it ran under the caps mlx-dfloat installs on a Mac of that size.
    row = tier_row_from_generate_report(
        {**REPORT, "limits": {**REPORT["limits"], "applied": "tier-caps"}}, source="s"
    )
    assert row.limits_note == "mlx-dfloat caps for the tier"


def test_a_report_from_a_run_under_tier_defaults_says_mlx_defaults():
    # Bug caught: a report written before the tier caps existed (applied "tier-defaults": MLX's
    # own limits, wired 0) labelled with mlx-dfloat's caps, which that run never had.
    row = tier_row_from_generate_report(
        {**REPORT, "limits": {**REPORT["limits"], "applied": "tier-defaults"}}, source="s"
    )
    assert row.limits_note == "MLX defaults for the tier"


def test_a_report_with_an_unknown_limits_path_is_refused():
    # Bug caught: an unrecognised `applied` value silently given a note (any note would be a guess).
    with pytest.raises(DFloatFormatError, match="applied"):
        tier_row_from_generate_report(
            {**REPORT, "limits": {**REPORT["limits"], "applied": "tier-guess"}}, source="s"
        )


def test_overhead_line_names_the_model_and_size():
    # Red when: the scenario key is printed raw instead of "FLUX.1-dev, 1024²".
    text = render_overhead_block(
        {"flux1-dev-1024": _summary()},
        caption="c",
        reproducers={"flux1-dev-1024": "r"},
        cache_limit_note="n",
    )
    assert text.splitlines()[0].startswith("FLUX.1-dev, 1024², per-block evaluation: ")


def test_overhead_block_gives_each_scenario_its_command_and_a_skipped_preflight():
    # Bug caught: one scenario's command printed for all of them (the schnell line would cite the
    # dev command), or a run that skipped its launch check shown like one that passed it.
    text = render_overhead_block(
        {"flux1-dev-1024": _summary(), "flux1-schnell-1024": _summary()},
        caption="c",
        reproducers={"flux1-dev-1024": "uv run dev", "flux1-schnell-1024": "uv run schnell"},
        preflight_skipped={"flux1-dev-1024": ("not_charging",)},
        cache_limit_note="n",
    )
    lines = text.splitlines()
    dev = lines.index(next(x for x in lines if x.startswith("FLUX.1-dev")))
    schnell = lines.index(next(x for x in lines if x.startswith("FLUX.1-schnell")))
    assert lines[dev + 2] == "Command: `uv run dev` (preflight skipped: not_charging)"
    assert lines[schnell + 2] == "Command: `uv run schnell`"
    assert text.count("preflight skipped") == 1


def test_caption_keeps_the_dirty_suffix():
    # Red when: caption cuts the git value to 7 characters including the suffix.
    prov = {
        "device_info": {"device_name": "Chip", "memory_size": 32 * GIB},
        "macos": "1",
        "mlx": "2",
        "mflux": "3",
        "git": "abc1234def-dirty",
    }
    assert "git abc1234-dirty, d" in caption(prov, date="d")


# The limits a 16 GB tier's run reads back after installing that Mac's caps, by hand: 8 GiB wired,
# 10 GiB memory, 10 GiB cache (the memory cap binds).
TIER16_CAPS = {"memory": 10_737_418_240, "cache": 10_737_418_240, "wired": 8_589_934_592}


def _abort(**over):
    # ceiling: 16 GiB * 2 // 3 - 1.5 GiB = 9_842_633_386 (bench/capped.py:87-99).
    base = {
        "reason": "memory",
        "elapsed": 5.6,
        "ceiling": 9_842_633_386,
        "peak_watched": 9_900_000_000,
        "peak_footprint": 9_800_000_000,
        "peak_mlx": 9_700_000_000,
        "context": {
            "model": "z-image-turbo",
            "tier_gb": 16,
            "label": "CAPPED",
            "height": 1024,
            "width": 1024,
            "limits": TIER16_CAPS,
        },
    }
    return {**base, **over}


def test_a_zimage_report_row_renders_the_model_label():
    # Bug caught: a Z-Image row printed as the raw name "z-image-turbo" (the table knew only FLUX.1 names).
    text = render_tier_table([_row(model="z-image-turbo"), _row(model="z-image")])
    assert "| Z-Image-Turbo |" in text
    assert "| Z-Image |" in text


def test_a_watchdog_stop_renders_as_a_capped_row_with_unrecorded_df11_size():
    # Bug caught: an abort record that cannot reach the README, or one rendered as "target".
    # By hand: 9_842_633_386 / 2**30 = 9.1667 -> 9.17; 9_900_000_000 -> 9.22; 9.8e9 -> 9.13; 9.7e9 -> 9.03.
    row = tier_row_from_abort_artifact(_abort(), source="bench/results/tiers/aborts/t.json")
    assert row.status == "stopped by the watchdog"
    assert row.df11_bytes is None
    assert (
        "| 16 GB | 9.17 GiB | Z-Image-Turbo | not recorded | at least 9.22 GiB (stopped after 5.6 s) "
        "| at least 9.13 GiB | at least 9.03 GiB "
        "| CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier "
        "| `bench/results/tiers/aborts/t.json` |"
    ) in render_tier_table([row])


def _capped_abort_note(limits):
    """The limits note of a CAPPED abort row whose run context records ``limits`` (None: no key)."""
    context = {k: v for k, v in _abort()["context"].items() if k != "limits"}
    if limits is not None:
        context["limits"] = limits
    return tier_row_from_abort_artifact(_abort(context=context), source="s").limits_note


def test_a_capped_abort_under_a_wired_cap_says_it_ran_under_mlx_dfloats_caps():
    # Bug caught: the note read from the label alone, or from the memory limit (a CAPPED run under
    # MLX's defaults also has one), instead of the wired cap only mlx-dfloat's caps install.
    assert _capped_abort_note(TIER16_CAPS) == "mlx-dfloat caps for the tier"


def test_a_capped_abort_with_no_wired_limit_says_it_ran_under_mlx_defaults():
    # Bug caught: an artifact from a run that kept MLX's defaults (wired 0, as CAPPED runs did
    # before the tier caps existed) labelled with caps it never ran under.
    defaults = {"memory": 17_179_869_184, "cache": 10_880_583_815, "wired": 0}
    assert _capped_abort_note(defaults) == "MLX defaults for the tier"


def test_a_capped_abort_without_recorded_limits_says_not_recorded():
    # Bug caught: an artifact written before the run context carried `limits` given a note it
    # cannot back, either way.
    assert _capped_abort_note(None) == "not recorded"


def test_a_measured_abort_keeps_the_host_caps_note():
    # Bug caught: the wired-limit reading applied to every label, so a host-tier stop (which also
    # records a wired cap) read "mlx-dfloat caps for the tier".
    context = {**_abort()["context"], "label": "MEASURED", "tier_gb": 32}
    row = tier_row_from_abort_artifact(_abort(context=context), source="s")
    assert row.limits_note == "host caps"


def test_an_abort_without_an_mlx_peak_renders_it_as_not_recorded():
    # Bug caught: the watchdog's peak_mlx is None until a sample read MLX; int(None) crashing the README render.
    row = tier_row_from_abort_artifact(_abort(peak_mlx=None), source="s")
    assert "| at least 9.13 GiB | not recorded | CAPPED |" in render_tier_table([row])


@pytest.mark.parametrize("missing", ["tier_gb", "label", "model"])
def test_an_abort_artifact_without_the_tier_context_is_refused(missing):
    # Bug caught: a stop that cannot say which tier it was rendered under a guessed one.
    context = {k: v for k, v in _abort()["context"].items() if k != missing}
    with pytest.raises(DFloatFormatError, match=missing):
        tier_row_from_abort_artifact(_abort(context=context), source="s")
    with pytest.raises(DFloatFormatError, match="context"):
        tier_row_from_abort_artifact(
            {k: v for k, v in _abort().items() if k != "context"}, source="s"
        )
