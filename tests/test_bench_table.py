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
    # names the fit budget a watchdog ceiling or the MLX column without what it counts.
    # By hand: 16_195_141_095 / 2**30 = 15.0829 -> "15.08 GiB"; int(22.96 * GIB) / 2**30 rounds to 22.96
    # (a truncating renderer would print 22.95).
    text = render_tier_table([_row()])
    assert text.startswith(
        "| Mac | Fit budget (budget − reserve) | Model | DF11 size | Peak (watched) | Peak footprint "  # noqa: RUF001
        "| Peak MLX (active + cache) | Label | Status | Limits | Result |"
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
    assert "not measured" in text


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
    # Review Focus 5: a README without exactly one marker pair must not be silently rewritten.
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
    # mx.get_peak_memory (17.4 GiB here) instead of the watchdog's active + cache peak (19.19 GiB).
    row = tier_row_from_generate_report(REPORT, source="bench/results/tiers/schnell-1024.json")
    assert row.ceiling_bytes == int(22.96 * GIB)
    assert row.df11_bytes == 16_081_241_447 + 113_899_648
    assert row.mlx_peak_bytes == int(19.19 * GIB)
    assert row.status == "target"
    assert row.limits_note == "host caps"


def test_tier_row_reads_a_report_the_generate_command_actually_wrote():
    # Bug caught: the consumer reading a field the producer never writes, so every real report fails
    # with a missing-field error (a hand-built REPORT would hide it).
    path = Path(__file__).parent / "fixtures" / "generate_report_schnell_1024.json"
    report = json.loads(path.read_text())
    row = tier_row_from_generate_report(report, source="schnell-1024.json")
    assert row.watched_peak_bytes == 21266019216
    assert row.footprint_peak_bytes == 21266019216
    # the report's own mlx_peak_bytes (the watchdog's active + cache peak), not the largest
    # phase's active-only mlx_peak (18725825760, denoise's)
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


def test_proof_record_and_paragraph():
    # Red when: proof_from_files takes the cap from the artifact's peak, or the paragraph omits the cap or either size.
    passed = {
        "height": 512,
        "width": 512,
        "watched_peak_bytes": int(17.0 * GIB),
        "memory_ceiling_bytes": int(18.5 * GIB),
        "label": "PROOF",
    }
    abort = {
        "reason": "memory",
        "ceiling": int(18.5 * GIB),
        "verdict_counter": "footprint",
        "peak_watched": int(18.6 * GIB),
    }
    proof = proof_from_files(passed, abort, source_dir="bench/results/harness-proof")
    assert isinstance(proof, ProofRecord)
    assert proof.cap_bytes == int(18.5 * GIB)
    assert proof.pass_size == 512
    text = render_proof_paragraph(proof)
    for needle in ("18.50 GiB", "512", "1024", "memory"):
        assert needle in text


def test_proof_refuses_a_mismatched_cap_a_non_proof_label_or_a_non_square_size():
    # Red when: proof_from_files stops comparing the pass report's cap with the artifact's ceiling.
    passed = {
        "height": 512,
        "width": 512,
        "watched_peak_bytes": 1,
        "memory_ceiling_bytes": 100,
        "label": "PROOF",
    }
    abort = {
        "reason": "memory",
        "ceiling": 100,
        "verdict_counter": "footprint",
        "peak_watched": 101,
    }
    with pytest.raises(DFloatFormatError, match="cap"):
        proof_from_files(passed, {**abort, "ceiling": 99}, source_dir="d")
    with pytest.raises(DFloatFormatError, match="PROOF"):
        proof_from_files({**passed, "label": "MEASURED"}, abort, source_dir="d")
    with pytest.raises(DFloatFormatError, match="square"):
        proof_from_files({**passed, "width": 256}, abort, source_dir="d")


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


def test_tier_row_tier_defaults_note():
    # Red when: limits_note ignores limits["applied"].
    row = tier_row_from_generate_report(
        {**REPORT, "limits": {**REPORT["limits"], "applied": "tier-defaults"}}, source="s"
    )
    assert row.limits_note == "tier defaults"


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
