"""scripts/bench_table.py: collect result files, render the README blocks, the --check mode."""

import json
from pathlib import Path

import pytest
from scripts import bench_table as bt

from mlx_dfloat.bench.capped import GIB
from mlx_dfloat.errors import DFloatFormatError

REPO = Path(__file__).resolve().parents[1]

H = "a" * 64
NONE = "No result files yet."

README = (
    "# t\n\n## Measured numbers\n\nText.\n\n"
    f"<!-- bench:tier-table -->\n{NONE}\n<!-- /bench:tier-table -->\n\n"
    f"<!-- bench:overhead -->\n{NONE}\n<!-- /bench:overhead -->\n\n"
    f"<!-- bench:harness-proof -->\n{NONE}\n<!-- /bench:harness-proof -->\n\nend\n"
)

TIER = {
    "model": "schnell",
    "label": "MEASURED",
    "limits": {
        "tier": {"tier_gb": 32, "ceiling_bytes": 24_000_000_000},
        "applied": "host-caps",
    },
    "sizes": {"compressed": 10 * GIB, "extras": 0},
    "watched_peak_bytes": 20 * GIB,
    "footprint_peak_bytes": 20 * GIB,
    "mlx_peak_bytes": 18 * GIB,
    "peaks": {"denoise": {"mlx_peak": 17 * GIB}},
}
PASS = {
    "label": "PROOF",
    "model": "schnell",
    "height": 512,
    "width": 512,
    "memory_ceiling_bytes": 18 * GIB,
    "watched_peak_bytes": 17 * GIB,
}
# Every key the watchdog's _fire writes, with the context generate hands it.
ABORT = {
    "reason": "memory",
    "footprint": 19 * GIB,
    "peak_footprint": 19 * GIB,
    "ceiling": 18 * GIB,
    "elapsed": 50.0,
    "budget": 3600.0,
    "rss": 7 * GIB,
    "mlx_active": 17 * GIB,
    "mlx_cache": 1 * GIB,
    "verdict_memory": 19 * GIB,
    "verdict_counter": "footprint",
    "peak_watched": 19 * GIB,
    "peak_mlx": 18 * GIB,
    "context": {"model": "schnell", "height": 1024, "width": 1024, "seed": 42, "steps": 4},
}
PROV = {
    "date": "2026-09-30",
    "git": "abc1234def",
    "mlx": "0.32.2",
    "mflux": "0.20.0",
    "macos": "27.0",
    "device_info": {"device_name": "Apple M1 Max", "memory_size": 32 * GIB},
}


def _child(mode, rnd, step):
    return {
        "mode": mode,
        "round": rnd,
        "exit_code": 0,
        "step_s": step,
        "warmup": 0,
        "launches_per_step": [0] * len(step),
        "launches_expected_per_step": 0,
        "step_footprint_peak_bytes": 1,
        "step_mlx_peak_bytes": 1,
        "step_watched_peak_bytes": 1,
        "footprint_peak_bytes": 1,
        "mlx_peak_memory_bytes": 1,
        "watched_peak_bytes": 1,
        "scenario_hash": H,
        "label": "MEASURED",
        "limits": {},
    }


def _scenario(
    root: Path,
    name: str,
    *,
    reproducer="uv run x",
    cache=2_500_000_000,
    skipped=None,
    missing=(),
    stopped=None,
    label="MEASURED",
    prov=None,
):
    d = root / name
    d.mkdir(parents=True)
    (d / "report.json").write_text(
        json.dumps(
            {
                "scenario": {"name": name, "cache_limit_bytes": cache},
                "provenance": prov or PROV,
                "reproducer": reproducer,
                "skipped_preflight": skipped is not None,
                "failed_gates": list(skipped or []),
                "missing": list(missing),
                "stopped": stopped,
            }
        )
    )
    # df11 1.2 vs control 1.0 -> per-block overhead +20.0 % by hand.
    for mode, step in (("df11", [1.2, 1.2, 1.2]), ("control", [1.0, 1.0, 1.0])):
        child = {**_child(mode, 1, step), "label": label}
        (d / f"round1-{mode}.json").write_text(json.dumps(child))
    return d


def _full_root(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    (root / "tiers").mkdir(parents=True)
    (root / "tiers" / "b.json").write_text(json.dumps(TIER))
    (root / "tiers" / "a.json").write_text(json.dumps({**TIER, "model": "dev"}))
    (root / "harness-proof").mkdir()
    (root / "harness-proof" / "pass-512.json").write_text(json.dumps(PASS))
    (root / "harness-proof" / "abort.json").write_text(json.dumps(ABORT))
    _scenario(root, "flux1-schnell-1024", reproducer="uv run schnell", skipped=["busy"])
    _scenario(root, "flux1-dev-1024", reproducer="uv run dev", cache=1_400_000_000)
    return root


def test_collect_on_an_empty_root_finds_nothing(tmp_path):
    c = bt.collect(tmp_path)
    assert (c.tier_rows, c.summaries, c.proof, c.provenance, c.reproducers, c.date) == (
        (),
        {},
        None,
        None,
        {},
        "",
    )


def test_collect_sorts_and_takes_provenance_from_the_first_scenario(tmp_path):
    # Bug caught: sorting by mtime or listing order (tiers b before a; the dev scenario must
    # supply the caption and cache limit because "flux1-dev-1024" < "flux1-schnell-1024"); or a
    # scenario's command or skipped preflight read from another scenario's report.
    c = bt.collect(_full_root(tmp_path))
    assert [r.model for r in c.tier_rows] == ["dev", "schnell"]
    assert c.tier_rows[0].source == "bench/results/tiers/a.json"
    assert list(c.summaries) == ["flux1-dev-1024", "flux1-schnell-1024"]
    assert c.reproducers == {"flux1-dev-1024": "uv run dev", "flux1-schnell-1024": "uv run schnell"}
    assert c.preflight_skipped == {"flux1-schnell-1024": ("busy",)}
    assert c.date == "2026-09-30"
    assert c.cache_limits == {"flux1-dev-1024": 1_400_000_000, "flux1-schnell-1024": 2_500_000_000}
    assert c.provenance is not None
    assert c.proof is not None
    assert c.proof.pass_size == 512
    assert c.proof.abort_size == 1024


def test_collect_reads_the_abort_size_from_the_artifacts_context(tmp_path):
    # Bug caught: ignoring the artifact's own record and printing a default size.
    root = _full_root(tmp_path)
    context = {**ABORT["context"], "height": 768, "width": 768}
    (root / "harness-proof" / "abort.json").write_text(json.dumps({**ABORT, "context": context}))
    proof = bt.collect(root).proof
    assert proof is not None
    assert proof.abort_size == 768


def test_an_abort_artifact_without_its_runs_size_is_exit_2(tmp_path, capsys):
    # Bug caught: an artifact from a watchdog that was not told the run's size rendered with an
    # assumed size (the old 1024 default) instead of refused.
    root = _full_root(tmp_path)
    bare = {k: v for k, v in ABORT.items() if k != "context"}
    (root / "harness-proof" / "abort.json").write_text(json.dumps(bare))
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert _run(root, readme) == 2
    assert "run context" in capsys.readouterr().err
    assert readme.read_text() == README


def test_collect_skips_a_proof_dir_with_only_one_file(tmp_path):
    # Bug caught: building a proof from a lone pass report (or crashing on the missing abort).
    root = _full_root(tmp_path)
    (root / "harness-proof" / "abort.json").unlink()
    assert bt.collect(root).proof is None


def test_render_readme_splices_all_three_blocks(tmp_path):
    root = _full_root(tmp_path)
    (root / "flux1-dev-1024" / "report.json").write_text(
        (root / "flux1-dev-1024" / "report.json")
        .read_text()
        .replace('"cache_limit_bytes": 1400000000', '"cache_limit_bytes": 2500000000')
    )
    out = bt.render_readme(README, bt.collect(root), date="2026-09-30")
    # By hand: 1.2 / 1.0 - 1 = +20.0 %; cache 2_500_000_000 B -> "2.5 GB"; caption date passed in.
    assert "| 32 GB |" in out
    assert "per-block evaluation: +20.0 %" in out
    assert out.count("under a 2.5 GB MLX buffer-cache limit") == 1
    assert "Harness proof: under one 18.00 GiB cap" in out
    assert (
        out.count(
            "Apple M1 Max, 32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, git abc1234, 2026-09-30"
        )
        == 1
    )
    assert NONE not in out


def test_scenarios_with_different_cache_limits_each_get_their_own_note(tmp_path):
    # Bug caught: one cache-limit note (the first scenario's 1.4 GB) printed for every scenario,
    # so the schnell numbers read as measured under a limit they did not run under.
    out = bt.render_readme(README, bt.collect(_full_root(tmp_path)), date="2026-09-30")
    assert "FLUX.1-dev, 1024²: DF11 and the mflux q8 step both run under a 1.4 GB" in out
    assert "FLUX.1-schnell, 1024²: DF11 and the mflux q8 step both run under a 2.5 GB" in out


def test_scenarios_measured_at_different_commits_each_get_their_own_caption(tmp_path):
    # Bug caught: the first scenario's caption (its git and date) printed for a scenario measured
    # at another commit.
    root = tmp_path / "results"
    _scenario(root, "flux1-dev-1024")
    _scenario(root, "flux1-schnell-1024", prov={**PROV, "git": "fedcba9876", "date": "2026-10-02"})
    out = bt.render_readme(README, bt.collect(root), date="unused")
    assert (
        "FLUX.1-dev, 1024²: Apple M1 Max, 32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, git abc1234, 2026-09-30"
        in out
    )
    assert (
        "FLUX.1-schnell, 1024²: Apple M1 Max, 32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, git fedcba9, 2026-10-02"
        in out
    )


def test_collect_refuses_a_scenario_with_missing_runs(tmp_path):
    # Bug caught: an interrupted scenario pooled from its completed rounds and rendered as if whole.
    root = tmp_path / "results"
    _scenario(root, "flux1-dev-1024", missing=["round 3: q8"])
    with pytest.raises(DFloatFormatError, match=r"flux1-dev-1024.*round 3: q8"):
        bt.collect(root)


def test_collect_refuses_a_stopped_scenario(tmp_path):
    # Bug caught: a scenario stopped by a child's watchdog abort rendered from what it had.
    root = tmp_path / "results"
    stopped = {"round": 1, "condition": "control", "exit_code": 70, "abort": "x/abort.json"}
    _scenario(root, "flux1-dev-1024", stopped=stopped)
    with pytest.raises(DFloatFormatError, match=r"flux1-dev-1024.*stopped"):
        bt.collect(root)


def test_collect_skips_a_capped_scenario_with_a_note(tmp_path):
    # Bug caught: a --tier run's CAPPED children rendered as an unlabelled overhead line, next to
    # the host's MEASURED ones.
    root = tmp_path / "results"
    _scenario(root, "flux1-dev-1024")
    _scenario(root, "flux1-dev-1024-tier24", label="CAPPED")
    c = bt.collect(root)
    assert list(c.summaries) == ["flux1-dev-1024"]
    assert any("flux1-dev-1024-tier24" in n and "CAPPED" in n for n in c.notes)


def test_render_readme_on_no_results_leaves_the_placeholder(tmp_path):
    # Bug caught: an empty overhead block that drops the placeholder line or crashes on no reports.
    assert bt.render_readme(README, bt.collect(tmp_path), date="") == README


def _run(root, readme, *extra):
    return bt.main(["--results-root", str(root), "--readme", str(readme), *extra])


def test_check_exits_1_on_a_stale_block_and_0_when_fresh(tmp_path, capsys):
    # Bug caught: --check that compares nothing (always 0) or rewrites the file instead of reporting.
    root = _full_root(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert _run(root, readme, "--check") == 1
    assert "tier-table" in capsys.readouterr().out
    assert readme.read_text() == README
    assert _run(root, readme) == 0
    assert _run(root, readme, "--check") == 0


def test_check_names_only_the_block_that_differs(tmp_path, capsys):
    # Bug caught: reporting every block as stale, or none by name.
    root = _full_root(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert _run(root, readme) == 0
    text = readme.read_text().replace("| 32 GB |", "| 99 GB |")
    readme.write_text(text)
    assert _run(root, readme, "--check") == 1
    out = capsys.readouterr().out
    assert "tier-table" in out
    assert "overhead" not in out


def test_main_exits_2_on_a_readme_without_markers_or_a_bad_result(tmp_path, capsys):
    # Bug caught: surfacing DFloatFormatError as a traceback (exit 1 is reserved for "stale").
    readme = tmp_path / "README.md"
    readme.write_text("no markers\n")
    assert _run(tmp_path, readme) == 2
    assert "bench:tier-table" in capsys.readouterr().err
    readme.write_text(README)
    (tmp_path / "tiers").mkdir()
    (tmp_path / "tiers" / "x.json").write_text("{not json")
    assert _run(tmp_path, readme, "--check") == 2


def test_an_unexpected_error_in_a_result_file_is_exit_2(tmp_path, capsys):
    # Bug caught: a child JSON with "round": null raising TypeError out of main (Python exits 1,
    # the code this tool reserves for a stale README).
    root = tmp_path / "results"
    d = _scenario(root, "flux1-dev-1024")
    (d / "round1-df11.json").write_text(json.dumps({**_child("df11", 1, [1.2]), "round": None}))
    readme = tmp_path / "README.md"
    readme.write_text(README)
    assert _run(root, readme) == 2
    assert "TypeError" in capsys.readouterr().err
    assert readme.read_text() == README


def test_a_readme_that_cannot_be_written_is_exit_2(tmp_path, capsys):
    # Bug caught: the README write outside the guarded block, so a read-only checkout raises
    # PermissionError out of main (exit 1, the "stale" code).
    root = _full_root(tmp_path)
    folder = tmp_path / "ro"
    folder.mkdir()
    readme = folder / "README.md"
    readme.write_text(README)
    folder.chmod(0o500)
    try:
        assert _run(root, readme) == 2
    finally:
        folder.chmod(0o700)
    assert "README" in capsys.readouterr().err
    assert readme.read_text() == README


def test_the_committed_reports_render_each_command_and_the_dev_runs_skipped_preflight():
    # Against the real committed reports. Bug caught: the dev run's --skip-preflight (it ran while
    # macOS held the battery at 80 % without charging) missing from the README block, or the
    # schnell command missing from it.
    collected = bt.collect(REPO / "bench" / "results")
    assert collected.preflight_skipped == {"flux1-dev-1024": ("not_charging",)}
    block = bt.render_readme(README, collected, date=collected.date)
    for name in ("flux1-dev-1024", "flux1-schnell-1024"):
        assert f"bench/scenarios/{name}.toml`" in block
    assert "flux1-dev-1024.toml` (preflight skipped: not_charging)" in block
