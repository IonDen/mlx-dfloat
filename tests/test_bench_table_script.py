"""scripts/bench_table.py: collect result files, render the README blocks, the --check mode."""

import json
from pathlib import Path

from scripts import bench_table as bt

from mlx_dfloat.bench.capped import GIB

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
    "height": 512,
    "width": 512,
    "memory_ceiling_bytes": 18 * GIB,
    "watched_peak_bytes": 17 * GIB,
}
ABORT = {
    "reason": "memory",
    "ceiling": 18 * GIB,
    "verdict_counter": "footprint",
    "peak_watched": 19 * GIB,
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


def _scenario(root: Path, name: str, *, reproducer="uv run x", cache=2_500_000_000, skipped=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "report.json").write_text(
        json.dumps(
            {
                "scenario": {"name": name, "cache_limit_bytes": cache},
                "provenance": PROV,
                "reproducer": reproducer,
                "skipped_preflight": skipped is not None,
                "failed_gates": list(skipped or []),
            }
        )
    )
    # df11 1.2 vs control 1.0 -> per-block overhead +20.0 % by hand.
    (d / "round1-df11.json").write_text(json.dumps(_child("df11", 1, [1.2, 1.2, 1.2])))
    (d / "round1-control.json").write_text(json.dumps(_child("control", 1, [1.0, 1.0, 1.0])))


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
    assert c.cache_limit_bytes == 1_400_000_000
    assert c.provenance is not None
    assert c.proof is not None
    assert c.proof.pass_size == 512
    assert c.proof.abort_size == 1024


def test_collect_reads_the_abort_size_when_the_artifact_carries_it(tmp_path):
    # Bug caught: ignoring the artifact's own size and always printing the default 1024.
    root = _full_root(tmp_path)
    (root / "harness-proof" / "abort.json").write_text(json.dumps({**ABORT, "size": 768}))
    proof = bt.collect(root).proof
    assert proof is not None
    assert proof.abort_size == 768


def test_collect_skips_a_proof_dir_with_only_one_file(tmp_path):
    # Bug caught: building a proof from a lone pass report (or crashing on the missing abort).
    root = _full_root(tmp_path)
    (root / "harness-proof" / "abort.json").unlink()
    assert bt.collect(root).proof is None


def test_render_readme_splices_all_three_blocks(tmp_path):
    out = bt.render_readme(README, bt.collect(_full_root(tmp_path)), date="2026-09-30")
    # By hand: 1.2 / 1.0 - 1 = +20.0 %; cache 1_400_000_000 B -> "1.4 GB"; caption date passed in.
    assert "| 32 GB |" in out
    assert "per-block evaluation: +20.0 %" in out
    assert "under a 1.4 GB MLX buffer-cache limit" in out
    assert "Harness proof: under one 18.00 GiB cap" in out
    assert (
        "Apple M1 Max, 32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, git abc1234, 2026-09-30" in out
    )
    assert NONE not in out


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
