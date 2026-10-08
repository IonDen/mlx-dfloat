"""The scenario orchestrator: out dirs, embeddings reuse, the child plan and key, the report, ``main``.

``main`` is driven with fakes for the preflight sample, the snapshot resolver, the encoder and the
children; a fake child parses its argv with the step bench's own ``parse_args`` and keys its JSON
with the step bench's own ``current_key``, so resume is checked against the real key the child
writes. Nothing here loads a model, downloads or starts a subprocess.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest
import scripts.bench_flux1 as bf1
import scripts.bench_flux_step as bfs
from scripts.encode_prompt import parse_args as encode_parse_args

from mlx_dfloat.bench.preflight import Preflight
from mlx_dfloat.bench.results import Summary
from mlx_dfloat.bench.scenario import scenario_from_mapping, scenario_hash
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "bench/scenarios/flux1-schnell-1024.toml"
GIB = 1024**3
DF11_REV = "51a428b928197e0531cb93d6e438941e2d0b247e"
BASE_REV = "741f7c3ce8b383c54771c7003378a50191e9efe9"
SPEC = {
    "name": "flux-test",
    "model": "schnell",
    "df11_repo": "DFloat11/FLUX.1-schnell-DF11",
    "df11_revision": DF11_REV,
    "base_repo": "black-forest-labs/FLUX.1-schnell",
    "base_revision": BASE_REV,
    "prompt": "a lighthouse",
    "seed": 42,
    "steps": 2,
    "warmup": 1,
    "size": 256,
    "rounds": 2,
    "cache_limit_bytes": 2_500_000_000,
    "conditions": ["df11", "control", "q8"],
    "wall_budget_s": 600,
}
ENCODER_PATTERNS = ["text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*"]
HOST = {"host_ram_bytes": 32 * GIB, "host_recommended_bytes": 24 * GIB}
GOOD_PREFLIGHT = Preflight(
    ac_power=True,
    battery_percent=None,
    charging=None,
    charger_watts=96,
    cpu_speed_limit=100,
    lid_open=None,
    free_disk_bytes=100 * GIB,
    memory_free_percent=60,
    busy_processes=(),
)


def _toml(data):
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k} = [" + ", ".join(f'"{x}"' for x in v) + "]")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def _scenario_file(tmp_path, **changes):
    path = tmp_path / "scenario.toml"
    path.write_text(_toml({**SPEC, **changes}))
    return path


def _write_embeds(path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), {"prompt_embeds": mx.zeros((1, 4))}, metadata=meta)


# --- pure helpers ---------------------------------------------------------------------------------


def test_out_dir_is_the_scenario_name_and_a_tier_gets_its_own_dir(tmp_path):
    # Bug caught: a tier run writing into the host run's directory, where its CAPPED children would
    # conflict with (or, keyed alike, resume as) the MEASURED ones.
    s = scenario_from_mapping(SPEC)
    assert bf1.out_dir_for(s, tmp_path, tier_gb=None) == tmp_path / "flux-test"
    assert bf1.out_dir_for(s, tmp_path, tier_gb=24) == tmp_path / "flux-test-tier24"


def test_out_dir_refuses_a_directory_that_resolves_outside_the_results_root(tmp_path):
    # Bug caught: a scenario dir that is a symlink to elsewhere accepted, so the bench writes its
    # records (and moves embeddings aside) outside the results root it was given.
    root = tmp_path / "res"
    root.mkdir()
    (tmp_path / "elsewhere").mkdir()
    (root / "flux-test").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(DFloatFormatError, match="outside"):
        bf1.out_dir_for(scenario_from_mapping(SPEC), root, tier_gb=None)


def test_embeds_key_names_prompt_model_and_base_revision():
    # Bug caught: a key without the base revision, so embeddings from another snapshot's encoders
    # are reused for a scenario that pins a different base.
    assert bf1.embeds_key(scenario_from_mapping(SPEC)) == {
        "prompt": "a lighthouse",
        "model": "schnell",
        "base_revision": BASE_REV,
    }


MATCHING_META = {
    "prompt": "a lighthouse",
    "model": "schnell",
    "base_revision": BASE_REV,
    "seed": "42",
    "synthetic": "false",
}


@pytest.mark.parametrize(
    ("changes", "want"),
    [
        ({}, True),
        ({"prompt": "a lighthouse at dusk"}, False),
        ({"model": "dev"}, False),
        ({"base_revision": "0" * 40}, False),
        ({"base_revision": None}, False),  # an embeddings file from before the field existed
    ],
)
def test_embeds_match_needs_every_key_field(changes, want):
    # Bug caught (per row): a field left out of the comparison, so embeddings of another prompt,
    # model or encoder snapshot are timed under this scenario; or a missing field read as a match.
    meta = {k: v for k, v in {**MATCHING_META, **changes}.items() if v is not None}
    key = bf1.embeds_key(scenario_from_mapping(SPEC))
    assert bf1.embeds_match(meta, key) is want


@pytest.mark.parametrize(
    ("conditions", "want"),
    [
        (["df11", "control"], ENCODER_PATTERNS),
        (["df11", "q8"], ["transformer/*", *ENCODER_PATTERNS]),
    ],
)
def test_base_patterns_add_the_transformer_only_for_q8(conditions, want):
    # Bug caught: the q8 build's BF16 transformer shards never checked before the children start
    # (the q8 child would fail minutes in), or a non-q8 scenario refused for lacking 23.8 GB of
    # shards it never reads.
    assert bf1.base_patterns(scenario_from_mapping({**SPEC, "conditions": conditions})) == want


def _child_json(path, *, key, round_no, mode):
    write = {"round": round_no, "mode": mode, "exit_code": 0, "step_s": [1.0], "key": key}
    path.write_text(json.dumps(write))


def test_plan_children_skips_complete_runs_in_interleaved_order(tmp_path):
    # Bug caught: condition-major order (all df11 rounds, then all control), which loses the pairing
    # against machine drift; or a complete child re-run (a finished result overwritten).
    s = scenario_from_mapping(SPEC)
    key = {"model": "schnell"}
    _child_json(tmp_path / "round1-control.json", key=key, round_no=1, mode="control")
    plan = bf1.plan_children(s, tmp_path, key=key)
    assert plan == [
        (1, "df11", tmp_path / "round1-df11.json"),
        (1, "q8", tmp_path / "round1-q8.json"),
        (2, "df11", tmp_path / "round2-df11.json"),
        (2, "control", tmp_path / "round2-control.json"),
        (2, "q8", tmp_path / "round2-q8.json"),
    ]


def test_plan_children_refuses_a_dir_whose_child_key_differs(tmp_path):
    # Bug caught: plan_children skipping the resume-key check, so a run with another
    # mlx version or source hash resumes into (and pools with) results it did not produce.
    s = scenario_from_mapping(SPEC)
    _child_json(tmp_path / "round2-q8.json", key={"model": "dev"}, round_no=2, mode="q8")
    with pytest.raises(bfs.BenchError, match=r"round2-q8\.json"):
        bf1.plan_children(s, tmp_path, key={"model": "schnell"})


def _argv(tmp_path, scenario_file, *, tier):
    return bf1.child_argv(
        scenario_file,
        condition="q8",
        round_no=2,
        out=tmp_path / "out" / "round2-q8.json",
        df11=tmp_path / DF11_REV,
        embeds=tmp_path / "out" / "embeds.safetensors",
        tier=tier,
    )


@pytest.mark.parametrize("tier", [None, 24])
def test_child_argv_round_trips_through_the_step_bench_parser(tmp_path, tier):
    # Bug caught: a child argv carrying a flag the scenario fixes (the child refuses to start), a
    # dropped --round or --tier (the round's JSON or the tier's limits go missing), or a module
    # other than the step bench.
    scenario_file = _scenario_file(tmp_path)
    argv = _argv(tmp_path, scenario_file, tier=tier)
    assert argv[1:3] == ["-m", "scripts.bench_flux_step"]
    assert bfs.conflicting_flags(argv[3:]) == []
    args = bfs.parse_args(argv[3:])
    assert (args.mode, args.round, args.tier) == ("q8", 2, tier)
    assert args.out == (tmp_path / "out" / "round2-q8.json").resolve()
    assert args.df11 == (tmp_path / DF11_REV).resolve()
    assert args.scenario_hash == scenario_hash(scenario_from_mapping(SPEC))


def test_child_argv_paths_are_absolute(tmp_path, monkeypatch):
    # Bug caught: relative paths passed on; the child runs with the repository root as cwd and
    # would resolve them there, writing its JSON outside the out dir.
    scenario_file = _scenario_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = bf1.child_argv(
        Path("scenario.toml"),
        condition="df11",
        round_no=1,
        out=Path("out/round1-df11.json"),
        df11=Path(DF11_REV),
        embeds=Path("out/embeds.safetensors"),
        tier=None,
    )
    for flag, want in [
        ("--scenario", scenario_file),
        ("--out", tmp_path / "out/round1-df11.json"),
        ("--df11", tmp_path / DF11_REV),
        ("--embeds", tmp_path / "out/embeds.safetensors"),
    ]:
        assert argv[argv.index(flag) + 1] == str(want.resolve())


@pytest.mark.parametrize("tier", [None, 24])
def test_child_key_equals_the_key_the_child_writes(tmp_path, monkeypatch, tier):
    # The test that proves resume can work. Bug caught: any input of child_key differing from the
    # child's current_key (the scenario hash left out, tier None where the child keys the host
    # tier 32): every complete child would read as a conflict.
    meta = {"prompt": "a lighthouse", "root": "~/snap"}
    monkeypatch.setattr(bfs, "embeds_metadata", lambda path: meta)
    monkeypatch.setattr(bfs, "source_hash", lambda: "src-abc")
    monkeypatch.setattr(bfs, "host_memory", lambda: HOST)
    scenario_file = _scenario_file(tmp_path)
    argv = _argv(tmp_path, scenario_file, tier=tier)
    child = bfs.current_key(bfs.parse_args(argv[3:]))
    ours = bf1.child_key(
        scenario_from_mapping(SPEC),
        df11=tmp_path / DF11_REV,
        embeds=tmp_path / "out" / "embeds.safetensors",
        embeds_meta=meta,
        source="src-abc",
        mlx=mx.__version__,
        tier_gb=32 if tier is None else tier,
    )
    assert ours == child
    assert ours["tier_gb"] == (32 if tier is None else 24)


def test_child_key_resolves_a_relative_embeds_path_as_the_child_does(tmp_path, monkeypatch):
    # Bug caught: child_key keying the embeddings path as given; the child's parse_args resolves
    # it, so a relative path would key "out/embeds.safetensors" against the child's absolute one.
    monkeypatch.setattr(bfs, "embeds_metadata", lambda path: {})
    monkeypatch.setattr(bfs, "source_hash", lambda: "s")
    monkeypatch.chdir(tmp_path)
    argv = _argv(tmp_path, _scenario_file(tmp_path), tier=24)
    child = bfs.current_key(bfs.parse_args(argv[3:]))
    ours = bf1.child_key(
        scenario_from_mapping(SPEC),
        df11=Path(DF11_REV),
        embeds=Path("out/embeds.safetensors"),
        embeds_meta={},
        source="s",
        mlx=mx.__version__,
        tier_gb=24,
    )
    assert ours == child


def test_encode_argv_round_trips_through_the_encoder_parser(tmp_path):
    # Bug caught: the scenario's prompt or model not passed (the encoder's default prompt, or T5 at
    # schnell's 256 tokens for dev), or --synthetic.
    s = scenario_from_mapping({**SPEC, "model": "dev", "prompt": "two words"})
    argv = bf1.encode_argv(s, base_root=tmp_path / BASE_REV, out=tmp_path / "e.safetensors")
    assert argv[1:3] == ["-m", "scripts.encode_prompt"]
    args = encode_parse_args(argv[3:])
    assert (args.model, args.prompt, args.seed, args.synthetic) == ("dev", "two words", 42, False)
    assert args.root == tmp_path / BASE_REV
    assert args.out == tmp_path / "e.safetensors"


def test_scrub_home_rewrites_every_home_occurrence_in_nested_values():
    # Bug caught: only a leading home prefix rewritten (a ps command line names the home in its
    # middle), or a sibling directory sharing the prefix (/Users/ab2) mangled.
    home = "/Users/ab"
    data = {
        "a": ["python /Users/ab/x.py --df11 /Users/ab/snap", "/Users/ab2/y"],
        "b": {"c": "/Users/ab"},
        "n": 3,
    }
    assert bf1.scrub_home(data, home=home) == {
        "a": ["python ~/x.py --df11 ~/snap", "/Users/ab2/y"],
        "b": {"c": "~"},
        "n": 3,
    }


def test_report_payload_carries_every_record_without_the_home_dir():
    # Bug caught: a report missing the scenario, its hash, the tier's limits or the reproducer (a
    # table row would have no recipe to cite), or a busy process line leaking the home directory.
    s = scenario_from_mapping(SPEC)
    summary = Summary(
        scenario_hash="h",
        rounds_seen=1,
        conditions={},
        overhead={"per-block": 0.05},
        eval_cost_s=None,
        q8_ratio=None,
        paired_rounds={"per-block": [1]},
        pair_n={"per-block": 4},
    )
    home = str(Path.home())
    payload = bf1.report_payload(
        s,
        summary,
        missing=["round 2: q8"],
        preflight={"busy_processes": [f"{home}/venv/bin/python -m mflux"]},
        failed_gates=["busy"],
        skipped_preflight=True,
        provenance={"git": "abc"},
        stopped=None,
        reproducer="uv run --group bench python -m scripts.bench_flux1 s.toml",
        tier_gb=32,
        limits={"tier_gb": 32},
    )
    assert payload["scenario"]["prompt"] == "a lighthouse"
    assert payload["scenario_hash"] == scenario_hash(s)
    assert payload["summary"]["overhead"] == {"per-block": 0.05}
    assert payload["missing"] == ["round 2: q8"]
    assert payload["preflight"]["busy_processes"] == ["~/venv/bin/python -m mflux"]
    assert payload["failed_gates"] == ["busy"]
    assert payload["skipped_preflight"] is True
    assert payload["stopped"] is None
    assert payload["tier_gb"] == 32
    assert payload["limits"] == {"tier_gb": 32}
    assert payload["provenance"] == {"git": "abc"}
    assert payload["reproducer"].endswith("scripts.bench_flux1 s.toml")
    assert home + "/" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("scenario_file", "tier", "want"),
    [
        (COMMITTED, None, "bench/scenarios/flux1-schnell-1024.toml"),
        (COMMITTED, 24, "bench/scenarios/flux1-schnell-1024.toml --tier 24"),
    ],
)
def test_reproducer_names_the_repo_relative_scenario_and_the_tier(scenario_file, tier, want):
    # Bug caught: an absolute path in the reproducer (it names the user and does not run on another
    # checkout), or the --tier dropped (the command would reproduce the host run instead).
    assert bf1.reproducer(scenario_file, tier=tier) == (
        f"uv run --group bench python -m scripts.bench_flux1 {want}"
    )


@pytest.mark.parametrize(
    ("root", "want"),
    [
        (Path("/fresh/results"), " --results-root /fresh/results"),
        (Path.home() / "fresh", " --results-root '~/fresh'"),
        (REPO / "bench" / "results", " --results-root bench/results"),
        (None, ""),
    ],
)
def test_reproducer_records_the_results_root_only_when_one_was_given(root, want):
    # Bug caught: the reproducer dropping --results-root (the recorded command would rerun into the
    # committed bench/results instead of where the run went), adding one nobody passed, or writing
    # the home directory.
    assert bf1.reproducer(COMMITTED, tier=None, results_root=root) == (
        "uv run --group bench python -m scripts.bench_flux1 "
        f"bench/scenarios/flux1-schnell-1024.toml{want}"
    )


def test_parse_args_tells_a_given_results_root_from_the_default(tmp_path):
    # Bug caught: the default root reported as given (every reproducer would carry
    # --results-root bench/results), or a given one lost before the reproducer sees it.
    given = bf1.parse_args([str(COMMITTED), "--results-root", str(tmp_path)])
    default = bf1.parse_args([str(COMMITTED)])
    assert (given.results_root, given.results_root_given) == (tmp_path.resolve(), True)
    assert (default.results_root, default.results_root_given) == (
        bf1.DEFAULT_RESULTS_ROOT.resolve(),
        False,
    )


# --- main with fakes ------------------------------------------------------------------------------

STEP_S = {"df11": [1.1, 1.1], "control": [1.0, 1.0], "q8": [2.2, 2.2]}


class Rig:
    """Fakes for main's injectables; every call is recorded."""

    def __init__(self, tmp_path, monkeypatch, *, codes=None, preflight=GOOD_PREFLIGHT, samples=()):
        monkeypatch.setattr(bfs, "host_memory", lambda: HOST)
        self.samples = list(samples)  # returned first, one per call, then `preflight` for good
        self.sampled = 0
        self.snaps = tmp_path / "hub"
        self.children = []
        self.encodes = []
        self.resolved = []
        self.codes = dict(codes or {})
        self.preflight = preflight

    def sample(self):
        self.sampled += 1
        return self.samples.pop(0) if self.samples else self.preflight

    def resolve(self, repo_id, revision, *, allow_patterns):
        self.resolved.append((repo_id, revision, list(allow_patterns)))
        root = self.snaps / repo_id.replace("/", "--") / revision
        root.mkdir(parents=True, exist_ok=True)
        return root

    def encode(self, argv):
        args = encode_parse_args(argv[3:])
        self.encodes.append(args.prompt)
        meta = {
            "prompt": args.prompt,
            "model": args.model,
            "seed": str(args.seed),
            "base_revision": args.root.name,
        }
        _write_embeds(args.out, meta)
        return 0

    def run_child(self, argv):
        args = bfs.parse_args(argv[3:])
        self.children.append((args.round, args.mode))
        code = self.codes.get((args.round, args.mode), 0)
        if code in (70, 71):
            (args.out.parent / "abort.json").write_text("{}")
        if code != 0 or (args.round, args.mode) in self.codes:
            return code
        warmup_launches = [0] * args.warmup
        args.out.write_text(
            json.dumps(
                {
                    "key": bfs.current_key(args),
                    "round": args.round,
                    "mode": args.mode,
                    "exit_code": 0,
                    "step_s": STEP_S.get(args.mode, [1.0, 1.0]),
                    "warmup": args.warmup,
                    "launches_per_step": [*warmup_launches, 0, 0],
                    "launches_expected_per_step": 0,
                    "step_footprint_peak_bytes": 1,
                    "step_mlx_peak_bytes": 1,
                    "step_watched_peak_bytes": 1,
                    "footprint_peak_bytes": 1,
                    "mlx_peak_memory_bytes": 1,
                    "watched_peak_bytes": 1,
                    "scenario_hash": args.scenario_hash,
                    "label": "MEASURED",
                    "limits": {},
                }
            )
        )
        return 0

    def main(self, argv):
        return bf1.main(
            argv,
            sample=self.sample,
            run_child=self.run_child,
            resolve=self.resolve,
            encode=self.encode,
        )


def _report(out_dir):
    return json.loads((out_dir / "report.json").read_text())


def test_main_happy_path_runs_every_child_and_writes_the_report(tmp_path, monkeypatch):
    # Bug caught: a child skipped or run out of the interleaved order, the report without the
    # reproducer or the pooled overhead, or the DF11 repo resolved with a pattern subset (the child
    # would find groups missing).
    rig = Rig(tmp_path, monkeypatch)
    code = rig.main([str(COMMITTED), "--results-root", str(tmp_path / "res")])
    assert code == 0
    conditions = ["df11", "control", "df11-depth2", "control-depth2", "control-noeval", "q8"]
    assert rig.children == [(r, c) for r in (1, 2, 3) for c in conditions]
    assert rig.resolved[0] == ("DFloat11/FLUX.1-schnell-DF11", DF11_REV, ["*"])
    assert rig.resolved[1] == (
        "black-forest-labs/FLUX.1-schnell",
        BASE_REV,
        ["transformer/*", *ENCODER_PATTERNS],
    )
    out = tmp_path / "res" / "flux1-schnell-1024"
    rep = _report(out)
    assert rep["reproducer"] == (
        "uv run --group bench python -m scripts.bench_flux1 bench/scenarios/flux1-schnell-1024.toml"
        f" --results-root {bfs.redact_home(str((tmp_path / 'res').resolve()))}"
    )
    # df11 1.1 s against control 1.0 s: +10 %; df11 1.1 s against q8 2.2 s: 0.5.
    assert rep["summary"]["overhead"]["per-block"] == pytest.approx(0.1)
    assert rep["summary"]["q8_ratio"] == pytest.approx(0.5)
    assert rep["missing"] == []
    assert rep["stopped"] is None
    assert rep["skipped_preflight"] is False
    assert rep["failed_gates"] == []
    assert rep["tier_gb"] == 32
    assert rep["limits"]["tier_gb"] == 32
    assert rep["scenario"]["name"] == "flux1-schnell-1024"
    assert json.loads((out / "preflight.json").read_text())["failed_gates"] == []


def test_main_resumes_with_nothing_to_run_and_no_re_encode(tmp_path, monkeypatch):
    # Bug caught: the orchestrator's key differing from the children's (every rerun would stop on
    # a conflict), complete children re-run, or matching embeddings encoded again.
    rig = Rig(tmp_path, monkeypatch)
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path / "res")]
    assert rig.main(argv) == 0
    assert len(rig.children) == 6
    assert rig.main(argv) == 0
    assert len(rig.children) == 6
    assert rig.encodes == ["a lighthouse"]


def test_main_re_encodes_mismatched_embeddings_and_keeps_the_old_file(tmp_path, monkeypatch):
    # Bug caught: embeddings of another prompt reused because the file exists, or the old file
    # overwritten instead of moved aside.
    rig = Rig(tmp_path, monkeypatch)
    out = tmp_path / "res" / "flux-test"
    _write_embeds(out / "embeds.safetensors", {**MATCHING_META, "prompt": "old"})
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path / "res")]
    assert rig.main(argv) == 0
    assert rig.encodes == ["a lighthouse"]
    assert bfs.embeds_metadata(out / "embeds.safetensors")["prompt"] == "a lighthouse"
    assert bfs.embeds_metadata(out / "embeds.previous.safetensors")["prompt"] == "old"


def test_main_reuses_matching_embeddings(tmp_path, monkeypatch):
    # Bug caught: a matching file encoded again (a minute and ~11 GB of encoder per launch).
    rig = Rig(tmp_path, monkeypatch)
    out = tmp_path / "res" / "flux-test"
    _write_embeds(out / "embeds.safetensors", MATCHING_META)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path / "res")]) == 0
    assert rig.encodes == []


def test_main_stops_on_an_encoder_failure(tmp_path, monkeypatch, capsys):
    # Bug caught: an encoder failure ignored, so every child starts on a missing embeddings file.
    rig = Rig(tmp_path, monkeypatch)
    monkeypatch.setattr(rig, "encode", lambda argv: 2)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert rig.children == []
    err = capsys.readouterr().err
    assert "the prompt encoder (scripts.encode_prompt) exited 2" in err
    assert "Traceback" not in err  # an expected refusal, not a crash


def test_main_refuses_on_a_failed_preflight_and_runs_nothing(tmp_path, monkeypatch, capsys):
    # Bug caught: the gate result ignored (children launched while another heavy job holds the GPU),
    # or the refused launch writing into the out dir (a committed preflight.json overwritten by a
    # run that never started).
    busy = GOOD_PREFLIGHT.__class__(
        **{**GOOD_PREFLIGHT.as_dict(), "busy_processes": ("python (pid 7)",)}
    )
    rig = Rig(tmp_path, monkeypatch, preflight=busy)
    code = rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path / "res")])
    assert code == 2
    assert rig.children == []
    assert rig.resolved == []
    assert not (tmp_path / "res" / "flux-test").exists()
    err = capsys.readouterr().err
    assert "busy" in err
    assert "python (pid 7)" in err


def test_main_stops_before_a_child_when_the_gate_fails_mid_run_and_a_rerun_resumes(
    tmp_path, monkeypatch, capsys
):
    # Bug caught: the gate sampled only at launch, so a run that started on a charged battery keeps
    # timing children after the adapter fell behind; or the stop not recorded (a partial set would
    # read as complete); or a rerun redoing the children that finished.
    on_battery = GOOD_PREFLIGHT.__class__(**{**GOOD_PREFLIGHT.as_dict(), "ac_power": False})
    # launch, then before children 1 and 2 the gate passes; before child 3 it fails
    rig = Rig(
        tmp_path,
        monkeypatch,
        samples=[GOOD_PREFLIGHT, GOOD_PREFLIGHT, GOOD_PREFLIGHT, on_battery],
    )
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]
    assert rig.main(argv) == 2
    assert rig.children == [(1, "df11"), (1, "control")]
    rep = _report(tmp_path / "flux-test")
    assert rep["stopped"] == {
        "round": 1,
        "condition": "q8",
        "reason": "gate",
        "failed_gates": ["ac_power"],
        "exit_code": None,
        "abort": None,
    }
    assert rep["missing"] == ["round 1: q8", "round 2: df11", "round 2: control", "round 2: q8"]
    assert "ac_power" in capsys.readouterr().err
    assert rig.main(argv) == 0
    assert rig.children[2:] == [(1, "q8"), (2, "df11"), (2, "control"), (2, "q8")]
    assert _report(tmp_path / "flux-test")["stopped"] is None


def test_main_with_skip_preflight_samples_the_gate_once(tmp_path, monkeypatch):
    # Bug caught: --skip-preflight still stopping mid-run on a gate the user chose to skip.
    on_battery = GOOD_PREFLIGHT.__class__(**{**GOOD_PREFLIGHT.as_dict(), "ac_power": False})
    rig = Rig(tmp_path, monkeypatch, preflight=on_battery)
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path), "--skip-preflight"]
    assert rig.main(argv) == 0
    assert len(rig.children) == 6
    assert rig.sampled == 1


def test_main_skip_preflight_runs_and_records_the_failed_gates(tmp_path, monkeypatch):
    # Bug caught: --skip-preflight not recorded, so a run on battery reads like a gated one.
    on_battery = GOOD_PREFLIGHT.__class__(**{**GOOD_PREFLIGHT.as_dict(), "ac_power": False})
    rig = Rig(tmp_path, monkeypatch, preflight=on_battery)
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path), "--skip-preflight"]
    assert rig.main(argv) == 0
    assert len(rig.children) == 6
    rep = _report(tmp_path / "flux-test")
    assert rep["skipped_preflight"] is True
    assert rep["failed_gates"] == ["ac_power"]


def test_main_stops_at_a_watchdog_abort_and_names_the_artifact(tmp_path, monkeypatch):
    # Bug caught: the orchestration carrying on after a memory abort (the next child starts under
    # the same pressure), the orchestrator exiting 70 itself, or the abort artifact not named.
    rig = Rig(tmp_path, monkeypatch, codes={(1, "control"): 70})
    code = rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)])
    assert code == 2
    assert rig.children == [(1, "df11"), (1, "control")]
    rep = _report(tmp_path / "flux-test")
    assert rep["stopped"]["round"] == 1
    assert rep["stopped"]["condition"] == "control"
    assert rep["stopped"]["exit_code"] == 70
    assert rep["stopped"]["reason"] == "child_exit"
    assert rep["stopped"]["abort"].endswith("flux-test/abort.json")
    assert rep["missing"] == [
        "round 1: control",
        "round 1: q8",
        "round 2: df11",
        "round 2: control",
        "round 2: q8",
    ]


def test_main_a_child_failure_has_no_abort_artifact(tmp_path, monkeypatch):
    # Bug caught: an exit-2 child reported with an abort.json path it never wrote.
    rig = Rig(tmp_path, monkeypatch, codes={(1, "df11"): 2})
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert _report(tmp_path / "flux-test")["stopped"]["abort"] is None


def test_main_a_child_exiting_zero_without_a_result_is_not_complete(tmp_path, monkeypatch):
    # Bug caught: exit 0 for a run whose result files are missing (the table would be built from a
    # partial set without a word).
    rig = Rig(tmp_path, monkeypatch, codes={(2, "q8"): 0})
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert _report(tmp_path / "flux-test")["missing"] == ["round 2: q8"]


def test_main_refuses_a_resume_conflict_before_any_child(tmp_path, monkeypatch, capsys):
    # Bug caught: the conflict found only after children ran (or never), mixing two recipes' runs;
    # or the refused launch rewriting the committed preflight.json with today's machine sample.
    rig = Rig(tmp_path, monkeypatch)
    out = tmp_path / "flux-test"
    out.mkdir()
    _child_json(out / "round1-df11.json", key={"model": "dev"}, round_no=1, mode="df11")
    committed = '{"preflight": {"battery_percent": 100}, "failed_gates": []}\n'
    (out / "preflight.json").write_text(committed)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert rig.children == []
    assert "round1-df11.json" in capsys.readouterr().err
    assert (out / "preflight.json").read_text() == committed
    assert sorted(p.name for p in out.iterdir()) == ["preflight.json", "round1-df11.json"]


def test_main_refuses_a_resume_conflict_before_the_encoder_runs(tmp_path, monkeypatch, capsys):
    # Bug caught: the conflict check placed after the embeddings step, so a run that is going to
    # refuse first spends about a minute and ~11 GiB on the T5+CLIP encode it then throws away.
    rig = Rig(tmp_path, monkeypatch)
    out = tmp_path / "flux-test"
    out.mkdir()
    _child_json(out / "round2-q8.json", key={"model": "dev"}, round_no=2, mode="q8")
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert rig.encodes == []
    assert rig.children == []
    assert "round2-q8.json" in capsys.readouterr().err


def test_main_a_missing_snapshot_is_exit_2_with_the_download_hint(tmp_path, monkeypatch, capsys):
    # Bug caught: a missing snapshot downloaded behind the user's back, or an uncaught error
    # (exit 1, the bit-mismatch code).
    rig = Rig(tmp_path, monkeypatch)

    def missing(repo_id, revision, *, allow_patterns):
        raise DFloatIntegrationError(f"{repo_id} is not cached; hf download {repo_id}")

    monkeypatch.setattr(rig, "resolve", missing)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert rig.children == []
    assert rig.encodes == []
    assert "hf download" in capsys.readouterr().err


def test_main_tier_run_uses_its_own_dir_the_tier_and_its_limits(tmp_path, monkeypatch):
    # Bug caught: --tier not forwarded to the children (they would run under the host caps while the
    # report claims the tier), or the tier results written into the host run's directory.
    rig = Rig(tmp_path, monkeypatch)
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path), "--tier", "24"]
    assert rig.main(argv) == 0
    rep = _report(tmp_path / "flux-test-tier24")
    assert rep["tier_gb"] == 24
    assert rep["limits"]["tier_gb"] == 24
    assert rep["limits"]["is_host"] is False
    child = json.loads((tmp_path / "flux-test-tier24" / "round1-df11.json").read_text())
    assert child["key"]["tier_gb"] == 24
    assert " --tier 24 --results-root " in rep["reproducer"]


def test_main_a_tier_above_the_host_is_exit_2(tmp_path, monkeypatch):
    # Bug caught: an unsupported tier raising out of main (exit 1) or running on the host caps.
    rig = Rig(tmp_path, monkeypatch)
    argv = [str(_scenario_file(tmp_path)), "--results-root", str(tmp_path), "--tier", "64"]
    assert rig.main(argv) == 2
    assert rig.children == []


def test_main_a_bad_scenario_is_exit_2_before_the_preflight(tmp_path, monkeypatch):
    # Bug caught: a scenario error surfacing as a traceback (exit 1) or after the machine was probed.
    rig = Rig(tmp_path, monkeypatch)
    sampled = []
    monkeypatch.setattr(rig, "sample", lambda: sampled.append(1) or GOOD_PREFLIGHT)
    bad = _scenario_file(tmp_path, size=1000)
    assert rig.main([str(bad), "--results-root", str(tmp_path)]) == 2
    assert sampled == []


def test_main_re_encodes_an_unreadable_embeddings_file(tmp_path, monkeypatch):
    # Bug caught: a truncated embeddings file (an encoder killed mid-write) crashing every launch
    # instead of being moved aside and encoded again.
    rig = Rig(tmp_path, monkeypatch)
    out = tmp_path / "flux-test"
    out.mkdir()
    (out / "embeds.safetensors").write_text("not a safetensors file")
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 0
    assert rig.encodes == ["a lighthouse"]
    assert (out / "embeds.previous.safetensors").read_text() == "not a safetensors file"


def test_main_refuses_embeddings_the_encoder_wrote_for_another_base(tmp_path, monkeypatch, capsys):
    # Bug caught: the encoder's output trusted without a check, so embeddings from another snapshot
    # (an encoder that ignored --root) are timed under this scenario.
    rig = Rig(tmp_path, monkeypatch)

    def wrong_base(argv):
        out = encode_parse_args(argv[3:]).out
        _write_embeds(out, {**MATCHING_META, "base_revision": "0" * 40})
        return 0

    monkeypatch.setattr(rig, "encode", wrong_base)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2
    assert rig.children == []
    assert "does not match" in capsys.readouterr().err


def test_main_an_unexpected_error_is_exit_2_never_1(tmp_path, monkeypatch):
    # Bug caught: an unexpected exception escaping main; Python would exit 1, which this project
    # reserves for a real bit mismatch.
    rig = Rig(tmp_path, monkeypatch)

    def cannot_start(argv):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(rig, "run_child", cannot_start)
    assert rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path)]) == 2


def test_main_refuses_a_tier_below_one(tmp_path, monkeypatch, capsys):
    # Bug caught: --tier 0 accepted by the orchestrator, which would probe the machine, create
    # an out dir and leave the refusal to the first child's parser. It must refuse up front.
    rig = Rig(tmp_path, monkeypatch)
    sampled = []
    monkeypatch.setattr(rig, "sample", lambda: sampled.append(1) or GOOD_PREFLIGHT)
    with pytest.raises(SystemExit) as exc:
        rig.main([str(_scenario_file(tmp_path)), "--results-root", str(tmp_path), "--tier", "0"])
    assert exc.value.code == 2
    assert sampled == []
    assert not (tmp_path / "flux-test-tier0").exists()
    assert "--tier must be >= 1" in capsys.readouterr().err
