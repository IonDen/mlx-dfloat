"""The FLUX.1 scenario bench: one scenario file in, the step bench's conditions run, a summary out.

Given a scenario TOML (``mlx_dfloat.bench.scenario``), this orchestrator:

1. samples the launch gate (``mlx_dfloat.bench.preflight``) and writes ``preflight.json``; a failed
   gate stops the run with exit 2 unless ``--skip-preflight`` is given, which the report records;
2. resolves the pinned DF11 snapshot and the pinned base snapshot from the local Hugging Face cache
   only (``scripts._q8_rig.pinned_snapshot``, never a download: a missing snapshot is exit 2 with
   the ``hf download`` command that fetches it). The base resolves its encoders and tokenizers,
   plus ``transformer/*`` when ``q8`` is one of the conditions;
3. makes sure ``embeds.safetensors`` in the out dir holds the scenario's prompt encoded for its
   model by its pinned base (``scripts.encode_prompt`` as a subprocess when the file is missing or
   its metadata differs; a mismatched file is moved to ``embeds.previous.safetensors`` first);
4. runs the step bench (``scripts.bench_flux_step --scenario``) once per condition and round, as
   serial subprocesses interleaved per round in the scenario's order. Before any child starts, the
   existing ``round{r}-{condition}.json`` files are checked against this run's key (the key each
   child writes: the scenario hash, the resolved tier, the checkpoint and embeddings, the source
   hash, the mlx version); a file with another key stops the run with exit 2 naming it, and a
   complete file is skipped, so an interrupted run resumes. A child's non-zero exit stops the run
   (exit 2); for a watchdog abort (70 memory, 71 wall) the report names the child's ``abort.json``;
5. pools the complete children (``mlx_dfloat.bench.results``) and writes ``report.json``: the
   scenario and its hash, the summary, the missing runs, the preflight sample and failed gates,
   provenance, where the run stopped, the command that reproduces it, the tier and its limits.

``--tier GB`` runs every child under a smaller Mac's limits (``mlx_dfloat.bench.capped``) into its
own out dir, ``<results root>/<name>-tier<GB>``; without it the out dir is ``<results root>/<name>``.
The home directory is written as ``~`` everywhere in the records.

Usage (from the repository root of a synced checkout; the children need mflux):
    uv run --group bench python -m scripts.bench_flux1 bench/scenarios/flux1-schnell-1024.toml \
        [--tier GB] [--skip-preflight] [--results-root DIR]
Exit codes: 0 every run complete; 2 a failed preflight, a scenario or format error, a resume
conflict, a missing snapshot, an encoder failure or a child's non-zero exit. The orchestrator never
exits 70/71 itself (a child's watchdog abort is reported as exit 2).
"""

import argparse
import dataclasses
import shlex
import subprocess
import sys
import traceback
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# These imports load mlx and the step bench's helpers but allocate no GPU memory and no model: the
# process stays at a few hundred MB, far under the busy gate's 1 GiB RSS threshold, so its own
# command line (which matches the gate's "bench_" pattern) does not trip the gate it samples.
try:
    import mlx.core as mx
    import scripts.bench_flux_step as bfs
    from scripts._bench_common import provenance, write_json_atomic
    from scripts._q8_rig import pinned_snapshot
    from scripts.verify_checkpoint import source_hash

    from mlx_dfloat._scrub import scrub_home
    from mlx_dfloat.bench import preflight
    from mlx_dfloat.bench.capped import host_tier_gb
    from mlx_dfloat.bench.results import (
        Summary,
        expected_missing,
        load_results,
        summarise,
    )
    from mlx_dfloat.bench.scenario import Scenario, load_scenario, scenario_hash
    from mlx_dfloat.errors import DFloatError, DFloatFormatError
except Exception as exc:  # a broken environment is a tool error (2)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_ERROR = 0, 2
WATCHDOG_EXITS = (70, 71)  # the child's footprint ceiling and wall budget aborts
ENCODER_PATTERNS: tuple[str, ...] = (
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
)
EMBEDS_NAME = "embeds.safetensors"
EMBEDS_PREVIOUS = "embeds.previous.safetensors"
DEFAULT_RESULTS_ROOT = _REPO / "bench" / "results"

RunArgv = Callable[[list[str]], int]
Resolve = Callable[..., Path]


# --- pure helpers ---------------------------------------------------------------------------------


def out_dir_for(scenario: Scenario, results_root: Path, *, tier_gb: int | None) -> Path:
    """``results_root / name``, or ``results_root / f"{name}-tier{tier_gb}"`` for a ``--tier`` run."""
    name = scenario.name if tier_gb is None else f"{scenario.name}-tier{tier_gb}"
    return Path(results_root) / name


def embeds_path(out_dir: Path) -> Path:
    """Where a scenario's prompt embeddings live: ``out_dir / "embeds.safetensors"``."""
    return Path(out_dir) / EMBEDS_NAME


def embeds_key(scenario: Scenario) -> dict[str, str]:
    """The embeddings metadata a scenario needs: its prompt, model and pinned base revision."""
    return {
        "prompt": scenario.prompt,
        "model": scenario.model,
        "base_revision": scenario.base_revision,
    }


def embeds_match(meta: Mapping[str, str], key: Mapping[str, str]) -> bool:
    """Whether an embeddings file's metadata carries every field of ``key`` with the same value."""
    return all(field in meta and meta[field] == value for field, value in key.items())


def base_patterns(scenario: Scenario) -> list[str]:
    """The base snapshot's files a scenario needs: the encoders, plus the transformer for ``q8``."""
    head = ["transformer/*"] if "q8" in scenario.conditions else []
    return [*head, *ENCODER_PATTERNS]


def plan_children(
    scenario: Scenario,
    out_dir: Path,
    *,
    key: Mapping[str, Any],
    ignore: Collection[str] = (),
) -> list[tuple[int, str, Path]]:
    """The pending (round, condition, result path) runs, interleaved per round in the scenario's order.

    Key fields named in ``ignore`` are left out of the conflict check (the embeddings metadata,
    before the embeddings file exists).

    Raises:
        BenchError: An existing child file in ``out_dir`` was written under another key (the
            message lists every such file and its differing fields).
    """
    conditions = list(scenario.conditions)
    conflicts = [
        (r, m, kept)
        for r, m, fields in bfs.resume_conflicts(out_dir, scenario.rounds, key, modes=conditions)
        if (kept := [f for f in fields if f not in ignore])
    ]
    if conflicts:
        lines = [
            f"{bfs.redact_home(str(bfs.run_path(out_dir, r, m)))}: differs in {fields}"
            for r, m, fields in conflicts
        ]
        raise bfs.BenchError(
            "existing results were written by another run; use a fresh results root or move "
            "them aside:\n  " + "\n  ".join(lines)
        )
    plan: list[tuple[int, str, Path]] = []
    for r, m in bfs.interleaved(scenario.rounds, modes=conditions):
        path = bfs.run_path(out_dir, r, m)
        if not bfs.result_is_complete(path):
            plan.append((r, m, path))
    return plan


def child_key(
    scenario: Scenario,
    *,
    df11: Path,
    embeds: Path,
    embeds_meta: Mapping[str, str],
    source: str,
    mlx: str,
    tier_gb: int,
) -> dict[str, Any]:
    """The key each child of this run writes (the step bench's ``current_key`` from the same inputs).

    ``tier_gb`` is the resolved tier: the requested ``--tier``, or the host's own tier without one.
    The embeddings path is resolved, as the child's ``parse_args`` resolves it.
    """
    return bfs.run_key(
        model=scenario.model,
        size=scenario.size,
        steps=scenario.steps,
        warmup=scenario.warmup,
        seed=scenario.seed,
        df11=df11,
        embeds=Path(embeds).resolve(),
        embeds_meta=embeds_meta,
        source=source,
        mlx=mlx,
        cache_limit=scenario.cache_limit_bytes,
        scenario_hash=scenario_hash(scenario),
        tier_gb=tier_gb,
    )


def child_argv(
    scenario_file: Path,
    *,
    condition: str,
    round_no: int,
    out: Path,
    df11: Path,
    embeds: Path,
    tier: int | None,
) -> list[str]:
    """One child's command line (run with the repository root as cwd, so every path is absolute).

    Only ``--scenario`` fixes the recipe: no flag the scenario fixes is ever passed.
    """
    cmd = [
        sys.executable,
        "-m",
        "scripts.bench_flux_step",
        *("--scenario", str(Path(scenario_file).resolve())),
        *("--mode", condition, "--round", str(round_no)),
        *("--out", str(Path(out).resolve())),
        *("--df11", str(Path(df11).resolve())),
        *("--embeds", str(Path(embeds).resolve())),
    ]
    if tier is not None:
        cmd += ["--tier", str(tier)]
    return cmd


def encode_argv(scenario: Scenario, *, base_root: Path, out: Path) -> list[str]:
    """The encoder's command line for the scenario's prompt, model and seed from ``base_root``."""
    return [
        sys.executable,
        "-m",
        "scripts.encode_prompt",
        *("--model", scenario.model, "--prompt", scenario.prompt, "--seed", str(scenario.seed)),
        *("--root", str(base_root), "--out", str(out)),
    ]


def reproducer(scenario_file: Path, *, tier: int | None, results_root: Path | None = None) -> str:
    """The command that reproduces the run, with its paths relative to the repository.

    ``results_root`` is the ``--results-root`` the run was given, or None when it used the
    default; the command records it only when it was given.
    """
    words = ["uv", "run", "--group", "bench", "python", "-m", "scripts.bench_flux1"]
    words.append(bfs.shown_path(scenario_file))
    if tier is not None:
        words += ["--tier", str(tier)]
    if results_root is not None:
        words += ["--results-root", bfs.shown_path(results_root)]
    return shlex.join(words)


def report_payload(
    scenario: Scenario,
    summary: Summary,
    *,
    missing: Sequence[str],
    preflight: Mapping[str, Any],
    failed_gates: Sequence[str],
    skipped_preflight: bool,
    provenance: Mapping[str, Any],
    stopped: Mapping[str, Any] | None,
    reproducer: str,
    tier_gb: int,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    """The ``report.json`` record, with the home directory written as ``~`` everywhere."""
    payload = {
        "scenario": dataclasses.asdict(scenario),
        "scenario_hash": scenario_hash(scenario),
        "summary": dataclasses.asdict(summary),
        "missing": list(missing),
        "preflight": dict(preflight),
        "failed_gates": list(failed_gates),
        "skipped_preflight": skipped_preflight,
        "provenance": dict(provenance),
        "stopped": None if stopped is None else dict(stopped),
        "reproducer": reproducer,
        "tier_gb": tier_gb,
        "limits": dict(limits),
    }
    scrubbed: dict[str, Any] = scrub_home(payload)
    return scrubbed


# --- the run --------------------------------------------------------------------------------------


def _run_in_repo(argv: list[str]) -> int:
    return subprocess.run(argv, cwd=_REPO, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: the scenario file and the optional tier, gate and results root."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("scenario", type=Path, help="scenario TOML (bench/scenarios/*.toml)")
    p.add_argument("--tier", type=int, default=None, help="emulate a Mac of this many GB")
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="run even when a launch gate fails (the report records the failed gates)",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="directory holding one out dir per scenario (default: bench/results)",
    )
    args = p.parse_args(argv)
    if args.tier is not None and args.tier < 1:
        p.error("--tier must be >= 1")
    args.results_root_given = args.results_root is not None
    args.results_root = Path(args.results_root or DEFAULT_RESULTS_ROOT).resolve()
    return args


def _embeddings(
    scenario: Scenario, out_dir: Path, base: Path, encode: RunArgv
) -> tuple[Path, dict[str, str]]:
    """The out dir's embeddings file and its metadata, encoded first when missing or mismatched."""
    path = embeds_path(out_dir)
    key = embeds_key(scenario)
    if path.exists():
        try:
            meta: dict[str, str] | None = bfs.embeds_metadata(path)
        except (OSError, ValueError, RuntimeError):
            meta = None  # an unreadable file is re-encoded like a mismatched one
        if meta is not None and embeds_match(meta, key):
            return path, meta
        path.replace(out_dir / EMBEDS_PREVIOUS)  # kept, never deleted
    print(f"encoding the prompt into {bfs.redact_home(str(path))}", flush=True)
    code = encode(encode_argv(scenario, base_root=base, out=path))
    if code != 0:
        raise bfs.BenchError(f"the prompt encoder (scripts.encode_prompt) exited {code}")
    meta = bfs.embeds_metadata(path)
    if not embeds_match(meta, key):
        raise bfs.BenchError(
            f"the prompt encoder wrote embeddings whose metadata does not match {key}"
        )
    return path, meta


def _stopped(out_dir: Path, round_no: int, condition: str, code: int) -> dict[str, Any]:
    abort = out_dir / "abort.json" if code in WATCHDOG_EXITS else None
    return {
        "round": round_no,
        "condition": condition,
        "exit_code": code,
        "abort": None if abort is None else bfs.redact_home(str(abort)),
    }


def _print_summary(payload: Mapping[str, Any], report_path: Path) -> None:
    summary = payload["summary"]
    for label, value in summary["overhead"].items():
        print(f"overhead {label}: {value:+.2%}")
    if summary["eval_cost_s"] is not None:
        print(f"eval policy cost (control - control-noeval): {summary['eval_cost_s']:+.3f} s")
    if summary["q8_ratio"] is not None:
        print(f"q8 ratio (df11 / q8 step time): {summary['q8_ratio']:.3f}")
    if payload["missing"]:
        print(f"missing: {', '.join(payload['missing'])}")
    if payload["stopped"] is not None:
        print(f"stopped: {payload['stopped']}", file=sys.stderr)
    print(f"report: {bfs.redact_home(str(report_path))}")


def _run(
    args: argparse.Namespace,
    scenario: Scenario,
    out_dir: Path,
    gate: Mapping[str, Any],
    failed: Sequence[str],
    *,
    run_child: RunArgv,
    resolve: Resolve,
    encode: RunArgv,
) -> int:
    host = bfs.host_memory()
    tier_gb = host_tier_gb(host["host_ram_bytes"]) if args.tier is None else args.tier
    limits = bfs.limits_for_process(args.tier, **host)
    df11 = resolve(scenario.df11_repo, scenario.df11_revision, allow_patterns=["*"])
    base = resolve(
        scenario.base_repo, scenario.base_revision, allow_patterns=base_patterns(scenario)
    )
    key_inputs: dict[str, Any] = {
        "df11": df11,
        "embeds": embeds_path(out_dir),
        "source": source_hash(),
        "mlx": mx.__version__,
        "tier_gb": tier_gb,
    }
    # The resume check runs twice: first on every key field but the embeddings metadata (not known
    # until the file exists), so a conflicting run refuses before the encoder spends a minute and
    # ~11 GiB; then on the whole key once the embeddings are in place.
    plan_children(
        scenario,
        out_dir,
        key=child_key(scenario, embeds_meta={}, **key_inputs),
        ignore=("embeds_meta",),
    )
    embeds, meta = _embeddings(scenario, out_dir, base, encode)
    key = child_key(scenario, embeds_meta=meta, **key_inputs)
    todo = plan_children(scenario, out_dir, key=key)
    total = scenario.rounds * len(scenario.conditions)
    print(f"{total - len(todo)} run(s) already complete, {len(todo)} to run")
    stopped: dict[str, Any] | None = None
    for round_no, condition, path in todo:
        argv = child_argv(
            args.scenario,
            condition=condition,
            round_no=round_no,
            out=path,
            df11=df11,
            embeds=embeds,
            tier=args.tier,
        )
        print(f"round {round_no} {condition}: {shlex.join(argv)}", flush=True)
        code = run_child(argv)
        if code != 0:
            stopped = _stopped(out_dir, round_no, condition, code)
            print(f"error: round {round_no} {condition} exited {code}; stopping", file=sys.stderr)
            break
    results = load_results(out_dir)
    missing = expected_missing(results, conditions=scenario.conditions, rounds=scenario.rounds)
    payload = report_payload(
        scenario,
        summarise(results),
        missing=missing,
        preflight=gate,
        failed_gates=failed,
        skipped_preflight=bool(args.skip_preflight),
        # The orchestrator installs no caps: it loads no model.
        provenance=provenance((0, 0)),
        stopped=stopped,
        reproducer=reproducer(
            args.scenario,
            tier=args.tier,
            results_root=args.results_root if args.results_root_given else None,
        ),
        tier_gb=tier_gb,
        limits=limits.as_dict(),
    )
    report_path = out_dir / "report.json"
    write_json_atomic(report_path, payload)
    _print_summary(payload, report_path)
    return EXIT_ERROR if stopped is not None or missing else EXIT_OK


def main(
    argv: list[str] | None = None,
    *,
    sample: Callable[[], preflight.Preflight] = preflight.sample,
    check: Callable[[preflight.Preflight], list[str]] = preflight.check,
    run_child: RunArgv = _run_in_repo,
    resolve: Resolve = pinned_snapshot,
    encode: RunArgv = _run_in_repo,
) -> int:
    """Entry point: preflight, snapshots, embeddings, the children in order, then the report.

    The callables are injectable so the flow is tested without a machine probe, the Hub cache,
    the encoder or a child process.
    """
    args = parse_args(argv)
    try:
        scenario = load_scenario(args.scenario)
    except DFloatFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    out_dir = out_dir_for(scenario, args.results_root, tier_gb=args.tier)
    # The gate is sampled first, before any MLX device query, snapshot lookup, encoder or child:
    # this process's command line matches the busy patterns ("bench_"), but its RSS is far under
    # the 1 GiB threshold here, so only another heavy job can trip the busy gate.
    gate_sample = sample()
    failed = list(check(gate_sample))
    gate = gate_sample.as_dict()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        out_dir / "preflight.json",
        scrub_home(
            {
                "preflight": gate,
                "failed_gates": failed,
                "skipped_preflight": bool(args.skip_preflight),
            }
        ),
    )
    if failed and not args.skip_preflight:
        print(
            f"error: the launch gate failed: {', '.join(failed)} (see preflight.json; "
            "--skip-preflight runs anyway and records it)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    try:
        return _run(
            args,
            scenario,
            out_dir,
            gate,
            failed,
            run_child=run_child,
            resolve=resolve,
            encode=encode,
        )
    except (bfs.BenchError, DFloatError) as exc:  # an expected refusal: the message says it all
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # any other setup or report error is a tool error (2), never 1
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
