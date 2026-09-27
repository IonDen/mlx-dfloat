"""Control validation: does the step bench's control cost what a plain resident-BF16 transformer costs?

The step bench (``scripts/bench_flux_step.py``) times DF11 against a control that hands every block
the same two pre-decoded weight dicts. This script checks that control on a reduced-depth FLUX.1
transformer (``--double 4 --single 8`` by default, 12 blocks), small enough that every block's
group can be decoded once and kept resident as BF16. It runs three modes, one per process, with
the same seam and the per-block eval policy:

- ``bf16``: the 12 groups decoded once by the Metal kernel into per-block resident dicts
  (``ResidentProvider``); no launches in the step. The plain BF16 transformer.
- ``control``: the step bench's control (``ReuseProvider`` over one decoded double and one decoded
  single block); no launches.
- ``df11``: just-in-time decode (``DF11Provider``), one launch per block per step (12).

Every mode loads the same 12 compressed groups first, so the footprint baseline is the same;
``bf16`` holds its 12 decoded groups on top. The step itself, its timing window, the warm-up, the
parity conditions, the embeddings and latents handling and the run JSON are the step bench's
(``step_inputs``, ``time_steps``, ``run_one``); the key adds the depth.

Without ``--mode`` the script orchestrates: rounds interleave ``bf16, control, df11``, one child at
a time (``python -m scripts.bench_control_validation --mode M --round R --out FILE``, run from the
repository root), each writing ``OUT/round{r}-{mode}.json``. A run whose complete JSON exists is
skipped; an existing JSON whose key differs from this run's (or has none) stops with exit 2
naming the fields. A child's non-zero exit stops the orchestration with exit 2 and suppresses the
verdict. ``OUT/report.json`` holds each round's medians ``T_bf16``, ``T_control``, ``T_df11``, the
medians and spreads pooled over the timed steps of the rounds in which all three completed, and:

- ``control_vs_bf16 = |T_bf16 - T_control| / T_control``, and the verdict ``within_spread``:
  whether it is at most ``max(spread_bf16, spread_control)``, the larger pooled spread
  (``(max - min) / median`` over those steps, so it spans step-to-step and run-to-run variation);
- ``df11_vs_bf16 = (T_df11 - T_bf16) / T_bf16``, DF11's overhead over plain BF16.

Usage (from the repository root of a synced checkout, ``--group bench``):
    uv run python -m scripts.bench_control_validation --df11 DIR --embeds FILE --out DIR \
        [--double 4] [--single 8] [--rounds 3] [--steps 5] [--warmup 2] [--model schnell|dev] \
        [--size 1024] [--seed 42] [--wall-budget S]
Exit codes: 0 ok, 2 any error (a failed parity condition, a decode status error, a rig error, a
key conflict, a child's failure), 70/71 watchdog abort (footprint ceiling / wall budget).
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

try:
    import mlx.core as mx
    from scripts import bench_flux_step as step_bench
    from scripts._bench_common import Timing, overhead, provenance, write_json_atomic
    from scripts._flux_rig import (
        DF11Provider,
        ResidentProvider,
        WeightProvider,
        build_transformer,
        load_resident_set,
    )
    from scripts._watchdog import Watchdog
    from scripts.bench_flux_step import (
        EXIT_ERROR,
        EXIT_OK,
        GUIDANCE,
        T5_LENGTHS,
        BenchError,
        pooled_stats,
        run_path,
    )
    from scripts.verify_checkpoint import source_hash

    from mlx_dfloat.format import DF11Checkpoint, open_checkpoint
except Exception as exc:  # a broken environment is a tool error (2)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

# bf16 and control, the pair the verdict compares, run back to back.
MODES: tuple[str, ...] = ("bf16", "control", "df11")
POLICY = "per-block"
MAX_DOUBLE, MAX_SINGLE = 19, 38  # FLUX.1's depth


# --- pure parts (unit-tested without mflux) ---------------------------------------------------------


def expected_launches(mode: str, *, n_double: int, n_single: int, steps: int) -> int:
    """Decode launches ``steps`` steps of ``mode`` must make: one per block per step for df11, else 0.

    Raises:
        BenchError: Unknown mode.
    """
    if mode not in MODES:
        raise BenchError(f"unknown mode {mode!r}; choose from {MODES}")
    return (n_double + n_single) * steps if mode == "df11" else 0


def interleaved(rounds: int) -> list[tuple[int, str]]:
    """The (round, mode) sequence: every mode once per round, in ``MODES`` order."""
    return step_bench.interleaved(rounds, modes=MODES)


def pending_runs(out_dir: Path, rounds: int) -> list[tuple[int, str]]:
    """The interleaved runs whose complete JSON is not in ``out_dir`` yet (the resume set)."""
    return step_bench.pending_runs(out_dir, rounds, modes=MODES)


def resume_conflicts(
    out_dir: Path, rounds: int, key: Mapping[str, Any]
) -> list[tuple[int, str, list[str]]]:
    """Existing run files whose key differs from ``key``: (round, mode, differing fields)."""
    return step_bench.resume_conflicts(out_dir, rounds, key, modes=MODES)


def read_results(out_dir: Path, rounds: int) -> list[dict[str, Any]]:
    """The complete run JSONs of ``out_dir`` in interleaved order (incomplete ones are left out)."""
    return step_bench.read_results(out_dir, rounds, modes=MODES)


def run_key(*, double: int, single: int, **common: Any) -> dict[str, Any]:
    """The step bench's key (``bench_flux_step.run_key(**common)``) plus the reduced depth."""
    return {**step_bench.run_key(**common), "double": double, "single": single}


def control_vs_bf16(t_bf16: float, t_control: float) -> float:
    """How far the control's time is from plain BF16's: ``|t_bf16 - t_control| / t_control``.

    Raises:
        ValueError: ``t_control`` is zero or negative.
    """
    if t_control <= 0:
        raise ValueError(f"the control time must be positive, got {t_control}")
    return abs(t_bf16 - t_control) / t_control


def within_spread(deviation: float, spread_bf16: float, spread_control: float) -> bool:
    """The verdict: the deviation is at most the larger of the two modes' spreads."""
    return deviation <= max(spread_bf16, spread_control)


def df11_vs_bf16(t_df11: float, t_bf16: float) -> float:
    """DF11's overhead over plain BF16: ``(t_df11 - t_bf16) / t_bf16``."""
    return overhead(t_df11, t_bf16)


def report(
    results: Sequence[Mapping[str, Any]], *, stopped: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Per-round medians, pooled medians and spreads, the verdict and the df11 anchor.

    ``results`` are run JSONs with ``round``, ``mode``, ``step_s`` and optionally ``verify_s``.
    Pooling uses only the rounds in which all three modes completed (``complete_rounds``); with
    none, or when the orchestration ``stopped`` early, the verdict fields are None (the per-round
    medians and the pooled numbers stay).

    Raises:
        BenchError: A result has no timed steps.
    """
    by_round: dict[int, dict[str, Mapping[str, Any]]] = {}
    for r in results:
        if not r["step_s"]:
            raise BenchError(f"round {r['round']} {r['mode']}: no timed steps")
        by_round.setdefault(int(r["round"]), {})[str(r["mode"])] = r
    rounds = {
        rnd: {
            f"T_{mode}": Timing(reps=tuple(float(s) for s in modes[mode]["step_s"])).median
            for mode in MODES
            if mode in modes
        }
        for rnd, modes in sorted(by_round.items())
    }
    complete = sorted(rnd for rnd, modes in by_round.items() if all(m in modes for m in MODES))
    pooled: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        stats = pooled_stats(by_round, mode, complete)
        if stats is not None:
            pooled[mode] = stats
    verdict: dict[str, Any] = {
        "control_vs_bf16": None,
        "spread_bound": None,
        "within_spread": None,
        "df11_vs_bf16": None,
    }
    if complete and stopped is None:
        bf16, control, df11 = (pooled[m] for m in MODES)
        deviation = control_vs_bf16(bf16["median"], control["median"])
        verdict = {
            "control_vs_bf16": deviation,
            "spread_bound": max(bf16["spread"], control["spread"]),
            "within_spread": within_spread(deviation, bf16["spread"], control["spread"]),
            "df11_vs_bf16": df11_vs_bf16(df11["median"], bf16["median"]),
        }
    return {
        "rounds": rounds,
        "complete_rounds": complete,
        "pooled": pooled,
        **verdict,
        "stopped": dict(stopped) if stopped is not None else None,
    }


def child_command(
    *,
    mode: str,
    round_no: int,
    out: Path,
    df11: Path,
    embeds: Path,
    double: int,
    single: int,
    steps: int,
    warmup: int,
    model: str,
    size: int,
    seed: int,
    wall_budget: float,
) -> list[str]:
    """The subprocess argv for one mode of one round (run with the repository root as cwd)."""
    return [
        sys.executable,
        "-m",
        "scripts.bench_control_validation",
        *("--mode", mode),
        *("--round", str(round_no)),
        *("--out", str(out)),
        *("--df11", str(df11)),
        *("--embeds", str(embeds)),
        *("--double", str(double)),
        *("--single", str(single)),
        *("--steps", str(steps)),
        *("--warmup", str(warmup)),
        *("--model", model),
        *("--size", str(size)),
        *("--seed", str(seed)),
        *("--wall-budget", str(wall_budget)),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: orchestrate (``--out DIR``) or, with ``--mode``, one run (``--out FILE``).

    Paths are made absolute, so children started from the repository root see the same files and
    write the same key as the orchestrator.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=MODES, help="run this one mode (set by the orchestrator)")
    p.add_argument("--round", type=int, default=None, help="round number (set by the orchestrator)")
    p.add_argument(
        "--out", type=Path, required=True, help="directory (orchestrate) or JSON file (--mode)"
    )
    p.add_argument("--rounds", type=int, default=3, help="rounds of the three modes")
    p.add_argument("--df11", type=Path, required=True, help="DF11 checkpoint directory")
    p.add_argument(
        "--embeds", type=Path, required=True, help="safetensors from scripts.encode_prompt"
    )
    p.add_argument("--double", type=int, default=4, help="double blocks of the reduced transformer")
    p.add_argument("--single", type=int, default=8, help="single blocks of the reduced transformer")
    p.add_argument("--steps", type=int, default=5, help="timed steps")
    p.add_argument("--warmup", type=int, default=2, help="untimed steps before the timed ones")
    p.add_argument("--model", choices=tuple(T5_LENGTHS), default="schnell")
    p.add_argument("--size", type=int, default=1024, help="image side in pixels (multiple of 16)")
    p.add_argument("--seed", type=int, default=42, help="seed of the packed latent noise")
    p.add_argument(
        "--wall-budget", type=float, default=3600.0, help="seconds before the watchdog aborts"
    )
    args = p.parse_args(argv)
    if not (1 <= args.double <= MAX_DOUBLE and 1 <= args.single <= MAX_SINGLE):
        # The control reuses one decoded block of each kind, so both kinds need at least one.
        p.error(f"--double must be in 1..{MAX_DOUBLE} and --single in 1..{MAX_SINGLE}")
    if args.steps < 1 or args.warmup < 1 or args.rounds < 0:
        # The first step carries the real-group pipeline compile and the lazy scheduler
        # construction, so at least one warm-up step is required.
        p.error("--steps and --warmup must be >= 1, --rounds >= 0")
    args.out, args.df11, args.embeds = (
        Path(x).resolve() for x in (args.out, args.df11, args.embeds)
    )
    return args


# --- one mode ---------------------------------------------------------------------------------------


def current_key(args: argparse.Namespace) -> dict[str, Any]:
    """This run's resume key (reads the embeddings metadata, the source hash and the mlx version)."""
    return run_key(
        model=args.model,
        size=args.size,
        steps=args.steps,
        warmup=args.warmup,
        seed=args.seed,
        df11=args.df11,
        embeds=args.embeds,
        embeds_meta=step_bench.embeds_metadata(args.embeds),
        source=source_hash(),
        mlx=mx.__version__,
        double=args.double,
        single=args.single,
    )


def make_provider(
    mode: str, ckpt: DF11Checkpoint, resident: Mapping[str, Any], shapes: Mapping[str, Any]
) -> WeightProvider:
    """The mode's provider.

    ``bf16`` decodes every group once with the Metal backend, evaluating block by block (one lazy
    eval over all of them would allocate every decode up front), and keeps the dicts resident;
    ``control`` and ``df11`` are the step bench's providers.
    """
    if mode != "bf16":
        return step_bench.make_provider(mode, ckpt, resident, shapes)
    decoder = DF11Provider(resident, {n: ckpt.groups[n].matrix_names for n in shapes})
    per_block: dict[str, dict[str, mx.array]] = {}
    for name in shapes:
        weights = decoder.weights_for(name, shapes[name])
        mx.eval(weights)
        per_block[name] = weights
    decoder.verify()
    return ResidentProvider(per_block)


def run_mode(
    args: argparse.Namespace, watchdog: Watchdog, *, limits_recorded: bool
) -> dict[str, Any]:
    """Build the reduced transformer, load its groups, warm up, check parity, time; the result dict.

    Raises:
        ParityError: A parity condition failed before the timed loop.
        BenchError: A timed step's launches differ from the mode's, or the final latents are not finite.
    """
    mode = args.mode
    timings: dict[str, float] = {}
    ckpt = open_checkpoint(args.df11)
    start = time.perf_counter()
    transformer, shapes = build_transformer(
        args.model, ckpt, n_double=args.double, n_single=args.single
    )
    timings["build_s"] = time.perf_counter() - start
    start = time.perf_counter()
    resident = load_resident_set(ckpt, names=shapes)  # the reduced depth's groups only
    timings["load_resident_s"] = time.perf_counter() - start
    start = time.perf_counter()
    provider = make_provider(mode, ckpt, resident, shapes)
    timings["provider_s"] = time.perf_counter() - start
    transformer.attach(provider, shapes, eval_policy=POLICY)
    config, latents, prompt, pooled, inputs = step_bench.step_inputs(args)
    measured = step_bench.time_steps(
        transformer,
        provider,
        config,
        latents,
        prompt,
        pooled,
        warmup=args.warmup,
        steps=args.steps,
        per_step=expected_launches(mode, n_double=args.double, n_single=args.single, steps=1),
        limits_recorded=limits_recorded,
        watchdog=watchdog,
        label=mode,
    )
    return {
        "exit_code": EXIT_OK,
        "mode": mode,
        "policy": POLICY,
        "round": args.round,
        "model": args.model,
        "size": args.size,
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "guidance": GUIDANCE,
        "n_double": args.double,
        "n_single": args.single,
        "groups_resident": len(resident),
        **measured,
        **inputs,
        "timings_s": timings,
        "provenance": provenance(),
    }


# --- orchestration ----------------------------------------------------------------------------------


def _print_report(rep: Mapping[str, Any]) -> None:
    if rep["stopped"] is not None:
        st = rep["stopped"]
        print(f"stopped at round {st['round']} {st['mode']} (exit {st['exit_code']}): no verdict")
    for rnd, medians in rep["rounds"].items():
        text = ", ".join(f"{name} {value:.3f} s" for name, value in medians.items())
        print(f"round {rnd}: {text}")
    for mode, stats in rep["pooled"].items():
        print(
            f"pooled {mode}: median {stats['median']:.3f} s spread {stats['spread']:.3f} "
            f"(n={stats['n']})"
        )
    if rep["within_spread"] is None:
        if rep["stopped"] is None:
            print("verdict: none (no round in which all three modes completed)")
        return
    deviation, bound = rep["control_vs_bf16"], rep["spread_bound"]
    if rep["within_spread"]:
        print(
            f"verdict: the control matches plain BF16: |T_bf16 - T_control| / T_control = "
            f"{deviation:.2%}, within the spread {bound:.2%}"
        )
    else:
        print(
            f"verdict: the control does NOT match plain BF16: |T_bf16 - T_control| / T_control = "
            f"{deviation:.2%}, outside the spread {bound:.2%}"
        )
    print(f"df11 vs bf16: (T_df11 - T_bf16) / T_bf16 = {rep['df11_vs_bf16']:+.2%}")


def orchestrate(args: argparse.Namespace) -> int:
    """Run the pending (round, mode) subprocesses in order, then report; exit 2 on a child's failure."""
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    key = current_key(args)
    conflicts = resume_conflicts(out_dir, args.rounds, key)
    if conflicts:
        for r, m, fields in conflicts:
            print(
                f"error: {run_path(out_dir, r, m)} was written by a run with other {fields}; "
                "use a fresh --out",
                file=sys.stderr,
            )
        return EXIT_ERROR
    todo = pending_runs(out_dir, args.rounds)
    print(
        f"{len(interleaved(args.rounds)) - len(todo)} run(s) already complete, {len(todo)} to run"
    )
    stopped: dict[str, Any] | None = None
    for round_no, mode in todo:
        cmd = child_command(
            mode=mode,
            round_no=round_no,
            out=run_path(out_dir, round_no, mode),
            df11=args.df11,
            embeds=args.embeds,
            double=args.double,
            single=args.single,
            steps=args.steps,
            warmup=args.warmup,
            model=args.model,
            size=args.size,
            seed=args.seed,
            wall_budget=args.wall_budget,
        )
        print(f"round {round_no} {mode}: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd, cwd=_REPO, check=False).returncode
        if code != 0:
            stopped = {"round": round_no, "mode": mode, "exit_code": code}
            print(f"error: round {round_no} {mode} exited {code}; stopping", file=sys.stderr)
            break
    results = read_results(out_dir, args.rounds)
    rep = report(results, stopped=stopped)
    rep.update(
        {
            "key": key,
            "rounds_requested": args.rounds,
            "runs_complete": [(r["round"], r["mode"]) for r in results],
            "provenance": provenance(),
        }
    )
    write_json_atomic(out_dir / "report.json", rep)
    _print_report(rep)
    return EXIT_ERROR if stopped is not None else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.mode is None:
        return orchestrate(args)
    return step_bench.run_one(args, measure=run_mode, key=current_key)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
