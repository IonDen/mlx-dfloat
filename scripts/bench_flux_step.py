"""The FLUX.1 step bench: DF11 just-in-time decode against a decoded control, one mode per process.

Each process runs one mode of a full-depth (19 + 38 block) FLUX.1 transformer built by the rig
(``scripts/_flux_rig.py``): ``df11`` decodes every block's DF11 group with the Metal kernel as the
block runs (``DF11Provider``, 57 launches per step); ``control`` hands every block the same two
pre-decoded weight dicts, one double and one single block, decoded once by that same kernel
(``ReuseProvider``, no launches). Both load and evaluate the whole compressed set first, so the
footprint baseline is the same; the control's two decoded blocks are its only extra. ``-depth2``
variants run the depth-2 eval policy and ``control-noeval`` runs the control with no eval inside
the step, so ``control - control-noeval`` is the eval policy's own cost. No text encoder or VAE is
loaded: the prompt embeddings come from ``--embeds`` (``scripts/encode_prompt.py``). Activation
dtypes follow mflux exactly: the latents stay the float32 ``create_noise`` returns and the
embeddings keep the dtype the encoder produced (T5 float32, CLIP bfloat16; a synthetic file is
float32), so every Linear promotes its bf16 weights to float32 as upstream does. Both dtypes are
recorded in the JSON.

A step is exactly mflux's loop body: ``scale_model_input``, the transformer, ``scheduler.step``,
``mx.eval(latents)``, timed as a whole with ``time.perf_counter``; ``verify_step()`` (the deferred
decode status reads, which upstream never does at step time) runs after the stop and is timed on
its own as ``verify_s``. ``--warmup`` steps run untimed, then ``--steps`` timed. Between them the
parity conditions are asserted: the compressed set is loaded, the memory caps and cache limit are
recorded, nothing is pending, the launches so far match the mode, and the latents are finite. A
failed condition is exit 2 with the names in the JSON. Every timed step's launch count must equal
the mode's expectation (0 or 57), and the final latents must be finite.

``--orchestrate`` runs the five modes as subprocesses, interleaved per round in the order
``df11, control, df11-depth2, control-depth2, control-noeval``, one at a time, each writing
``DIR/round{r}-{mode}.json``; a run whose complete JSON exists is skipped, so an interrupted
orchestration resumes. Every JSON carries a ``key`` (model, size, steps, warmup, seed, the
resolved checkpoint path, the embeddings path and metadata, the source hash, the mlx version); an
existing JSON whose key differs from this run's, or has none, stops the orchestration with exit 2
naming the fields, before anything runs, so different runs never mix or overwrite. It then prints
per-round paired overheads (``overhead(t_df11, t_control)`` for per-block and depth2), the pooled
medians with spreads over the timed steps of the rounds in which both modes of a pair completed,
and the eval-policy cost, and writes ``DIR/report.json``. A child's non-zero exit stops the
orchestration with exit 2; the report records which run and suppresses the pooled overheads.

Usage (from the repository root of a synced checkout, ``--group bench``):
    uv run python -m scripts.bench_flux_step --mode MODE --df11 DIR --embeds FILE --out FILE \
        [--steps 5] [--warmup 2] [--model schnell|dev] [--size 1024] [--seed 42] [--wall-budget S]
    uv run python -m scripts.bench_flux_step --orchestrate --rounds 3 --out-dir DIR --df11 DIR \
        --embeds FILE [--steps 5] [--warmup 2] [--model schnell|dev] [--size 1024] [--wall-budget S]
Exit codes: 0 ok, 2 any error (a non-finite latent, a failed parity condition, a decode status
error, a rig error, a child's failure), 70/71 watchdog abort (footprint ceiling / wall budget).
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx
    from scripts._bench_common import (
        Timing,
        overhead,
        parity_conditions,
        provenance,
        resume_key_diff,
        write_json_atomic,
    )
    from scripts._flux_rig import (
        DOUBLE_PREFIX,
        FLUX_CACHE_LIMIT,
        SINGLE_PREFIX,
        DF11Provider,
        ReuseProvider,
        WeightProvider,
        build_transformer,
        load_resident_set,
    )
    from scripts._watchdog import Watchdog, default_ceiling, phys_footprint
    from scripts.verify_checkpoint import source_hash

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat.format import DF11Checkpoint, open_checkpoint
except Exception as exc:  # a broken environment is a tool error (2)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_ERROR = 0, 2
# One round runs the modes in this order: each df11 mode right before its control, so a drift in
# machine state lands on both sides of a pair.
MODES: tuple[str, ...] = ("df11", "control", "df11-depth2", "control-depth2", "control-noeval")
_POLICIES = {
    "df11": "per-block",
    "control": "per-block",
    "df11-depth2": "depth2",
    "control-depth2": "depth2",
    "control-noeval": "none",
}
# (label, df11 mode, control mode): the paired overheads the report computes.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("per-block", "df11", "control"),
    ("depth2", "df11-depth2", "control-depth2"),
)
N_DOUBLE, N_SINGLE = 19, 38  # FLUX.1 at full depth
T5_DIM, POOLED_DIM = 4096, 768
T5_LENGTHS = {"schnell": 256, "dev": 512}  # mflux 0.20 ModelConfig.max_sequence_length
# mflux's CLI default (cli/defaults/defaults.py GUIDANCE_SCALE). schnell's transformer has no
# guidance embedder (supports_guidance=False), so the value is inert there; dev embeds it.
GUIDANCE = 3.5


class BenchError(ValueError):
    """An input, mode or condition problem of this bench (exit 2)."""


class ParityError(BenchError):
    """A parity condition failed before the timed loop; ``failed`` names the conditions."""

    def __init__(self, failed: Sequence[str]) -> None:
        """Keep the failed condition names."""
        super().__init__(f"parity conditions failed before the timed loop: {list(failed)}")
        self.failed = list(failed)


# --- pure parts (unit-tested without mflux) ---------------------------------------------------------


def mode_policy(mode: str) -> str:
    """The rig eval policy a mode runs under.

    Raises:
        BenchError: Unknown mode.
    """
    try:
        return _POLICIES[mode]
    except KeyError:
        raise BenchError(f"unknown mode {mode!r}; choose from {MODES}") from None


def is_df11(mode: str) -> bool:
    """Whether the mode decodes just in time (launches) rather than reusing decoded weights."""
    mode_policy(mode)
    return mode.startswith("df11")


def expected_launches(mode: str, *, n_double: int, n_single: int, steps: int) -> int:
    """Decode launches ``steps`` steps of ``mode`` must make: one per block per step, or none."""
    return (n_double + n_single) * steps if is_df11(mode) else 0


def interleaved(rounds: int) -> list[tuple[int, str]]:
    """The (round, mode) sequence of an orchestration: every mode once per round, in ``MODES`` order."""
    return [(r, mode) for r in range(1, rounds + 1) for mode in MODES]


def run_path(out_dir: Path, round_no: int, mode: str) -> Path:
    """Where one run's JSON goes: ``DIR/round{r}-{mode}.json``."""
    return out_dir / f"round{round_no}-{mode}.json"


def result_is_complete(path: Path) -> bool:
    """Whether ``path`` holds a finished successful run: JSON with ``exit_code`` 0 and step times."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("exit_code") == 0 and bool(data.get("step_s"))


def pending_runs(out_dir: Path, rounds: int) -> list[tuple[int, str]]:
    """The interleaved runs whose complete JSON is not in ``out_dir`` yet (the resume set)."""
    return [
        (r, m) for r, m in interleaved(rounds) if not result_is_complete(run_path(out_dir, r, m))
    ]


def run_key(
    *,
    model: str,
    size: int,
    steps: int,
    warmup: int,
    seed: int,
    df11: Path,
    embeds: Path,
    embeds_meta: Mapping[str, str],
    source: str,
    mlx: str,
) -> dict[str, Any]:
    """The settings a run's JSON is keyed on; two runs may share an out dir only when they agree.

    The checkpoint path is resolved so the same checkpoint reached from another cwd matches; the
    embeddings metadata (prompt, seed, model, synthetic, versions) identifies the file's content.
    """
    return {
        "model": model,
        "size": size,
        "steps": steps,
        "warmup": warmup,
        "seed": seed,
        "df11": str(Path(df11).resolve()),
        "embeds": str(embeds),
        "embeds_meta": dict(embeds_meta),
        "source": source,
        "mlx": mlx,
    }


def resume_conflicts(
    out_dir: Path, rounds: int, key: Mapping[str, Any]
) -> list[tuple[int, str, list[str]]]:
    """Existing run files whose key differs from ``key``: (round, mode, differing fields).

    A file that cannot be parsed or has no key differs in ``["key"]``. A matching file, complete
    or not, is no conflict (``pending_runs`` decides whether it is re-run). Missing files are
    no conflict.
    """
    conflicts: list[tuple[int, str, list[str]]] = []
    for r, m in interleaved(rounds):
        path = run_path(out_dir, r, m)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            data = None
        stored = data.get("key") if isinstance(data, dict) else None
        diff = resume_key_diff(stored, key)
        if diff:
            conflicts.append((r, m, diff))
    return conflicts


def child_command(
    *,
    mode: str,
    round_no: int,
    out: Path,
    df11: Path,
    embeds: Path,
    steps: int,
    warmup: int,
    model: str,
    size: int,
    seed: int,
    wall_budget: float,
) -> list[str]:
    """The subprocess argv for one mode of one round (never ``--orchestrate``)."""
    return [
        sys.executable,
        "-m",
        "scripts.bench_flux_step",
        "--mode",
        mode,
        "--round",
        str(round_no),
        "--out",
        str(out),
        "--df11",
        str(df11),
        "--embeds",
        str(embeds),
        "--steps",
        str(steps),
        "--warmup",
        str(warmup),
        "--model",
        model,
        "--size",
        str(size),
        "--seed",
        str(seed),
        "--wall-budget",
        str(wall_budget),
    ]


def check_embeds_shapes(
    prompt_shape: Sequence[int], pooled_shape: Sequence[int], model: str
) -> None:
    """Refuse embeddings of another token length or width than ``model``'s encoders produce.

    Raises:
        BenchError: The shapes are not ``(1, T5 length, 4096)`` and ``(1, 768)``.
    """
    want_prompt = (1, T5_LENGTHS[model], T5_DIM)
    want_pooled = (1, POOLED_DIM)
    if tuple(prompt_shape) != want_prompt:
        raise BenchError(f"prompt_embeds has shape {tuple(prompt_shape)}, expected {want_prompt}")
    if tuple(pooled_shape) != want_pooled:
        raise BenchError(
            f"pooled_prompt_embeds has shape {tuple(pooled_shape)}, expected {want_pooled}"
        )


def report(
    results: Sequence[Mapping[str, Any]], *, stopped: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Paired overheads per round, pooled medians with spreads, and the eval-policy cost.

    ``results`` are run JSONs with ``round``, ``mode`` and ``step_s`` (the timed steps), and
    optionally ``verify_s`` (the status validation timed outside the step window, pooled to
    ``verify_median_s``, None when no result of the mode recorded it). A round's pair is reported
    only when both of its modes are present. Pooling uses only the rounds in which both modes of a
    pair completed (``paired_rounds`` lists them per pair, and ``"eval-policy"`` for control with
    control-noeval); a mode with no such round is not pooled. The overheads come from the pooled
    medians; ``eval_policy_cost_s`` is the control median minus the control-noeval median over
    their shared rounds, None when there is none. When the orchestration ``stopped`` early the
    pooled overheads and the eval-policy cost are suppressed (the per-round pairs stay).

    Raises:
        BenchError: A result has no timed steps.
    """
    by_round: dict[int, dict[str, Mapping[str, Any]]] = {}
    for r in results:
        if not r["step_s"]:
            raise BenchError(f"round {r['round']} {r['mode']}: no timed steps")
        by_round.setdefault(int(r["round"]), {})[str(r["mode"])] = r

    def median_of(mode: str, rnd: int) -> float:
        return Timing(reps=tuple(float(s) for s in by_round[rnd][mode]["step_s"])).median

    def pool(mode: str, over: Sequence[int]) -> dict[str, Any] | None:
        if not over:
            return None
        steps = [float(s) for rnd in over for s in by_round[rnd][mode]["step_s"]]
        verify = [float(s) for rnd in over for s in by_round[rnd][mode].get("verify_s", ())]
        t = Timing(reps=tuple(steps))
        return {
            "median": t.median,
            "spread": t.spread,
            "n": len(steps),
            "verify_median_s": Timing(reps=tuple(verify)).median if verify else None,
        }

    rounds = {
        rnd: {
            label: overhead(median_of(d, rnd), median_of(c, rnd))
            for label, d, c in PAIRS
            if d in modes and c in modes
        }
        for rnd, modes in sorted(by_round.items())
    }
    paired = {
        label: sorted(rnd for rnd, modes in by_round.items() if d in modes and c in modes)
        for label, d, c in (*PAIRS, ("eval-policy", "control", "control-noeval"))
    }
    pooled: dict[str, dict[str, Any]] = {}
    for label, d, c in PAIRS:
        for mode in (d, c):
            stats = pool(mode, paired[label])
            if stats is not None:
                pooled[mode] = stats
    noeval = pool("control-noeval", paired["eval-policy"])
    if noeval is not None:
        pooled["control-noeval"] = noeval
    overheads = {
        label: overhead(pooled[d]["median"], pooled[c]["median"])
        for label, d, c in PAIRS
        if paired[label]
    }
    cost = None
    if noeval is not None:
        control = pool("control", paired["eval-policy"])
        assert control is not None  # the same rounds as noeval, by construction
        cost = control["median"] - noeval["median"]
    if stopped is not None:
        overheads, cost = {}, None
    return {
        "rounds": rounds,
        "pooled": pooled,
        "overhead": overheads,
        "paired_rounds": paired,
        "eval_policy_cost_s": cost,
        "stopped": dict(stopped) if stopped is not None else None,
    }


def read_results(out_dir: Path, rounds: int) -> list[dict[str, Any]]:
    """The complete run JSONs of ``out_dir`` in interleaved order (incomplete ones are left out)."""
    results: list[dict[str, Any]] = []
    for r, m in interleaved(rounds):
        path = run_path(out_dir, r, m)
        if result_is_complete(path):
            results.append(json.loads(path.read_text()))
    return results


# --- one mode ---------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: one mode (``--mode``, ``--out``) or ``--orchestrate`` (``--out-dir``)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=MODES, help="the one mode this process runs")
    p.add_argument("--round", type=int, default=None, help="round number (set by --orchestrate)")
    p.add_argument("--out", type=Path, help="JSON file of this mode's result")
    p.add_argument("--orchestrate", action="store_true", help="run every mode as subprocesses")
    p.add_argument("--rounds", type=int, default=3, help="rounds of the five modes to orchestrate")
    p.add_argument("--out-dir", type=Path, help="directory of the per-run JSONs and report.json")
    p.add_argument("--df11", type=Path, required=True, help="DF11 checkpoint directory")
    p.add_argument(
        "--embeds", type=Path, required=True, help="safetensors from scripts.encode_prompt"
    )
    p.add_argument("--steps", type=int, default=5, help="timed steps")
    p.add_argument("--warmup", type=int, default=2, help="untimed steps before the timed ones")
    p.add_argument("--model", choices=tuple(T5_LENGTHS), default="schnell")
    p.add_argument("--size", type=int, default=1024, help="image side in pixels (multiple of 16)")
    p.add_argument("--seed", type=int, default=42, help="seed of the packed latent noise")
    p.add_argument(
        "--wall-budget", type=float, default=3600.0, help="seconds before the watchdog aborts"
    )
    args = p.parse_args(argv)
    if args.orchestrate:
        if args.out_dir is None:
            p.error("--orchestrate needs --out-dir")
        if args.mode is not None or args.out is not None:
            p.error("--orchestrate takes no --mode or --out")
    elif args.mode is None or args.out is None:
        p.error("a single run needs --mode and --out (or use --orchestrate)")
    if args.steps < 1 or args.warmup < 1 or args.rounds < 0:
        # The first step carries the real-group pipeline compile and the lazy scheduler
        # construction, so at least one warm-up step is required.
        p.error("--steps and --warmup must be >= 1, --rounds >= 0")
    return args


def embeds_metadata(path: Path) -> dict[str, str]:
    """The string metadata of an embeddings file (part of the resume key)."""
    _data, meta = mx.load(str(path), return_metadata=True)
    return dict(meta)


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
        embeds_meta=embeds_metadata(args.embeds),
        source=source_hash(),
        mlx=mx.__version__,
    )


def load_embeds(path: Path, model: str) -> tuple[mx.array, mx.array, dict[str, str]]:
    """Read and shape-check the prompt embeddings in the dtype the encoder produced; returns their metadata.

    No cast: mflux passes the T5 output (float32) and the CLIP output as they are, and the
    transformer's Linears promote their bf16 weights to the activation dtype.

    Raises:
        BenchError: A tensor is missing or has the wrong shape.
    """
    data, meta = mx.load(str(path), return_metadata=True)
    try:
        prompt, pooled = data["prompt_embeds"], data["pooled_prompt_embeds"]
    except KeyError as exc:
        raise BenchError(f"{path}: no {exc} tensor") from exc
    check_embeds_shapes(prompt.shape, pooled.shape, model)
    mx.eval(prompt, pooled)
    return prompt, pooled, dict(meta)


def make_provider(
    mode: str, ckpt: DF11Checkpoint, resident: Mapping[str, Any], shapes: Mapping[str, Any]
) -> WeightProvider:
    """The mode's provider.

    ``DF11Provider`` for df11 modes; otherwise a ``ReuseProvider`` over one double and one single
    block decoded once by that same Metal backend (the control's only extra memory).
    """
    decoder = DF11Provider(resident, {n: ckpt.groups[n].matrix_names for n in shapes})
    if is_df11(mode):
        return decoder
    double_name, single_name = f"{DOUBLE_PREFIX}.0", f"{SINGLE_PREFIX}.0"
    double = decoder.weights_for(double_name, shapes[double_name])
    single = decoder.weights_for(single_name, shapes[single_name])
    mx.eval(double, single)
    decoder.verify()
    return ReuseProvider(double, single)


def denoise_step(
    transformer: Any,
    config: Any,
    latents: mx.array,
    prompt: mx.array,
    pooled: mx.array,
    t: int,
) -> tuple[mx.array, float, float]:
    """The upstream loop body for step ``t``, timed to its eval; then ``verify_step()``, timed on its own.

    Returns ``(latents, step seconds, verify seconds)``. The measured window holds exactly what
    upstream runs per step; the status validation is DF11's separate, reported cost.
    """
    start = time.perf_counter()
    latents = config.scheduler.scale_model_input(latents, t)
    noise = transformer(
        t=t, config=config, hidden_states=latents, prompt_embeds=prompt, pooled_prompt_embeds=pooled
    )
    latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
    mx.eval(latents)
    stop = time.perf_counter()
    transformer.verify_step()
    return latents, stop - start, time.perf_counter() - stop


def run_mode(
    args: argparse.Namespace, watchdog: Watchdog, *, limits_recorded: bool
) -> dict[str, Any]:
    """Build, load, warm up, assert the parity conditions, time the steps; the result dict.

    Raises:
        ParityError: A parity condition failed before the timed loop.
        BenchError: A timed step's launches differ from the mode's, or the final latents are not finite.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

    mode, policy = args.mode, mode_policy(args.mode)
    timings: dict[str, float] = {}
    ckpt = open_checkpoint(args.df11)
    start = time.perf_counter()
    transformer, shapes = build_transformer(args.model, ckpt, n_double=N_DOUBLE, n_single=N_SINGLE)
    timings["build_s"] = time.perf_counter() - start
    start = time.perf_counter()
    resident = load_resident_set(ckpt)  # every group, in every mode: the same footprint baseline
    timings["load_resident_s"] = time.perf_counter() - start
    start = time.perf_counter()
    provider = make_provider(mode, ckpt, resident, shapes)
    timings["provider_s"] = time.perf_counter() - start
    transformer.attach(provider, shapes, eval_policy=policy)

    model_config = ModelConfig.schnell() if args.model == "schnell" else ModelConfig.dev()
    config = Config(
        model_config,
        num_inference_steps=args.warmup + args.steps,
        height=args.size,
        width=args.size,
        guidance=GUIDANCE,
        scheduler="linear",
    )
    prompt, pooled, embeds_meta = load_embeds(args.embeds, args.model)
    latents = FluxLatentCreator.create_noise(
        args.seed, args.size, args.size
    )  # float32, as upstream
    mx.eval(latents)
    latent_dtype = str(latents.dtype)
    embeds_dtype = {"prompt_embeds": str(prompt.dtype), "pooled_prompt_embeds": str(pooled.dtype)}

    per_step = expected_launches(mode, n_double=N_DOUBLE, n_single=N_SINGLE, steps=1)
    launches_per_step: list[int] = []
    footprint_peak = phys_footprint()

    def run_steps(first: int, count: int) -> tuple[list[float], list[float]]:
        nonlocal latents, footprint_peak
        seconds: list[float] = []
        verify: list[float] = []
        for t in range(first, first + count):
            before = provider.launches
            latents, took, checked = denoise_step(transformer, config, latents, prompt, pooled, t)
            seconds.append(took)
            verify.append(checked)
            launches_per_step.append(provider.launches - before)
            footprint_peak = max(footprint_peak, phys_footprint())
        return seconds, verify

    warmup_s, warmup_verify_s = run_steps(0, args.warmup)
    failed = parity_conditions(
        compressed_loaded=provider.groups_evaluated,
        limits_recorded=limits_recorded,
        nothing_pending=not getattr(provider, "pending", []),
        launches_expected=provider.launches
        == expected_launches(mode, n_double=N_DOUBLE, n_single=N_SINGLE, steps=args.warmup),
        finite=bool(mx.isfinite(latents).all().item()),
    )
    if failed:
        raise ParityError(failed)
    step_s, verify_s = run_steps(args.warmup, args.steps)
    if any(n != per_step for n in launches_per_step[args.warmup :]):
        raise BenchError(
            f"timed steps made {launches_per_step[args.warmup :]} launches; {mode} expects "
            f"{per_step} per step"
        )
    if not bool(mx.isfinite(latents).all().item()):
        raise BenchError("the final latents are not finite")
    timing = Timing(reps=tuple(step_s))
    return {
        "exit_code": EXIT_OK,
        "mode": mode,
        "policy": policy,
        "round": args.round,
        "model": args.model,
        "size": args.size,
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "guidance": GUIDANCE,
        "n_double": N_DOUBLE,
        "n_single": N_SINGLE,
        "step_s": step_s,
        "warmup_s": warmup_s,
        "median_s": timing.median,
        "spread": timing.spread,
        "verify_s": verify_s,
        "warmup_verify_s": warmup_verify_s,
        "verify_median_s": Timing(reps=tuple(verify_s)).median,
        "launches_per_step": launches_per_step,
        "launches_expected_per_step": per_step,
        "launches_total": provider.launches,
        "footprint_peak_bytes": max(footprint_peak, watchdog.peak_footprint),
        "watchdog_peak_footprint_bytes": watchdog.peak_footprint,
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "embeds": {"path": str(args.embeds), "metadata": embeds_meta},
        "latent_dtype": latent_dtype,
        "embeds_dtype": embeds_dtype,
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
        "timings_s": timings,
        "provenance": provenance(),
    }


def run_one(args: argparse.Namespace) -> int:
    """Run one mode under the caps, the cache limit and the watchdog; write ``--out``."""
    caps = list(install_memory_caps())
    mx.set_cache_limit(FLUX_CACHE_LIMIT)
    cache_limit = int(mx.set_cache_limit(FLUX_CACHE_LIMIT))  # the limit now in force
    limits_recorded = caps[0] > 0 and cache_limit == FLUX_CACHE_LIMIT
    args.out.parent.mkdir(parents=True, exist_ok=True)
    watchdog = Watchdog(args.out.parent, ceiling=default_ceiling(), budget=args.wall_budget).start()
    summary: dict[str, Any]
    key: dict[str, Any] | None = None
    try:
        key = current_key(args)
        summary = run_mode(args, watchdog, limits_recorded=limits_recorded)
    except ParityError as exc:
        summary = {
            "exit_code": EXIT_ERROR,
            "error": str(exc),
            "parity_conditions_failed": exc.failed,
        }
    except Exception as exc:  # any failure is exit 2, recorded
        summary = {"exit_code": EXIT_ERROR, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    finally:
        watchdog.stop()
    summary.update(
        {
            "mode": args.mode,
            "round": args.round,
            "memory_caps_gb": caps,
            "cache_limit_bytes": cache_limit,
            "key": key,
        }
    )
    try:
        write_json_atomic(args.out, summary)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: cannot write {args.out} ({exc})", file=sys.stderr)
        summary["exit_code"] = EXIT_ERROR
    code = int(summary["exit_code"])
    if code == EXIT_OK:
        print(
            f"ok: {args.mode} ({summary['policy']}) median {summary['median_s']:.3f} s "
            f"spread {summary['spread']:.3f} over {args.steps} steps "
            f"(+ {summary['verify_median_s']:.4f} s status validation, outside the window), "
            f"{summary['launches_expected_per_step']} launches/step, footprint peak "
            f"{summary['footprint_peak_bytes'] / 1024**3:.2f} GiB"
        )
    else:
        print(f"exit {code}: {summary.get('error')}", file=sys.stderr)
    return code


# --- orchestration ----------------------------------------------------------------------------------


def _print_report(rep: Mapping[str, Any]) -> None:
    if rep["stopped"] is not None:
        st = rep["stopped"]
        print(
            f"stopped at round {st['round']} {st['mode']} (exit {st['exit_code']}): "
            "pooled overheads suppressed"
        )
    for rnd, pairs in rep["rounds"].items():
        pairs_text = ", ".join(f"{label} {value:+.1%}" for label, value in pairs.items())
        print(f"round {rnd}: {pairs_text or 'no complete pair'}")
    for mode, stats in rep["pooled"].items():
        verify = stats["verify_median_s"]
        verify_text = f", validation {verify:.4f} s" if verify is not None else ""
        print(
            f"pooled {mode}: median {stats['median']:.3f} s spread {stats['spread']:.3f} "
            f"(n={stats['n']}{verify_text})"
        )
    for label, value in rep["overhead"].items():
        print(f"overhead {label}: {value:+.1%}")
    cost = rep["eval_policy_cost_s"]
    if cost is not None:
        print(f"eval policy cost (control - control-noeval): {cost:+.3f} s")


def orchestrate(args: argparse.Namespace) -> int:
    """Run the pending (round, mode) subprocesses in order, then report; exit 2 on a child's failure."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    key = current_key(args)
    conflicts = resume_conflicts(out_dir, args.rounds, key)
    if conflicts:
        for r, m, fields in conflicts:
            print(
                f"error: {run_path(out_dir, r, m)} was written by a run with other {fields}; "
                "use a fresh --out-dir",
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
            steps=args.steps,
            warmup=args.warmup,
            model=args.model,
            size=args.size,
            seed=args.seed,
            wall_budget=args.wall_budget,
        )
        print(f"round {round_no} {mode}: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd, check=False).returncode
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
            "steps": args.steps,
            "warmup": args.warmup,
            "model": args.model,
            "size": args.size,
            "provenance": provenance(),
        }
    )
    write_json_atomic(out_dir / "report.json", rep)
    _print_report(rep)
    return EXIT_ERROR if stopped is not None else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return orchestrate(args) if args.orchestrate else run_one(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
