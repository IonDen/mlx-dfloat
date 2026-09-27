"""The FLUX.1 step bench: DF11 just-in-time decode against a decoded control, one mode per process.

Each process runs one mode of a full-depth (19 + 38 block) FLUX.1 transformer built by the rig
(``scripts/_flux_rig.py``): ``df11`` decodes every block's DF11 group with the Metal kernel as the
block runs (``DF11Provider``, 57 launches per step); ``control`` hands every block the same two
pre-decoded weight dicts, one double and one single block, decoded once by that same kernel
(``ReuseProvider``, no launches). Both load and evaluate the whole compressed set first, so the
footprint baseline is the same; the control's two decoded blocks are its only extra. ``-depth2``
variants run the depth-2 eval policy: its look-ahead is bounded by MLX's command-buffer window (the
encoding thread blocks once enough committed buffers are in flight, so ``async_eval`` of block i
returns only near its end and the next block's decode never runs alongside block i's matmuls on the
same in-order stream); what it hides is host-side work done while the GPU runs block i, above all
the allocation of decode outputs that miss the buffer cache. ``control-noeval`` runs the control
with no eval inside the step, so ``control - control-noeval`` is the eval policy's own cost.
``--trace`` records per-block phase stamps; ``--cache-limit`` sets the MLX buffer-cache limit (at
the default 1.4 GB the decode outputs miss the exact-size cache at every double/single switch, which
costs 1.29 s per schnell step; 2.5 GB holds one buffer of each size). ``--modes`` runs a subset,
including the two look-ahead experiment modes (``df11-prefetch`` decodes the next block on a
second GPU stream while this one runs, ``df11-prefetch-inline`` on the default stream); measured
2026-09-27, neither beats per-block once the cache limit fits. No text encoder or VAE is
loaded: the prompt embeddings come from ``--embeds`` (``scripts/encode_prompt.py``). Activation
dtypes follow mflux exactly: the latents stay the float32 ``create_noise`` returns and the
embeddings keep the dtype the encoder produced (T5 float32, CLIP bfloat16; a synthetic file is
float32), so every Linear promotes its bf16 weights to float32 as upstream does. Both dtypes are
recorded in the JSON.

A step is exactly mflux's loop body: ``scale_model_input``, the transformer, ``scheduler.step``,
``mx.eval(latents)``, timed as a whole with ``time.perf_counter``; ``verify_step()`` (the deferred
decode status reads, which upstream never does at step time) runs after the stop and is timed on
its own as ``verify_s``. ``--warmup`` steps run untimed, then ``--steps`` timed. Between them the
parity conditions are asserted: every block's compressed group is resident, both memory caps and
the cache limit are in force, the launches so far match the mode, and the latents are finite. A
failed condition is exit 2 with the names in the JSON. Every timed step's launch count must equal
the mode's expectation (0 or 57), and the final latents must be finite. The MLX peak and the
watchdog's footprint peak are reset right before the timed steps, so the JSON carries the timed
steps' own peaks (``step_*``) next to the process-lifetime ones.

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
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

try:
    import mlx.core as mx
    from scripts._bench_common import (
        Timing,
        move_stale_abort_aside,
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
        PrefetchProvider,
        ReuseProvider,
        Tracer,
        WeightProvider,
        build_transformer,
        load_resident_set,
        summarize_trace,
    )
    from scripts._watchdog import Watchdog, default_ceiling, phys_footprint
    from scripts.verify_checkpoint import source_hash

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat.decode import DecodeResult
    from mlx_dfloat.format import DF11Checkpoint, MxGroup, open_checkpoint
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
# Experiment modes, run only when ``--modes`` names them: the look-ahead decode of the next block,
# on a second GPU stream (``df11-prefetch``) or on the default one (``df11-prefetch-inline``).
EXTRA_MODES: tuple[str, ...] = ("df11-prefetch", "df11-prefetch-inline")
_POLICIES = {
    "df11": "per-block",
    "control": "per-block",
    "df11-depth2": "depth2",
    "control-depth2": "depth2",
    "control-noeval": "none",
    "df11-prefetch": "per-block",
    "df11-prefetch-inline": "per-block",
}
# (label, df11 mode, control mode): the paired overheads the report computes.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("per-block", "df11", "control"),
    ("depth2", "df11-depth2", "control-depth2"),
    ("prefetch", "df11-prefetch", "control"),
    ("prefetch-inline", "df11-prefetch-inline", "control"),
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
        raise BenchError(f"unknown mode {mode!r}; choose from {MODES + EXTRA_MODES}") from None


def is_df11(mode: str) -> bool:
    """Whether the mode decodes just in time (launches) rather than reusing decoded weights."""
    mode_policy(mode)
    return mode.startswith("df11")


def expected_launches(mode: str, *, n_double: int, n_single: int, steps: int) -> int:
    """Decode launches ``steps`` steps of ``mode`` must make: one per block per step, or none."""
    return (n_double + n_single) * steps if is_df11(mode) else 0


def limits_in_force(
    caps: Sequence[int], cache_limit: int, *, expected: int = FLUX_CACHE_LIMIT
) -> bool:
    """Whether both memory caps installed (non-zero GB) and the MLX cache limit is ``expected``.

    ``caps`` is what ``install_memory_caps`` returned; a 0 means that cap failed to install.
    ``expected`` is the rig's limit unless the run asked for another (``--cache-limit``).
    """
    return len(caps) == 2 and all(c > 0 for c in caps) and cache_limit == expected


def compressed_set_loaded(resident: Mapping[str, Any], shapes: Mapping[str, Any]) -> bool:
    """Whether every block of the transformer has a resident compressed group with elements.

    ``GroupArrays.to_mx`` evaluates every array it builds, so presence and a non-zero element
    count are what remains to check.
    """
    return all(name in resident and resident[name].n_elements > 0 for name in shapes)


def interleaved(rounds: int, *, modes: Sequence[str] = MODES) -> list[tuple[int, str]]:
    """The (round, mode) sequence of an orchestration: every mode once per round, in ``modes`` order."""
    return [(r, mode) for r in range(1, rounds + 1) for mode in modes]


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


def pending_runs(
    out_dir: Path, rounds: int, *, modes: Sequence[str] = MODES
) -> list[tuple[int, str]]:
    """The interleaved runs whose complete JSON is not in ``out_dir`` yet (the resume set)."""
    return [
        (r, m)
        for r, m in interleaved(rounds, modes=modes)
        if not result_is_complete(run_path(out_dir, r, m))
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
    cache_limit: int,
) -> dict[str, Any]:
    """The settings a run's JSON is keyed on; two runs may share an out dir only when they agree.

    The checkpoint path is resolved so the same checkpoint reached from another cwd matches; the
    embeddings metadata (prompt, seed, model, synthetic, versions) identifies the file's content.
    The MLX cache limit is part of the key because it changes what a step allocates.
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
        "cache_limit": cache_limit,
    }


def resume_conflicts(
    out_dir: Path, rounds: int, key: Mapping[str, Any], *, modes: Sequence[str] = MODES
) -> list[tuple[int, str, list[str]]]:
    """Existing run files whose key differs from ``key``: (round, mode, differing fields).

    A file that cannot be parsed or has no key differs in ``["key"]``. A matching file, complete
    or not, is no conflict (``pending_runs`` decides whether it is re-run). Missing files are
    no conflict.
    """
    conflicts: list[tuple[int, str, list[str]]] = []
    for r, m in interleaved(rounds, modes=modes):
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
    cache_limit: int,
    trace: bool,
) -> list[str]:
    """The subprocess argv for one mode of one round (never ``--orchestrate``; run with the repository root as cwd)."""
    cmd = [
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
        "--cache-limit",
        str(cache_limit),
    ]
    if trace:
        cmd.append("--trace")
    return cmd


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


def pooled_stats(
    by_round: Mapping[int, Mapping[str, Mapping[str, Any]]], mode: str, over: Sequence[int]
) -> dict[str, Any] | None:
    """Median and spread of ``mode``'s timed steps pooled over the rounds ``over``; None for no round.

    ``by_round`` maps round -> mode -> run JSON. ``verify_median_s`` pools ``verify_s`` the same
    way and is None when none of those runs recorded it.
    """
    if not over:
        return None
    steps = [float(s) for rnd in over for s in by_round[rnd][mode]["step_s"]]
    verify = [float(s) for rnd in over for s in by_round[rnd][mode].get("verify_s", ())]
    traced = [t for rnd in over for t in by_round[rnd][mode].get("trace_steps", ())]
    t = Timing(reps=tuple(steps))
    return {
        "median": t.median,
        "spread": t.spread,
        "n": len(steps),
        "verify_median_s": Timing(reps=tuple(verify)).median if verify else None,
        "trace": trace_medians(traced) if traced else None,
    }


def trace_medians(steps: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-phase medians over per-step trace summaries (``summarize_trace`` dicts)."""
    keys = list(steps[0]) if steps else []
    return {k: Timing(reps=tuple(float(s[k]) for s in steps)).median for k in keys}


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
        return pooled_stats(by_round, mode, over)

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


def read_results(
    out_dir: Path, rounds: int, *, modes: Sequence[str] = MODES
) -> list[dict[str, Any]]:
    """The complete run JSONs of ``out_dir`` in interleaved order (incomplete ones are left out)."""
    results: list[dict[str, Any]] = []
    for r, m in interleaved(rounds, modes=modes):
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
    p.add_argument("--mode", choices=MODES + EXTRA_MODES, help="the one mode this process runs")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=MODES + EXTRA_MODES,
        default=list(MODES),
        help="the modes an orchestration runs each round, in this order (default: the five)",
    )
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
        "--cache-limit",
        type=int,
        default=FLUX_CACHE_LIMIT,
        help="MLX buffer-cache limit in bytes for this process (part of the resume key)",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="record per-block phase timestamps (decode / encode / eval wait) into the run JSON",
    )
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
    # Children run with the repository root as cwd, so every path they receive must be absolute.
    for name in ("out", "out_dir", "df11", "embeds"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
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
        cache_limit=args.cache_limit,
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


Decode = Callable[[MxGroup], DecodeResult]


def make_provider(
    mode: str,
    ckpt: DF11Checkpoint,
    resident: Mapping[str, Any],
    shapes: Mapping[str, Any],
    *,
    decode: Decode | None = None,
) -> WeightProvider:
    """The mode's provider.

    ``DF11Provider`` for df11 modes; otherwise a ``ReuseProvider`` over one double and one single
    block decoded once by that same backend (the control's only extra memory). ``decode`` is the
    Metal backend by default; tests inject a counting reference decode.
    """
    decoder = DF11Provider(
        resident, {n: ckpt.groups[n].matrix_names for n in shapes}, decode=decode
    )
    if mode.startswith("df11-prefetch"):
        stream = mx.new_stream(mx.gpu) if mode == "df11-prefetch" else None
        return PrefetchProvider(decoder, shapes, stream=stream)
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


def step_inputs(
    args: argparse.Namespace,
) -> tuple[Any, mx.array, mx.array, mx.array, dict[str, Any]]:
    """The mflux ``Config``, the float32 noise latents and the embeddings of a run, plus their record.

    Returns ``(config, latents, prompt, pooled, record)``; ``record`` holds the embeddings path and
    metadata and the latent and embedding dtypes, for the run JSON.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

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
    record = {
        "embeds": {"path": str(args.embeds), "metadata": embeds_meta},
        "latent_dtype": str(latents.dtype),
        "embeds_dtype": {
            "prompt_embeds": str(prompt.dtype),
            "pooled_prompt_embeds": str(pooled.dtype),
        },
    }
    return config, latents, prompt, pooled, record


def time_steps(
    transformer: Any,
    provider: WeightProvider,
    config: Any,
    latents: mx.array,
    prompt: mx.array,
    pooled: mx.array,
    *,
    warmup: int,
    steps: int,
    per_step: int,
    compressed_loaded: bool,
    limits_recorded: bool,
    watchdog: Watchdog,
    label: str,
    tracer: Tracer | None = None,
) -> dict[str, Any]:
    """Warm up, assert the parity conditions, time the steps; the measured fields of the run JSON.

    With a ``tracer`` (the one attached to the transformer's seam) the result also carries
    ``trace_events`` (the timed steps' block events), ``trace_steps`` (one ``summarize_trace``
    dict per timed step) and ``trace_medians``.

    ``per_step`` is the decode launches every step must make (``label`` names the mode in errors);
    ``compressed_loaded`` and ``limits_recorded`` are the two conditions the caller establishes
    (``compressed_set_loaded``, ``limits_in_force``). The provider's deferred status words are read
    by ``verify_step`` inside every step, so "nothing pending" is not a condition here; the
    queue-then-drain contract is tested on the provider itself.

    Raises:
        ParityError: A parity condition failed before the timed loop.
        BenchError: A timed step's launches differ from ``per_step``, or the final latents are not
            finite.
    """
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

    warmup_s, warmup_verify_s = run_steps(0, warmup)
    failed = parity_conditions(
        compressed_loaded=compressed_loaded,
        limits_recorded=limits_recorded,
        launches_expected=provider.launches == per_step * warmup,
        finite=bool(mx.isfinite(latents).all().item()),
    )
    if failed:
        raise ParityError(failed)
    # The timed steps' own peaks, apart from the load and warm-up spike: snapshot the lifetime
    # peaks so far, then start both counters over.
    lifetime_footprint_peak = max(footprint_peak, watchdog.peak_footprint)
    lifetime_mlx_peak = int(mx.get_peak_memory())
    mx.reset_peak_memory()
    watchdog.reset_peak()
    footprint_peak = phys_footprint()
    step_s, verify_s = run_steps(warmup, steps)
    if any(n != per_step for n in launches_per_step[warmup:]):
        raise BenchError(
            f"timed steps made {launches_per_step[warmup:]} launches; {label} expects "
            f"{per_step} per step"
        )
    if not bool(mx.isfinite(latents).all().item()):
        raise BenchError("the final latents are not finite")
    timing = Timing(reps=tuple(step_s))
    traced: dict[str, Any] = {}
    if tracer is not None:
        timed_steps = range(warmup, warmup + steps)
        trace_steps = [
            summarize_trace([e for e in tracer.events if e.step == s], step=tracer.steps[s])
            for s in timed_steps
        ]
        traced = {
            "trace_events": [dataclasses.asdict(e) for e in tracer.events if e.step >= warmup],
            "trace_steps": trace_steps,
            "trace_medians": trace_medians(trace_steps),
        }
    return {
        **traced,
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
        "step_footprint_peak_bytes": max(footprint_peak, watchdog.peak_footprint),
        "step_mlx_peak_bytes": int(mx.get_peak_memory()),
        "footprint_peak_bytes": max(
            lifetime_footprint_peak, footprint_peak, watchdog.peak_footprint
        ),
        "watchdog_peak_footprint_bytes": max(lifetime_footprint_peak, watchdog.peak_footprint),
        "mlx_peak_memory_bytes": max(lifetime_mlx_peak, int(mx.get_peak_memory())),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
    }


def run_mode(
    args: argparse.Namespace, watchdog: Watchdog, *, limits_recorded: bool
) -> dict[str, Any]:
    """Build, load, warm up, assert the parity conditions, time the steps; the result dict.

    Raises:
        ParityError: A parity condition failed before the timed loop.
        BenchError: A timed step's launches differ from the mode's, or the final latents are not finite.
    """
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
    tracer = Tracer() if args.trace else None
    transformer.attach(provider, shapes, eval_policy=policy, tracer=tracer)
    config, latents, prompt, pooled, inputs = step_inputs(args)
    measured = time_steps(
        transformer,
        provider,
        config,
        latents,
        prompt,
        pooled,
        warmup=args.warmup,
        steps=args.steps,
        per_step=expected_launches(mode, n_double=N_DOUBLE, n_single=N_SINGLE, steps=1),
        compressed_loaded=compressed_set_loaded(resident, shapes),
        limits_recorded=limits_recorded,
        watchdog=watchdog,
        label=mode,
        tracer=tracer,
    )
    return {
        "exit_code": EXIT_OK,
        "mode": mode,
        "policy": policy,
        "traced": tracer is not None,
        "round": args.round,
        "model": args.model,
        "size": args.size,
        "steps": args.steps,
        "warmup": args.warmup,
        "seed": args.seed,
        "guidance": GUIDANCE,
        "n_double": N_DOUBLE,
        "n_single": N_SINGLE,
        **measured,
        **inputs,
        "timings_s": timings,
        "provenance": provenance(),
    }


Measure = Callable[..., dict[str, Any]]


def run_one(
    args: argparse.Namespace,
    *,
    measure: Measure | None = None,
    key: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
    cache_limit: int = FLUX_CACHE_LIMIT,
) -> int:
    """Run one mode under the caps, the cache limit and the watchdog; write ``--out``.

    ``measure(args, watchdog, limits_recorded=...)`` returns the run's result dict (default
    ``run_mode``) and ``key(args)`` its resume key (default ``current_key``); another bench with the
    same run discipline (the reduced-depth control validation) passes its own. ``cache_limit`` is
    the MLX buffer-cache limit to set for the process (``--cache-limit``; the rig's by default).
    """
    measure = run_mode if measure is None else measure
    key_of = current_key if key is None else key
    caps = list(install_memory_caps())
    mx.set_cache_limit(cache_limit)
    in_force = int(mx.set_cache_limit(cache_limit))  # the limit now in force
    limits_recorded = limits_in_force(caps, in_force, expected=cache_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    move_stale_abort_aside(args.out.parent)
    watchdog = Watchdog(args.out.parent, ceiling=default_ceiling(), budget=args.wall_budget).start()
    summary: dict[str, Any]
    this_key: dict[str, Any] | None = None
    try:
        this_key = key_of(args)
        summary = measure(args, watchdog, limits_recorded=limits_recorded)
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
            "cache_limit_bytes": in_force,
            "key": this_key,
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
    for mode, stats in rep["pooled"].items():
        trace = stats.get("trace")
        if trace:
            phases = ", ".join(f"{k[:-2]} {v:.3f}" for k, v in trace.items() if k.endswith("_s"))
            print(f"trace {mode} (median s/step): {phases}")
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
    modes: list[str] = list(args.modes)
    conflicts = resume_conflicts(out_dir, args.rounds, key, modes=modes)
    if conflicts:
        for r, m, fields in conflicts:
            print(
                f"error: {run_path(out_dir, r, m)} was written by a run with other {fields}; "
                "use a fresh --out-dir",
                file=sys.stderr,
            )
        return EXIT_ERROR
    todo = pending_runs(out_dir, args.rounds, modes=modes)
    print(
        f"{len(interleaved(args.rounds, modes=modes)) - len(todo)} run(s) already complete, "
        f"{len(todo)} to run"
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
            cache_limit=args.cache_limit,
            trace=args.trace,
        )
        print(f"round {round_no} {mode}: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd, cwd=_REPO, check=False).returncode
        if code != 0:
            stopped = {"round": round_no, "mode": mode, "exit_code": code}
            print(f"error: round {round_no} {mode} exited {code}; stopping", file=sys.stderr)
            break
    results = read_results(out_dir, args.rounds, modes=modes)
    rep = report(results, stopped=stopped)
    rep.update(
        {
            "key": key,
            "modes": modes,
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
    if not args.orchestrate:
        return run_one(args, cache_limit=args.cache_limit)
    try:
        return orchestrate(args)
    except Exception as exc:  # a setup or report error is a tool error (2), never 1
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
