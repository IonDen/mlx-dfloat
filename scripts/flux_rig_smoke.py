"""One FLUX.1 denoise step through the rig on a reduced transformer: the first unit of the step bench.

Builds mflux's ``Transformer`` with ``--double`` joint and ``--single`` single blocks (every extra
loaded, every block matrix a placeholder), loads those blocks' DF11 groups, and runs ``--steps``
steps in two modes: ``df11`` (``DF11Provider``: one Metal decode per block, the chosen eval policy)
and ``control`` (``ResidentProvider`` over the same decoded weights, no launches), each after one
untimed warm-up step. Every output must be finite. The report carries the launches, the per-step
times (labelled: not a measurement), the footprint peak, the cache limit and whether the two modes'
outputs are bit-identical (reported, not asserted).

Gated heavy-run unit: it loads real weights and dispatches the decode kernel. Expected at 1+1 blocks
and 256 px: about 2 GB and seconds.

Usage (from the repository root of a synced checkout, ``--group bench``):
    uv run python -m scripts.flux_rig_smoke --df11 DIR (--embeds FILE | --synthetic) \
        [--model schnell|dev] [--double 1] [--single 1] [--size 256] [--steps 1] \
        [--policy per-block|depth2|none] [--out DIR] [--wall-budget S] [--seed N]
``--embeds`` is a safetensors file with ``prompt_embeds`` and ``pooled_prompt_embeds``.
Exit codes: 0 finite outputs, 2 a non-finite output or an input, format or tool error (1 is
reserved for a real bit mismatch and never used here), 70/71 watchdog abort (footprint ceiling /
wall budget). The guidance is the step bench's (mflux's 3.5 default; inert for schnell) and is
recorded in the report.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx
    from scripts._bench_common import move_stale_abort_aside, provenance, write_json_atomic
    from scripts._flux_rig import (
        EVAL_POLICIES,
        FLUX_CACHE_LIMIT,
        DF11Provider,
        ResidentProvider,
        RigError,
        build_transformer,
        decode_resident,
        load_resident_set,
    )
    from scripts._watchdog import Watchdog, default_ceiling, phys_footprint
    from scripts.bench_flux_step import GUIDANCE, N_DOUBLE, N_SINGLE

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.flux1.names import flux_name_map
except Exception as exc:  # a broken environment is a tool error (2)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_ERROR = 0, 2
LATENT_CHANNELS = 64  # packed FLUX.1 latents: (1, (H/16)*(W/16), 64)
POOLED_DIM = 768
T5_DIM = 4096


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--df11", type=Path, required=True, help="DF11 checkpoint directory")
    p.add_argument("--model", choices=("schnell", "dev"), default="schnell")
    p.add_argument("--double", type=int, default=1, help="joint transformer blocks to build")
    p.add_argument("--single", type=int, default=1, help="single transformer blocks to build")
    p.add_argument("--size", type=int, default=256, help="image side in pixels (multiple of 16)")
    p.add_argument("--steps", type=int, default=1, help="denoise steps to run per mode")
    embeds = p.add_mutually_exclusive_group(required=True)
    embeds.add_argument(
        "--embeds", type=Path, help="safetensors with prompt_embeds, pooled_prompt_embeds"
    )
    embeds.add_argument("--synthetic", action="store_true", help="seeded random embeddings")
    p.add_argument("--policy", choices=EVAL_POLICIES, default="per-block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("_artifacts/flux_rig_smoke"))
    p.add_argument(
        "--wall-budget", type=float, default=900.0, help="seconds before the watchdog aborts"
    )
    args = p.parse_args(argv)
    if not (1 <= args.double <= N_DOUBLE and 1 <= args.single <= N_SINGLE):
        p.error(f"--double must be in 1..{N_DOUBLE} and --single in 1..{N_SINGLE}")
    return args


def verdict(*, finite: bool) -> int:
    """The exit code for the outputs: 0 when every output is finite, else 2 (1 means a bit mismatch)."""
    return EXIT_OK if finite else EXIT_ERROR


def config_kwargs(args: argparse.Namespace) -> dict[str, int | float]:
    """The keywords of mflux's ``Config``: at least four scheduler steps, the image side, the bench guidance."""
    return {
        "num_inference_steps": max(4, args.steps),
        "height": args.size,
        "width": args.size,
        "guidance": GUIDANCE,
    }


def make_inputs(args: argparse.Namespace) -> tuple[mx.array, mx.array, mx.array]:
    """Packed latents plus prompt embeddings (from ``--embeds`` or seeded noise), dtypes as mflux passes them.

    No cast: upstream feeds the float32 ``create_noise`` latents and the encoders' outputs as they
    are (T5 float32, CLIP bfloat16); the transformer's Linears promote their bf16 weights.

    Raises:
        RigError: The embeddings file lacks a key or has an unexpected shape.
    """
    keys = mx.random.split(mx.random.key(args.seed), 3)
    n_tokens = (args.size // 16) ** 2
    hidden = mx.random.normal((1, n_tokens, LATENT_CHANNELS), key=keys[0])
    if args.synthetic:
        seq = 256 if args.model == "schnell" else 512
        prompt = mx.random.normal((1, seq, T5_DIM), key=keys[1])
        pooled = mx.random.normal((1, POOLED_DIM), key=keys[2])
    else:
        data = mx.load(str(args.embeds))
        try:
            prompt, pooled = data["prompt_embeds"], data["pooled_prompt_embeds"]
        except KeyError as exc:
            raise RigError(f"{args.embeds}: no {exc} tensor") from exc
        if prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[2] != T5_DIM:
            raise RigError(
                f"{args.embeds}: prompt_embeds has shape {prompt.shape}, expected (1, N, {T5_DIM})"
            )
        if tuple(pooled.shape) != (1, POOLED_DIM):
            raise RigError(
                f"{args.embeds}: pooled_prompt_embeds has shape {pooled.shape}, expected (1, {POOLED_DIM})"
            )
    inputs = (hidden, prompt, pooled)
    mx.eval(*inputs)
    return inputs


STEP_S_NOTE = "not a measurement (single step; JIT and warm-up effects)"


def run_steps(
    transformer: Any, config: Any, inputs: tuple[mx.array, mx.array, mx.array], steps: int
) -> tuple[mx.array, list[float]]:
    """One untimed warm-up step, then ``steps`` steps (t = 0, 1, ...) timed to their final eval.

    Every step ends with ``verify_step()``, which reads the decode status words after the eval.
    """
    hidden, prompt, pooled = inputs
    times: list[float] = []
    out = hidden
    for t in range(-1, steps):  # -1 is the warm-up
        start = time.perf_counter()
        out = transformer(
            t=max(t, 0),
            config=config,
            hidden_states=hidden,
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
        )
        mx.eval(out)
        transformer.verify_step()
        if t >= 0:
            times.append(time.perf_counter() - start)
    return out, times


def smoke(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, object]:
    """Build, load, run both modes, and return the summary (its ``exit_code`` is the verdict)."""
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    timings: dict[str, float] = {}
    ckpt = open_checkpoint(args.df11)
    start = time.perf_counter()
    transformer, shapes = build_transformer(
        args.model, ckpt, n_double=args.double, n_single=args.single
    )
    timings["build_s"] = time.perf_counter() - start
    active_after_build = int(mx.get_active_memory())
    names = list(shapes)
    start = time.perf_counter()
    resident = load_resident_set(ckpt, names)
    timings["load_resident_s"] = time.perf_counter() - start
    model_config = ModelConfig.schnell() if args.model == "schnell" else ModelConfig.dev()
    config = Config(model_config, **config_kwargs(args))
    inputs = make_inputs(args)

    df11 = DF11Provider(resident, {n: ckpt.groups[n].matrix_names for n in names}, flux_name_map())
    transformer.attach(df11, shapes, eval_policy=args.policy)
    out_df11, df11_times = run_steps(transformer, config, inputs, args.steps)
    launches_df11 = df11.launches
    if launches_df11 != (args.steps + 1) * len(names):  # + the warm-up step
        raise RigError(
            f"{launches_df11} launches for {args.steps} + 1 steps of {len(names)} blocks"
        )
    finite_df11 = bool(mx.isfinite(out_df11).all().item())

    control = ResidentProvider(decode_resident(df11, shapes))  # one block evaluated at a time
    transformer.attach(control, shapes, eval_policy=args.policy)
    out_control, control_times = run_steps(transformer, config, inputs, args.steps)
    finite_control = bool(mx.isfinite(out_control).all().item())
    equal = bool(mx.array_equal(out_df11.view(mx.uint16), out_control.view(mx.uint16)))

    finite = finite_df11 and finite_control
    return {
        "exit_code": verdict(finite=finite),
        "model": args.model,
        "guidance": GUIDANCE,
        "blocks": names,
        "size": args.size,
        "steps": args.steps,
        "policy": args.policy,
        "embeds": "synthetic" if args.synthetic else str(args.embeds),
        "seed": args.seed,
        "output_shape": list(out_df11.shape),
        "latent_dtype": str(inputs[0].dtype),
        "embeds_dtype": {
            "prompt_embeds": str(inputs[1].dtype),
            "pooled_prompt_embeds": str(inputs[2].dtype),
        },
        "step_s_note": STEP_S_NOTE,
        "df11": {
            "launches": launches_df11,
            "launches_per_step": len(names),
            "step_s": df11_times,
            "finite": finite_df11,
        },
        "control": {
            "launches": control.launches,
            "launches_per_step": 0,
            "step_s": control_times,
            "finite": finite_control,
        },
        "cache_limit_bytes": FLUX_CACHE_LIMIT,
        "outputs_bit_identical": equal,
        "active_after_build_bytes": active_after_build,
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "timings_s": timings,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the smoke under the watchdog and write ``smoke.json`` to ``--out``."""
    args = parse_args(argv)
    caps = list(install_memory_caps())
    mx.set_cache_limit(FLUX_CACHE_LIMIT)
    args.out.mkdir(parents=True, exist_ok=True)
    move_stale_abort_aside(args.out)
    watchdog = Watchdog(args.out, ceiling=default_ceiling(), budget=args.wall_budget).start()
    try:
        summary = smoke(args, watchdog)
        summary["provenance"] = provenance(caps)
    except Exception as exc:  # any unexpected failure is a tool error (2), never the verdict (0/1)
        summary = {"exit_code": EXIT_ERROR, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    finally:
        watchdog.stop()
    summary["memory_caps_gb"] = caps
    summary["cache_limit_bytes"] = FLUX_CACHE_LIMIT
    try:
        write_json_atomic(args.out / "smoke.json", summary)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: cannot write smoke.json ({exc})", file=sys.stderr)
        summary["exit_code"] = EXIT_ERROR
    code = int(summary["exit_code"])  # type: ignore[arg-type]
    if code == EXIT_OK:
        print(
            f"ok: {summary['df11']['launches']} launches over {args.steps} + 1 step(s), "  # type: ignore[index]
            f"df11 {summary['df11']['step_s']} s, control {summary['control']['step_s']} s "  # type: ignore[index]
            f"({STEP_S_NOTE}), "
            f"footprint peak {int(summary['footprint_peak_bytes']) / 1024**3:.2f} GiB, "  # type: ignore[call-overload]
            f"bit-identical {summary['outputs_bit_identical']}"
        )
    else:
        print(f"exit {code}: {summary.get('error', 'non-finite output')}", file=sys.stderr)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
