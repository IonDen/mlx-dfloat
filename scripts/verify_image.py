"""Image identity: DFloat11 latents against the same seam streaming the BF16 weights, for every BF16 family.

The ``df11`` side generates through ``DFloatFlux1``, ``DFloatZImage``, ``DFloatFlux2Klein`` or ``DFloatQwenImage21``
(the prompt encoded by its own encoders; the final latents captured by an after-loop callback and saved with the
embeddings and the image). The ``bf16`` side builds the same seamed transformer with its extras read from the base's BF16
transformer shards, drives it with a ``StreamingBF16Provider`` (each block's matrices read from
the shards as the block runs, never all resident) through exactly mflux's loop body on the saved
embeddings and the same seed, then decodes with the base's VAE. ``compare`` checks the two latent
files bit for bit (an unsigned view as wide as the dtype) and that the DF11 latents are not degenerate (finite,
variance above zero, moved away from the initial noise).

Both sides run under the footprint watchdog. ``--orchestrate`` runs df11 and bf16 as subprocesses,
one at a time, skipping a side whose ``result.json`` already carries this run's key, then compares
the two sides' saved latents in-process. For FLUX.2 Klein, ``--base`` is the snapshot of the model's own BF16
repository (a distilled model's transformer lives in the distilled repository, not in the base one). The Klein
bf16 side's copy of mflux's loop body is measurement glue, run by the identity check and not unit-tested: the df11
side runs mflux's own loop, so a drift between the two shows as a false mismatch (exit 1), never as a false pass.
For Qwen-Image 2.1, ``--base`` must hold the BF16 transformer (``transformer/``: its index and both shards) for the
bf16 side and ``--orchestrate``; the df11 side needs only the text encoder, the VAE and the DF11 file. Both Qwen sides
run the forward pass uncompiled, and the bf16 side's copy of mflux's Qwen loop body is measurement glue in the same
sense as Klein's.
Usage (from the repository root of a synced checkout, ``--group bench``):
    uv run python -m scripts.verify_image --orchestrate --model schnell --df11 DIR --base DIR --out DIR \
        [--prompt "..."] [--seed 42] [--steps 4] [--size 1024] [--guidance 3.5] [--negative-prompt "..."] [--eval-policy per-block]
Exit codes: 0 equal and non-degenerate, 1 the latents differ, 2 any error, 70/71 watchdog abort.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

try:
    import mlx.core as mx
    from scripts._bench_common import (
        move_stale_abort_aside,
        provenance,
        resume_key_diff,
        write_json_atomic,
    )
    from scripts._watchdog import Watchdog, default_ceiling, phys_footprint
    from scripts.verify_checkpoint import source_hash

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.integrate.providers import StreamingBF16Provider
    from mlx_dfloat.mflux.families import MODELS, entry
    from mlx_dfloat.mflux.flux1 import init as base_init
    from mlx_dfloat.mflux.flux1.names import flux_name_map
    from mlx_dfloat.mflux.flux1.transformer import (
        base_extras,
        base_transformer_index,
        build_transformer,
    )
except Exception as exc:  # a broken environment is a tool error (2), never the mismatch code (1)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_MISMATCH, EXIT_ERROR = 0, 1, 2
SIDES: tuple[str, ...] = ("df11", "bf16")
DEFAULT_PROMPT = (
    "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing "
    "boat far out on the water"
)


class VerifyImageError(Exception):
    """An input or tool problem of this check (exit 2), distinct from a real latent mismatch (exit 1)."""


# --- pure parts (unit-tested without mflux) -----------------------------------------------------


# Models with no BF16 side to compare against, refused before any side runs.
NO_BF16_ORIGINAL: dict[str, str] = {
    "z-image-turbo": "its original transformer is FP32, so there is no BF16 side to compare with; "
    "check its checkpoint with verify_remote_group --cast-fp32-to-bf16 instead",
}


def klein_base_problem(model: str, base: Path) -> str | None:
    """Why ``base`` cannot be a FLUX.2 Klein model's BF16 side, or None (and None for every other family).

    The base and distilled Klein repositories share the text encoder and VAE but not the transformer, which the
    bf16 side streams from ``base``. BFL's ``model_index.json`` marks a distilled repository with
    ``"is_distilled": true``; mflux tells the variants apart by the name (``flux2_generate.py:74``: distilled = no
    "base" in the model name), and so does this check with the registry's repository name.
    """
    e = entry(model)
    if e.family != "flux2":
        return None
    distilled = "base" not in e.base_repo.lower()
    index_path = base / "model_index.json"
    try:
        index = json.loads(index_path.read_text())
    except FileNotFoundError:
        return (
            f"--base {base}: missing model_index.json: pass the full snapshot of {e.base_repo} (the bf16 side also "
            "needs its transformer/; the model_index.json tells a base from a distilled transformer)"
        )
    except (OSError, ValueError) as exc:
        return (
            f"{index_path}: {type(exc).__name__}; the bf16 side needs the snapshot of {e.base_repo} "
            "(its model_index.json tells a base from a distilled transformer)"
        )
    if not isinstance(index, dict) or bool(index.get("is_distilled", False)) != distilled:
        found = "distilled" if isinstance(index, dict) and index.get("is_distilled") else "base"
        return (
            f"--base {base} holds a {found} FLUX.2 Klein transformer, but {e.label} is "
            f"{'distilled' if distilled else 'a base model'}: pass the snapshot of {e.base_repo}"
        )
    return None


def qwen21_base_problem(model: str, base: Path) -> str | None:
    """Why ``base`` cannot feed a Qwen-Image 2.1 bf16 side, or None (and None for every other family).

    The bf16 side streams the BF16 transformer from ``base/transformer``: its ``*.safetensors.index.json`` and every
    shard the index names must be there, each named by a plain file name (as the streaming reader requires). The df11
    side never reads it.
    """
    e = entry(model)
    if e.family != "qwen21":
        return None
    root = base / "transformer"
    need = (
        f"--base {base}: the bf16 side streams the BF16 transformer from transformer/ "
        f"(download {e.base_repo}'s transformer/* first)"
    )
    indexes = sorted(root.glob("*.safetensors.index.json")) if root.is_dir() else []
    if len(indexes) != 1:
        return f"{need}; found {len(indexes)} weight indexes in {root}"
    try:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        shards = sorted({str(v) for v in weight_map.values()})
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"{indexes[0]}: unreadable weight index ({type(exc).__name__}); {need}"
    unsafe = [s for s in shards if Path(s).name != s]
    if unsafe:
        return f"{indexes[0]}: a shard name that is not a plain file name: {unsafe[0]!r}"
    missing = [s for s in shards if not (root / s).is_file()]
    if not shards or missing:
        return f"{need}; the index names {len(shards)} shards, missing {missing}"
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line: one side (``df11``, ``bf16`` or ``compare``), or ``--orchestrate``."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--side", choices=(*SIDES, "compare"), help="the one side this process runs")
    mode.add_argument("--orchestrate", action="store_true", help="run df11, bf16, then compare")
    p.add_argument("--model", choices=tuple(MODELS), default="schnell")
    p.add_argument("--df11", required=True, help="DF11 checkpoint directory")
    p.add_argument(
        "--base", required=True, help="base repository directory (encoders, VAE and transformer)"
    )
    p.add_argument("--out", type=Path, required=True, help="directory for both sides' outputs")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="guidance (default per model, as mflux: FLUX.1 3.5; Z-Image unset: the model's own rule; "
        "FLUX.2 Klein 1.0, base models with CFG at 4; Qwen-Image 2.1 1.0, CFG above 1 with --negative-prompt)",
    )
    p.add_argument(
        "--negative-prompt",
        default=None,
        help="the negative prompt (used by z-image, a base model, and qwen-image-2.1 above guidance 1)",
    )
    p.add_argument("--eval-policy", choices=("per-block", "depth2"), default="per-block")
    p.add_argument(
        "--wall-budget", type=float, default=7200.0, help="seconds before the watchdog aborts"
    )
    args = p.parse_args(argv)
    if args.model in NO_BF16_ORIGINAL:
        p.error(f"--model {args.model}: {NO_BF16_ORIGINAL[args.model]}")
    if entry(args.model).family == "flux2" and args.negative_prompt is not None:
        p.error(
            f"--negative-prompt: {args.model} takes none (mflux's FLUX.2 Klein encodes its own blank negative)"
        )
    problem = klein_base_problem(args.model, Path(args.base).expanduser())
    if problem is not None:
        p.error(f"--model {args.model}: {problem}")
    # Only the runs that stream the BF16 transformer need it; the df11 side and compare never read it.
    if args.orchestrate or args.side == "bf16":
        problem = qwen21_base_problem(args.model, Path(args.base).expanduser())
        if problem is not None:
            p.error(f"--model {args.model}: {problem}")
    if args.guidance is None:  # resolved before run_key, so a stored FLUX.1 side keeps its 3.5
        args.guidance = entry(args.model).default_guidance
    # Absolute paths: the children run with the repository root as their cwd.
    args.df11 = str(Path(args.df11).expanduser().resolve())
    args.base = str(Path(args.base).expanduser().resolve())
    args.out = args.out.expanduser().resolve()
    return args


def run_key(args: argparse.Namespace) -> dict[str, Any]:
    """This run's resume key: every setting that changes the latents, plus the source and mlx version."""
    key: dict[str, Any] = {
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "size": args.size,
        "prompt": args.prompt,
        "guidance": args.guidance,
        "eval_policy": args.eval_policy,
        "df11": str(Path(args.df11).expanduser()),
        "base": str(Path(args.base).expanduser()),
        "source": source_hash(),
        "mlx": mx.__version__,
    }
    if args.negative_prompt is not None:  # absent when unset: FLUX.1 keys keep their old fields
        key["negative_prompt"] = args.negative_prompt
    return key


def nondegenerate(latents: mx.array, noise: mx.array) -> list[str]:
    """The names of the checks ``latents`` fails: ``finite``, ``variance``, ``moved_from_noise``.

    A non-finite tensor returns immediately with only ``"finite"``: variance and the noise
    comparison are meaningless once a value is NaN or infinite.
    """
    failed: list[str] = []
    if not bool(mx.isfinite(latents).all().item()):
        failed.append("finite")
        return failed
    if float(mx.var(latents.astype(mx.float32)).item()) <= 0.0:
        failed.append("variance")
    if latents.shape == noise.shape and compare_latents(latents, noise):
        failed.append("moved_from_noise")
    return failed


_UNSIGNED_BY_BYTES = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}


def compare_latents(a: mx.array, b: mx.array) -> bool:
    """Whether ``a`` and ``b`` share a shape, a dtype and every bit (never a float ``==``, which equates ±0.0).

    The bits are compared through the unsigned integer as wide as the dtype (``uint16`` for bf16, ``uint32`` for
    float32): the latents are bf16 or end float32, and a view of another width fails on an odd length.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    unsigned = _UNSIGNED_BY_BYTES[a.itemsize]
    return bool(mx.array_equal(a.view(unsigned), b.view(unsigned)))


def verdict(*, equal: bool, degenerate: list[str]) -> int:
    """The exit code: 2 when the df11 latents are degenerate (nothing meaningful was compared), else 0/1."""
    if degenerate:
        return EXIT_ERROR
    return EXIT_OK if equal else EXIT_MISMATCH


def child_command(args: argparse.Namespace, side: str) -> list[str]:
    """The subprocess argv for one side (never ``--orchestrate``; run with the repository root as cwd)."""
    command = [
        sys.executable,
        "-m",
        "scripts.verify_image",
        "--side",
        side,
        "--model",
        args.model,
        "--df11",
        args.df11,
        "--base",
        args.base,
        "--out",
        str(args.out),
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--steps",
        str(args.steps),
        "--size",
        str(args.size),
        "--eval-policy",
        args.eval_policy,
        "--wall-budget",
        str(args.wall_budget),
    ]
    if args.guidance is not None:
        command += ["--guidance", str(args.guidance)]
    if args.negative_prompt is not None:
        command += ["--negative-prompt", args.negative_prompt]
    return command


def _read_json(path: Path) -> Any | None:
    """The parsed JSON at ``path``, or ``None`` when it's missing or not valid JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def side_complete(path: Path, key: Mapping[str, Any]) -> bool:
    """Whether ``path / "result.json"`` holds a successful run keyed exactly like ``key`` (the resume check)."""
    data = _read_json(path / "result.json")
    if not isinstance(data, dict):
        return False
    return data.get("exit_code") == EXIT_OK and not resume_key_diff(data.get("key"), key)


def sides_ready(out_dir: Path, key: Mapping[str, Any]) -> list[str]:
    """The problems blocking a compare: a side missing its result, not exit 0, or keyed differently than ``key``.

    Empty exactly when both ``df11`` and ``bf16`` are complete and keyed exactly like ``key`` — the
    gate ``run_compare`` checks before it will issue the 0/1 verdict, so a stale, failed or
    different-run side is never silently compared.
    """
    problems: list[str] = []
    for side in SIDES:
        data = _read_json(out_dir / side / "result.json")
        if not isinstance(data, dict):
            problems.append(f"{side}: no result.json")
            continue
        if data.get("exit_code") != EXIT_OK:
            problems.append(f"{side}: exit_code {data.get('exit_code')!r}, not 0")
            continue
        diff = resume_key_diff(data.get("key"), key)
        if diff:
            problems.append(f"{side}: key differs in {diff}")
    return problems


def _sha256_file(path: Path) -> str:
    """The sha256 hex digest of ``path``'s bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixels_identical(a: Path, b: Path) -> bool:
    """Whether two image files decode to the same size, mode and pixels (their metadata ignored)."""
    from PIL import Image

    with Image.open(a) as image_a, Image.open(b) as image_b:
        return (
            image_a.size == image_b.size
            and image_a.mode == image_b.mode
            and image_a.tobytes() == image_b.tobytes()
        )


def max_phase_peak(peaks: Mapping[str, Any]) -> int:
    """The largest per-phase MLX peak in a model report's ``peaks`` (0 when no phase was recorded).

    The model resets MLX's process-wide peak counter at every phase boundary, so the counter read
    at the end holds only the last phase; the per-phase values are the whole record.
    """
    return max(
        (int(v["mlx_peak"]) for v in peaks.values() if isinstance(v, Mapping)),
        default=0,
    )


# --- the df11 side (mflux-touching; not unit-tested here) ---------------------------------------


class _LatentCapture:
    """An mflux after-loop subscriber that keeps the final packed latents every subscriber is handed."""

    def __init__(self) -> None:
        """Start with no captured latents."""
        self.latents: mx.array | None = None

    def call_after_loop(self, seed: int, prompt: str, latents: mx.array, config: Any) -> None:
        """Keep ``latents``; mflux calls every after-loop subscriber this way regardless of the rest."""
        del seed, prompt, config
        self.latents = latents


def run_df11(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Generate through ``DFloatFlux1``, capture the final latents, and save the image, latents and embeddings.

    Raises:
        VerifyImageError: The after-loop callback never ran (mflux's own loop never called it).
    """
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    out_dir = args.out / "df11"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DFloatFlux1(
        args.model, df11_path=args.df11, base_path=args.base, eval_policy=args.eval_policy
    )
    capture = _LatentCapture()
    model.callbacks.register(capture)
    image = model.generate_image(
        args.seed,
        args.prompt,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
    )
    captured = capture.latents
    if captured is None:
        raise VerifyImageError("the after-loop callback never ran; no latents were captured")
    mx.eval(captured)
    # overwrite=True: without it, GeneratedImage.save renames around an existing image.png
    # (image-1.png, ...) instead of replacing it, so a rerun after a failed attempt would leave a
    # stale image.png behind while latents.safetensors/embeds.safetensors/result.json move on.
    image.save(str(out_dir / "image.png"), export_json_metadata=False, overwrite=True)
    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": captured})
    prompt_embeds, pooled_prompt_embeds = model.prompt_cache[args.prompt]
    mx.save_safetensors(
        str(out_dir / "embeds.safetensors"),
        {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds},
    )
    noise = FluxLatentCreator.create_noise(args.seed, args.size, args.size)
    degenerate = nondegenerate(captured, noise)
    report = model.report()
    return {
        "exit_code": EXIT_OK if not degenerate else EXIT_ERROR,
        "degenerate": degenerate,
        "report": report,
        "output_shape": list(captured.shape),
        "output_dtype": str(captured.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_max_over_phases": max_phase_peak(report["peaks"]),
    }


# --- the bf16 side (mflux-touching; not unit-tested here) ---------------------------------------


def run_bf16(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Stream the base's BF16 transformer shards through the seam, mflux's exact loop body, then the VAE.

    Raises:
        VerifyImageError: The df11 side's saved embeddings are not present yet.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.utils.image_util import ImageUtil

    out_dir = args.out / "bf16"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = args.out / "df11" / "embeds.safetensors"
    if not embeds_path.exists():
        raise VerifyImageError(f"{embeds_path}: run the df11 side first")

    df11_root = Path(args.df11).expanduser()
    base_root = Path(args.base).expanduser()
    ckpt = open_checkpoint(df11_root)
    index = base_transformer_index(base_root / "transformer")
    model_config = ModelConfig.from_name(model_name=args.model, base_model=None)
    transformer, shapes = build_transformer(model_config, ckpt, extras=base_extras(index, ckpt))
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in shapes}, flux_name_map()
    )
    transformer.attach(provider, shapes, eval_policy=args.eval_policy)

    embeds = mx.load(str(embeds_path))
    prompt_embeds, pooled_prompt_embeds = embeds["prompt_embeds"], embeds["pooled_prompt_embeds"]
    config = Config(
        model_config,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
        scheduler="linear",
    )
    latents = FluxLatentCreator.create_noise(args.seed, args.size, args.size)
    mx.eval(latents, prompt_embeds, pooled_prompt_embeds)

    for t in config.time_steps:
        # Exactly mflux's own Flux1.generate_image step body (scale, transformer, scheduler step,
        # eval); verify_step() is our seam's own contract and a no-op for this provider (it defers
        # nothing), kept for the same reason bench_flux_step.denoise_step keeps it after every step.
        latents = config.scheduler.scale_model_input(latents, t)
        noise = transformer(
            t=t,
            config=config,
            hidden_states=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        mx.eval(latents)
        transformer.verify_step()

    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": latents})

    vae = base_init.load_vae(base_root)
    mx.clear_cache()
    mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
    unpacked = FluxLatentCreator.unpack_latents(latents, args.size, args.size)
    decoded = vae.decode(unpacked)  # (B, C, 1, H, W): VAE.decode's raw output, not VAEUtil's
    # PIL.Image.save always overwrites an existing file at this path; no equivalent to
    # GeneratedImage.save's rename-around-a-stale-file behaviour to guard against here.
    ImageUtil.to_pil(decoded[:, :, 0, :, :]).save(str(out_dir / "image.png"))

    return {
        "exit_code": EXIT_OK,
        "reads": provider.reads,
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_process": int(mx.get_peak_memory()),  # never reset on this side
    }


# --- the Z-Image sides (mflux-touching; exercised by the identity run, not unit-tested here) ------


def run_df11_zimage(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Generate through ``DFloatZImage``, capture the final latents, and save the image, latents and embeddings.

    The embeddings file holds ``cap_feats`` and, when classifier-free guidance ran, ``negative_cap_feats``.

    Raises:
        VerifyImageError: The after-loop callback never ran (mflux's own loop never called it).
    """
    from mflux.models.z_image.latent_creator import ZImageLatentCreator

    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    out_dir = args.out / "df11"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DFloatZImage(
        args.model, df11_path=args.df11, base_path=args.base, eval_policy=args.eval_policy
    )
    capture = _LatentCapture()
    model.callbacks.register(capture)
    image = model.generate_image(
        args.seed,
        args.prompt,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    captured = capture.latents
    if captured is None:
        raise VerifyImageError("the after-loop callback never ran; no latents were captured")
    mx.eval(captured)
    image.save(str(out_dir / "image.png"), export_json_metadata=False, overwrite=True)
    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": captured})
    wanted = model.cfg_prompts(
        args.prompt, negative_prompt=args.negative_prompt, guidance=args.guidance
    )
    embeds = {"cap_feats": model._embeddings[wanted[0]][0]}
    if len(wanted) > 1:
        embeds["negative_cap_feats"] = model._embeddings[wanted[1]][0]
    mx.save_safetensors(str(out_dir / "embeds.safetensors"), embeds)
    noise = ZImageLatentCreator.create_noise(args.seed, args.size, args.size)
    degenerate = nondegenerate(captured, noise)
    report = model.report()
    return {
        "exit_code": EXIT_OK if not degenerate else EXIT_ERROR,
        "degenerate": degenerate,
        "report": report,
        "output_shape": list(captured.shape),
        "output_dtype": str(
            captured.dtype
        ),  # bf16, or float32 when the scheduler's arithmetic promoted it
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_max_over_phases": max_phase_peak(report["peaks"]),
    }


def run_bf16_zimage(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Stream the base's BF16 transformer shards through the seam and mflux's Z-Image loop body, then the VAE.

    Raises:
        VerifyImageError: The df11 side's saved embeddings are not present yet.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.z_image.latent_creator import ZImageLatentCreator
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.utils.image_util import ImageUtil

    from mlx_dfloat.mflux._compile import uncompiled
    from mlx_dfloat.mflux.zimage import init as zinit
    from mlx_dfloat.mflux.zimage.names import zimage_name_map
    from mlx_dfloat.mflux.zimage.transformer import (
        ZIMAGE_BASE_INDEX,
        base_extras,
        build_transformer,
    )

    out_dir = args.out / "bf16"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = args.out / "df11" / "embeds.safetensors"
    if not embeds_path.exists():
        raise VerifyImageError(f"{embeds_path}: run the df11 side first")

    df11_root = Path(args.df11).expanduser()
    base_root = Path(args.base).expanduser()
    ckpt = open_checkpoint(df11_root)
    index = base_transformer_index(base_root / "transformer", index_file=ZIMAGE_BASE_INDEX)
    build = build_transformer(ckpt, extras=base_extras(index, ckpt), nonblock_from_extras=True)
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in build.shapes}, zimage_name_map()
    )
    build.transformer.attach(
        provider, build.shapes, eval_policy=args.eval_policy, verify_in_call=True
    )

    embeds = mx.load(str(embeds_path))
    text_encodings = embeds["cap_feats"]
    negative_encodings = embeds.get("negative_cap_feats")  # present only when guidance ran

    model_config = ModelConfig.from_name(model_name=args.model, base_model=None)
    supports_guidance = bool(model_config.supports_guidance)
    guidance = args.guidance if supports_guidance and args.guidance is not None else 0.0
    # mflux z_image.py:69-70: the scheduler default is a function of whether the model supports guidance.
    scheduler = "flow_match_euler_discrete" if supports_guidance else "linear"
    config = Config(
        model_config=model_config,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=guidance,
        scheduler=scheduler,
    )
    latents = ZImageLatentCreator.create_noise(args.seed, args.size, args.size)
    predict = uncompiled(ZImage._predict, build.transformer)
    mx.eval(latents, text_encodings)

    for t in config.time_steps:
        # mflux z_image.py:108-129: sigma, timestep, predict, scheduler step, eval.
        sigma_t = config.scheduler.sigmas[t].reshape((1,))
        timestep = mx.ones_like(sigma_t) - sigma_t
        noise = predict(
            latents=latents,
            timestep=timestep,
            sigmas=config.scheduler.sigmas,
            text_encodings=text_encodings,
            negative_encodings=negative_encodings,
            guidance=config.guidance,
        )
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        mx.eval(latents)
        build.transformer.verify_step()

    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": latents})

    vae = zinit.load_vae(base_root)
    mx.clear_cache()
    mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
    unpacked = ZImageLatentCreator.unpack_latents(latents, args.size, args.size)
    decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=None)
    ImageUtil.to_pil(decoded).save(str(out_dir / "image.png"))

    return {
        "exit_code": EXIT_OK,
        "reads": provider.reads,
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_process": int(mx.get_peak_memory()),  # never reset on this side
    }


# --- the FLUX.2 Klein sides (mflux-touching; exercised by the identity run, not unit-tested here) -----


def run_df11_flux2(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Generate through ``DFloatFlux2Klein``, capture the final latents, and save the image, latents and embeddings.

    The embeddings file holds ``prompt_embeds`` and ``text_ids`` and, when classifier-free guidance ran (a guidance
    above 1.0), ``negative_prompt_embeds`` and ``negative_text_ids`` for mflux's blank negative.

    Raises:
        VerifyImageError: The after-loop callback never ran (mflux's own loop never called it).
    """
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    out_dir = args.out / "df11"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DFloatFlux2Klein(
        args.model, df11_path=args.df11, base_path=args.base, eval_policy=args.eval_policy
    )
    capture = _LatentCapture()
    model.callbacks.register(capture)
    image = model.generate_image(
        args.seed,
        args.prompt,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
    )
    captured = capture.latents
    if captured is None:
        raise VerifyImageError("the after-loop callback never ran; no latents were captured")
    mx.eval(captured)
    image.save(str(out_dir / "image.png"), export_json_metadata=False, overwrite=True)
    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": captured})
    wanted = model.cfg_prompts(args.prompt, guidance=args.guidance)
    prompt_embeds, text_ids = model._embeddings[wanted[0]]
    embeds = {"prompt_embeds": prompt_embeds, "text_ids": text_ids}
    if len(wanted) > 1:
        embeds["negative_prompt_embeds"], embeds["negative_text_ids"] = model._embeddings[wanted[1]]
    mx.save_safetensors(str(out_dir / "embeds.safetensors"), embeds)
    noise = Flux2LatentCreator.prepare_packed_latents(
        seed=args.seed, height=args.size, width=args.size, batch_size=1
    )[0]
    degenerate = nondegenerate(captured, noise)
    report = model.report()
    return {
        "exit_code": EXIT_OK if not degenerate else EXIT_ERROR,
        "degenerate": degenerate,
        "report": report,
        "output_shape": list(captured.shape),
        "output_dtype": str(captured.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_max_over_phases": max_phase_peak(report["peaks"]),
    }


def run_bf16_flux2(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Stream the base's BF16 transformer through the seam and mflux's FLUX.2 Klein loop body, then the VAE.

    The transformer is one file (4B) or shards with an index (9B); its five non-block matrices load as plain BF16
    weights, the blocks stream as they run.

    Raises:
        VerifyImageError: The df11 side's saved embeddings are not present yet.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.utils.image_util import ImageUtil

    from mlx_dfloat.mflux._compile import uncompiled
    from mlx_dfloat.mflux.flux2 import init as finit
    from mlx_dfloat.mflux.flux2 import transformer as ktf
    from mlx_dfloat.mflux.flux2.names import klein_name_map

    out_dir = args.out / "bf16"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = args.out / "df11" / "embeds.safetensors"
    if not embeds_path.exists():
        raise VerifyImageError(f"{embeds_path}: run the df11 side first")

    df11_root = Path(args.df11).expanduser()
    base_root = Path(args.base).expanduser()
    ckpt = open_checkpoint(df11_root)
    index = ktf.base_transformer_files_index(base_root / "transformer")
    model_config = ModelConfig.from_name(model_name=args.model, base_model=None)
    names = klein_name_map()
    build = ktf.build_transformer(
        ckpt,
        transformer_overrides=model_config.transformer_overrides,
        name_map=names,
        extras=ktf.base_extras(index, ckpt),
        nonblock_from_extras=True,
    )
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in build.shapes}, names
    )
    build.transformer.attach(
        provider, build.shapes, eval_policy=args.eval_policy, verify_in_call=True
    )

    embeds = mx.load(str(embeds_path))
    prompt_embeds, text_ids = embeds["prompt_embeds"], embeds["text_ids"]
    negative_prompt_embeds = embeds.get("negative_prompt_embeds")  # present only when guidance ran
    negative_text_ids = embeds.get("negative_text_ids")
    config = Config(
        model_config=model_config,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
        scheduler="flow_match_euler_discrete",
    )
    latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
        seed=args.seed, height=args.size, width=args.size, batch_size=1
    )
    predict = uncompiled(Flux2Klein._predict, build.transformer)
    mx.eval(latents, prompt_embeds, text_ids)

    for t in config.time_steps:
        # mflux flux2_klein.py:91-112: predict, scheduler step, eval.
        noise = predict(
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance=args.guidance,
            timestep=config.scheduler.timesteps[t],
        )
        latents = config.scheduler.step(
            noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
        )
        mx.eval(latents)
        build.transformer.verify_step()

    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": latents})

    vae = finit.load_vae(base_root)
    mx.clear_cache()
    mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
    # mflux flux2_klein.py:117-129: unpatchify to (B, C, H, W) and decode without tiling.
    packed = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1])
    decoded = vae.decode_packed_latents(packed.transpose(0, 3, 1, 2), tiling_config=None)
    ImageUtil.to_pil(decoded).save(str(out_dir / "image.png"))

    return {
        "exit_code": EXIT_OK,
        "reads": provider.reads,
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_process": int(mx.get_peak_memory()),  # never reset on this side
    }


# --- the Qwen-Image 2.1 sides (mflux-touching; exercised by the identity run, not unit-tested here) -----------


def qwen21_embeds(
    prompt_cache: Mapping[str, tuple[mx.array, mx.array]], prompts: tuple[str, ...]
) -> dict[str, mx.array]:
    """The embeddings file of a Qwen df11 side: the prompt's pair, and the negative's when CFG ran (two prompts).

    ``prompts`` is ``DFloatQwenImage21.cfg_prompts``'s answer for the call; the pairs are ``(embeds, mask)`` as
    mflux's prompt cache holds them.
    """
    embeds, mask = prompt_cache[prompts[0]]
    out = {"prompt_embeds": embeds, "prompt_mask": mask}
    if len(prompts) > 1:
        out["negative_prompt_embeds"], out["negative_prompt_mask"] = prompt_cache[prompts[1]]
    return out


def run_df11_qwen21(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Generate through ``DFloatQwenImage21``, capture the final latents, and save the image, latents and embeddings.

    The embeddings file holds ``prompt_embeds`` and ``prompt_mask`` and, when classifier-free guidance ran (a
    guidance above 1.0 with a non-empty negative prompt), ``negative_prompt_embeds`` and ``negative_prompt_mask``.

    Raises:
        VerifyImageError: The after-loop callback never ran (mflux's own loop never called it).
    """
    from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator

    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    out_dir = args.out / "df11"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = DFloatQwenImage21(
        args.model, df11_path=args.df11, base_path=args.base, eval_policy=args.eval_policy
    )
    capture = _LatentCapture()
    model.callbacks.register(capture)
    image = model.generate_image(
        args.seed,
        args.prompt,
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    captured = capture.latents
    if captured is None:
        raise VerifyImageError("the after-loop callback never ran; no latents were captured")
    mx.eval(captured)
    image.save(str(out_dir / "image.png"), export_json_metadata=False, overwrite=True)
    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": captured})
    wanted = model.cfg_prompts(
        args.prompt, negative_prompt=args.negative_prompt, guidance=args.guidance
    )
    mx.save_safetensors(
        str(out_dir / "embeds.safetensors"), qwen21_embeds(model.prompt_cache, wanted)
    )
    noise = Qwen21LatentCreator.create_noise(args.seed, args.size, args.size)
    degenerate = nondegenerate(captured, noise)
    report = model.report()
    return {
        "exit_code": EXIT_OK if not degenerate else EXIT_ERROR,
        "degenerate": degenerate,
        "report": report,
        "output_shape": list(captured.shape),
        "output_dtype": str(captured.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_max_over_phases": max_phase_peak(report["peaks"]),
    }


def run_bf16_qwen21(args: argparse.Namespace, watchdog: Watchdog) -> dict[str, Any]:
    """Stream the base's BF16 transformer shards through the seam and mflux's Qwen-Image 2.1 loop body, then the VAE.

    ``modulation.1`` loads from the base as a plain weight; the blocks stream as they run, the forward pass
    uncompiled as on the df11 side (``EagerForward`` sets mflux's ``_step_fn`` to the plain method).

    Raises:
        VerifyImageError: The df11 side's saved embeddings are not present yet.
    """
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
    from mflux.utils.image_util import ImageUtil

    from mlx_dfloat.mflux.qwen21 import init as qinit
    from mlx_dfloat.mflux.qwen21 import transformer as qtf
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    out_dir = args.out / "bf16"
    out_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = args.out / "df11" / "embeds.safetensors"
    if not embeds_path.exists():
        raise VerifyImageError(f"{embeds_path}: run the df11 side first")

    df11_root = Path(args.df11).expanduser()
    base_root = Path(args.base).expanduser()
    ckpt = open_checkpoint(df11_root)
    index = qtf.base_transformer_files_index(base_root / "transformer")
    build = qtf.build_transformer(
        ckpt, extras=qtf.base_extras(index, ckpt), nonblock_from_extras=True
    )
    provider = StreamingBF16Provider(
        index, {n: ckpt.groups[n].matrix_names for n in build.shapes}, qwen21_name_map()
    )
    transformer = build.transformer
    transformer.attach(provider, build.shapes, eval_policy=args.eval_policy, verify_in_call=True)

    embeds = mx.load(str(embeds_path))
    prompt_embeds, prompt_mask = embeds["prompt_embeds"], embeds["prompt_mask"]
    negative_prompt_embeds = embeds.get("negative_prompt_embeds")  # present only when CFG ran
    negative_prompt_mask = embeds.get("negative_prompt_mask")
    do_true_cfg = negative_prompt_embeds is not None
    config = Config(
        model_config=ModelConfig.from_name(model_name=args.model, base_model=None),
        num_inference_steps=args.steps,
        height=args.size,
        width=args.size,
        guidance=args.guidance,
        scheduler="linear",
    )
    # mflux qwen_image_21.py: txt2img noise, cast to the model precision.
    latents = Qwen21LatentCreator.create_noise(args.seed, args.size, args.size).astype(
        ModelConfig.precision
    )
    mx.eval(latents, prompt_embeds, prompt_mask)

    for t in config.time_steps:
        # mflux qwen_image_21.py's step body: scale, positive call, negative call when CFG ran, combine, step, eval.
        latents = config.scheduler.scale_model_input(latents, t)
        noise = transformer(
            t=t,
            config=config,
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
        )
        if do_true_cfg:
            noise_negative = transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_hidden_states_mask=negative_prompt_mask,
            )
            noise = noise_negative + config.guidance * (noise - noise_negative)
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
        mx.eval(latents)
        # No verify_step(): attach(..., verify_in_call=True) checks inside each call, and this provider defers nothing.

    if not qtf.is_eager(transformer):
        raise VerifyImageError("the bf16 side's forward pass was compiled; the df11 side's is not")
    mx.save_safetensors(str(out_dir / "latents.safetensors"), {"latents": latents})

    vae = qinit.load_vae(base_root)
    mx.clear_cache()
    mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
    unpacked = Qwen21LatentCreator.unpack_latents(
        latents=latents, height=args.size, width=args.size
    )
    decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=None)
    ImageUtil.to_pil(decoded).save(str(out_dir / "image.png"))

    return {
        "exit_code": EXIT_OK,
        "reads": provider.reads,
        "cfg_calls_per_step": 2 if do_true_cfg else 1,
        "output_shape": list(latents.shape),
        "output_dtype": str(latents.dtype),
        "footprint_peak_bytes": max(watchdog.peak_footprint, phys_footprint()),
        "mlx_peak_bytes_process": int(mx.get_peak_memory()),  # never reset on this side
    }


SIDE_RUNNERS: dict[str, tuple[Callable[..., dict[str, Any]], Callable[..., dict[str, Any]]]] = {
    "flux1": (run_df11, run_bf16),
    "zimage": (run_df11_zimage, run_bf16_zimage),
    "flux2": (run_df11_flux2, run_bf16_flux2),
    "qwen21": (run_df11_qwen21, run_bf16_qwen21),
}


def runner_for(model: str, side: str) -> Callable[..., dict[str, Any]]:
    """The function that runs ``side`` (``df11`` or ``bf16``) for ``model``'s family."""
    df11_runner, bf16_runner = SIDE_RUNNERS[entry(model).family]
    return df11_runner if side == "df11" else bf16_runner


# --- compare, orchestration and the CLI ----------------------------------------------------------


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Compare both sides' saved latents bit for bit, and report the df11 side's own degeneracy.

    Refuses (``exit_code`` 2, never 0 or 1) unless ``sides_ready`` finds both sides complete and
    keyed exactly like this run — a missing, failed or different-run side is never compared. The
    two PNGs are also decoded and compared pixel for pixel (``pixels_identical``); their sha256
    hashes are informational only, since the df11 PNG carries mflux's embedded metadata.

    Raises:
        OSError: A ready side's saved latents or image file is missing.
        ValueError: A ready side's result.json turned invalid between ``sides_ready`` and here.
    """
    key = run_key(args)
    df11_dir, bf16_dir = args.out / "df11", args.out / "bf16"
    problems = sides_ready(args.out, key)
    df11_result = _read_json(df11_dir / "result.json")
    bf16_result = _read_json(bf16_dir / "result.json")
    df11_key = df11_result.get("key") if isinstance(df11_result, dict) else None
    bf16_key = bf16_result.get("key") if isinstance(bf16_result, dict) else None
    if problems:
        return {
            "exit_code": EXIT_ERROR,
            "error": "cannot compare: " + "; ".join(problems),
            "df11_key": df11_key,
            "bf16_key": bf16_key,
        }
    a = mx.load(str(df11_dir / "latents.safetensors"))["latents"]
    b = mx.load(str(bf16_dir / "latents.safetensors"))["latents"]
    equal = compare_latents(a, b)
    degenerate = list(df11_result.get("degenerate", [])) if isinstance(df11_result, dict) else []
    return {
        "exit_code": verdict(equal=equal, degenerate=degenerate),
        "equal": equal,
        "degenerate": degenerate,
        "df11_key": df11_key,
        "bf16_key": bf16_key,
        "pixels_identical": pixels_identical(df11_dir / "image.png", bf16_dir / "image.png"),
        "df11_png_sha256": _sha256_file(df11_dir / "image.png"),
        "bf16_png_sha256": _sha256_file(bf16_dir / "image.png"),
    }


def _write_compare(args: argparse.Namespace) -> int:
    """Run the compare side and write ``out/compare.json`` (an error verdict on failure); return the exit code."""
    try:
        summary = run_compare(args)
    except Exception as exc:  # any failure is a tool error (2), never the mismatch code (1)
        summary = {"exit_code": EXIT_ERROR, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    try:
        write_json_atomic(args.out / "compare.json", summary)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: cannot write compare.json ({exc})", file=sys.stderr)
        summary["exit_code"] = EXIT_ERROR
    code = int(summary["exit_code"])
    if code == EXIT_OK:
        print("ok: the latents are bit-identical and non-degenerate")
    elif code == EXIT_MISMATCH:
        print("mismatch: the df11 and bf16 latents differ", file=sys.stderr)
    else:
        print(f"exit {code}: {summary.get('error', summary.get('degenerate'))}", file=sys.stderr)
    return code


def _move_stale_result_aside(side_dir: Path, key: Mapping[str, Any]) -> None:
    """Move a previous ``result.json`` in ``side_dir`` to ``result.previous.json`` if its key differs from ``key``.

    Mirrors ``move_stale_abort_aside``: a side directory left over from a run with different
    settings must never be mistaken for this run's outputs (its stale image/latents/embeddings
    would otherwise sit next to a fresh ``result.json`` until this run overwrites them), and the
    old record is kept, never deleted.
    """
    result_path = side_dir / "result.json"
    if not result_path.exists():
        return
    data = _read_json(result_path)
    stored_key = data.get("key") if isinstance(data, dict) else None
    if resume_key_diff(stored_key, key):
        result_path.replace(side_dir / "result.previous.json")


def run_side(args: argparse.Namespace) -> int:
    """Run one heavy side (``df11`` or ``bf16``) under the memory caps and the footprint watchdog; write ``result.json``."""
    side = args.side
    side_dir = args.out / side
    caps = list(install_memory_caps())
    side_dir.mkdir(parents=True, exist_ok=True)
    key = run_key(args)
    _move_stale_result_aside(side_dir, key)
    move_stale_abort_aside(side_dir)
    watchdog = Watchdog(side_dir, ceiling=default_ceiling(), budget=args.wall_budget).start()
    run = runner_for(args.model, side)
    try:
        summary = run(args, watchdog)
    except Exception as exc:  # any failure is a tool error (2), never the mismatch code (1)
        summary = {"exit_code": EXIT_ERROR, "error": f"{type(exc).__name__}: {exc}"}
        traceback.print_exc()
    finally:
        watchdog.stop()
    summary.update({"key": key, "memory_caps_gb": caps, "provenance": provenance(caps)})
    try:
        write_json_atomic(side_dir / "result.json", summary)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: cannot write {side_dir / 'result.json'} ({exc})", file=sys.stderr)
        summary["exit_code"] = EXIT_ERROR
    code = int(summary["exit_code"])
    if code == EXIT_OK:
        print(f"ok: {side} wrote {side_dir}")
    else:
        print(f"exit {code}: {summary.get('error', summary.get('degenerate'))}", file=sys.stderr)
    return code


def orchestrate(args: argparse.Namespace) -> int:
    """Run the pending df11/bf16 subprocesses in order, then compare in-process; exit 2 on a child's failure."""
    args.out.mkdir(parents=True, exist_ok=True)
    key = run_key(args)
    for side in SIDES:
        side_dir = args.out / side
        if side_complete(side_dir, key):
            print(f"{side}: already complete, skipping")
            continue
        cmd = child_command(args, side)
        print(f"{side}: {' '.join(cmd)}", flush=True)
        code = subprocess.run(cmd, cwd=_REPO, check=False).returncode
        if code != 0:
            print(f"error: {side} exited {code}; stopping", file=sys.stderr)
            return EXIT_ERROR
    return _write_compare(args)


def main(argv: list[str] | None = None) -> int:
    """Entry point: one side, or ``--orchestrate`` (df11, then bf16, then compare)."""
    args = parse_args(argv)
    try:
        if args.orchestrate:
            return orchestrate(args)
        if args.side == "compare":
            return _write_compare(args)
        return run_side(args)
    except Exception as exc:  # a setup error is a tool error (2), never the mismatch code (1)
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
