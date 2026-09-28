"""``mlx-dfloat generate``: one FLUX.1 image from a DFloat11 transformer, with memory caps and a watchdog.

Flag names follow ``mflux-generate`` so a pasted command works; the options that path cannot
honour are parsed only to refuse them with a reason (exit 2).
"""

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_dfloat._memory_caps import install_memory_caps
from mlx_dfloat._watchdog import Watchdog, default_ceiling
from mlx_dfloat.errors import DFloatError

EXIT_OK, EXIT_ERROR = 0, 2
DEFAULT_STEPS = {"schnell": 4, "dev": 25, "krea-dev": 25}  # mflux 0.20's per-model defaults
DEFAULT_GUIDANCE = 3.5  # mflux-generate's default; schnell ignores it
REFUSED: dict[str, str] = {
    "--quantize": "quantisation on top of DFloat11 changes the output the format exists to keep",
    "--lora-paths": "LoRA is not on the DFloat11 path",
    "--lora-scales": "LoRA is not on the DFloat11 path",
    "--lora": "LoRA is not on the DFloat11 path",
    "--lora-style": "LoRA is not on the DFloat11 path",
    "--image-path": "img2img is not on the DFloat11 path",
    "--image-strength": "img2img is not on the DFloat11 path",
    "--image": "img2img is not on the DFloat11 path",
    "--pid-decode": "the PiD decoder loads an 8 GB caption encoder next to the compressed set",
    "--controlnet-image-path": "ControlNet is another model class, not provided on the DFloat11 path",
    "--controlnet-strength": "ControlNet is another model class, not provided on the DFloat11 path",
}
_REFUSED_ALIASES = {"--quantize": ["-q"]}


def cache_limit_bytes(text: str) -> int:
    """A positive byte count, ``2.5e9`` accepted.

    Raises:
        ValueError: Not a number, or not positive.
    """
    value = int(float(text))
    if value <= 0:
        raise ValueError(f"cache limit must be positive, got {text!r}")
    return value


def add_generate_parser(sub: Any) -> argparse.ArgumentParser:
    """Register ``generate`` on a subparsers object."""
    p: argparse.ArgumentParser = sub.add_parser(
        "generate",
        help="generate one FLUX.1 image from a DFloat11 transformer",
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(run=run)
    return p


def build_parser() -> argparse.ArgumentParser:
    """A standalone parser for ``generate`` (tests parse with it)."""
    p = argparse.ArgumentParser(prog="mlx-dfloat generate", description=__doc__)
    _add_arguments(p)
    return p


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", choices=tuple(DEFAULT_STEPS), default="schnell")
    p.add_argument("--prompt", required=True, help="the text to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="denoise steps (default per model: schnell 4, dev 25)",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--guidance",
        type=float,
        default=None,
        help=f"guidance (default {DEFAULT_GUIDANCE}; schnell ignores it)",
    )
    p.add_argument("--scheduler", default="linear")
    p.add_argument(
        "--negative-prompt", default=None, help="accepted and ignored, as mflux does for FLUX.1"
    )
    p.add_argument(
        "--output", default="image.png", help="the image file; an existing file is not overwritten"
    )
    p.add_argument(
        "--metadata", action="store_true", help="also write mflux's JSON metadata sidecar"
    )
    p.add_argument(
        "--df11",
        default=None,
        help="DFloat11 checkpoint: a directory or a Hub id (default per model)",
    )
    p.add_argument(
        "--base", default=None, help="base repository for the encoders and VAE (default per model)"
    )
    p.add_argument("--eval-policy", choices=("per-block", "depth2"), default="per-block")
    p.add_argument(
        "--cache-limit",
        type=cache_limit_bytes,
        default=None,
        help="MLX buffer-cache limit in bytes (derived per call by default)",
    )
    p.add_argument(
        "--no-fit-check",
        action="store_true",
        help="skip the memory fit estimate (a warning instead of a refusal)",
    )
    p.add_argument("--report", type=Path, default=None, help="write the run report as JSON")
    p.add_argument(
        "--wall-budget",
        type=float,
        default=3600.0,
        help="seconds before the watchdog aborts (exit 71)",
    )
    for flag, reason in REFUSED.items():
        p.add_argument(
            flag,
            *_REFUSED_ALIASES.get(flag, []),
            nargs="*",
            default=None,
            help=argparse.SUPPRESS,
            metavar=reason,
        )


def refused_option(args: argparse.Namespace) -> str | None:
    """The first refused flag present on the command line, with its reason; None when there is none."""
    for flag, reason in REFUSED.items():
        if getattr(args, flag.lstrip("-").replace("-", "_")) is not None:
            return f"{flag}: {reason}"
    return None


def _model_class() -> Callable[..., Any]:
    from mlx_dfloat.mflux.flux1.model import (
        DFloatFlux1,  # raises DFloatDependencyError without mflux
    )

    return DFloatFlux1


def run(
    args: argparse.Namespace,
    *,
    model_factory: Callable[..., Any] | None = None,
    install_caps: Callable[[], tuple[int, int]] = install_memory_caps,
    watchdog_factory: Callable[..., Any] = Watchdog,
) -> int:
    """Refuse, cap, watch, build, generate, save, report. Returns the exit code."""
    refused = refused_option(args)
    if refused is not None:
        print(f"error: {refused}", file=sys.stderr)
        return EXIT_ERROR
    if args.negative_prompt:
        print(
            "warning: --negative-prompt is ignored: FLUX.1 has no negative branch", file=sys.stderr
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    caps = list(install_caps())
    model_kwargs = {
        "model": args.model,
        "df11_path": args.df11,
        "base_path": args.base,
        "eval_policy": args.eval_policy,
        "cache_limit": args.cache_limit,
        "fit_check": not args.no_fit_check,
    }
    report: dict[str, Any] = {
        "exit_code": EXIT_ERROR,
        "output": str(output),
        "memory_caps_gb": caps,
        "model_kwargs": model_kwargs,
    }
    try:
        watchdog = watchdog_factory(
            output.parent, ceiling=default_ceiling(), budget=args.wall_budget
        ).start()
    except DFloatError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    try:
        factory = model_factory if model_factory is not None else _model_class()
        model = factory(**model_kwargs)
        image = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=args.steps if args.steps is not None else DEFAULT_STEPS[args.model],
            height=args.height,
            width=args.width,
            guidance=args.guidance if args.guidance is not None else DEFAULT_GUIDANCE,
            scheduler=args.scheduler,
        )
        image.save(str(output), export_json_metadata=args.metadata)
        report.update(exit_code=EXIT_OK, **model.report())
    except DFloatError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # anything else is still a tool error (2), never a silent success
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog.stop()
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, default=str))
    if report["exit_code"] == EXIT_OK:
        print(f"ok: {output}")
    return int(report["exit_code"])
