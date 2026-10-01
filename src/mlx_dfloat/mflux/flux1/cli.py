"""``mlx-dfloat generate``: one FLUX.1 image from a DFloat11 transformer, with memory caps and a watchdog.

Flag names follow ``mflux-generate`` so a pasted command works; the options that path cannot
honour are parsed only to refuse them with a reason (exit 2). ``--tier GB`` runs under a smaller
Mac's MLX limits, with that tier's watchdog ceiling and fit budget (the host's own tier keeps the
host caps); ``--memory-ceiling BYTES`` sets the watchdog ceiling alone, under the host caps.
"""

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_dfloat._memory_caps import install_memory_caps
from mlx_dfloat._scrub import scrub_home
from mlx_dfloat._watchdog import Watchdog, default_ceiling, phys_footprint
from mlx_dfloat.bench import capped
from mlx_dfloat.bench.capped import TierLimits, host_tier_gb, limits_record, tier_limits
from mlx_dfloat.errors import DFloatError

EXIT_OK, EXIT_ERROR = 0, 2
DEFAULT_STEPS = {"schnell": 4, "dev": 25, "krea-dev": 25}  # mflux 0.20's per-model defaults
DEFAULT_GUIDANCE = 3.5  # mflux-generate's default; schnell ignores it
FOOTPRINT_PEAK_LABEL = "OS phys_footprint, sampled every 0.05 s by the watchdog"
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


def _positive_bytes(text: str, what: str) -> int:
    try:
        value = int(float(text))
    except OverflowError as exc:  # "inf"
        raise ValueError(f"{what} must be a finite byte count, got {text!r}") from exc
    if value <= 0:
        raise ValueError(f"{what} must be positive, got {text!r}")
    return value


def cache_limit_bytes(text: str) -> int:
    """A positive byte count, ``2.5e9`` accepted.

    Raises:
        ValueError: Not a finite number, or not positive.
    """
    return _positive_bytes(text, "cache limit")


def memory_ceiling_bytes(text: str) -> int:
    """A positive byte count for the watchdog ceiling, ``1.8e10`` accepted.

    Raises:
        ValueError: Not a finite number, or not positive.
    """
    return _positive_bytes(text, "memory ceiling")


def tier_gb(text: str) -> int:
    """A tier in whole GB, at least 1.

    Raises:
        ValueError: Not an integer, or below 1.
    """
    value = int(text)
    if value < 1:
        raise ValueError(f"tier must be at least 1 GB, got {text!r}")
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
        "--output",
        default="image.png",
        help="the image file; an existing file is kept and the new one gets a numbered name",
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
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write the run report as JSON (an existing file is replaced)",
    )
    p.add_argument(
        "--wall-budget",
        type=float,
        default=3600.0,
        help="seconds before the watchdog aborts (exit 71)",
    )
    p.add_argument(
        "--tier",
        type=tier_gb,
        default=None,
        metavar="GB",
        help="run under a smaller Mac's MLX limits, watchdog ceiling and fit budget "
        "(default: this host's own tier and caps)",
    )
    p.add_argument(
        "--memory-ceiling",
        type=memory_ceiling_bytes,
        default=None,
        metavar="BYTES",
        help="a lower watchdog ceiling alone, under the host caps (not with --tier)",
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


def ceiling_for(
    args: argparse.Namespace,
    *,
    host_ram_bytes: int,
    host_recommended_bytes: int,
    default_ceiling_bytes: int,
) -> tuple[int, TierLimits, str]:
    """The watchdog ceiling, the limits to install and the report label for these flags (pure).

    ``--memory-ceiling`` gives that ceiling under the host's limits (``PROOF``); ``--tier`` gives
    the tier's ceiling and limits, except the host's own tier, which keeps ``default_ceiling_bytes``
    (``MEASURED``); neither gives the host's limits and ``default_ceiling_bytes`` (``MEASURED``).

    Raises:
        ValueError: Both ``--tier`` and ``--memory-ceiling`` were given, or ``--memory-ceiling`` is
            above ``default_ceiling_bytes`` (it may only lower the ceiling).
        DFloatUnsupportedError: The tier is larger than the host.
    """
    tier, ceiling = args.tier, args.memory_ceiling
    if tier is not None and ceiling is not None:
        raise ValueError(
            "--tier and --memory-ceiling cannot be combined: --tier sets a tier's limits and "
            "ceiling, --memory-ceiling the watchdog ceiling alone under the host caps"
        )
    limits = tier_limits(
        host_tier_gb(host_ram_bytes) if tier is None else tier,
        host_ram_bytes=host_ram_bytes,
        host_recommended_bytes=host_recommended_bytes,
    )
    if ceiling is not None:
        if ceiling > default_ceiling_bytes:
            raise ValueError(
                f"--memory-ceiling {ceiling} is above this host's ceiling "
                f"{default_ceiling_bytes / 1024**3:.1f} GiB: it may only lower the watchdog ceiling"
            )
        return ceiling, limits, "PROOF"
    return (default_ceiling_bytes if limits.is_host else limits.ceiling_bytes), limits, limits.label


def _steps(args: argparse.Namespace) -> int:
    """The denoise steps this call runs: ``--steps``, else the model's mflux default."""
    return int(args.steps) if args.steps is not None else DEFAULT_STEPS[args.model]


def _run_context(args: argparse.Namespace) -> dict[str, Any]:
    """What the watchdog's abort artifact records about the run it may stop."""
    return {
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "steps": _steps(args),
    }


def _host_facts() -> tuple[int, int]:
    """This Mac's RAM and recommended working set, from MLX's device info (0 when not reported)."""
    info = mx.device_info()
    return int(info.get("memory_size", 0)), int(info.get("max_recommended_working_set_size", 0))


def _model_class() -> Callable[..., Any]:
    from mlx_dfloat.mflux.flux1.model import (
        DFloatFlux1,  # raises DFloatDependencyError without mflux
    )

    return DFloatFlux1


def _output_resolver() -> Callable[[Path], Path]:
    """The saved name by mflux's own rule (an existing file gets a numbered name next to it)."""
    from mflux.utils.image_util import ImageUtil

    resolve: Callable[[Path], Path] = ImageUtil.resolve_output_path
    return resolve


def _finish(args: argparse.Namespace, report: dict[str, Any]) -> int:
    """Write the report (when ``--report`` was given) and announce success. The one path every exit uses."""
    if args.report is not None:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(scrub_home(report), indent=1, default=str))
        except OSError as exc:
            print(f"error: the report was not written: {exc}", file=sys.stderr)
            return EXIT_ERROR
    if report["exit_code"] == EXIT_OK:
        peak = report.get("footprint_peak_bytes")
        suffix = f" (footprint peak {peak / 1024**3:.2f} GiB)" if peak is not None else ""
        print(f"ok: {report['output']}{suffix}")
    return int(report["exit_code"])


def run(
    args: argparse.Namespace,
    *,
    model_factory: Callable[..., Any] | None = None,
    install_caps: Callable[[], tuple[int, int]] = install_memory_caps,
    watchdog_factory: Callable[..., Any] = Watchdog,
    resolve_output: Callable[[Path], Path] | None = None,
    host_facts: Callable[[], tuple[int, int]] = _host_facts,
    ceiling_default: Callable[[], int] = default_ceiling,
    apply_limits: Callable[[TierLimits], object] = capped.apply,
    read_limits: Callable[[], dict[str, int]] = capped.current_limits,
) -> int:
    """Refuse, cap, watch, build, generate, save, report. Returns the exit code.

    ``resolve_output`` maps the requested file to the one written (default: mflux's rule, looked
    up lazily next to the model class). ``host_facts`` returns this host's RAM and recommended
    working set; ``ceiling_default`` the host's watchdog ceiling; ``apply_limits`` installs a
    smaller tier's limits (in place of ``install_caps``); ``read_limits`` reads the limits in force.
    """
    refused = refused_option(args)
    if refused is not None:
        print(f"error: {refused}", file=sys.stderr)
        return EXIT_ERROR
    if args.negative_prompt:
        print(
            "warning: --negative-prompt is ignored: FLUX.1 has no negative branch", file=sys.stderr
        )
    output = Path(args.output)
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
        "memory_caps_gb": None,
        "model_kwargs": model_kwargs,
        "height": args.height,
        "width": args.width,
        "memory_ceiling_bytes": args.memory_ceiling,
        "tier_gb": None,
        "label": None,
        "watchdog_ceiling_bytes": None,
        "limits": None,
    }
    try:
        host_ram, host_recommended = host_facts()
        try:
            ceiling, limits, label = ceiling_for(
                args,
                host_ram_bytes=host_ram,
                host_recommended_bytes=host_recommended,
                default_ceiling_bytes=ceiling_default(),
            )
        except ValueError as exc:
            if isinstance(exc, DFloatError):  # DFloatUnsupportedError: the tier above the host
                raise
            print(f"error: {exc}", file=sys.stderr)
            report["error"] = str(exc)
            return _finish(args, report)
        report.update(tier_gb=limits.tier_gb, label=label, watchdog_ceiling_bytes=ceiling)
        output.parent.mkdir(parents=True, exist_ok=True)
        if limits.is_host:
            report["memory_caps_gb"] = list(install_caps())
            applied = "host-caps"
        else:
            apply_limits(limits)
            applied = "tier-defaults"
            model_kwargs["budget_bytes"] = limits.ceiling_bytes
        effective = read_limits()
        report["limits"] = limits_record(
            limits,
            effective_memory_limit=effective["memory"],
            effective_cache_limit=effective["cache"],
            effective_wired_limit=effective["wired"],
            applied=applied,
        )
        watchdog = watchdog_factory(
            output.parent, ceiling=ceiling, budget=args.wall_budget, context=_run_context(args)
        ).start()
    except (DFloatError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
        return _finish(args, report)
    except Exception as exc:  # symmetric with the block below: never a silent exit 1
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
        return _finish(args, report)
    try:
        factory = model_factory if model_factory is not None else _model_class()
        resolve = resolve_output if resolve_output is not None else _output_resolver()
        model = factory(**model_kwargs)
        image = model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=_steps(args),
            height=args.height,
            width=args.width,
            guidance=args.guidance if args.guidance is not None else DEFAULT_GUIDANCE,
            scheduler=args.scheduler,
        )
        final = Path(resolve(output))
        image.save(str(final), export_json_metadata=args.metadata, overwrite=True)
        if not final.is_file() or final.stat().st_size == 0:
            # mflux's save logs a write failure and returns normally
            raise DFloatError(f"{final}: the image was not written")
        report["output"] = str(final)
        report.update(exit_code=EXIT_OK, **model.report())
    except DFloatError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        report["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # anything else is still a tool error (2), never a silent success
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        watchdog.stop()
    final_sample = phys_footprint()
    report["footprint_peak_bytes"] = max(watchdog.peak_footprint, final_sample)
    report["footprint_peak_label"] = FOOTPRINT_PEAK_LABEL
    report["watched_peak_bytes"] = max(watchdog.peak_watched, final_sample)
    report["mlx_peak_bytes"] = watchdog.peak_mlx
    return _finish(args, report)
