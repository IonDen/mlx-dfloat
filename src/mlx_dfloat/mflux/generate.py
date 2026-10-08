"""``mlx-dfloat generate``: one image from a DFloat11 transformer, with memory caps and a watchdog.

The command's ``--help`` text is ``DESCRIPTION`` below, kept free of reST markup because argparse
prints it as is.
"""

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_dfloat._memory_caps import install_memory_caps
from mlx_dfloat._scrub import scrub_home
from mlx_dfloat._watchdog import Watchdog, default_ceiling, phys_footprint
from mlx_dfloat.bench import capped
from mlx_dfloat.bench.capped import TierLimits, host_tier_gb, limits_record, tier_limits
from mlx_dfloat.errors import DFloatError
from mlx_dfloat.mflux.families import FAMILIES, MODELS, entry, family_of

EXIT_OK, EXIT_ERROR = 0, 2
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
    "--pid-decode": "mflux's alternative image decoder (PiD) loads an 8 GB caption encoder next to the compressed set",
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


DESCRIPTION = (
    "mlx-dfloat generate: one image from a DFloat11 transformer, with memory caps and a watchdog. "
    "Flag names follow mflux-generate so a pasted command works; the options that path cannot "
    "honour are parsed only to refuse them with a reason (exit 2). --tier GB runs under a smaller "
    "Mac's MLX limits, with that tier's watchdog ceiling and fit budget (the host's own tier keeps "
    "the host's own limits); --memory-ceiling BYTES sets the watchdog ceiling alone, under the "
    "host's own limits."
)


def add_generate_parser(sub: Any) -> argparse.ArgumentParser:
    """Register ``generate`` on a subparsers object."""
    p: argparse.ArgumentParser = sub.add_parser(
        "generate",
        help="generate one image from a DFloat11 transformer",
        description=DESCRIPTION,
    )
    _add_arguments(p)
    p.set_defaults(run=run)
    return p


def build_parser() -> argparse.ArgumentParser:
    """A standalone parser for ``generate`` (tests parse with it)."""
    p = argparse.ArgumentParser(prog="mlx-dfloat generate", description=DESCRIPTION)
    _add_arguments(p)
    return p


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", choices=tuple(MODELS), default="schnell")
    p.add_argument("--prompt", required=True, help="the text to generate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="denoise steps (default per model, as mflux: schnell 4, dev 25, z-image 50, z-image-turbo 9, "
        "FLUX.2 Klein base 50, distilled 4, qwen-image-2.1 40, ernie-image 50, ernie-image-turbo 8)",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="guidance (default per model, as mflux: FLUX.1 dev 3.5, ignored by schnell and z-image-turbo; "
        "z-image 0, its model card suggests about 4; FLUX.2 Klein 1.0, where the base models take another "
        "value such as 4 and the distilled ones refuse it; Qwen-Image 2.1 1.0, where classifier-free guidance "
        "needs a value above 1 and --negative-prompt; ERNIE-Image 4.0 (classifier-free guidance above 1, with or "
        "without --negative-prompt); ERNIE-Image-Turbo 1.0 only)",
    )
    p.add_argument(
        "--scheduler",
        default=None,
        help="scheduler (default per model: FLUX.1, Qwen-Image 2.1 and ERNIE-Image linear; FLUX.2 Klein always runs "
        "flow_match_euler_discrete and refuses the flag)",
    )
    p.add_argument(
        "--negative-prompt",
        default=None,
        help="used by z-image, qwen-image-2.1 and ernie-image with --guidance above 1; accepted and ignored for the "
        "other models, as mflux does",
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
    p.add_argument(
        "--eval-policy",
        choices=("per-block", "depth2"),
        default="per-block",
        help="per-block: evaluate after each block (default); depth2: one block behind",
    )
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
        "(default: this host's own tier and limits)",
    )
    p.add_argument(
        "--memory-ceiling",
        type=memory_ceiling_bytes,
        default=None,
        metavar="BYTES",
        help="a lower watchdog ceiling alone, under the host's own limits (not with --tier)",
    )
    family_flags = {f: r for fam in FAMILIES.values() for f, r in fam.refused_flags.items()}
    for flag, reason in {**REFUSED, **family_flags}.items():
        if flag in p._option_string_actions:
            continue  # a real option for the other families (FLUX.2 Klein refuses --scheduler); refused_option reads it
        p.add_argument(
            flag,
            *_REFUSED_ALIASES.get(flag, []),
            nargs="*",
            default=None,
            help=argparse.SUPPRESS,
            metavar=reason,
        )


def refused_option(args: argparse.Namespace) -> str | None:
    """The first refused flag present on the command line, with its reason; None when there is none.

    The common refusals apply to every model; a family's own list only to its models.
    """
    for flag, reason in {**REFUSED, **family_of(args.model).refused_flags}.items():
        if getattr(args, flag.lstrip("-").replace("-", "_")) is not None:
            return f"{flag}: {reason}"
    return None


def fixed_guidance_refusal(args: argparse.Namespace) -> str | None:
    """Why ``--guidance`` is refused for this model, or None.

    A distilled model runs at its one guidance only (FLUX.2 Klein's distilled models at 1.0, as mflux's own command
    enforces); an absent ``--guidance`` is never refused.
    """
    e = entry(args.model)
    if e.fixed_guidance is None or args.guidance is None or args.guidance == e.fixed_guidance:
        return None
    return (
        f"--guidance: {e.label} is a distilled model and runs at guidance {e.fixed_guidance} only "
        "(use a base model for guidance)"
    )


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
            "ceiling, --memory-ceiling the watchdog ceiling alone under the host's own limits"
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


def empty_prompt_refusal(args: argparse.Namespace) -> str | None:
    """Why an empty or blank ``--prompt`` is refused for this model, or None.

    ERNIE-Image refuses an empty or blank prompt as a user error (its tokenizer gives an empty prompt no tokens at all,
    not even a start token); the other families' mflux pipelines encode an empty prompt as it is.
    """
    if entry(args.model).family != "ernie" or args.prompt.strip():
        return None
    return (
        f"--prompt {args.prompt!r}: an empty or blank prompt is refused as a user error (an empty prompt gives "
        "ERNIE-Image's tokenizer no tokens at all)"
    )


def _steps(args: argparse.Namespace) -> int:
    """The denoise steps this call runs: ``--steps``, else the model's mflux default."""
    return int(args.steps) if args.steps is not None else entry(args.model).default_steps


def _guidance(args: argparse.Namespace) -> float | None:
    """The guidance this call passes: ``--guidance``, else the model's default (None: the model's own rule)."""
    return float(args.guidance) if args.guidance is not None else entry(args.model).default_guidance


def _scheduler(args: argparse.Namespace) -> str | None:
    """The scheduler this call passes: ``--scheduler``, else the model's default (None: its own rule)."""
    return args.scheduler if args.scheduler is not None else entry(args.model).default_scheduler


def _run_context(
    args: argparse.Namespace, *, tier_gb: int, label: str, limits: Mapping[str, int]
) -> dict[str, Any]:
    """What the watchdog's abort artifact records about the run it may stop.

    ``limits`` is the MLX ``memory`` / ``cache`` / ``wired`` limits read back after the install,
    so a stopped run still says which limits it ran under.
    """
    return {
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "steps": _steps(args),
        "tier_gb": tier_gb,
        "label": label,
        "limits": {name: int(limits[name]) for name in ("memory", "cache", "wired")},
    }


def _phase_of(built: Mapping[str, Any]) -> str | None:
    """The phase a run is in: ``"build"`` before the model exists, then the model's ``open_phase``."""
    if "model" not in built:
        return "build"
    phase = getattr(built["model"], "open_phase", None)
    return None if phase is None else str(phase)


def _host_facts() -> tuple[int, int]:
    """This Mac's RAM and recommended working set, from MLX's device info (0 when not reported)."""
    info = mx.device_info()
    return int(info.get("memory_size", 0)), int(info.get("max_recommended_working_set_size", 0))


def _model_class(name: str) -> Callable[..., Any]:
    """The model class of a registered name (raises DFloatDependencyError without mflux)."""
    return family_of(name).load_model_class()


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
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Refuse, cap, watch, build, generate, save, report. Returns the exit code.

    ``resolve_output`` maps the requested file to the one written (default: mflux's rule, looked
    up lazily next to the model class). ``host_facts`` returns this host's RAM and recommended
    working set; ``ceiling_default`` the host's watchdog ceiling; ``apply_limits`` installs a
    smaller tier's limits (in place of ``install_caps``); ``read_limits`` reads the limits in force;
    ``clock`` gives the report's ``elapsed_seconds`` (read at the start and after the run).
    The watchdog's abort artifact names the phase open when it fired: ``"build"`` until the model
    exists, then the model's own phase (``encode``, ``set_load``, ``denoise``, ``vae``, or None
    between them).
    """
    started = clock()
    refused = refused_option(args) or fixed_guidance_refusal(args) or empty_prompt_refusal(args)
    if refused is not None:
        print(f"error: {refused}", file=sys.stderr)
        return EXIT_ERROR
    if args.negative_prompt and not entry(args.model).uses_negative_prompt:
        # A base FLUX.2 Klein above guidance 1 does run a negative branch (mflux's blank " "); it takes no custom one.
        why = (
            "this model takes no custom negative prompt"
            if entry(args.model).family == "flux2"
            else "this model has no negative branch"
        )
        print(f"warning: --negative-prompt is ignored: {why}", file=sys.stderr)
    elif args.negative_prompt and (_guidance(args) or 0.0) <= 1.0:
        default = entry(args.model).default_guidance or 0  # None (Z-Image): mflux's own rule, 0
        print(
            "warning: --negative-prompt has no effect: classifier-free guidance runs only above "
            f"guidance 1.0 (the default is {default}). Pass --guidance above 1.0 to enable it.",
            file=sys.stderr,
        )
    if (
        entry(args.model).cfg_needs_negative
        and (_guidance(args) or 0.0) > 1.0
        and not args.negative_prompt
    ):
        print(
            "warning: --guidance above 1.0 runs no classifier-free guidance for "
            f"{entry(args.model).label} without --negative-prompt (as mflux); pass --negative-prompt "
            "to enable it",
            file=sys.stderr,
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
    built: dict[str, Any] = {}  # the model, once it exists (the watchdog reads its phase)
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
            applied = "tier-caps"
            model_kwargs["budget_bytes"] = limits.fit_budget_bytes  # a real Mac's, not the ceiling
        effective = read_limits()
        report["limits"] = limits_record(
            limits,
            effective_memory_limit=effective["memory"],
            effective_cache_limit=effective["cache"],
            effective_wired_limit=effective["wired"],
            applied=applied,
        )
        watchdog = watchdog_factory(
            output.parent,
            ceiling=ceiling,
            budget=args.wall_budget,
            context=_run_context(args, tier_gb=limits.tier_gb, label=label, limits=effective),
            live_context={"phase": lambda: _phase_of(built)},
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
        factory = model_factory if model_factory is not None else _model_class(args.model)
        resolve = resolve_output if resolve_output is not None else _output_resolver()
        model = built["model"] = factory(**model_kwargs)
        call: dict[str, Any] = {
            "seed": args.seed,
            "prompt": args.prompt,
            "num_inference_steps": _steps(args),
            "height": args.height,
            "width": args.width,
            "guidance": _guidance(args),
            "scheduler": _scheduler(args),
        }
        if entry(args.model).uses_negative_prompt:
            call["negative_prompt"] = args.negative_prompt
        image = model.generate_image(**call)
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
    report["elapsed_seconds"] = round(clock() - started, 3)
    return _finish(args, report)
