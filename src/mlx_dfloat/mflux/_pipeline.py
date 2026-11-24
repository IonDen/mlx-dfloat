"""Family-free pieces of an mflux model class: refusals, the phase tracker, the VAE pool guard and the call plan.

Each model class (one per mflux family) runs the same prelude around mflux's ``generate_image``: refuse what the
DFloat11 path cannot honour, plan the call's memory, track the phases, and empty the buffer pool before the VAE
decode. The family supplies its measured numbers (``PhaseConstants``) and sizes; nothing here imports mflux.
"""

import contextlib
import datetime
import functools
import gc
import inspect
import logging
import math
import os
import platform
import stat
import sys
import tempfile
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

import mlx.core as mx

from mlx_dfloat import _guard
from mlx_dfloat._memory_caps import (
    caps_for_recommended_bytes,
    known_wired_limit,
    remember_wired_limit,
)
from mlx_dfloat._watchdog import (
    Counters,
    Watchdog,
    default_ceiling,
    phys_footprint,
    read_counters,
)
from mlx_dfloat.errors import DFloatResourceError, DFloatUnsupportedError, DFloatWatchdogError
from mlx_dfloat.integrate.memory import CallPlan
from mlx_dfloat.mflux import _drain
from mlx_dfloat.mflux._phases import (
    FamilySizes,
    PhaseConstants,
    activation_allowance,
    cache_limit_for,
    fit_for,
)
from mlx_dfloat.mflux._watchconfig import check_watchdog_arg


def refuse(name: str, reason: str) -> NoReturn:
    """Raise the refusal for an argument or a feature this path does not run.

    Raises:
        DFloatUnsupportedError: Always.
    """
    raise DFloatUnsupportedError(f"{name}: {reason}; not on the DFloat11 path in this version")


def check_eval_policy(policy: str, policies: tuple[str, ...]) -> None:
    """Refuse an eval policy outside ``policies``.

    Raises:
        DFloatUnsupportedError: ``policy`` is not one of ``policies``.
    """
    if policy not in policies:
        raise DFloatUnsupportedError(f"eval_policy {policy!r}: choose from {policies}")


def refuse_construction_args(
    *,
    quantize: int | None,
    lora_paths: list[str] | None,
    lora_scales: list[float] | None,
    bake_lora: bool,
) -> None:
    """Refuse the mflux constructor arguments that would change the checkpoint's output (accepted only to refuse).

    Raises:
        DFloatUnsupportedError: ``quantize`` set, any LoRA path or scale, or ``bake_lora=False``.
    """
    if quantize is not None:
        refuse(
            "quantize",
            "quantisation on top of DFloat11 changes the output the format exists to keep",
        )
    if lora_paths:
        refuse("lora_paths", "LoRA of any kind")
    if lora_scales:
        refuse("lora_scales", "LoRA of any kind")
    if not bake_lora:
        refuse("bake_lora", "LoRA of any kind")


def refuse_precision_args(*, float32: bool, compute_precision: Any) -> None:
    """Refuse mflux's precision options (Z-Image's ``float32`` stream, ``compute_precision``), accepted only to refuse.

    ``compute_precision`` casts the attention and feed-forward weights in place and ``float32`` changes the
    transformer's stream dtype; neither run was checked or measured on this path.

    Raises:
        DFloatUnsupportedError: ``compute_precision`` set or ``float32=True``.
    """
    if compute_precision is not None:
        refuse(
            "compute_precision",
            "a lower compute precision changes the output the format exists to keep",
        )
    if float32:
        refuse("float32", "the float32 transformer stream")


# What an armed call's watchdog reads, looked up at call time so tests can put fakes in front of them: the counter
# reader, the host's default ceiling in bytes, and the process exit (None: the watchdog module's own ``_exit``).
_counters: Callable[[], Counters] = read_counters
_default_ceiling: Callable[[], int] = default_ceiling
_exit_process: Callable[[int], object] | None = None


def _artifact_root() -> Path:
    """The folder an armed call writes its abort record to when its config names none."""
    return Path(tempfile.gettempdir()) / "mlx-dfloat"


def _artifact_name() -> str:
    """A record name unique per abort: ``abort-<UTC to the microsecond>Z-<pid>.json``."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"abort-{stamp}Z-{os.getpid()}.json"


def clear_call_frames(exc: BaseException, *, outer: BaseException | None) -> None:
    """Release the locals of the frames ``exc`` and every exception in its chain (causes and contexts) passed through.

    ``outer`` is the exception the caller was already handling when the call began (``sys.exc_info()`` at entry): the
    walk stops there, so the caller's own frames keep their locals. A seen-set ends a chain that loops back.
    """
    seen: set[int] = set() if outer is None else {id(outer)}
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


def _versions() -> dict[str, str | None]:
    """The mlx and mflux versions (mflux None when it is not installed)."""
    try:
        mflux: str | None = metadata.version("mflux")
    except metadata.PackageNotFoundError:
        mflux = None
    return {"mlx": mx.__version__, "mflux": mflux}  # type: ignore[attr-defined]


_CALL_FIELDS = ("seed", "height", "width", "num_inference_steps", "guidance")


def _call_context(
    model: Any, method: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """What the abort record says about the call: model, class, method, the size and seed arguments, versions."""
    context: dict[str, Any] = {
        "model": getattr(model, "_model", None),
        "class": type(model).__name__,
        "method": method.__name__,
    }
    try:
        bound = inspect.signature(method).bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:  # the method itself will refuse these arguments
        arguments: dict[str, Any] = {}
    else:
        arguments = bound.arguments
    for name in _CALL_FIELDS:
        if name in arguments:
            value = arguments[name]
            context[name] = (
                value if value is None or isinstance(value, int | float) else repr(value)
            )
    context["versions"] = _versions()
    context["macos"] = platform.mac_ver()[0]
    return context


def _private_root_refusal(root: Path) -> str | None:
    """Make the default record folder private (0o700); why it cannot be trusted, or None when it can.

    The default folder sits under the shared temp dir, where another user could have placed it first: a symlink, or a
    folder that user owns, is refused.
    """
    with contextlib.suppress(FileExistsError):
        root.mkdir(mode=0o700)
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode):
        return f"the record folder {root} is a symlink; nothing was written"
    if not stat.S_ISDIR(info.st_mode):
        return f"the record folder {root} is not a folder; nothing was written"
    if info.st_uid != os.getuid():
        return f"the record folder {root} is owned by uid {info.st_uid}, not this user; nothing was written"
    # An own folder made earlier with a looser mode is made private too.
    root.chmod(0o700)
    return None


class _DefaultFolderWatchdog(Watchdog):
    """The watchdog of a call whose config names no folder: it writes only into a private folder this user owns."""

    def _write(self, record: dict[str, Any]) -> Path | None:
        try:
            refusal = _private_root_refusal(self.out_dir)
        # This runs on the sampler thread: a failed check costs the record, never the trip.
        except Exception as exc:
            refusal = f"{type(exc).__name__}: {exc}"
        if refusal is not None:
            record["artifact_error"] = refusal
            return None
        return super()._write(record)


def _trip_at_arm(watchdog: Watchdog) -> None:
    """One reading before the sampler starts: at or above the stop line, drain, read again, and trip if still there.

    Memory the process already holds counts, so a call that starts over the line stops at ``call_start`` before any
    work. Pages an earlier call freed leave the footprint only after the next GPU submission, so a first reading at or
    above the line is followed by the drain (``_drain.drain``) and a second reading, and only that second one decides.
    A call below the line makes one reading and no drain. A reading that fails leaves the call to the sampler. Uses
    the watchdog's own sample and trip, so the record is the one a sampled trip writes.

    Raises:
        DFloatWatchdogError: The reading after the drain is at or above the stop line.
    """

    def over_the_line() -> dict[str, float | str | None] | None:
        _reason, sample = watchdog._sample()
        memory = sample["verdict_memory"]
        if not isinstance(memory, int | float) or memory < watchdog.ceiling - watchdog.stop_margin:
            return None
        return sample

    if over_the_line() is None:
        return
    _drain.drain(lambda: _counters().footprint)
    sample = over_the_line()
    if sample is None:
        return
    watchdog._trip("memory", sample)
    with _guard.scope(watchdog):
        _guard.checkpoint("call_start")


def _after_abort(model: Any, exc: BaseException, *, outer: BaseException | None) -> None:
    """Free what the stopped call held: its frames' locals, MLX's buffer pool, then the compressed set if resident.

    The pool is emptied in every case, also when the stop came during the set load (the groups loaded so far are
    freed into it), after waiting for the GPU so no free is still pending. It ends with the drain (``_drain.drain``):
    one tiny evaluation, then a poll of the footprint while it falls, for 0.25 s at most, and a last clear. Each step
    runs on its own: a step that fails is attached to ``exc`` as a note and the next step still runs, so ``exc`` is
    never replaced.
    """

    def drop_set() -> None:
        lifecycle = getattr(model, "_lifecycle", None)
        if lifecycle is not None and lifecycle.set_resident:
            try:
                lifecycle.drop_set()
            except DFloatResourceError as reclaim:
                exc.add_note(
                    "dropping the compressed set after the abort failed: "
                    f"{type(reclaim).__name__}: {reclaim}"
                )

    steps: list[tuple[str, Callable[[], object]]] = [
        ("clearing the call's frames", lambda: clear_call_frames(exc, outer=outer)),
        ("waiting for the GPU", mx.synchronize),
        ("collecting garbage", gc.collect),
        ("emptying the buffer pool", mx.clear_cache),
        ("dropping the compressed set", drop_set),
        ("draining the footprint", lambda: _drain.drain(lambda: _counters().footprint)),
    ]
    for what, step in steps:
        try:
            step()
        except Exception as failure:
            exc.add_note(f"{what} after the abort failed: {type(failure).__name__}: {failure}")


_wired_refusal_warned = False


@contextmanager
def call_caps(limits: Any = mx) -> Iterator[tuple[int, int]]:
    """Run the block under the ``mlx-dfloat`` command's memory caps when no wired cap is in force; restore after.

    The Python API starts at MLX's default wired limit 0 (no cap), while every VAE term was measured under the caps
    the command installs (``install_memory_caps``: with wired 0 a FLUX.2 Klein VAE decode's MLX peak rose from 6.89
    to 9.49 GiB). A wired limit already set (the command's own caps, a CAPPED tier's, an outer call's) is left
    untouched. mlx 0.32.2 has no getters: the wired limit is read by setting it to 0 and back, once per process; after
    that the limit this package installed or read is remembered (``_memory_caps.known_wired_limit``), so a call with
    the compressed set resident does not touch it. Where MLX refuses the wired cap (the system's wired limit was
    lowered), the memory cap is still installed and one warning is issued, as ``install_memory_caps`` goes on without
    it. ``limits`` is the object holding the setters and the device report (``mlx.core``). Yields the (wired, memory)
    caps in bytes installed for the block, 0 for one that was not.
    """
    global _wired_refusal_warned
    wired_now = known_wired_limit(limits)
    if wired_now is None:
        wired_now = int(limits.set_wired_limit(0))
        limits.set_wired_limit(wired_now)
        remember_wired_limit(limits, wired_now)
    if wired_now != 0:
        yield (0, 0)
        return
    try:
        recommended = int(limits.device_info().get("max_recommended_working_set_size", 0))
    except Exception:  # a device or backend that reports nothing: no caps, as install_memory_caps
        recommended = 0
    wired, memory = caps_for_recommended_bytes(recommended)
    if wired == 0:
        yield (0, 0)
        return
    previous_memory = int(limits.set_memory_limit(memory))
    try:
        try:
            limits.set_wired_limit(wired)
        except Exception as exc:  # MLX refuses a wired limit above what the system allows
            wired = 0
            if not _wired_refusal_warned:
                _wired_refusal_warned = True
                warnings.warn(
                    f"MLX refused the wired memory cap ({exc}); the call runs with the memory cap only",
                    stacklevel=4,
                )
        remember_wired_limit(limits, wired)
        try:
            yield (wired, memory)
        finally:
            if wired:
                limits.set_wired_limit(wired_now)
            remember_wired_limit(limits, wired_now)
    finally:
        limits.set_memory_limit(previous_memory)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def with_call_caps(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Wrap a model's ``generate_image`` or ``encode`` in ``call_caps`` and the model's memory watchdog.

    The outermost decorated call arms a raise-mode watchdog from ``model.watchdog`` (a ``WatchdogConfig``; None runs
    unwatched) and makes it the active guard; a nested decorated call (an ``encode`` inside ``generate_image``, or
    another model's call) arms none and only lowers the active ceiling for its own duration when its config's is
    lower. Before the sampler starts, one reading of memory (what the process already holds counts): at or above the
    stop line the call raises at ``call_start`` before the method runs, after the abort cleanup below (which drops a
    set an earlier call left resident). After the method returns, one more checkpoint
    (``call_end``) catches a trip in a phase with none; the watchdog then stops, and a trip that landed between that
    checkpoint and the stop raises too, so no abort record is ever left behind a normal return. On
    ``DFloatWatchdogError`` the call's frames are cleared, MLX's buffer pool is emptied and a resident compressed set
    is dropped before the error leaves; another error that leaves a tripped call gets the same cleanup and a note
    naming the trip and its record. The watchdog stops in every case.

    Raises:
        DFloatWatchdogError: Memory crossed the stop line, at the arm-time reading or by a checkpoint.
        DFloatUnsupportedError: The config's ceiling is above the host's default ceiling (RAM minus 4 GiB), or its
            stop margin is not below the ceiling in force (the host's when the config names none); refused before
            anything runs.
    """

    @functools.wraps(method)
    def capped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        model = args[0]
        # Checked again here: ``model.watchdog`` is reassignable after the constructor's check.
        config = check_watchdog_arg(getattr(model, "watchdog", None))
        if config is not None:
            host = _default_ceiling()
            if config.ceiling is not None and config.ceiling > host:
                raise DFloatUnsupportedError(
                    f"watchdog ceiling {config.ceiling} bytes is above this Mac's ceiling of {host} bytes "
                    "(its memory minus 4 GiB)"
                )
            ceiling = host if config.ceiling is None else config.ceiling
            if config.stop_margin >= ceiling:
                raise DFloatUnsupportedError(
                    f"watchdog stop margin {config.stop_margin} bytes is not below the ceiling of {ceiling} "
                    "bytes"
                )
        if _guard.active() is not None:
            with call_caps(), _guard.tightened(None if config is None else config.ceiling):
                return method(*args, **kwargs)
        if config is None:
            with call_caps():
                return method(*args, **kwargs)
        outer = sys.exc_info()[1]
        # A folder the caller names is used as given; the default one, under the shared temp dir, is checked.
        watchdog_class = Watchdog if config.artifact_dir else _DefaultFolderWatchdog
        watchdog = watchdog_class(
            config.artifact_dir or _artifact_root(),
            ceiling=_default_ceiling() if config.ceiling is None else config.ceiling,
            budget=config.wall_budget,
            mode="raise",
            stop_margin=config.stop_margin,
            grace=config.grace,
            counters=_counters,
            exit_process=_exit_process,
            artifact_name=_artifact_name(),
            context=_call_context(model, method, args, kwargs),
            live_context={"phase": lambda: getattr(model, "open_phase", None)},
        )
        try:
            # Before the sampler, the caps and the method: nothing has run yet.
            _trip_at_arm(watchdog)
        except DFloatWatchdogError as exc:
            # The same cleanup as any abort: a set left resident by an earlier call would keep every later call over
            # the line.
            _after_abort(model, exc, outer=outer)
            raise
        with call_caps():
            try:
                watchdog.start()  # inside the try: an interrupt right after the start still stops it
                with _guard.scope(watchdog):
                    try:
                        result = method(*args, **kwargs)
                        _guard.checkpoint("call_end")
                        watchdog.stop()
                        # A trip after the checkpoint, before the stop: raise it, never a record behind a return.
                        if watchdog.tripped.is_set():
                            del result
                            _guard.checkpoint("call_end")
                    except DFloatWatchdogError as exc:
                        _after_abort(model, exc, outer=outer)
                        raise
                    except Exception as exc:
                        # Stopped first, as on the return path: no trip lands after the check below.
                        watchdog.stop()
                        if watchdog.tripped.is_set():
                            trip = watchdog.trip or {}
                            phase = (trip.get("context") or {}).get("phase")
                            where = (
                                "record not written"
                                if watchdog.artifact is None
                                else f"record: {watchdog.artifact}"
                            )
                            exc.add_note(
                                "the memory watchdog had tripped before this error "
                                f"({trip.get('reason')}, phase {phase}); {where}"
                            )
                            _after_abort(model, exc, outer=outer)
                        raise
            finally:
                watchdog.stop()
        return result

    return capped


class PhaseTracker:
    """Footprint and MLX peak per phase, sampled at the phase boundaries; one phase open at a time."""

    def __init__(self) -> None:
        """Start with no phase open and no peaks."""
        self.peaks: dict[str, dict[str, int]] = {}
        self._open: str | None = None

    @property
    def open_phase(self) -> str | None:
        """The phase running now; None between phases."""
        return self._open

    def begin(self, name: str) -> None:
        """Open ``name``: reset MLX's peak counter and sample the footprint (a phase begun again starts over)."""
        mx.reset_peak_memory()  # resets to zero, not to what is active
        self._open = name
        self.peaks[name] = {
            "footprint_start": phys_footprint(),
            "mlx_peak": 0,
            "footprint_end": 0,
            "active_at_start": int(mx.get_active_memory()),
        }

    def end(self, name: str) -> None:
        """Close ``name`` if it is the open phase (an end for another phase is ignored)."""
        if self._open != name:
            return
        record = self.peaks[name]
        record["mlx_peak"] = max(int(mx.get_peak_memory()), record["active_at_start"])
        record["footprint_end"] = phys_footprint()
        self._open = None


class VaePoolGuard:
    """mflux after-loop subscriber: close the denoise phase, drop the set if planned, empty and cap the pool."""

    def __init__(
        self,
        *,
        plan: Callable[[], CallPlan | None],
        end_denoise: Callable[[], None],
        drop_set: Callable[[], None],
        begin_vae: Callable[[], None],
    ) -> None:
        """Bind the call's plan (read when the loop ends) and the model's phase and set callbacks."""
        self._plan = plan
        self._end_denoise = end_denoise
        self._drop_set = drop_set
        self._begin_vae = begin_vae

    def call_after_loop(self, seed: int, prompt: str, latents: Any, config: Any) -> None:
        """Runs once per generation, after the last denoise step and before the VAE decode."""
        del seed, prompt, latents, config
        self._end_denoise()
        plan = self._plan()
        if plan is not None and plan.drop_set_before_vae:
            self._drop_set()
        mx.clear_cache()
        mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
        self._begin_vae()


def plan_call_for(
    *,
    constants: PhaseConstants,
    sizes: FamilySizes,
    largest: Mapping[str, int],
    policy: str,
    cache_limit_override: int | None,
    fit_check: bool,
    budget: int,
    height: int,
    width: int,
    text_tokens: int,
    log: logging.Logger,
) -> CallPlan:
    """The cache limit, the fit estimate and the VAE strategy of one call (sizes rounded down to multiples of 16).

    The set stays resident through the VAE decode when the VAE phase fits the budget with it; otherwise the call
    drops it before decoding. Sizes above the measured pixel count are an extrapolation, refused unless
    ``fit_check`` is off.

    Raises:
        DFloatResourceError: The size is above the measured ceiling, or the predicted peak exceeds the budget
            even with the set dropped, and ``fit_check`` is on.
    """
    height, width = 16 * (height // 16), 16 * (width // 16)
    ceiling = constants.max_measured_pixels
    if height * width > ceiling:
        side = math.isqrt(ceiling)
        if fit_check:
            raise DFloatResourceError(
                f"{height}x{width}: above the measured ceiling of {side}x{side} ({ceiling} pixels) on this "
                "path; pass fit_check=False to run on an extrapolated estimate"
            )
        log.warning(
            "%dx%d: no measurement above %dx%d on this path; the estimate is an extrapolation",
            height,
            width,
            side,
            side,
        )
    allowance = activation_allowance(constants, height=height, width=width, text_tokens=text_tokens)
    if len(largest) == 1:
        # One block kind: the limit holds one decoded group (two under depth2, the look-ahead's too), and the
        # activations freed beside it must fit as well, or they may evict the decoded buffer and each block's output
        # be allocated fresh. A prediction: Qwen-Image 2.1's A/B of real steps below this limit measured no slowdown.
        groups = 2 if policy == "depth2" else 1
        derived_minimum = groups * max(largest.values()) + allowance
        minimum_is = (
            "the largest decoded group twice, for depth2's look-ahead, and the activation allowance"
            if groups == 2
            else "the largest decoded group and the activation allowance"
        )
        consequence = "the cache may allocate each block's decode output fresh"
    else:
        derived_minimum = sum(sorted(largest.values(), reverse=True)[:2])
        minimum_is = "the two largest decoded groups"
        consequence = "every block's decode output will be allocated fresh"
    limit = cache_limit_for(
        constants,
        largest,
        policy=policy,
        height=height,
        width=width,
        text_tokens=text_tokens,
        override=cache_limit_override,
    )
    if limit < derived_minimum:
        log.warning(
            "cache_limit %d is below the derived minimum %d (%s): %s",
            limit,
            derived_minimum,
            minimum_is,
            consequence,
        )
    common: dict[str, Any] = {
        "sizes": sizes,
        "largest": largest,
        "policy": policy,
        "cache_limit": limit,
        "allowance": allowance,
        "budget": budget,
        "height": height,
        "width": width,
        "text_tokens": text_tokens,
    }
    estimate = fit_for(constants, **common)
    drop_set_before_vae = estimate.phases["vae"] > budget
    if drop_set_before_vae:
        estimate = fit_for(constants, vae_with_set=False, **common)
    if not estimate.fits:
        phases = ", ".join(f"{k} {v / 1024**3:.1f} GiB" for k, v in estimate.phases.items())
        message = (
            f"predicted peak {estimate.peak_bytes / 1024**3:.1f} GiB in the {estimate.peak_phase} phase "
            f"exceeds the budget {estimate.budget_bytes / 1024**3:.1f} GiB ({phases})"
        )
        if fit_check:
            raise DFloatResourceError(f"{message}; pass fit_check=False to run anyway")
        log.warning("%s; running anyway (fit_check=False)", message)
    return CallPlan(cache_limit=limit, estimate=estimate, drop_set_before_vae=drop_set_before_vae)
