"""Family-free pieces of an mflux model class: refusals, the phase tracker, the VAE pool guard and the call plan.

Each model class (one per mflux family) runs the same prelude around mflux's ``generate_image``: refuse what the
DFloat11 path cannot honour, plan the call's memory, track the phases, and empty the buffer pool before the VAE
decode. The family supplies its measured numbers (``PhaseConstants``) and sizes; nothing here imports mflux.
"""

import functools
import logging
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, NoReturn, ParamSpec, TypeVar

import mlx.core as mx

from mlx_dfloat._memory_caps import (
    caps_for_recommended_bytes,
    known_wired_limit,
    remember_wired_limit,
)
from mlx_dfloat._watchdog import phys_footprint
from mlx_dfloat.errors import DFloatResourceError, DFloatUnsupportedError
from mlx_dfloat.integrate.memory import CallPlan
from mlx_dfloat.mflux._phases import (
    FamilySizes,
    PhaseConstants,
    activation_allowance,
    cache_limit_for,
    fit_for,
)


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
    """Wrap a model's ``generate_image`` or ``encode`` in ``call_caps``: the whole call runs capped."""

    @functools.wraps(method)
    def capped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with call_caps():
            return method(*args, **kwargs)

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
