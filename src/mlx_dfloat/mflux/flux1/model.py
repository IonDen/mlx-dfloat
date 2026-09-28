"""``DFloatFlux1``: mflux's FLUX.1 pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``Flux1`` so its loop, scheduler, callbacks, VAE decode and image
metadata run unchanged; only construction and the prelude of ``generate_image`` differ. The text
encoders and the compressed transformer are never resident together (see ``lifecycle``). The
Python API installs no memory caps and no watchdog; the ``mlx-dfloat`` command does both.
"""

import logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_dfloat._watchdog import phys_footprint
from mlx_dfloat.errors import DFloatFormatError, DFloatResourceError, DFloatUnsupportedError
from mlx_dfloat.format import DF11Checkpoint, MxGroup, open_checkpoint
from mlx_dfloat.integrate.coverage import load_resident_set
from mlx_dfloat.integrate.memory import FitEstimate, budget_bytes, largest_decoded_bytes
from mlx_dfloat.integrate.names import NameMap, Shapes
from mlx_dfloat.integrate.providers import DF11Provider
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux.flux1 import init as base_init
from mlx_dfloat.mflux.flux1.init import BaseComponents, ResolvedRepo
from mlx_dfloat.mflux.flux1.lifecycle import Lifecycle
from mlx_dfloat.mflux.flux1.memory import (
    FluxSizes,
    activation_allowance,
    cache_limit_for,
    fit_for,
    sizes_for,
    text_tokens,
)
from mlx_dfloat.mflux.flux1.names import (
    DOUBLE_PREFIX,
    SINGLE_PREFIX,
    check_flux_groups,
    flux_name_map,
)
from mlx_dfloat.mflux.flux1.transformer import build_transformer

require_mflux()
from mflux.models.flux.variants.txt2img.flux import Flux1  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.flux1")

MODELS: dict[str, tuple[str, str]] = {
    "schnell": ("DFloat11/FLUX.1-schnell-DF11", "black-forest-labs/FLUX.1-schnell"),
    "dev": ("DFloat11/FLUX.1-dev-DF11", "black-forest-labs/FLUX.1-dev"),
    "krea-dev": ("DFloat11/FLUX.1-Krea-dev-DF11", "black-forest-labs/FLUX.1-Krea-dev"),
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
RETAINED_SLACK_BYTES = 2 * 1024**3
_PHASES = ("encode", "set_load", "denoise", "vae")


@dataclass(frozen=True, slots=True, kw_only=True)
class CallPlan:
    """What one generate call will do about memory: the cache limit, the estimate, and whether the set is dropped before the VAE decode."""

    cache_limit: int
    estimate: FitEstimate
    drop_set_before_vae: bool


def _refuse(name: str, reason: str) -> None:
    raise DFloatUnsupportedError(f"{name}: {reason}; not on the DFloat11 path in this version")


class _VaePoolGuard:
    """mflux after-loop subscriber: close the denoise phase, clear and cap the buffer pool before the VAE decode."""

    def __init__(self, model: "DFloatFlux1") -> None:
        self._model = model

    def call_after_loop(self, seed: int, prompt: str, latents: mx.array, config: Any) -> None:
        """Runs once per generation, after the last denoise step and before the VAE decode.

        Drops the compressed set first when the call was planned that way (the resident set plus the
        float32 decode would exceed the budget; measured 23.29 GiB at 1024² on a 32 GB Mac).
        """
        del seed, prompt, latents, config
        self._model._phase_end("denoise")
        if self._model._plan is not None and self._model._plan.drop_set_before_vae:
            self._model._lifecycle.drop_set()
        mx.clear_cache()
        mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
        self._model._phase_begin("vae")


class DFloatFlux1(Flux1):  # type: ignore[misc]  # mflux ships no type information
    """FLUX.1 (schnell, dev, Krea-dev) generation from a DFloat11 transformer through mflux.

    Bit-identical to mflux's own BF16 pipeline; the transformer stays compressed and each block is
    decoded on the GPU as it runs. Text encoders and the compressed set are never resident
    together: a new prompt after a generation drops the set, reloads the encoders, encodes and
    reloads the set; ``encode(*prompts)`` pays that once for several prompts.
    """

    def __init__(
        self,
        model: str = "schnell",
        *,
        df11_path: str | None = None,
        base_path: str | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        quantize: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        """Resolve the checkpoint and the base repository and build the model (nothing is read from disk yet).

        Raises:
            DFloatUnsupportedError: ``quantize``, ``lora_paths``, ``lora_scales`` or ``bake_lora=False``
                (accepted only to refuse), an unknown ``model``, or an eval policy other than
                ``"per-block"`` / ``"depth2"``.
            DFloatFormatError: The checkpoint is not a FLUX.1 DF11 checkpoint, or the base lacks the
                encoders or is a quantized save.
            DFloatAccessError: A gated repository this account may not read.
            DFloatDependencyError: The ``mlx-dfloat[mflux]`` extra is not installed.
        """
        if quantize is not None:
            _refuse(
                "quantize",
                "quantisation on top of DFloat11 changes the output the format exists to keep",
            )
        if lora_paths:
            _refuse("lora_paths", "LoRA of any kind")
        if lora_scales:
            _refuse("lora_scales", "LoRA of any kind")
        if not bake_lora:
            _refuse("bake_lora", "LoRA of any kind")
        if model not in MODELS:
            raise DFloatUnsupportedError(f"model {model!r}: this path runs {sorted(MODELS)}")
        from mflux.models.common.config.model_config import ModelConfig

        model_config = ModelConfig.from_name(model_name=model, base_model=None)
        df11_default, base_default = MODELS[model]
        df11 = base_init.resolve(df11_path or df11_default, patterns=base_init.DF11_PATTERNS)
        ckpt = open_checkpoint(df11.root)
        check_flux_groups(ckpt)
        base = base_init.resolve(base_path or base_default, patterns=base_init.BASE_PATTERNS)
        components = base_init.load_base(base.root, model_config)
        transformer, shapes = build_transformer(model_config, ckpt)
        self._assemble(
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            transformer=transformer,
            shapes=shapes,
            sizes=sizes_for(ckpt, base.root),
            name_map=None,
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatFlux1":
        """Assemble a model from already loaded parts (what ``__init__`` ends with)."""
        self = cls.__new__(cls)
        self._assemble(**parts)
        return self

    def _assemble(
        self,
        *,
        model_config: Any,
        ckpt: DF11Checkpoint,
        df11: ResolvedRepo,
        base: ResolvedRepo,
        components: BaseComponents,
        transformer: Any,
        shapes: Shapes,
        sizes: FluxSizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
    ) -> None:
        from mflux.models.flux.flux_initializer import FluxInitializer

        if eval_policy not in POLICIES:
            raise DFloatUnsupportedError(f"eval_policy {eval_policy!r}: choose from {POLICIES}")
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not Flux1.__init__: mflux's own initializer runs over the whole base
        FluxInitializer._init_config(
            self, model_config
        )  # prompt_cache, model_config, callbacks, tiling_config
        self.vae = components.vae
        self.t5_text_encoder = components.t5
        self.clip_text_encoder = components.clip
        self.tokenizers = components.tokenizers
        self.transformer = transformer
        self.bits = None
        self.lora_paths: list[str] = []
        self.lora_scales: list[float] = []
        self._ckpt, self._df11, self._base, self._shapes, self._sizes = (
            ckpt,
            df11,
            base,
            shapes,
            sizes,
        )
        self._names = flux_name_map() if name_map is None else name_map
        self._largest = largest_decoded_bytes(ckpt, self._names)
        self._policy, self._cache_limit_override, self._fit_check = (
            eval_policy,
            cache_limit,
            fit_check,
        )
        self._provider: DF11Provider | None = None
        self._plan: CallPlan | None = None
        self._peaks: dict[str, dict[str, int]] = {}
        self._open_phase: str | None = None
        self._launches = 0
        self._lifecycle = Lifecycle(
            load_encoders=self._load_encoders,
            unload_encoders=self._unload_encoders,
            encode=self._encode,
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache=self.prompt_cache,
            retained_bound_bytes=int(mx.get_active_memory()) + RETAINED_SLACK_BYTES,
            encoders_loaded=True,
        )
        self.callbacks.register(_VaePoolGuard(self))

    # --- lifecycle callbacks ------------------------------------------------------------------------

    def _load_encoders(self) -> None:
        self.t5_text_encoder, self.clip_text_encoder = base_init.load_encoders(self._base.root)

    def _unload_encoders(self) -> None:
        self.t5_text_encoder = None
        self.clip_text_encoder = None

    def _encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder

        pair: tuple[mx.array, mx.array] = PromptEncoder.encode_prompt(
            prompt,
            prompt_cache={},  # the lifecycle caches the evaluated pair itself
            t5_tokenizer=self.tokenizers["t5"],
            clip_tokenizer=self.tokenizers["clip"],
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )
        return pair

    def _load_set(self) -> None:
        resident: dict[str, MxGroup] = load_resident_set(self._ckpt)
        self._provider = DF11Provider(
            resident, {n: self._ckpt.groups[n].matrix_names for n in resident}, self._names
        )
        self.transformer.attach(
            self._provider, self._shapes, eval_policy=self._policy, verify_in_call=True
        )

    def _unload_set(self) -> None:
        if self._provider is not None:
            self._launches += self._provider.launches
        self.transformer.detach()
        self._provider = None

    # --- phases -----------------------------------------------------------------------------------

    def _phase_begin(self, name: str) -> None:
        mx.reset_peak_memory()
        self._open_phase = name
        self._peaks[name] = {"footprint_start": phys_footprint(), "mlx_peak": 0, "footprint_end": 0}

    def _phase_end(self, name: str) -> None:
        if self._open_phase != name:
            return
        self._peaks[name]["mlx_peak"] = int(mx.get_peak_memory())
        self._peaks[name]["footprint_end"] = phys_footprint()
        self._open_phase = None

    # --- the public surface -----------------------------------------------------------------------

    def plan_call(self, *, height: int, width: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call at this size (rounded down to multiples of 16, as mflux does).

        The set stays resident through the VAE decode when the estimate allows it; otherwise the call drops it
        before decoding and the next call reloads it.

        Raises:
            DFloatResourceError: The predicted peak exceeds the budget even with the set dropped, and ``fit_check`` is on.
        """
        height, width = 16 * (height // 16), 16 * (width // 16)
        tokens = text_tokens(self.model_config)
        derived_minimum = self._largest[DOUBLE_PREFIX] + self._largest[SINGLE_PREFIX]
        limit = cache_limit_for(
            self._largest,
            policy=self._policy,
            height=height,
            width=width,
            text_tokens=tokens,
            override=self._cache_limit_override,
        )
        if limit < derived_minimum:
            log.warning(
                "cache_limit %d is below the derived minimum %d (the two largest decoded groups): "
                "every block's decode output will be allocated fresh",
                limit,
                derived_minimum,
            )
        allowance = activation_allowance(height=height, width=width, text_tokens=tokens)
        budget = budget_bytes()
        estimate = fit_for(
            sizes=self._sizes,
            largest=self._largest,
            policy=self._policy,
            cache_limit=limit,
            allowance=allowance,
            budget=budget,
        )
        drop_set_before_vae = estimate.phases["vae"] > budget
        if drop_set_before_vae:
            estimate = fit_for(
                sizes=self._sizes,
                largest=self._largest,
                policy=self._policy,
                cache_limit=limit,
                allowance=allowance,
                budget=budget,
                vae_with_set=False,
            )
        if not estimate.fits:
            phases = ", ".join(f"{k} {v / 1024**3:.1f} GiB" for k, v in estimate.phases.items())
            message = (
                f"predicted peak {estimate.peak_bytes / 1024**3:.1f} GiB in the {estimate.peak_phase} phase "
                f"exceeds the budget {estimate.budget_bytes / 1024**3:.1f} GiB ({phases})"
            )
            if self._fit_check:
                raise DFloatResourceError(f"{message}; pass fit_check=False to run anyway")
            log.warning("%s; running anyway (fit_check=False)", message)
        plan = CallPlan(
            cache_limit=limit, estimate=estimate, drop_set_before_vae=drop_set_before_vae
        )
        self._plan = plan
        return plan

    def encode(self, *prompts: str) -> None:
        """Encode prompts now, so several generations pay the encoder reload once (drops a resident set first)."""
        self._lifecycle.ensure_embeddings(*prompts)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> Any:
        """Mflux's ``generate_image`` behind a prelude: refusals, cache limit, fit check, prompt encoding, set load.

        ``negative_prompt`` is accepted and ignored, as upstream does for FLUX.1.

        Raises:
            DFloatUnsupportedError: ``image_path`` / ``image_strength`` (img2img) or ``pid_decode``.
            DFloatResourceError: The fit estimate refuses the call, or memory stayed active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
        """
        if image_path is not None or image_strength is not None:
            _refuse("image_path/image_strength", "img2img")
        if pid_decode:
            _refuse(
                "pid_decode",
                "the PiD decoder loads an 8 GB caption encoder next to the compressed set",
            )
        plan = self.plan_call(height=height, width=width)
        self._phase_begin("encode")
        self._lifecycle.ensure_embeddings(prompt)
        self._phase_end("encode")
        self._phase_begin("set_load")
        self._lifecycle.ensure_set()
        self._phase_end("set_load")
        previous = mx.set_cache_limit(plan.cache_limit)
        self._phase_begin("denoise")
        try:
            return super().generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance=guidance,
                image_path=None,
                image_strength=None,
                scheduler=scheduler,
                negative_prompt=negative_prompt,
                pid_decode=False,
                pid_degrade_sigma=pid_degrade_sigma,
            )
        except DFloatFormatError:
            self._lifecycle.drop_set()  # a corrupt block: the retry starts from a clean load
            raise
        finally:
            mx.set_cache_limit(previous)
            self._phase_end("denoise")
            self._phase_end("vae")

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction.
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "model": self._model_name(),
            "df11": {
                "root": str(self._df11.root),
                "repo_id": self._df11.repo_id,
                "revision": self._df11.revision,
            },
            "base": {
                "root": str(self._base.root),
                "repo_id": self._base.repo_id,
                "revision": self._base.revision,
            },
            "eval_policy": self._policy,
            "cache_limit_in_force": None if self._plan is None else self._plan.cache_limit,
            "drop_set_before_vae": None if self._plan is None else self._plan.drop_set_before_vae,
            "fit": None
            if fit is None
            else {
                "label": "predicted",
                "phases": dict(fit.phases),
                "peak_phase": fit.peak_phase,
                "peak_bytes": fit.peak_bytes,
                "budget_bytes": fit.budget_bytes,
                "fits": fit.fits,
            },
            "peaks": {"label": "sampled at phase boundaries", **self._peaks},
            "decode_launches": launches,
            "lifecycle": self._lifecycle.counters.as_dict(),
            "versions": {"mlx": mx.__version__, "mflux": mflux_version},  # type: ignore[attr-defined]
        }

    def _model_name(self) -> str:
        """The short model name whose default DF11 repository this model came from, or ``"custom"``."""
        for name, (df11_repo, _base_repo) in MODELS.items():
            if df11_repo == self._df11.repo_id:
                return name
        return "custom"

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "DFloatFlux1":
        """A model by its mflux name; ``quantize`` is accepted only to refuse it."""
        return DFloatFlux1(model_name, quantize=quantize)

    def save_model(self, base_path: str) -> None:
        """Refused: the DFloat11 checkpoint is the saved form; there is nothing of mflux's to write.

        Raises:
            DFloatUnsupportedError: Always.
        """
        del base_path
        _refuse("save_model", "the DFloat11 repository is the checkpoint")

    def freeze(self, **kwargs: Any) -> None:
        """Freeze the components that are loaded (a dropped encoder is ``None``)."""
        del kwargs
        for module in (self.vae, self.transformer, self.t5_text_encoder, self.clip_text_encoder):
            if module is not None:
                module.freeze()
