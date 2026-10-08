"""``DFloatZImage``: mflux's Z-Image pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``ZImage`` so its loop, scheduler, callbacks, VAE decode and image metadata run
unchanged; only construction and the prelude of ``generate_image`` differ. The text encoder and the compressed
transformer are never resident together (see ``mlx_dfloat.mflux.lifecycle``). The Python API installs no memory
caps and no watchdog; the ``mlx-dfloat`` command does both.
"""

import logging
import traceback
from collections.abc import Callable
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_dfloat import _metal_decode
from mlx_dfloat._watchdog import phys_footprint
from mlx_dfloat.decode import DecodeResult
from mlx_dfloat.errors import (
    DFloatFormatError,
    DFloatIntegrationError,
    DFloatResourceError,
    DFloatUnsupportedError,
)
from mlx_dfloat.format import DF11Checkpoint, MxGroup, open_checkpoint
from mlx_dfloat.integrate.coverage import load_resident_set
from mlx_dfloat.integrate.memory import CallPlan, budget_bytes, largest_decoded_bytes
from mlx_dfloat.integrate.names import NameMap
from mlx_dfloat.integrate.providers import DF11Provider
from mlx_dfloat.integrate.resident import clear_nonblock, decode_nonblock, install_nonblock
from mlx_dfloat.mflux import families as registry
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._compile import predict_mode, uncompiled
from mlx_dfloat.mflux.lifecycle import Lifecycle
from mlx_dfloat.mflux.zimage import init as zinit
from mlx_dfloat.mflux.zimage.init import ResolvedRepo, ZImageComponents
from mlx_dfloat.mflux.zimage.memory import (
    MAX_MEASURED_PIXELS,
    ZImageSizes,
    activation_allowance,
    cache_limit_for,
    fit_for,
    sizes_for,
    text_tokens,
)
from mlx_dfloat.mflux.zimage.names import NONBLOCK_GROUPS, check_zimage_groups, zimage_name_map
from mlx_dfloat.mflux.zimage.transformer import ZImageBuild, build_transformer

require_mflux()
from mflux.models.z_image.variants.z_image import ZImage  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.zimage")

# (DF11 repository, base repository), derived from the registry.
MODELS: dict[str, tuple[str, str]] = {
    n: (e.df11_repo, e.base_repo) for n, e in registry.MODELS.items() if e.family == "zimage"
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
RETAINED_SLACK_BYTES = 2 * 1024**3


def _refuse(name: str, reason: str) -> None:
    raise DFloatUnsupportedError(f"{name}: {reason}; not on the DFloat11 path in this version")


def _check_eval_policy(eval_policy: str) -> None:
    """Refuse an eval policy outside ``POLICIES`` (checked in ``__init__`` and ``_assemble``)."""
    if eval_policy not in POLICIES:
        raise DFloatUnsupportedError(f"eval_policy {eval_policy!r}: choose from {POLICIES}")


class _VaePoolGuard:
    """mflux after-loop subscriber: close the denoise phase, clear and cap the buffer pool before the VAE decode."""

    def __init__(self, model: "DFloatZImage") -> None:
        self._model = model

    def call_after_loop(self, seed: int, prompt: str, latents: mx.array, config: Any) -> None:
        """Runs once per generation, after the last denoise step and before the VAE decode."""
        del seed, prompt, latents, config
        self._model._phase_end("denoise")
        if self._model._plan is not None and self._model._plan.drop_set_before_vae:
            self._model._lifecycle.drop_set()
        mx.clear_cache()
        mx.set_cache_limit(0)  # the transformer-shaped buffers cannot serve the decoder
        self._model._phase_begin("vae")


class DFloatZImage(ZImage):  # type: ignore[misc]  # mflux ships no type information
    """Z-Image and Z-Image-Turbo generation from a DFloat11 transformer through mflux.

    The transformer stays compressed and each block is decoded on the GPU as it runs (the one non-block matrix,
    the caption embedder, is decoded once when the set loads). For Z-Image the result is bit-identical to the same
    transformer run block by block from its BF16 weights. Z-Image-Turbo has no BF16 original (its published
    transformer is FP32): sampled groups of its checkpoint equal that original rounded to BF16, and its latents
    have not been compared with a reference. The text encoder and the compressed set are never resident together:
    a new prompt after a generation drops the set, reloads the encoder, encodes and reloads the set.
    """

    def __init__(
        self,
        model: str = "z-image-turbo",
        *,
        df11_path: str | None = None,
        base_path: str | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        quantize: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        """Resolve the checkpoint and the base repository and build the model.

        The checkpoint headers, the base's tokenizers and the transformer's extras are read; the encoder, VAE and
        compressed weights stay lazy until used. ``budget_bytes`` replaces the device's budget in every call's fit
        estimate and VAE decision.

        Raises:
            DFloatUnsupportedError: ``quantize``, ``lora_paths``, ``lora_scales`` or ``bake_lora=False``
                (accepted only to refuse), an eval policy other than ``"per-block"`` / ``"depth2"``, a
                ControlNet model, or an unknown ``model``.
            DFloatBackendError: The GPU decoder fails its check on the two packaged self-check groups (raised before
                anything is resolved or loaded), or Metal is unavailable.
            DFloatFormatError: The checkpoint is not a Z-Image DF11 checkpoint, or the base lacks a component.
            DFloatAccessError: A gated repository this account may not read.
            DFloatDependencyError: The ``mlx-dfloat[mflux]`` extra is not installed.
            DFloatIntegrationError: mflux's own weight definition or mapping does not shape the way this
                adapter expects (a name map ambiguity, an uncovered extra).
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
        _check_eval_policy(eval_policy)
        if "controlnet" in model:
            raise DFloatUnsupportedError(
                f"model {model!r}: ControlNet is another model class, not on the DFloat11 path"
            )
        if model not in MODELS:
            raise DFloatUnsupportedError(f"model {model!r}: this path runs {sorted(MODELS)}")
        # The GPU decoder proves itself on the two packaged self-check groups before anything is resolved, downloaded or
        # loaded, so a broken decoder refuses here and not minutes later at the first block.
        _metal_decode.ensure_canary(force_direct=False)
        from mflux.models.common.config.model_config import ModelConfig

        model_config = ModelConfig.from_name(model_name=model, base_model=None)
        df11_default, base_default = MODELS[model]
        # The default checkpoints live on one person's Hub account: pinned to the commit that was verified here.
        # A user's own checkpoint is taken as given.
        pin = None if df11_path else registry.MODELS[model].df11_revision
        df11 = zinit.resolve(df11_path or df11_default, patterns=zinit.DF11_PATTERNS, revision=pin)
        ckpt = open_checkpoint(df11.root)
        check_zimage_groups(ckpt)
        base = zinit.resolve(base_path or base_default, patterns=zinit.BASE_PATTERNS)
        components = zinit.load_base(base.root)
        build = build_transformer(ckpt)
        self._assemble(
            model=model,
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            build=build,
            sizes=sizes_for(ckpt, base.root),
            name_map=None,
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
            budget_bytes=budget_bytes,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatZImage":
        """Assemble a model from already loaded parts (what ``__init__`` ends with)."""
        self = cls.__new__(cls)
        self._assemble(**parts)
        return self

    def _assemble(
        self,
        *,
        model: str = "custom",
        model_config: Any,
        ckpt: DF11Checkpoint,
        df11: ResolvedRepo,
        base: ResolvedRepo,
        components: ZImageComponents,
        build: ZImageBuild,
        sizes: ZImageSizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        from mflux.models.z_image.z_image_initializer import ZImageInitializer

        _check_eval_policy(eval_policy)
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not ZImage.__init__: mflux's initializer runs over the whole base
        ZImageInitializer._init_config(self, model_config)  # model_config, callbacks, tiling_config
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizers = components.tokenizers
        self.transformer = build.transformer
        self.bits = None
        self.lora_paths: list[str] = []
        self.lora_scales: list[float] = []
        self._embeddings: dict[str, tuple[mx.array, ...]] = {}
        self._model = model
        self._ckpt, self._df11, self._base, self._sizes = ckpt, df11, base, sizes
        self._shapes, self._nonblock = build.shapes, build.nonblock
        self._decode = decode
        self._names = zimage_name_map() if name_map is None else name_map
        self._cfg_calls = 1
        self._largest = largest_decoded_bytes(ckpt, self._names, skip=frozenset(NONBLOCK_GROUPS))
        self._policy, self._cache_limit_override, self._fit_check = (
            eval_policy,
            cache_limit,
            fit_check,
        )
        self._budget_override = budget_bytes
        self._provider: DF11Provider | None = None
        self._plan: CallPlan | None = None
        self._peaks: dict[str, dict[str, int]] = {}
        self._open_phase: str | None = None
        self._launches = 0
        self._baseline_active = int(mx.get_active_memory())
        self._lifecycle = Lifecycle(
            load_encoders=self._load_encoders,
            unload_encoders=self._unload_encoders,
            encode=self._encode,
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache=self._embeddings,
            retained_bound=self._retained_bound,
            encoders_loaded=True,
        )
        self.callbacks.register(_VaePoolGuard(self))

    # --- lifecycle callbacks ------------------------------------------------------------------------

    def _retained_bound(self) -> int:
        """What may stay active after a drop: the assembly baseline, the cached embeddings, the VAE, slack."""
        embeddings = sum(int(a.nbytes) for arrays in self._embeddings.values() for a in arrays)
        flat: list[tuple[str, mx.array]] = list(tree_flatten(self.vae.parameters()))  # type: ignore[arg-type]
        vae = sum(int(v.nbytes) for _name, v in flat)
        return self._baseline_active + embeddings + vae + RETAINED_SLACK_BYTES

    def _load_encoders(self) -> None:
        self.text_encoder = zinit.load_text_encoder(self._base.root)

    def _unload_encoders(self) -> None:
        self.text_encoder = None

    def _encode(self, prompt: str) -> tuple[mx.array, ...]:
        from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder

        return (
            PromptEncoder.encode_prompt(
                prompt=prompt, tokenizer=self.tokenizers["z_image"], text_encoder=self.text_encoder
            ),
        )

    def _load_set(self) -> None:
        try:
            self._install_set()
        except BaseException:
            self._unload_set()  # nothing half-installed: the lifecycle believes no set is resident
            raise

    def _install_set(self) -> None:
        resident = load_resident_set(self._ckpt)
        nonblock = {g: self._ckpt.groups[g].matrix_names for g in NONBLOCK_GROUPS if g in resident}
        install_nonblock(
            self.transformer,
            decode_nonblock(
                {g: resident[g] for g in nonblock},
                nonblock,
                self._nonblock,
                self._names,
                decode=self._decode,
            ),
        )
        blocks = {n: g for n, g in resident.items() if n not in NONBLOCK_GROUPS}
        self._provider = DF11Provider(
            blocks,
            {n: self._ckpt.groups[n].matrix_names for n in blocks},
            self._names,
            decode=self._decode,
        )
        # Z-Image base with guidance calls the transformer twice per step: verify inside each call.
        self.transformer.attach(
            self._provider, self._shapes, eval_policy=self._policy, verify_in_call=True
        )

    def _unload_set(self) -> None:
        if self._provider is not None:
            self._launches += self._provider.launches
        self.transformer.detach()
        clear_nonblock(self.transformer, self._nonblock)
        self._provider = None

    @staticmethod
    def _predict(transformer: Any) -> Any:
        """Mflux's predict closure, uncompiled on every chip (the seam evaluates at block boundaries).

        Raises:
            DFloatIntegrationError: mflux's factory still compiled it.
        """
        return uncompiled(ZImage._predict, transformer)

    # --- phases -----------------------------------------------------------------------------------

    @property
    def open_phase(self) -> str | None:
        """The phase running now (``encode``, ``set_load``, ``denoise`` or ``vae``); None outside them."""
        return self._open_phase

    def _phase_begin(self, name: str) -> None:
        mx.reset_peak_memory()  # resets to zero, not to what is active
        self._open_phase = name
        self._peaks[name] = {
            "footprint_start": phys_footprint(),
            "mlx_peak": 0,
            "footprint_end": 0,
            "active_at_start": int(mx.get_active_memory()),
        }

    def _phase_end(self, name: str) -> None:
        if self._open_phase != name:
            return
        self._peaks[name]["mlx_peak"] = max(
            int(mx.get_peak_memory()), self._peaks[name]["active_at_start"]
        )
        self._peaks[name]["footprint_end"] = phys_footprint()
        self._open_phase = None

    # --- the public surface -----------------------------------------------------------------------

    def plan_call(self, *, height: int, width: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call at this size (rounded down to multiples of 16, as mflux does).

        The set stays resident through the VAE decode when the estimate allows it; otherwise the call drops it
        before decoding and the next call reloads it. Sizes above 1024² were not measured on this path; the
        estimate there is an extrapolation.

        Raises:
            DFloatResourceError: The size is above 1024² or the predicted peak exceeds the budget even with
                the set dropped, and ``fit_check`` is on.
        """
        height, width = 16 * (height // 16), 16 * (width // 16)
        if height * width > MAX_MEASURED_PIXELS:
            if self._fit_check:
                raise DFloatResourceError(
                    f"{height}x{width}: above the measured ceiling of 1024x1024 ({MAX_MEASURED_PIXELS} "
                    "pixels) on this path; pass fit_check=False to run on an extrapolated estimate"
                )
            log.warning(
                "%dx%d: no measurement above 1024² on this path; the estimate is an extrapolation",
                height,
                width,
            )
        tokens = text_tokens(self.model_config)
        derived_minimum = sum(sorted(self._largest.values(), reverse=True)[:2])
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
        budget = self._budget_override if self._budget_override is not None else budget_bytes()
        common: dict[str, Any] = {
            "sizes": self._sizes,
            "largest": self._largest,
            "policy": self._policy,
            "cache_limit": limit,
            "allowance": allowance,
            "budget": budget,
            "height": height,
            "width": width,
            "text_tokens": tokens,
        }
        estimate = fit_for(**common)
        drop_set_before_vae = estimate.phases["vae"] > budget
        if drop_set_before_vae:
            estimate = fit_for(vae_with_set=False, **common)
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

    def cfg_prompts(
        self, prompt: str, *, negative_prompt: str | None, guidance: float | None
    ) -> tuple[str, ...]:
        """The prompts a call encodes: one, or the prompt and its negative when classifier-free guidance runs.

        mflux 0.20's rules (``ZImage.generate_image`` and ``_encode_prompts``): a model without guidance support runs at guidance 0, guidance
        at or below 1 encodes the prompt alone, and a blank negative prompt is a single space.
        """
        effective = (
            0.0 if not self.model_config.supports_guidance or guidance is None else float(guidance)
        )
        if effective <= 1.0:
            return (prompt,)
        negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        return (prompt, negative)

    def _encode_prompts(
        self, *, prompt: str, negative_prompt: str | None, guidance: float
    ) -> tuple[mx.array, mx.array | None]:
        """Mflux's hook, answered from the cache (the encoder is not resident during the loop).

        Raises:
            DFloatIntegrationError: A prompt this call needs was not encoded first.
        """
        wanted = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        missing = [p for p in wanted if p not in self._embeddings]
        if missing:
            raise DFloatIntegrationError(f"prompt {missing[0]!r} was not encoded before the call")
        text = self._embeddings[wanted[0]][0]
        return text, (self._embeddings[wanted[1]][0] if len(wanted) > 1 else None)

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        negative_prompt: str | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> Any:
        """Mflux's ``generate_image`` behind a prelude: refusals, cache limit, fit check, prompt encoding, set load.

        Raises:
            DFloatUnsupportedError: ``image_path`` / ``image_strength`` (img2img) or ``pid_decode``.
            DFloatResourceError: The fit check refuses the call (the estimate, or a size above 1024²), or
                memory stayed active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
        """
        if image_path is not None or image_strength is not None:
            _refuse("image_path/image_strength", "img2img")
        if pid_decode:
            _refuse(
                "pid_decode",
                "mflux's alternative image decoder (PiD) loads an 8 GB caption encoder next to the compressed set",
            )
        plan = self.plan_call(height=height, width=width)
        prompts = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        self._cfg_calls = len(prompts)
        self._phase_begin("encode")
        self._lifecycle.ensure_embeddings(*prompts)
        self._phase_end("encode")
        self._phase_begin("set_load")
        self._lifecycle.ensure_set()
        self._phase_end("set_load")
        self._phase_begin("denoise")  # before set_cache_limit: a raise here must change nothing
        previous = mx.set_cache_limit(plan.cache_limit)
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
        except BaseException as exc:
            # The traceback keeps the seam frames, and with them decoded weights or the resident set, alive for as
            # long as the caller holds the exception (and past drop_set's own gc.collect()). Clear them for any
            # exception, and for the one it was raised while handling.
            traceback.clear_frames(exc.__traceback__)
            if exc.__context__ is not None and exc.__context__.__traceback__ is not None:
                traceback.clear_frames(exc.__context__.__traceback__)
            if isinstance(exc, DFloatFormatError):
                self._lifecycle.drop_set()  # a corrupt block: the retry starts from a clean load
            raise
        finally:
            mx.set_cache_limit(previous)
            self._phase_end("denoise")
            self._phase_end("vae")

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction.
        ``predict`` is what mflux's step factory returns under this class's compile bypass, checked when the report
        is built (``"uncompiled"``, or ``"compiled"`` if the bypass stopped working).
        ``cfg_calls_per_step`` is the last call's (1, or 2 when classifier-free guidance ran).
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "family": "zimage",
            "model": self._model,
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
            "cfg_calls_per_step": self._cfg_calls,
            "predict": predict_mode(ZImage._predict, self.transformer),
            "nonblock_groups": sorted(NONBLOCK_GROUPS),
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
            "sizes": asdict(self._sizes),
            "peaks": {"label": "sampled at phase boundaries", **self._peaks},
            "decode_launches": launches,
            "lifecycle": self._lifecycle.counters.as_dict(),
            "versions": {"mlx": mx.__version__, "mflux": mflux_version},  # type: ignore[attr-defined]
        }

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
        for module in (self.vae, self.transformer, self.text_encoder):
            if module is not None:
                module.freeze()
