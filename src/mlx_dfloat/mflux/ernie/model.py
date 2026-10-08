"""``DFloatErnieImage``: mflux's ERNIE-Image pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``ErnieImage`` so its loop, scheduler, callbacks, VAE decode and image metadata run
unchanged; only construction and the prelude of ``generate_image`` differ. The text encoder and the compressed
transformer are never resident together (see ``mlx_dfloat.mflux.lifecycle``): a call's prompts are encoded first into
a cache of mflux's own text batches, which the loop reads without touching the encoder.
"""

import json
import logging
import traceback
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_dfloat import _metal_decode
from mlx_dfloat.decode import DecodeResult
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.format import DF11Checkpoint, MxGroup, open_checkpoint
from mlx_dfloat.integrate.coverage import load_resident_set
from mlx_dfloat.integrate.memory import CallPlan, budget_bytes, largest_decoded_bytes
from mlx_dfloat.integrate.names import NameMap
from mlx_dfloat.integrate.providers import DF11Provider
from mlx_dfloat.integrate.resident import clear_nonblock, decode_nonblock, install_nonblock
from mlx_dfloat.mflux import _pipeline, require_mflux
from mlx_dfloat.mflux import families as registry
from mlx_dfloat.mflux._compile import uncompiled
from mlx_dfloat.mflux._phases import FamilySizes
from mlx_dfloat.mflux.ernie import init as einit
from mlx_dfloat.mflux.ernie import memory as emem
from mlx_dfloat.mflux.ernie.init import ErnieComponents, ResolvedRepo
from mlx_dfloat.mflux.ernie.names import NONBLOCK_GROUPS, ernie_name_map
from mlx_dfloat.mflux.ernie.transformer import ErnieBuild, build_transformer
from mlx_dfloat.mflux.lifecycle import Lifecycle

require_mflux()
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.ernie")

# (DF11 repository, base repository), derived from the registry.
MODELS: dict[str, tuple[str, str]] = {
    n: (e.df11_repo, e.base_repo) for n, e in registry.MODELS.items() if e.family == "ernie"
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
# The policies an ERNIE-Image run has measured; depth2 is accepted, and the report labels it unmeasured.
MEASURED_POLICIES: frozenset[str] = frozenset({"per-block"})
# What else may stay active across calls besides the assembly baseline, the cached batches and the VAE. It covers
# mflux's ``ErnieTransformer._pos_cache`` (transformer.py:120-122): up to 64 (cos, sin, attn_mask) triples keyed by
# image size, text length and the prompts' lengths, grown by one entry per new (size, prompt lengths) and evicted
# oldest-first past 64; it is not part of the set. One 1024² batch-2 entry at the 2048-token maximum holds 6_316_032 B
# (cos and sin (2, 6144, 1, 128) bf16, the mask (2, 1, 1, 6144) bf16; measured on mlx 0.32.2 / mflux 0.20.0), so 64
# entries hold at most 404_226_048 B (0.38 GiB), inside the slack. Above 1024² (only with fit_check=False) an entry grows
# with the image tokens, so the worst case grows with it.
RETAINED_SLACK_BYTES = 2 * 1024**3


def _clear_frames(exc: BaseException) -> None:
    """Release the locals of the frames ``exc`` and every exception in its chain (causes and contexts) passed through.

    The walk keeps a seen-set, so a chain that loops back on itself ends.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


class _EncoderNotResident:
    """Takes the text encoder's place while it is dropped: mflux calls the encoder only for a batch not cached."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: every batch a call needs is encoded before mflux's loop starts.

        Raises:
            DFloatIntegrationError: Always.
        """
        del args, kwargs
        raise DFloatIntegrationError(
            "the text encoder is not resident: a prompt was not encoded before the call "
            "(generate_image encodes the prompts it needs first; call encode() for any other use)"
        )


class DFloatErnieImage(ErnieImage):  # type: ignore[misc]  # mflux ships no type information
    """ERNIE-Image and ERNIE-Image-Turbo generation from a DFloat11 transformer through mflux.

    The transformer stays compressed and each of its 36 blocks is decoded on the GPU as it runs; the three non-block
    groups (the timestep MLP, the shared modulation, the final norm's projection) are decoded once when the set loads.
    The step function runs uncompiled on every chip (the per-block evaluation cannot run inside a compiled function).
    The decode is lossless.

    The text encoder and the compressed set are never resident together: a new prompt list after a generation drops
    the set, reloads the encoder, encodes and reloads the set.
    """

    def __init__(
        self,
        model: str = "ernie-image-turbo",
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

        The checkpoint header, the base's tokenizer and the transformer's extras are read; the encoder, VAE and
        compressed weights stay lazy until used. The default checkpoint and base are pinned to the revisions this
        package was checked against; a user's own paths are taken as given. ``budget_bytes`` replaces the device's
        budget in every call's fit estimate and VAE decision.

        Raises:
            DFloatUnsupportedError: ``quantize``, ``lora_paths``, ``lora_scales`` or ``bake_lora=False`` (accepted only
                to refuse), an eval policy other than ``"per-block"`` / ``"depth2"``, or an unknown ``model``.
            DFloatBackendError: The GPU decoder fails its check on the two packaged self-check groups (raised before
                anything is resolved or loaded), or Metal is unavailable.
            DFloatFormatError: The checkpoint is not an ERNIE-Image DF11 checkpoint, its block count differs from the
                model's, or the base lacks a component or is another model's.
            DFloatAccessError: A gated repository this account may not read.
            DFloatDependencyError: The ``mlx-dfloat[mflux]`` extra is not installed.
            DFloatIntegrationError: mflux's own weight definition or mapping does not shape the way this adapter
                expects.
        """
        _pipeline.refuse_construction_args(
            quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales, bake_lora=bake_lora
        )
        _pipeline.check_eval_policy(eval_policy, POLICIES)
        if model not in MODELS:
            raise DFloatUnsupportedError(f"model {model!r}: this path runs {sorted(MODELS)}")
        # The GPU decoder proves itself on the two packaged self-check groups before anything is resolved, downloaded or
        # loaded, so a broken decoder refuses here and not minutes later at the first block.
        _metal_decode.ensure_canary(force_direct=False)
        from mflux.models.common.config.model_config import ModelConfig

        model_config = ModelConfig.from_name(model_name=model, base_model=None)
        entry = registry.MODELS[model]
        df11_default, base_default = MODELS[model]
        pin = None if df11_path else entry.df11_revision
        df11 = einit.resolve(df11_path or df11_default, patterns=einit.DF11_PATTERNS, revision=pin)
        ckpt = open_checkpoint(df11.root)
        base_pin = None if base_path else entry.base_revision
        base = einit.resolve(
            base_path or base_default, patterns=einit.BASE_PATTERNS, revision=base_pin
        )
        # mflux builds the transformer with the model's overrides (ErnieImageInitializer._init_models).
        build = build_transformer(ckpt, transformer_overrides=model_config.transformer_overrides)
        components = einit.load_base(base.root)
        self._assemble(
            model=model,
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            build=build,
            sizes=emem.sizes_for(ckpt, base.root),
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
            budget_bytes=budget_bytes,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatErnieImage":
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
        components: ErnieComponents,
        build: ErnieBuild,
        sizes: FamilySizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer

        _pipeline.check_eval_policy(eval_policy, POLICIES)
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not ErnieImage.__init__: mflux's initializer loads the whole base
        # prompt_cache, model_config, callbacks, tiling_config (ernie_image_initializer.py:40-45)
        ErnieImageInitializer._init_config(self, model_config)
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizers = components.tokenizers
        self.transformer = build.transformer
        self.bits = None
        self.lora_paths: list[str] = []
        self.lora_scales: list[float] = []
        self._model = model
        self._constants = emem.CONSTANTS
        self._ckpt, self._df11, self._base, self._sizes = ckpt, df11, base, sizes
        self._shapes, self._nonblock = build.shapes, build.nonblock
        self._decode = decode
        self._names = ernie_name_map() if name_map is None else name_map
        self._batch = 1
        self._text_tokens: int | None = None
        self._predict_mode: str | None = None
        self._largest = largest_decoded_bytes(ckpt, self._names, skip=frozenset(NONBLOCK_GROUPS))
        self._policy, self._cache_limit_override, self._fit_check = (
            eval_policy,
            cache_limit,
            fit_check,
        )
        self._budget_override = budget_bytes
        self._provider: DF11Provider | None = None
        self._plan: CallPlan | None = None
        self._tracker = _pipeline.PhaseTracker()
        self._launches = 0
        self._baseline_active = int(mx.get_active_memory())
        # mflux's text batches, keyed by the call's prompt list (``_batch_key``); mflux's own prompt_cache stays empty.
        self._batches: dict[str, tuple[mx.array, ...]] = {}
        self._lifecycle = Lifecycle(
            load_encoders=self._load_encoders,
            unload_encoders=self._unload_encoders,
            encode=self._encode,
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache=self._batches,
            retained_bound=self._retained_bound,
            encoders_loaded=True,
        )
        self.callbacks.register(
            _pipeline.VaePoolGuard(
                plan=lambda: self._plan,
                end_denoise=lambda: self._tracker.end("denoise"),
                drop_set=lambda: self._lifecycle.drop_set(),
                begin_vae=lambda: self._tracker.begin("vae"),
            )
        )

    # --- lifecycle callbacks ------------------------------------------------------------------------

    @staticmethod
    def _batch_key(prompts: Sequence[str]) -> str:
        """The cache key of a call's prompt list (in mflux's batch order)."""
        return json.dumps(list(prompts))

    def _retained_bound(self) -> int:
        """What may stay active after a drop: the assembly baseline, the cached batches, the VAE, slack.

        The whole VAE counts, its encoder included: an upper bound, since mflux's decode reads only part of it and the
        rest stays unloaded under lazy loading, but which part depends on mflux's Flux2VAE layout.
        """
        batches = sum(int(a.nbytes) for arrays in self._batches.values() for a in arrays)
        flat: list[tuple[str, mx.array]] = list(tree_flatten(self.vae.parameters()))  # type: ignore[arg-type]
        vae = sum(int(v.nbytes) for _name, v in flat)
        return self._baseline_active + batches + vae + RETAINED_SLACK_BYTES

    def _load_encoders(self) -> None:
        self.text_encoder = einit.load_text_encoder(self._base.root)

    def _unload_encoders(self) -> None:
        # Not None: mflux would call it on a cache miss and fail with a TypeError far from the cause.
        self.text_encoder = _EncoderNotResident()

    def _encode(self, key: str) -> tuple[mx.array, ...]:
        """Mflux's text batch for the prompt list ``key`` names, built by mflux's own ``build_text_batch``."""
        from mflux.models.ernie_image.model.ernie_text_encoder.prompt_encoder import (
            ErniePromptEncoder,
        )

        text_bth, text_lens = ErniePromptEncoder.build_text_batch(
            prompts=json.loads(key),
            tokenizer=self.tokenizers["ernie"],
            text_encoder=self.text_encoder,
        )
        return text_bth, text_lens

    def _load_set(self) -> None:
        try:
            self._install_set()
        except BaseException as exc:
            # The traceback's frames hold the compressed set (_install_set's ``resident``) and any decoded non-block
            # weight for as long as the caller keeps the exception; the retry's memory check would then refuse.
            _clear_frames(exc)
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
        # One transformer call per step (classifier-free guidance is one batch-2 call); verified inside the call to
        # keep every family's invariant.
        self.transformer.attach(
            self._provider, self._shapes, eval_policy=self._policy, verify_in_call=True
        )

    def _unload_set(self) -> None:
        if self._provider is not None:
            self._launches += self._provider.launches
        self.transformer.detach()
        clear_nonblock(self.transformer, self._nonblock)
        self._provider = None

    # --- phases -----------------------------------------------------------------------------------

    @property
    def open_phase(self) -> str | None:
        """The phase running now (``encode``, ``set_load``, ``denoise`` or ``vae``); None outside them."""
        return self._tracker.open_phase

    def _budget(self) -> int:
        """The fit budget: the constructor's ``budget_bytes``, else the device's (read at each call)."""
        return self._budget_override if self._budget_override is not None else budget_bytes()

    # --- the public surface -----------------------------------------------------------------------

    @staticmethod
    def cfg_prompts(
        prompt: str, *, negative_prompt: str | None, guidance: float | None
    ) -> tuple[str, ...]:
        """The prompts a call encodes, in mflux's batch order: the prompt alone, or the negative then the prompt.

        mflux 0.20's rule: guidance is read as ``Config`` stores it (None is 0.0); above 1.0 the call runs
        classifier-free guidance with the negative prompt, a blank or missing one replaced by ``" "``.
        """
        g = 0.0 if guidance is None else float(guidance)
        if g <= 1.0:
            return (prompt,)
        negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        return (negative, prompt)

    def _entry(self) -> Any:
        """The registry entry of the model this instance runs, or None for a model assembled under another name."""
        return registry.MODELS.get(self._model)

    def _resolve_guidance(self, guidance: float | None) -> float:
        """``guidance``, or this model's default when None (mflux's per variant; 1.0 for an unregistered name)."""
        if guidance is not None:
            return guidance
        entry = self._entry()
        return (
            1.0
            if entry is None or entry.default_guidance is None
            else float(entry.default_guidance)
        )

    @staticmethod
    def _refuse_blank(prompt: str) -> None:
        """Refuse an empty or blank prompt as a user error.

        Raises:
            DFloatUnsupportedError: ``prompt`` is empty or only whitespace.
        """
        if not prompt.strip():
            raise DFloatUnsupportedError(
                f"prompt {prompt!r}: an empty or blank prompt is refused as a user error (an empty prompt gives "
                "ERNIE-Image's tokenizer no tokens at all)"
            )

    @_pipeline.with_call_caps
    def encode(
        self, prompt: str, *, negative_prompt: str | None = None, guidance: float | None = None
    ) -> None:
        """Encode a call's prompts now (drops a resident set first), so several generations pay the encoder once.

        ``guidance`` left as None takes this model's default, as ``generate_image`` does, so ``encode(p)`` prepares
        the batch ``generate_image(seed, p)`` runs. Runs under the command's memory caps unless a wired limit is set.

        Raises:
            DFloatUnsupportedError: An empty or blank ``prompt``.
        """
        self._refuse_blank(prompt)
        guidance = self._resolve_guidance(guidance)
        self._lifecycle.ensure_embeddings(
            self._batch_key(
                self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
            )
        )

    def _encode_prompts(
        self, *, prompt: str, negative_prompt: str | None, guidance: float
    ) -> tuple[mx.array, mx.array]:
        """Mflux's hook, answered from the batch cache (the encoder is not resident during the loop).

        Raises:
            DFloatIntegrationError: The call's prompts were not encoded first.
        """
        key = self._batch_key(
            self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        )
        if key not in self._batches:
            raise DFloatIntegrationError(f"prompts {key} were not encoded before the call")
        text_bth, text_lens = self._batches[key]
        return text_bth, text_lens

    def _predict(
        self, transformer: Any, text_bth: mx.array, text_lens: mx.array, latents: mx.array
    ) -> Any:
        """Mflux's predict closure, uncompiled on every chip (the seam evaluates at block boundaries).

        Raises:
            DFloatIntegrationError: mflux's factory still compiled it.
        """
        fn = uncompiled(ErnieImage._predict, transformer, text_bth, text_lens, latents)
        self._predict_mode = "uncompiled"
        return fn

    def plan_call(self, *, height: int, width: int, text_tokens: int, batch: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call (sizes rounded down to multiples of 16).

        ``text_tokens`` is the call's longest prompt in tokens (``memory.text_tokens_for``); ``batch`` is 2 when the
        call runs classifier-free guidance (one batch-2 transformer call), else 1, and selects the constants. The set
        stays resident through the VAE decode when the estimate allows it; otherwise the call drops it before decoding
        and the next call reloads it.

        Raises:
            DFloatResourceError: The size is above 1024² or the predicted peak exceeds the budget even with the set
                dropped, and ``fit_check`` is on.
        """
        plan = _pipeline.plan_call_for(
            constants=self._constants[batch],
            sizes=self._sizes,
            largest=self._largest,
            policy=self._policy,
            cache_limit_override=self._cache_limit_override,
            fit_check=self._fit_check,
            budget=self._budget(),
            height=height,
            width=width,
            text_tokens=text_tokens,
            log=log,
        )
        self._plan = plan
        return plan

    @_pipeline.with_call_caps
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
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

        ``guidance``, ``num_inference_steps`` and ``scheduler`` left as None take this model's defaults, mflux's per
        variant: ERNIE-Image 4.0 and 50 steps, which run classifier-free guidance; ERNIE-Image-Turbo 1.0 and 8 steps;
        ``"linear"`` for both. The same call through mflux's own ``ErnieImage`` class defaults to 1.0 and 8 steps for
        both. Above guidance 1.0 a call runs classifier-free guidance as one batch-2 transformer call, with the
        negative prompt or, when it is missing or blank, mflux's ``" "``; unlike the ``mlx-dfloat`` command, the
        Python API lets ERNIE-Image-Turbo run it too, as mflux's API does. Each call runs under the command's memory
        caps unless a wired limit is already set.

        Raises:
            DFloatUnsupportedError: An empty or blank ``prompt`` (a user error), ``image_path`` /
                ``image_strength`` (img2img) or ``pid_decode``.
            DFloatResourceError: The fit check refuses the call (the estimate, or a size above 1024²), or memory stayed
                active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
            DFloatIntegrationError: mflux compiled the step function despite the bypass.
        """
        self._refuse_blank(prompt)
        if image_path is not None or image_strength is not None:
            _pipeline.refuse("image_path/image_strength", "img2img")
        if pid_decode:
            _pipeline.refuse(
                "pid_decode",
                "mflux's alternative image decoder (PiD) loads its own caption encoder next to the compressed set",
            )
        # mflux's defaults per variant (the registry entry); a model assembled from parts under another name takes
        # mflux's class defaults (1.0, 8 steps, "linear").
        entry = self._entry()
        guidance = self._resolve_guidance(guidance)
        if num_inference_steps is None:
            num_inference_steps = 8 if entry is None else entry.default_steps
        if scheduler is None:
            scheduler = (
                "linear"
                if entry is None or entry.default_scheduler is None
                else entry.default_scheduler
            )
        prompts = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        batch = len(prompts)
        text_tokens = emem.text_tokens_for(self.tokenizers["ernie"], prompts)
        plan = self.plan_call(height=height, width=width, text_tokens=text_tokens, batch=batch)
        self._batch, self._text_tokens = batch, text_tokens
        self._tracker.begin("encode")
        try:
            self._lifecycle.ensure_embeddings(self._batch_key(prompts))
        finally:  # a failed encode closes its phase: the watchdog's context must not name it afterwards
            self._tracker.end("encode")
        self._check_encode_peak(plan)
        self._tracker.begin("set_load")
        try:
            self._lifecycle.ensure_set()
        finally:
            self._tracker.end("set_load")
        self._tracker.begin("denoise")  # before set_cache_limit: a raise here must change nothing
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
            _clear_frames(exc)
            if isinstance(exc, DFloatFormatError):
                self._lifecycle.drop_set()  # a corrupt block: the retry starts from a clean load
            raise
        finally:
            mx.set_cache_limit(previous)
            self._tracker.end("denoise")
            self._tracker.end("vae")

    def _check_encode_peak(self, plan: CallPlan) -> None:
        """Warn when the encode held more than the estimate's encode phase (language model + encode term) + slack.

        The estimate counts only the encoder's language-model tensors (``memory.encoder_bytes_used``); an mflux change
        that also loads the vision tower would make it under-predict the encode phase, and so may a prompt much longer
        than the one the encode term was measured on (the warning names the call's prompt length). The measure is
        MLX's peak minus what was active when the phase began, so an encode that first dropped a resident set reads
        low (no warning), never high.
        """
        record = self._tracker.peaks.get("encode")
        if record is None:
            return
        message = emem.encode_peak_warning(
            record["mlx_peak"] - record["active_at_start"],
            emem.encode_peak_bound(plan.estimate.phases["encode"], self._constants[self._batch]),
            text_tokens=self._text_tokens or 0,
        )
        if message is not None:
            warnings.warn(message, stacklevel=3)

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction.
        ``predict`` is how the last call's step function ran (``"uncompiled"``; None before any call). ``cfg_batch``
        and ``text_tokens`` are the last call's (2 when classifier-free guidance ran, as one batch-2 call; the longest
        prompt's tokens). ``sizes["encoders"]`` is the text encoder's language-model bytes, not its file size.
        ``eval_policy_status`` is ``"unmeasured"`` for depth2, which no ERNIE-Image run has measured.
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "family": "ernie",
            "model": self._model,
            "df11": {
                "root": str(self._df11.root),
                "repo_id": self._df11.repo_id,
                "revision": self._df11.revision,
                "config_source": self._ckpt.config_source,
            },
            "base": {
                "root": str(self._base.root),
                "repo_id": self._base.repo_id,
                "revision": self._base.revision,
            },
            "eval_policy": self._policy,
            "eval_policy_status": "measured" if self._policy in MEASURED_POLICIES else "unmeasured",
            "cfg_batch": self._batch,
            "text_tokens": self._text_tokens,
            "predict": self._predict_mode,
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
            "peaks": {"label": "sampled at phase boundaries", **self._tracker.peaks},
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
        _pipeline.refuse("save_model", "the DFloat11 repository is the checkpoint")

    def freeze(self, **kwargs: Any) -> None:
        """Freeze the components that are loaded (a dropped encoder is a stand-in, not a module)."""
        del kwargs
        for module in (self.vae, self.transformer, self.text_encoder):
            if not isinstance(module, _EncoderNotResident):
                module.freeze()
