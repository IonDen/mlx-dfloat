"""``DFloatQwenImage21``: mflux's Qwen-Image 2.1 pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``QwenImage21`` so its loop, scheduler, callbacks, VAE decode and image metadata run
unchanged; only construction and the prelude of ``generate_image`` differ. The text encoder and the compressed
transformer are never resident together (see ``mlx_dfloat.mflux.lifecycle``): prompts are encoded first into
mflux's own prompt cache, which its loop reads without touching the encoder. Each ``generate_image`` call runs under
the memory caps the ``mlx-dfloat`` command installs (unless a wired limit is already in force) and restores MLX's
limits afterwards; only the command adds a watchdog.
"""

import logging
import traceback
import warnings
from collections.abc import Callable
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
from mlx_dfloat.mflux._phases import FamilySizes
from mlx_dfloat.mflux.lifecycle import Lifecycle
from mlx_dfloat.mflux.qwen21 import init as qinit
from mlx_dfloat.mflux.qwen21 import memory as qmem
from mlx_dfloat.mflux.qwen21.init import Qwen21Components, ResolvedRepo
from mlx_dfloat.mflux.qwen21.names import NONBLOCK_GROUPS, qwen21_name_map
from mlx_dfloat.mflux.qwen21.transformer import Qwen21Build, build_transformer, is_eager

require_mflux()
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.qwen21")

# (DF11 repository, base repository), derived from the registry.
MODELS: dict[str, tuple[str, str]] = {
    n: (e.df11_repo, e.base_repo) for n, e in registry.MODELS.items() if e.family == "qwen21"
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
# The policies a Qwen-Image 2.1 run has measured (the de-risk, the MEASURED row, the identity check); depth2 is
# accepted, and the report labels it unmeasured.
MEASURED_POLICIES: frozenset[str] = frozenset({"per-block"})
RETAINED_SLACK_BYTES = 2 * 1024**3
# mflux 0.20.0 ``QwenImage21.generate_image``: its guidance and scheduler defaults.
DEFAULT_GUIDANCE = 1.0
DEFAULT_SCHEDULER = "linear"


def _clear_frames(exc: BaseException) -> None:
    """Release the locals of the frames ``exc`` and the exception it was raised while handling passed through."""
    traceback.clear_frames(exc.__traceback__)
    if exc.__context__ is not None and exc.__context__.__traceback__ is not None:
        traceback.clear_frames(exc.__context__.__traceback__)


class _EncoderNotResident:
    """Takes the text encoder's place while it is dropped: mflux calls the encoder only for a prompt not cached."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: every prompt a call needs is encoded before mflux's loop starts.

        Raises:
            DFloatIntegrationError: Always.
        """
        del args, kwargs
        raise DFloatIntegrationError(
            "the text encoder is not resident: a prompt was not encoded before the call "
            "(generate_image encodes the prompts it needs first; call encode() for any other use)"
        )


class DFloatQwenImage21(QwenImage21):  # type: ignore[misc]  # mflux ships no type information
    """Qwen-Image 2.1 generation from a DFloat11 transformer through mflux.

    The transformer stays compressed and each of its 32 blocks is decoded on the GPU as it runs; the shared
    modulation matrix is decoded once when the set loads. The forward pass runs uncompiled (mflux compiles it; the
    per-block evaluation cannot run inside a compiled function). The decode is lossless.

    The text encoder and the compressed set are never resident together: a new prompt after a generation drops the
    set, reloads the encoder, encodes and reloads the set.
    """

    def __init__(
        self,
        model: str = "qwen-image-2.1",
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
        compressed weights stay lazy until used. ``budget_bytes`` replaces the device's budget in every call's fit
        estimate and VAE decision.

        Raises:
            DFloatUnsupportedError: ``quantize``, ``lora_paths``, ``lora_scales`` or ``bake_lora=False``
                (accepted only to refuse), an eval policy other than ``"per-block"`` / ``"depth2"``, or an unknown
                ``model``.
            DFloatBackendError: The GPU decoder fails its check on the two packaged self-check groups (raised before
                anything is resolved or loaded), or Metal is unavailable.
            DFloatFormatError: The checkpoint is not a Qwen-Image 2.1 DF11 checkpoint (a config-less file of
                another layout included), its block count differs from the model's, or the base lacks a component
                or is another model's.
            DFloatAccessError: A gated repository this account may not read.
            DFloatDependencyError: The ``mlx-dfloat[mflux]`` extra is not installed.
            DFloatIntegrationError: mflux's own weight definition or mapping does not shape the way this
                adapter expects (a name map ambiguity, an uncovered extra).
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
        df11_default, base_default = MODELS[model]
        # The default checkpoint lives on one person's Hub account: pinned to the commit that was verified here.
        # A user's own checkpoint is taken as given.
        pin = None if df11_path else registry.MODELS[model].df11_revision
        df11 = qinit.resolve(df11_path or df11_default, patterns=qinit.DF11_PATTERNS, revision=pin)
        ckpt = open_checkpoint(df11.root)
        # The default base is pinned to the snapshot the recorded runs used; a user's own base is taken as given.
        base_pin = None if base_path else registry.MODELS[model].base_revision
        base = qinit.resolve(
            base_path or base_default, patterns=qinit.BASE_PATTERNS, revision=base_pin
        )
        # mflux builds the published transformer with its constructor defaults (Qwen21Initializer._init_models).
        build = build_transformer(ckpt)
        components = qinit.load_base(base.root)
        self._assemble(
            model=model,
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            build=build,
            sizes=qmem.sizes_for(ckpt, base.root),
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
            budget_bytes=budget_bytes,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatQwenImage21":
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
        components: Qwen21Components,
        build: Qwen21Build,
        sizes: FamilySizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

        _pipeline.check_eval_policy(eval_policy, POLICIES)
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not QwenImage21.__init__: mflux's initializer loads the whole base
        Qwen21Initializer._init_config(
            self, model_config
        )  # prompt_cache, model_config, callbacks, tiling_config
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizers = components.tokenizers
        self.transformer = build.transformer
        self.bits = None
        self._model = model
        self._constants = qmem.CONSTANTS
        self._ckpt, self._df11, self._base, self._sizes = ckpt, df11, base, sizes
        self._shapes, self._nonblock = build.shapes, build.nonblock
        self._decode = decode
        self._names = qwen21_name_map() if name_map is None else name_map
        self._cfg_calls = 1
        self._text_tokens: int | None = None
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
        # The pairs go into mflux's own prompt cache, keyed by the prompt as the caller wrote it: mflux's loop looks
        # them up there before it would touch the (dropped) encoder.
        self._lifecycle = Lifecycle(
            load_encoders=self._load_encoders,
            unload_encoders=self._unload_encoders,
            encode=self._encode,
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache=self.prompt_cache,
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

    def _retained_bound(self) -> int:
        """What may stay active after a drop: the assembly baseline, the cached pairs, the VAE, slack."""
        pairs = sum(int(a.nbytes) for arrays in self.prompt_cache.values() for a in arrays)
        flat: list[tuple[str, mx.array]] = list(tree_flatten(self.vae.parameters()))  # type: ignore[arg-type]
        vae = sum(int(v.nbytes) for _name, v in flat)
        return self._baseline_active + pairs + vae + RETAINED_SLACK_BYTES

    def _load_encoders(self) -> None:
        self.text_encoder = qinit.load_text_encoder(self._base.root)

    def _unload_encoders(self) -> None:
        # Not None: mflux would call it on a cache miss and fail with a TypeError far from the cause.
        self.text_encoder = _EncoderNotResident()

    def _encode(self, prompt: str) -> tuple[mx.array, ...]:
        """The prompt's embeddings and mask, as mflux's ``Qwen21PromptEncoder.encode_prompt`` computes them.

        It is called with a cache of its own, so its normalisation (a blank prompt encoded as ``" "``) stays mflux's;
        the lifecycle stores the pair under the prompt as given.
        """
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
            Qwen21PromptEncoder,
        )

        embeds, mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache={},
            tokenizer=self.tokenizers["qwen21"],
            text_encoder=self.text_encoder,
        )
        return embeds, mask

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
        # Classifier-free guidance calls the transformer twice per step: verify inside each call.
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

    @_pipeline.with_call_caps
    def encode(self, *prompts: str) -> None:
        """Encode prompts now, so several generations pay the encoder reload once (drops a resident set first)."""
        self._lifecycle.ensure_embeddings(*dict.fromkeys(prompts))

    def plan_call(self, *, height: int, width: int, text_tokens: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call at this size (rounded down to multiples of 16, as mflux does).

        ``text_tokens`` is the call's longest prompt in tokens (``memory.prompt_tokens``). The set stays resident
        through the VAE decode when the estimate allows it; otherwise the call drops it before decoding and the next
        call reloads it.

        Raises:
            DFloatResourceError: The size is above 1024² or the predicted peak exceeds the budget even with
                the set dropped, and ``fit_check`` is on.
        """
        plan = _pipeline.plan_call_for(
            constants=self._constants,
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

    def cfg_prompts(
        self, prompt: str, *, negative_prompt: str | None, guidance: float | None
    ) -> tuple[str, ...]:
        """The prompts a call encodes: the prompt, and the negative prompt when classifier-free guidance runs.

        mflux 0.20's rule (``QwenImage21.generate_image``): the negative branch runs, and the transformer is called
        twice per step, only for a guidance above 1.0 together with a non-empty negative prompt.
        """
        if guidance is None or guidance <= 1.0 or not negative_prompt:
            return (prompt,)
        return (prompt, negative_prompt)

    @_pipeline.with_call_caps
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 40,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = DEFAULT_GUIDANCE,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = DEFAULT_SCHEDULER,
        negative_prompt: str | None = None,
    ) -> Any:
        """Mflux's ``generate_image`` behind a prelude: refusals, cache limit, fit check, prompt encoding, set load.

        ``guidance`` or ``scheduler`` given as None take mflux's defaults (1.0, ``"linear"``). As in mflux,
        classifier-free guidance runs only for a guidance above 1.0 with a non-empty ``negative_prompt``.

        Raises:
            DFloatUnsupportedError: ``image_path`` / ``image_strength`` (img2img).
            DFloatResourceError: The fit check refuses the call (the estimate, or a size above 1024²), or
                memory stayed active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
            DFloatIntegrationError: mflux compiled the transformer's forward pass.
        """
        if image_path is not None or image_strength is not None:
            _pipeline.refuse("image_path/image_strength", "img2img")
        guidance = DEFAULT_GUIDANCE if guidance is None else guidance
        scheduler = DEFAULT_SCHEDULER if scheduler is None else scheduler
        prompts = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        tokenizer = self.tokenizers["qwen21"]
        text_tokens = max(qmem.prompt_tokens(tokenizer, p) for p in prompts)
        plan = self.plan_call(height=height, width=width, text_tokens=text_tokens)
        self._cfg_calls, self._text_tokens = len(prompts), text_tokens
        self._tracker.begin("encode")
        try:
            self.encode(*prompts)
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

        The estimate counts only the encoder's language-model tensors (``memory.encoder_bytes_used``); an mflux
        change that also loads the vision tower or the output head would make it under-predict the encode phase, and
        so may a prompt much longer than the 33 tokens the encode term was measured on (the warning names the call's
        prompt length; ``memory.encode_peak_warning``). The measure is MLX's peak minus what was active when the phase began, so an encode that first dropped a
        resident set reads low (no warning), never high.
        """
        record = self._tracker.peaks.get("encode")
        if record is None:
            return
        message = qmem.encode_peak_warning(
            record["mlx_peak"] - record["active_at_start"],
            qmem.encode_peak_bound(plan.estimate.phases["encode"], self._constants),
            text_tokens=self._text_tokens or 0,
        )
        if message is not None:
            warnings.warn(message, stacklevel=3)

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction.
        ``predict`` says whether the transformer's forward pass is the plain method (``"uncompiled"``) or a compiled
        one (``"compiled"``, which the next call refuses), checked when the report is built.
        ``cfg_calls_per_step`` and ``text_tokens`` are the last call's (2 calls when classifier-free guidance ran;
        the longest prompt's tokens). ``sizes["encoders"]`` is the text encoder's language-model bytes, not its file
        size. ``df11["config_source"]`` says where the checkpoint's config came from (a config-less file names the
        pinned layout it matched). ``eval_policy_status`` is ``"unmeasured"`` for depth2, which no Qwen-Image 2.1 run
        has measured.
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "family": "qwen21",
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
            "cfg_calls_per_step": self._cfg_calls,
            "text_tokens": self._text_tokens,
            "predict": "uncompiled" if is_eager(self.transformer) else "compiled",
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
