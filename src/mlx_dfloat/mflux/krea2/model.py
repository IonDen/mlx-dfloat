"""``DFloatKrea2``: mflux's Krea 2 pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``Krea2`` so its loop, scheduler, sampler, callbacks, VAE decode and image metadata run
unchanged; only construction and the prelude of ``generate_image`` differ. The text encoder and the compressed
transformer are never resident together (see ``mlx_dfloat.mflux.lifecycle``): a call's prompts are encoded first into
a per-prompt cache, which the loop reads without touching the encoder.
"""

import logging
import sys
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
from mlx_dfloat.mflux._compile import uncompiled
from mlx_dfloat.mflux._phases import FamilySizes
from mlx_dfloat.mflux.krea2 import init as kinit
from mlx_dfloat.mflux.krea2 import memory as kmem
from mlx_dfloat.mflux.krea2.init import Krea2Components, ResolvedRepo
from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS, check_variant, krea2_name_map
from mlx_dfloat.mflux.krea2.transformer import (
    Krea2Build,
    PerCallNonBlock,
    build_transformer,
    seam_transformer_class,
)
from mlx_dfloat.mflux.lifecycle import Lifecycle

require_mflux()
from mflux.models.krea2.variants.txt2img.krea2 import Krea2  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.krea2")

# (DF11 repository, base repository), derived from the registry.
MODELS: dict[str, tuple[str, str]] = {
    n: (e.df11_repo, e.base_repo) for n, e in registry.MODELS.items() if e.family == "krea2"
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
# The policies a Krea 2 run has measured; depth2 is accepted, and the report labels it unmeasured.
MEASURED_POLICIES: frozenset[str] = frozenset({"per-block"})
# What else may stay active across calls besides the assembly baseline, the cached embeddings and the VAE. It is
# larger than one transformer call's decoded non-block weights (1_325_400_064 B on the published model), so the
# retained-memory check does not catch one call's non-block decode left referenced after a drop; the per-call class's
# own cleanup and its tests guard that.
RETAINED_SLACK_BYTES = 2 * 1024**3


def _clear_frames(exc: BaseException, *, outer: BaseException | None = None) -> None:
    """Release the locals of the frames ``exc`` and every exception in its chain (causes and contexts) passed through.

    ``outer`` is the exception the caller was already handling when our call began (``sys.exc_info()`` at entry): the
    walk stops there, so the caller's own frames keep their locals. The walk keeps a seen-set, so a chain that loops
    back on itself ends.
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


class _EncoderNotResident:
    """Takes the text encoder's place while it is dropped: mflux calls the encoder only for a prompt not cached."""

    def _refuse(self) -> Any:
        raise DFloatIntegrationError(
            "the text encoder is not resident: a prompt was not encoded before the call "
            "(generate_image encodes the prompts it needs first; call encode() for any other use)"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: every prompt a call needs is encoded before mflux's loop starts.

        Raises:
            DFloatIntegrationError: Always.
        """
        del args, kwargs
        return self._refuse()

    def get_prompt_embeds(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse, as ``__call__`` does (mflux's prompt encoder calls this method).

        Raises:
            DFloatIntegrationError: Always.
        """
        del args, kwargs
        return self._refuse()


class DFloatKrea2(Krea2):  # type: ignore[misc]  # mflux ships no type information
    """Krea 2 Raw and Krea 2 Turbo generation from a DFloat11 transformer through mflux.

    The transformer stays compressed and each of its 28 blocks is decoded on the GPU as it runs; the seven non-block
    groups (the four text-fusion blocks, the timestep MLP and projection, the text MLP) are decoded at the start of
    each transformer call (twice per step when classifier-free guidance runs) and released before the first block's
    decode, once its inputs are evaluated, so they are not resident with the set between calls. Each call starts with
    MLX's buffer cache emptied and the OS footprint drained, so a guided step's second call does not build on the
    first call's cached activations (MLX would otherwise trim its cache only at the memory limit); at 1024² this
    keeps a guided Krea 2 Raw step's denoise peak at 21.14 GiB. The step function runs uncompiled on every chip (the
    per-block evaluation cannot run inside a compiled function). The decode is lossless.

    The text encoder and the compressed set are never resident together: a new prompt after a generation drops the
    set, reloads the encoder, encodes and reloads the set.
    """

    def __init__(
        self,
        model: str = "krea-2",
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
            DFloatFormatError: The checkpoint is not a Krea 2 DF11 checkpoint (the other Krea 2 model's included), its
                block count differs from the model's, or the base lacks a component or is another model's.
            DFloatAccessError: A gated repository this account may not read (both bases are gated).
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
        df11 = kinit.resolve(df11_path or df11_default, patterns=kinit.DF11_PATTERNS, revision=pin)
        ckpt = open_checkpoint(df11.root)
        # The other Krea 2 model's published file is refused here, before the base (about 16 GB) is resolved.
        check_variant(model, ckpt)
        base_pin = None if base_path else entry.base_revision
        base = kinit.resolve(
            base_path or base_default, patterns=kinit.BASE_PATTERNS, revision=base_pin
        )
        # mflux builds the published transformer with its constructor defaults (Krea2Initializer); the variant check
        # refuses the other Krea 2 model's published file. The non-block groups are decoded per call: with them
        # resident the denoise step of Krea 2 Raw at 1024² measured over the 32 GB fit line.
        build = build_transformer(
            ckpt, model=model, transformer_class=seam_transformer_class(per_call=True)
        )
        components = kinit.load_base(base.root)
        self._assemble(
            model=model,
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            build=build,
            sizes=kmem.sizes_for(ckpt, base.root, nonblock_per_call=True),
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
            budget_bytes=budget_bytes,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatKrea2":
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
        components: Krea2Components,
        build: Krea2Build,
        sizes: FamilySizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        from mflux.models.krea2.krea2_initializer import Krea2Initializer

        _pipeline.check_eval_policy(eval_policy, POLICIES)
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not Krea2.__init__: mflux's initializer loads the whole base
        # prompt_cache, model_config, callbacks, tiling_config = None (krea2_initializer.py:47-54)
        Krea2Initializer._init_config(self, model_config)
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizers = components.tokenizers
        self.transformer = build.transformer
        self.bits = None
        self.lora_paths: list[str] = []
        self.lora_scales: list[float] = []
        self._model = model
        self._constants = kmem.CONSTANTS
        self._ckpt, self._df11, self._base, self._sizes = ckpt, df11, base, sizes
        self._shapes, self._nonblock = build.shapes, build.nonblock
        # The build decides: a per-call transformer decodes the non-block groups at each call, any other keeps them
        # decoded while the set is resident.
        self._per_call = isinstance(build.transformer, PerCallNonBlock)
        if any(g in ckpt.groups for g in NONBLOCK_GROUPS) and self._per_call != (
            sizes.nonblock == 0
        ):
            mode = "per-call" if self._per_call else "resident"
            raise DFloatIntegrationError(
                f"the build runs the {mode} non-block decode but the sizes say nonblock={sizes.nonblock}: a "
                "per-call build plans with no resident non-block bytes (sizes_for(..., nonblock_per_call=True)), "
                "a resident one with them"
            )
        self._decode = decode
        self._names = krea2_name_map() if name_map is None else name_map
        self._cfg_calls = 1
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
        # The embeddings, one entry per prompt as encoded; mflux's own prompt_cache (keyed by the call's prompt,
        # negative prompt and guidance) stays empty, and the _encode_prompts override answers from this one.
        self._embeds: dict[str, tuple[mx.array, ...]] = {}
        self._lifecycle = Lifecycle(
            load_encoders=self._load_encoders,
            unload_encoders=self._unload_encoders,
            encode=self._encode,
            load_set=self._load_set,
            unload_set=self._unload_set,
            prompt_cache=self._embeds,
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
        """What may stay active after a drop: the assembly baseline, the cached embeddings, the VAE, slack."""
        embeds = sum(int(a.nbytes) for arrays in self._embeds.values() for a in arrays)
        flat: list[tuple[str, mx.array]] = list(tree_flatten(self.vae.parameters()))  # type: ignore[arg-type]
        vae = sum(int(v.nbytes) for _name, v in flat)
        return self._baseline_active + embeds + vae + RETAINED_SLACK_BYTES

    def _load_encoders(self) -> None:
        self.text_encoder = kinit.load_text_encoder(self._base.root)

    def _unload_encoders(self) -> None:
        # Not None: mflux would call it on a cache miss and fail with a TypeError far from the cause.
        self.text_encoder = _EncoderNotResident()

    def _encode(self, prompt: str) -> tuple[mx.array, ...]:
        """The prompt's embeddings, as mflux's ``Krea2PromptEncoder.encode_prompt`` computes them."""
        from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder

        return (
            Krea2PromptEncoder.encode_prompt(prompt, self.tokenizers["qwen3vl"], self.text_encoder),
        )

    def _load_set(self) -> None:
        outer = sys.exc_info()[1]
        try:
            self._install_set()
        except BaseException as exc:
            # The traceback's frames hold the compressed set (_install_set's ``resident``) and any decoded non-block
            # weight for as long as the caller keeps the exception; the retry's memory check would then refuse.
            _clear_frames(exc, outer=outer)
            self._unload_set()  # nothing half-installed: the lifecycle believes no set is resident
            raise

    def _install_set(self) -> None:
        resident = load_resident_set(self._ckpt)
        nonblock = {g: self._ckpt.groups[g].matrix_names for g in NONBLOCK_GROUPS if g in resident}
        if self._per_call:
            # Bound compressed; each transformer call decodes them and releases them after its first block.
            self.transformer.bind_nonblock(
                {g: resident[g] for g in nonblock}, self._nonblock, self._names, decode=self._decode
            )
        else:
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
        """Encode prompts now, so several generations pay the encoder reload once (drops a resident set first).

        Runs under the command's memory caps unless a wired limit is already set.

        Raises:
            DFloatUnsupportedError: An empty prompt (Krea 2's tokenizer gives it no tokens at all). A blank one such
                as ``" "``, mflux's own negative prompt, is encoded.
        """
        if any(p == "" for p in prompts):
            raise DFloatUnsupportedError(
                "an empty prompt is refused: Krea 2's tokenizer gives it no tokens at all"
            )
        self._lifecycle.ensure_embeddings(*dict.fromkeys(prompts))

    @staticmethod
    def cfg_prompts(
        prompt: str, *, negative_prompt: str | None, guidance: float
    ) -> tuple[str, ...]:
        """The prompts a call encodes: the prompt, then the negative one when classifier-free guidance runs.

        mflux 0.20's rule for Krea 2: the negative branch runs for any guidance other than 1.0 (below 1.0 too); a
        blank or missing negative prompt is encoded as ``" "``.
        """
        if guidance == 1.0:
            return (prompt,)
        negative = negative_prompt if negative_prompt and negative_prompt.strip() else " "
        return (prompt, negative)

    def _encode_prompts(
        self, *, prompt: str, negative_prompt: str | None, guidance: float
    ) -> tuple[mx.array, mx.array | None]:
        """Mflux's hook, answered from the per-prompt cache (the encoder is not resident during the loop).

        Raises:
            DFloatIntegrationError: A prompt of the call was not encoded first.
        """
        prompts = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        missing = [p for p in prompts if p not in self._embeds]
        if missing:
            raise DFloatIntegrationError(f"prompts {missing} were not encoded before the call")
        embeds = self._embeds[prompts[0]][0]
        return embeds, (self._embeds[prompts[1]][0] if len(prompts) == 2 else None)

    def _predict(
        self, transformer: Any, embeds: mx.array, neg_embeds: mx.array | None, guidance: float
    ) -> Any:
        """Mflux's predict closure, uncompiled on every chip (the seam evaluates at block boundaries).

        Raises:
            DFloatIntegrationError: mflux's factory still compiled it.
        """
        fn = uncompiled(Krea2._predict, transformer, embeds, neg_embeds, guidance)
        self._predict_mode = "uncompiled"
        return fn

    def _entry(self) -> Any:
        """The registry entry of the model this instance runs, or None for a model assembled under another name."""
        return registry.MODELS.get(self._model)

    def plan_call(self, *, height: int, width: int, text_tokens: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call (sizes rounded down to multiples of 16).

        ``text_tokens`` is the call's longer prompt in tokens (``memory.text_tokens_for``): a guided step runs one
        transformer call per prompt. The set stays resident through the VAE decode when the estimate allows it;
        otherwise the call drops it before decoding and the next call reloads it.

        Raises:
            DFloatResourceError: The size is above 1024² or the predicted peak exceeds the budget even with the set
                dropped, and ``fit_check`` is on.
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

    @staticmethod
    def _refuse_blank(prompt: str) -> None:
        """Refuse an empty or blank prompt as a user error.

        Raises:
            DFloatUnsupportedError: ``prompt`` is empty or only whitespace.
        """
        if not prompt.strip():
            raise DFloatUnsupportedError(
                f"prompt {prompt!r}: an empty or blank prompt is refused as a user error (an empty prompt gives "
                "Krea 2's tokenizer no tokens at all)"
            )

    @staticmethod
    def _resolve_scheduler(scheduler: str | None) -> str:
        """Mflux's scheduler resolution (None and ``"linear"`` become ``"er_sde"``), with an unknown name refused here.

        Raises:
            DFloatUnsupportedError: A scheduler Krea 2's pipeline does not run.
        """
        try:
            return str(Krea2._resolve_scheduler(scheduler))
        except ValueError:
            raise DFloatUnsupportedError(
                f"scheduler {scheduler!r}: Krea 2 runs er_sde, euler, linear (linear is er_sde)"
            ) from None

    @_pipeline.with_call_caps
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> Any:
        """Mflux's ``generate_image`` behind a prelude: refusals, cache limit, fit check, prompt encoding, set load.

        ``num_inference_steps``, ``guidance`` and ``scheduler`` left as None take this model's defaults, mflux's per
        variant: Krea 2 Raw 25 steps at guidance 1.0 and Krea 2 Turbo 8 steps at guidance 1.0, so one transformer call
        per step; ``"er_sde"`` for both. Krea 2 Raw's model card runs 52 steps at guidance 3.5. Classifier-free
        guidance runs for any guidance other than 1.0, below 1.0 too, as two transformer calls per step, with the
        negative prompt or, when it is missing or blank, mflux's ``" "``. Each call runs under the command's memory
        caps unless a wired limit is already set.

        Raises:
            DFloatUnsupportedError: An empty or blank ``prompt`` (a user error), a scheduler other than ``er_sde``,
                ``euler`` or ``linear``, ``image_path`` / ``image_strength`` (img2img) or ``pid_decode``; all before
                anything loads.
            DFloatResourceError: The fit check refuses the call (the estimate, or a size above 1024²), or memory stayed
                active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
            DFloatIntegrationError: mflux compiled the step function despite the bypass.
        """
        outer = sys.exc_info()[
            1
        ]  # the exception our caller is handling, if any: its frames are not ours to clear
        self._refuse_blank(prompt)
        if image_path is not None or image_strength is not None:
            _pipeline.refuse("image_path/image_strength", "img2img")
        if pid_decode:
            _pipeline.refuse(
                "pid_decode",
                "mflux's alternative image decoder (PiD) loads its own caption encoder next to the compressed set",
            )
        entry = self._entry()
        if guidance is None:
            guidance = (
                1.0 if entry is None or entry.default_guidance is None else entry.default_guidance
            )
        if num_inference_steps is None:
            num_inference_steps = 8 if entry is None else entry.default_steps
        if scheduler is None:
            scheduler = (
                "er_sde"
                if entry is None or entry.default_scheduler is None
                else entry.default_scheduler
            )
        # Before anything loads; mflux resolves the name again inside its own generate_image, the same way.
        self._resolve_scheduler(scheduler)
        prompts = self.cfg_prompts(prompt, negative_prompt=negative_prompt, guidance=guidance)
        text_tokens = kmem.text_tokens_for(self.tokenizers["qwen3vl"], prompts)
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
                negative_prompt=negative_prompt,
                image_path=None,
                image_strength=None,
                scheduler=scheduler,
                pid_decode=False,
                pid_degrade_sigma=pid_degrade_sigma,
            )
        except BaseException as exc:
            # The traceback keeps the seam frames, and with them decoded weights or the resident set, alive for as
            # long as the caller holds the exception (and past drop_set's own gc.collect()). Clear them for any
            # exception and every one in its chain raised in this call (not the one the caller was handling).
            _clear_frames(exc, outer=outer)
            if isinstance(exc, DFloatFormatError):
                self._lifecycle.drop_set()  # a corrupt block: the retry starts from a clean load
            raise
        finally:
            mx.set_cache_limit(previous)
            self._tracker.end("denoise")
            self._tracker.end("vae")

    def _check_encode_peak(self, plan: CallPlan) -> None:
        """Warn when the encode held more than the estimate's encode phase (evaluated language model + term) + slack.

        The estimate counts only the encoder's evaluated language-model tensors (``memory.encoder_bytes_used``); an
        mflux change that also loads the vision tower would make it under-predict the encode phase, and so may a prompt
        much longer than the one the encode term was measured on (``memory.encode_peak_warning``). The measure is MLX's
        peak minus what was active when the phase began, so an encode that first dropped a resident set reads low (no
        warning), never high.
        """
        record = self._tracker.peaks.get("encode")
        if record is None:
            return
        message = kmem.encode_peak_warning(
            record["mlx_peak"] - record["active_at_start"],
            kmem.encode_peak_bound(plan.estimate.phases["encode"], self._constants),
            text_tokens=self._text_tokens or 0,
        )
        if message is not None:
            warnings.warn(message, stacklevel=3)

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction. ``predict``
        is how the last call's step function ran (``"uncompiled"``; None before any call). ``cfg_calls_per_step`` and
        ``text_tokens`` are the last call's (2 calls when classifier-free guidance ran; the longer prompt's tokens).
        ``sizes["encoders"]`` is the text encoder's evaluated language-model bytes, not its file size. ``layout`` says
        where the checkpoint's config came from (the pinned layout a published file matched). ``nonblock`` is
        ``"per-call"`` (the seven non-block groups decoded at each transformer call; ``sizes["nonblock"]`` is 0) or
        ``"resident"`` (decoded once when the set loads).
        ``eval_policy_status`` is ``"unmeasured"`` for depth2, which no Krea 2 run has measured.
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "family": "krea2",
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
            "layout": self._ckpt.config_source,
            "nonblock": "per-call" if self._per_call else "resident",
            "eval_policy": self._policy,
            "eval_policy_status": "measured" if self._policy in MEASURED_POLICIES else "unmeasured",
            "cfg_calls_per_step": self._cfg_calls,
            "text_tokens": self._text_tokens,
            "predict": self._predict_mode,
            "nonblock_groups": sorted(g for g in NONBLOCK_GROUPS if g in self._ckpt.groups),
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
