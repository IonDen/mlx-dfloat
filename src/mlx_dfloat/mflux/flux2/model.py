"""``DFloatFlux2Klein``: mflux's FLUX.2 Klein pipeline over a DFloat11 transformer, decoded one block at a time.

The class subclasses mflux's ``Flux2Klein`` so its loop, scheduler, callbacks, VAE decode and image metadata run
unchanged; only construction and the prelude of ``generate_image`` differ. The text encoder and the compressed
transformer are never resident together (see ``mlx_dfloat.mflux.lifecycle``). Each ``generate_image`` call runs
under the memory caps the ``mlx-dfloat`` command installs (unless a wired limit is already in force) and restores
MLX's limits afterwards; only the command adds a watchdog.
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
from mlx_dfloat.mflux._compile import predict_mode, uncompiled
from mlx_dfloat.mflux._phases import FamilySizes
from mlx_dfloat.mflux.flux2 import init as finit
from mlx_dfloat.mflux.flux2 import memory as kmem
from mlx_dfloat.mflux.flux2.init import Flux2Components, ResolvedRepo
from mlx_dfloat.mflux.flux2.names import NONBLOCK_GROUPS, klein_name_map
from mlx_dfloat.mflux.flux2.transformer import Flux2Build, build_transformer, size_key
from mlx_dfloat.mflux.lifecycle import Lifecycle

require_mflux()
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein  # noqa: E402

log = logging.getLogger("mlx_dfloat.mflux.flux2")

# (DF11 repository, base repository), derived from the registry.
MODELS: dict[str, tuple[str, str]] = {
    n: (e.df11_repo, e.base_repo) for n, e in registry.MODELS.items() if e.family == "flux2"
}
POLICIES: tuple[str, ...] = ("per-block", "depth2")
RETAINED_SLACK_BYTES = 2 * 1024**3
# What a prompt encode may hold beyond the estimate's encode phase before the model warns. The 1024² trial runs
# (2026-10-08) measured MLX's encode peak minus what was active at the phase's start: 4B 6_635_239_256 B, 9B
# 11_997_978_456 B, i.e. 407_057_752 / 334_203_224 B over the used encoder layers. The bound (used layers + the
# call's allowance, at least 500_000_000 B, + this slack) keeps those runs under it at every size, and an encode that
# runs all 36 layers over it at 1024² (4B: 9 more layers of 201_861_632 B).
ENCODE_SLACK_BYTES = 256 * 1024**2
# mflux 0.20.0 ``Flux2Klein.generate_image``: the negative prompt it encodes, its guidance and scheduler defaults.
NEGATIVE_PROMPT = " "
DEFAULT_GUIDANCE = 1.0
DEFAULT_SCHEDULER = "flow_match_euler_discrete"


def _clear_frames(exc: BaseException) -> None:
    """Release the locals of the frames ``exc`` and the exception it was raised while handling passed through."""
    traceback.clear_frames(exc.__traceback__)
    if exc.__context__ is not None and exc.__context__.__traceback__ is not None:
        traceback.clear_frames(exc.__context__.__traceback__)


class DFloatFlux2Klein(Flux2Klein):  # type: ignore[misc]  # mflux ships no type information
    """FLUX.2 Klein (base and distilled, 4B and 9B) generation from a DFloat11 transformer through mflux.

    The transformer stays compressed and each block is decoded on the GPU as it runs; the five non-block matrices
    (the three modulations, the context embedder, the output norm's linear) are decoded once when the set loads.
    The decode is lossless; how far that has been checked differs per model. For FLUX.2-klein-base-4B, a 1024²
    generation's final latents and decoded pixels are bit-identical to the same transformer run block by block from
    its BF16 weights. For the distilled 4B and both 9B models, the GPU decode matches the CPU reference decoder on
    every matrix and sampled groups match the BF16 original bit for bit; no latents comparison has been run.

    The text encoder and the compressed set are never resident together: a new prompt after a generation drops the
    set, reloads the encoder, encodes and reloads the set.

    The distilled models (``flux2-klein-4b``, ``flux2-klein-9b``) are made for guidance 1.0. As in mflux's own Python
    API, ``generate_image`` does not enforce that; the ``mlx-dfloat`` command refuses another guidance for them.
    """

    def __init__(
        self,
        model: str = "flux2-klein-4b",
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

        The checkpoint headers, the base's tokenizer and the transformer's extras are read; the encoder, VAE and
        compressed weights stay lazy until used. ``budget_bytes`` replaces the device's budget in every call's fit
        estimate and VAE decision.

        Raises:
            DFloatUnsupportedError: ``quantize``, ``lora_paths``, ``lora_scales`` or ``bake_lora=False``
                (accepted only to refuse), an eval policy other than ``"per-block"`` / ``"depth2"``, the KV-cache
                model, or an unknown ``model``.
            DFloatBackendError: The GPU decoder fails its check on the two packaged self-check groups (raised before
                anything is resolved or loaded), or Metal is unavailable.
            DFloatFormatError: The checkpoint is not a FLUX.2 Klein DF11 checkpoint, its block counts differ from
                the model's, or the base lacks a component or is of another size.
            DFloatAccessError: A gated repository this account may not read.
            DFloatDependencyError: The ``mlx-dfloat[mflux]`` extra is not installed.
            DFloatIntegrationError: mflux's own weight definition or mapping does not shape the way this
                adapter expects (a name map ambiguity, an uncovered extra).
        """
        _pipeline.refuse_construction_args(
            quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales, bake_lora=bake_lora
        )
        _pipeline.check_eval_policy(eval_policy, POLICIES)
        if "kv" in model.lower():
            raise DFloatUnsupportedError(
                f"model {model!r}: FLUX.2 Klein's KV-cache variant is not on the DFloat11 path"
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
        df11 = finit.resolve(df11_path or df11_default, patterns=finit.DF11_PATTERNS, revision=pin)
        ckpt = open_checkpoint(df11.root)
        base = finit.resolve(base_path or base_default, patterns=finit.BASE_PATTERNS)
        build = build_transformer(ckpt, transformer_overrides=model_config.transformer_overrides)
        components = finit.load_base(base.root, model_config)
        self._assemble(
            model=model,
            model_config=model_config,
            ckpt=ckpt,
            df11=df11,
            base=base,
            components=components,
            build=build,
            sizes=kmem.sizes_for(ckpt, base.root),
            eval_policy=eval_policy,
            cache_limit=cache_limit,
            fit_check=fit_check,
            budget_bytes=budget_bytes,
        )

    @classmethod
    def _from_parts(cls, **parts: Any) -> "DFloatFlux2Klein":
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
        components: Flux2Components,
        build: Flux2Build,
        sizes: FamilySizes,
        name_map: NameMap | None = None,
        eval_policy: str = "per-block",
        cache_limit: int | None = None,
        fit_check: bool = True,
        budget_bytes: int | None = None,
        decode: Callable[[MxGroup], DecodeResult] | None = None,
    ) -> None:
        from mflux.models.flux2.flux2_initializer import Flux2Initializer

        _pipeline.check_eval_policy(eval_policy, POLICIES)
        nn.Module.__init__(self)  # type: ignore[attr-defined]  # not Flux2Klein.__init__: mflux's initializer loads the whole base
        Flux2Initializer._init_config(
            self, model_config
        )  # prompt_cache, model_config, callbacks, tiling_config
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.tokenizers = components.tokenizers
        self.transformer = build.transformer
        self.bits = None
        self.lora_paths: list[str] = []
        self.lora_scales: list[float] = []
        self._embeddings: dict[str, tuple[mx.array, ...]] = {}
        self._model = model
        self._size = size_key(model_config)
        self._constants = kmem.CONSTANTS[self._size]
        self._ckpt, self._df11, self._base, self._sizes = ckpt, df11, base, sizes
        self._shapes, self._nonblock = build.shapes, build.nonblock
        self._decode = decode
        self._names = klein_name_map() if name_map is None else name_map
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
        self._tracker = _pipeline.PhaseTracker()
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
        embeddings = sum(int(a.nbytes) for arrays in self._embeddings.values() for a in arrays)
        flat: list[tuple[str, mx.array]] = list(tree_flatten(self.vae.parameters()))  # type: ignore[arg-type]
        vae = sum(int(v.nbytes) for _name, v in flat)
        return self._baseline_active + embeddings + vae + RETAINED_SLACK_BYTES

    def _load_encoders(self) -> None:
        self.text_encoder = finit.load_text_encoder(self._base.root, self.model_config)

    def _unload_encoders(self) -> None:
        self.text_encoder = None

    def _encode(self, prompt: str) -> tuple[mx.array, ...]:
        """The prompt's embeddings and text ids, as ``Flux2Klein._encode_prompt_pair`` computes them.

        mflux 0.20.0 hard-codes 512 tokens and hidden states (9, 18, 27) there; the model config's token count and
        ``memory.TEXT_ENCODER_OUT_LAYERS`` carry the same values (a test pins them to mflux's source).
        """
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

        embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=kmem.text_tokens(self.model_config),
            text_encoder_out_layers=kmem.TEXT_ENCODER_OUT_LAYERS,
        )
        return embeds, text_ids

    def _load_set(self) -> None:
        try:
            self._install_set()
        except BaseException as exc:
            # The traceback's frames hold the compressed set (_install_set's ``resident``) and any decoded non-block
            # weights for as long as the caller keeps the exception; the retry's memory check would then refuse.
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
        # A base model with guidance calls the transformer twice per step: verify inside each call.
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
        return uncompiled(Flux2Klein._predict, transformer)

    # --- phases -----------------------------------------------------------------------------------

    @property
    def open_phase(self) -> str | None:
        """The phase running now (``encode``, ``set_load``, ``denoise`` or ``vae``); None outside them."""
        return self._tracker.open_phase

    def _budget(self) -> int:
        """The fit budget: the constructor's ``budget_bytes``, else the device's (read at each call)."""
        return self._budget_override if self._budget_override is not None else budget_bytes()

    # --- the public surface -----------------------------------------------------------------------

    def encode(self, *prompts: str) -> None:
        """Encode prompts now, so several generations pay the encoder reload once (drops a resident set first)."""
        self._lifecycle.ensure_embeddings(*dict.fromkeys(prompts))

    def plan_call(self, *, height: int, width: int) -> CallPlan:
        """The cache limit, the fit estimate and the VAE strategy of a call at this size (rounded down to multiples of 16, as mflux does).

        The set stays resident through the VAE decode when the estimate allows it; otherwise the call drops it
        before decoding and the next call reloads it. Sizes above 1024² were not measured on this path; the
        estimate there is an extrapolation.

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
            text_tokens=kmem.text_tokens(self.model_config),
            log=log,
        )
        self._plan = plan
        return plan

    def cfg_prompts(self, prompt: str, *, guidance: float | None) -> tuple[str, ...]:
        """The prompts a call encodes: the prompt, and mflux's blank negative ``" "`` when guidance is above 1.

        mflux 0.20's rule (``Flux2Klein.generate_image`` and ``_encode_prompt_pair``): the negative branch runs, and
        the transformer is called twice per step, only for a guidance above 1.0.
        """
        if guidance is None or guidance <= 1.0:
            return (prompt,)
        return (prompt, NEGATIVE_PROMPT)

    def _encode_prompt_pair(
        self, *, prompt: str, negative_prompt: str | None, guidance: float | None
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        """Mflux's hook, answered from the cache (the encoder is not resident during the loop).

        Raises:
            DFloatIntegrationError: A prompt this call needs was not encoded first.
        """
        negative = guidance is not None and guidance > 1.0 and negative_prompt is not None
        wanted = [prompt, negative_prompt] if negative else [prompt]
        missing = [p for p in wanted if p not in self._embeddings]
        if missing:
            raise DFloatIntegrationError(f"prompt {missing[0]!r} was not encoded before the call")
        embeds, text_ids = self._embeddings[prompt]
        if not negative:
            return embeds, text_ids, None, None
        negative_embeds, negative_ids = self._embeddings[str(negative_prompt)]
        return embeds, text_ids, negative_embeds, negative_ids

    @_pipeline.with_call_caps
    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = DEFAULT_GUIDANCE,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = DEFAULT_SCHEDULER,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> Any:
        """Mflux's ``generate_image`` behind a prelude: refusals, cache limit, fit check, prompt encoding, set load.

        ``guidance`` or ``scheduler`` given as None take mflux's defaults (1.0, ``"flow_match_euler_discrete"``). A
        guidance above 1.0 runs classifier-free guidance against mflux's blank negative prompt, on a distilled model
        too (mflux's Python API allows it; only the ``mlx-dfloat`` command refuses it there).

        Raises:
            DFloatUnsupportedError: ``image_path`` / ``image_strength`` (img2img) or ``pid_decode``.
            DFloatResourceError: The fit check refuses the call (the estimate, or a size above 1024²), or
                memory stayed active after a drop.
            DFloatFormatError: A block's decode reported an error; the set is dropped for a clean retry.
        """
        if image_path is not None or image_strength is not None:
            _pipeline.refuse("image_path/image_strength", "img2img")
        if pid_decode:
            _pipeline.refuse(
                "pid_decode",
                "mflux's alternative image decoder (PiD) loads its own caption encoder next to the compressed set",
            )
        guidance = DEFAULT_GUIDANCE if guidance is None else guidance
        scheduler = DEFAULT_SCHEDULER if scheduler is None else scheduler
        plan = self.plan_call(height=height, width=width)
        prompts = self.cfg_prompts(prompt, guidance=guidance)
        self._cfg_calls = len(prompts)
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
        """Warn when the encode held more than the estimate's encode phase (used encoder layers + allowance) + slack.

        Klein's estimate counts only the encoder layers its prompt embedding reads (``memory.encoder_bytes_used``);
        an mflux change that evaluates every layer would make the estimate under-predict the encode phase. The
        measure is MLX's peak minus what was active when the phase began, so an encode that first dropped a resident
        set reads low (no warning), never high.
        """
        record = self._tracker.peaks.get("encode")
        if record is None:
            return
        held = record["mlx_peak"] - record["active_at_start"]
        bound = plan.estimate.phases["encode"] - self._constants.overhead_bytes + ENCODE_SLACK_BYTES
        if held > bound:
            warnings.warn(
                f"the prompt encode held {held / 1024**3:.2f} GiB, over the {bound / 1024**3:.2f} GiB the fit "
                "estimate allows: mflux may now evaluate every encoder layer (FLUX.2 Klein reads hidden states "
                f"{', '.join(map(str, kmem.TEXT_ENCODER_OUT_LAYERS))} only), so the estimate under-predicts the "
                "encode phase",
                stacklevel=3,
            )

    def report(self) -> dict[str, Any]:
        """What this model is and what its calls cost: repositories, policy, limits, estimate, peaks, counts, versions.

        Peaks are sampled at phase boundaries (labelled ``"sampled"``); the fit estimate is a prediction (its
        constants are measured at 1024² per size). ``predict`` is what mflux's step factory returns under this
        class's compile bypass, checked when the report is built (``"uncompiled"``, or ``"compiled"`` if the bypass
        stopped working). ``cfg_calls_per_step`` is the last call's (1, or 2 when
        guidance above 1.0 ran the negative branch). ``sizes["encoders"]`` is the text encoder's bytes a prompt
        encode makes resident (the layers Klein reads), not its file size.
        """
        launches = self._launches + (self._provider.launches if self._provider is not None else 0)
        fit = None if self._plan is None else self._plan.estimate
        try:
            mflux_version: str | None = metadata.version("mflux")
        except metadata.PackageNotFoundError:
            mflux_version = None
        return {
            "family": "flux2",
            "model": self._model,
            "size": self._size,
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
            "predict": predict_mode(Flux2Klein._predict, self.transformer),
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
        """Freeze the components that are loaded (a dropped encoder is ``None``)."""
        del kwargs
        for module in (self.vae, self.transformer, self.text_encoder):
            if module is not None:
                module.freeze()
