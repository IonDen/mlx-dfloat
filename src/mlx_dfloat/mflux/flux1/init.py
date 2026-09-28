"""The FLUX.1 base repository without its transformer: VAE, T5, CLIP and tokenizers, resolved and loaded.

mflux's ``FluxInitializer.init`` wants every component under one path, so it would download and
size the 23.8 GB BF16 transformer; this module loads the three other components through mflux's
own loader and applier with weight definitions that name only them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.errors import DFloatAccessError, DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.mflux import require_mflux

DF11_PATTERNS: tuple[str, ...] = ("*.safetensors", "config.json")
ENCODER_PATTERNS: tuple[str, ...] = (
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "text_encoder_2/*.safetensors",
    "text_encoder_2/*.json",
)
VAE_PATTERNS: tuple[str, ...] = ("vae/*.safetensors", "vae/*.json")
TOKENIZER_PATTERNS: tuple[str, ...] = ("tokenizer/**", "tokenizer_2/**")
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS
_SHA_LENGTH = 40


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRepo:
    """Where a repository's files are, and which Hub repository and commit they came from (if any)."""

    root: Path
    repo_id: str | None
    revision: str | None


@dataclass(slots=True, kw_only=True)
class BaseComponents:
    """The loaded base components: mflux modules (weights still lazy) and the two tokenizers."""

    vae: Any
    t5: Any
    clip: Any
    tokenizers: dict[str, Any]


def is_hub_id(spec: str) -> bool:
    """Whether ``spec`` is an ``org/name`` Hub id rather than a path (an existing path always wins)."""
    if Path(spec).expanduser().exists():
        return False
    return "/" in spec and spec.count("/") == 1 and not spec.startswith(("./", "../", "~/", "/"))


def hub_revision(root: Path) -> str | None:
    """The commit SHA of a Hub cache snapshot directory (``…/snapshots/<sha>``); None for any other directory."""
    return root.name if root.parent.name == "snapshots" and len(root.name) == _SHA_LENGTH else None


def hub_error(exc: Exception, repo_id: str) -> Exception:
    """Translate a Hub failure: gated or unauthorised → access error, unknown repo → format error; else ``exc``."""
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    hint = "accept the licence on the Hub and run `hf auth login`"
    if isinstance(exc, GatedRepoError):
        return DFloatAccessError(f"{repo_id}: this repository is gated; {hint}")
    if isinstance(exc, RepositoryNotFoundError | RevisionNotFoundError):
        return DFloatFormatError(
            f"{repo_id}: not a model repository on the Hub (or private without access)"
        )
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403):
            return DFloatAccessError(f"{repo_id}: the Hub refused access ({status}); {hint}")
    return exc


def _snapshot_download(*, repo_id: str, allow_patterns: list[str]) -> str:
    """``huggingface_hub.snapshot_download`` behind one name a test can replace."""
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))


def resolve(spec: str, *, patterns: tuple[str, ...]) -> ResolvedRepo:
    """A local directory as is, or a Hub id fetched (or found in the cache) with ``patterns`` only.

    Raises:
        DFloatFormatError: Neither a directory nor a Hub id, or the Hub has no such repository.
        DFloatAccessError: The repository is gated or this account may not read it.
    """
    if not is_hub_id(spec):
        root = Path(spec).expanduser()
        if not root.is_dir():
            raise DFloatFormatError(f"{spec}: not a directory or a Hub repository id")
        return ResolvedRepo(root=root, repo_id=None, revision=hub_revision(root))
    try:
        root = Path(_snapshot_download(repo_id=spec, allow_patterns=list(patterns)))
    except Exception as exc:
        translated = hub_error(exc, spec)
        if translated is exc:
            raise
        raise translated from exc
    return ResolvedRepo(root=root, repo_id=spec, revision=hub_revision(root))


def refuse_quantized(weights: Any, root: Path) -> None:
    """Refuse an mflux-saved quantized base: the applier would honour its stored bits even with ``quantize_arg=None``.

    Raises:
        DFloatFormatError: The weights carry a quantization level.
    """
    level = weights.meta_data.quantization_level
    if level is not None:
        raise DFloatFormatError(
            f"{root}: an mflux-saved {level}-bit model; the base must hold the BF16 encoders and VAE"
        )


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named FLUX.1 components (mflux's own ``ComponentDefinition``s)."""
    require_mflux()
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    components = [c for c in FluxWeightDefinition.get_components() if c.name in names]
    if [c.name for c in components] != list(names):
        raise DFloatIntegrationError(f"mflux's FLUX.1 weight definition does not define {names}")
    return type(
        "DFloatFluxBaseDefinition",
        (),
        {
            "get_components": staticmethod(lambda: list(components)),
            "get_tokenizers": staticmethod(FluxWeightDefinition.get_tokenizers),
            "get_download_patterns": staticmethod(lambda: list(patterns)),
            "quantization_predicate": staticmethod(FluxWeightDefinition.quantization_predicate),
        },
    )


def encoders_definition() -> type:
    """The T5 and CLIP components of mflux's FLUX.1 definition, with their download patterns."""
    return _definition(("t5_encoder", "clip_encoder"), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's FLUX.1 definition, with its download patterns."""
    return _definition(("vae",), VAE_PATTERNS)


def _load_into(root: Path, definition: Any, models: dict[str, Any]) -> None:
    """Mflux's loader and applier over ``definition`` into ``models``; the weights stay lazy until evaluated.

    The ``LoadedWeights`` object is not kept: it references every lazily loaded array (9.5 GB of
    T5), and the encoders must be droppable.

    Raises:
        DFloatFormatError: The directory lacks the components (a checkpoint or model repository
            without them), or holds an mflux-saved quantized model.
    """
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    names = [c.name for c in definition.get_components()]
    try:
        weights = WeightLoader.load(
            weight_definition=definition,
            model_path=str(root),
            download_patterns=definition.get_download_patterns(),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise DFloatFormatError(
            f"{root}: not a FLUX.1 base with {names} (a DFloat11 checkpoint or model repository?): {exc}"
        ) from exc
    refuse_quantized(weights, root)
    bits = WeightApplier.apply_and_quantize(
        weights=weights, models=models, quantize_arg=None, weight_definition=definition
    )
    if bits is not None:
        raise DFloatIntegrationError(
            f"{root}: the applier quantized {names} to {bits} bits unasked"
        )


def load_encoders(root: Path) -> tuple[Any, Any]:
    """Fresh ``T5Encoder`` and ``CLIPEncoder`` modules with the base's weights (lazy)."""
    require_mflux()
    from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    t5, clip = T5Encoder(), CLIPEncoder()
    _load_into(root, encoders_definition(), {"t5_encoder": t5, "clip_encoder": clip})
    return t5, clip


def load_vae(root: Path) -> Any:
    """A fresh ``VAE`` module with the base's weights (lazy)."""
    require_mflux()
    from mflux.models.flux.model.flux_vae.vae import VAE

    vae = VAE()
    _load_into(root, vae_definition(), {"vae": vae})
    return vae


def load_tokenizers(root: Path, model_config: Any) -> dict[str, Any]:
    """The CLIP and T5 tokenizers; T5's length is the model's (256 schnell, 512 dev and Krea-dev).

    Raises:
        DFloatFormatError: The tokenizer files are missing.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=FluxWeightDefinition.get_tokenizers(),
                model_path=str(root),
                max_length_overrides={"t5": int(model_config.max_sequence_length)},
            )
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise DFloatFormatError(f"{root}: no usable FLUX.1 tokenizers: {exc}") from exc


def load_base(root: Path, model_config: Any) -> BaseComponents:
    """VAE, encoders and tokenizers of a base repository (everything lazy; nothing read from disk yet)."""
    t5, clip = load_encoders(root)
    return BaseComponents(
        vae=load_vae(root), t5=t5, clip=clip, tokenizers=load_tokenizers(root, model_config)
    )
