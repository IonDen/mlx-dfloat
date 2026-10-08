"""The FLUX.1 base repository without its transformer: VAE, T5, CLIP and tokenizers, resolved and loaded.

mflux's ``FluxInitializer.init`` wants every component under one path, so it would download and
size the 23.8 GB BF16 transformer; this module loads the three other components through mflux's
own loader and applier with weight definitions that name only them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._hub import ResolvedRepo as ResolvedRepo
from mlx_dfloat.mflux._hub import hub_error as hub_error
from mlx_dfloat.mflux._hub import hub_revision as hub_revision
from mlx_dfloat.mflux._hub import is_hub_id as is_hub_id
from mlx_dfloat.mflux._hub import load_into, subset_definition
from mlx_dfloat.mflux._hub import refuse_quantized as refuse_quantized
from mlx_dfloat.mflux._hub import resolve as resolve

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


@dataclass(slots=True, kw_only=True)
class BaseComponents:
    """The loaded base components: mflux modules (weights still lazy) and the two tokenizers."""

    vae: Any
    t5: Any
    clip: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named FLUX.1 components."""
    require_mflux()
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    return subset_definition(FluxWeightDefinition, names, patterns)


def encoders_definition() -> type:
    """The T5 and CLIP components of mflux's FLUX.1 definition, with their download patterns."""
    return _definition(("t5_encoder", "clip_encoder"), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's FLUX.1 definition, with its download patterns."""
    return _definition(("vae",), VAE_PATTERNS)


def _load_into(root: Path, definition: Any, models: dict[str, Any]) -> None:
    """``_hub.load_into`` for the FLUX.1 base."""
    load_into(root, definition, models, what="FLUX.1 base")


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
