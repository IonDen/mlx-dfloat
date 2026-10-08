"""The Z-Image base repository without its transformer: text encoder, VAE and tokenizer, resolved and loaded.

mflux's ``ZImageInitializer.init`` wants every component under one path, so it would download and size the
transformer; this module loads the other components through mflux's own loader and applier with weight definitions
that name only them.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import require_mflux
from mlx_dfloat.mflux._hub import ResolvedRepo as ResolvedRepo
from mlx_dfloat.mflux._hub import load_into, subset_definition
from mlx_dfloat.mflux._hub import resolve as resolve

DF11_PATTERNS: tuple[str, ...] = ("*.safetensors", "config.json")
ENCODER_PATTERNS: tuple[str, ...] = ("text_encoder/*.safetensors", "text_encoder/*.json")
VAE_PATTERNS: tuple[str, ...] = ("vae/*.safetensors", "vae/*.json")
TOKENIZER_PATTERNS: tuple[str, ...] = ("tokenizer/*",)
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS


@dataclass(slots=True, kw_only=True)
class ZImageComponents:
    """The loaded base components: mflux modules (weights still lazy) and the tokenizers."""

    vae: Any
    text_encoder: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named Z-Image components."""
    require_mflux()
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    return subset_definition(ZImageWeightDefinition, names, patterns)


def encoder_definition() -> type:
    """The text encoder component of mflux's Z-Image definition, with its download patterns."""
    return _definition(("text_encoder",), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's Z-Image definition, with its download patterns."""
    return _definition(("vae",), VAE_PATTERNS)


def load_text_encoder(root: Path) -> Any:
    """A fresh ``TextEncoder`` module with the base's weights (lazy)."""
    require_mflux()
    from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

    encoder = TextEncoder()
    load_into(root, encoder_definition(), {"text_encoder": encoder}, what="Z-Image base")
    return encoder


def load_vae(root: Path) -> Any:
    """A fresh ``VAE`` module with the base's weights (lazy)."""
    require_mflux()
    from mflux.models.z_image.model.z_image_vae.vae import VAE

    vae = VAE()
    load_into(root, vae_definition(), {"vae": vae}, what="Z-Image base")
    return vae


def load_tokenizers(root: Path) -> dict[str, Any]:
    """The Z-Image tokenizers.

    Raises:
        DFloatFormatError: The tokenizer files are missing.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=ZImageWeightDefinition.get_tokenizers(), model_path=str(root)
            )
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise DFloatFormatError(f"{root}: no usable Z-Image tokenizers: {exc}") from exc


def load_base(root: Path) -> ZImageComponents:
    """Text encoder, VAE and tokenizers of a base repository (everything lazy; nothing read from disk yet)."""
    return ZImageComponents(
        vae=load_vae(root), text_encoder=load_text_encoder(root), tokenizers=load_tokenizers(root)
    )
