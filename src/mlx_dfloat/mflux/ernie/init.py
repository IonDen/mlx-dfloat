"""The ERNIE-Image base repository without its transformer: text encoder, VAE and tokenizer, resolved and loaded.

mflux's ``ErnieImageInitializer.init`` wants every component under one path, so it would download and size the
transformer; this module loads the other components through mflux's own loader and applier with weight definitions
that name only them. The download patterns never name ``transformer/``, the prompt enhancer ``pe/`` (7.66 GB, which
mflux never uses) or its tokenizer ``pe_tokenizer/``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx.utils import tree_flatten

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import _hub, require_mflux
from mlx_dfloat.mflux._hub import ResolvedRepo as ResolvedRepo
from mlx_dfloat.mflux._hub import resolve as resolve

DF11_PATTERNS: tuple[str, ...] = ("*.safetensors", "config.json")
ENCODER_PATTERNS: tuple[str, ...] = ("text_encoder/*.safetensors", "text_encoder/*.json")
VAE_PATTERNS: tuple[str, ...] = ("vae/*.safetensors", "vae/*.json")
# The tokenizer's files, named one by one (never "tokenizer/*" or mflux's "tokenizer/**", which would fetch anything a
# later revision drops in there). Both base snapshots hold the first two (2026-10-08); the other three are files
# transformers' AutoTokenizer reads when present, named so a later revision that adds them still loads.
TOKENIZER_PATTERNS: tuple[str, ...] = (
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
)
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS

_WHAT = "ERNIE-Image base"
# Parameters no checkpoint holds: built at construction, so a base never replaces them. mflux 0.20.0 has none for
# either module (checked by building both and listing their parameters): the text encoder's RoPE frequencies are the
# underscore attribute ``_inv_freq`` (text_encoder.py:137-138), which MLX does not count as a parameter, and every
# other encoder parameter sits under ``language_model.model.``, the prefix the base file holds; the VAE's batch-norm
# statistics come from the file, as for FLUX.2 Klein (flux2_weight_mapping.py:202-209).
COMPUTED: dict[str, frozenset[str]] = {
    "text_encoder": frozenset(),
    "vae": frozenset(),
}
_SHOWN = 3  # missing names quoted in the error


@dataclass(slots=True, kw_only=True)
class ErnieComponents:
    """The loaded base components: mflux modules (weights still lazy) and the tokenizers."""

    vae: Any
    text_encoder: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named ERNIE-Image components."""
    require_mflux()
    from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition

    return _hub.subset_definition(ErnieWeightDefinition, names, patterns)


def encoder_definition() -> type:
    """The text encoder component of mflux's ERNIE-Image definition (its language-model prefix filter kept)."""
    return _definition(("text_encoder",), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's ERNIE-Image definition, with its download patterns."""
    return _definition(("vae",), VAE_PATTERNS)


def _shapes(module: Any) -> dict[str, tuple[int, ...]]:
    """Every parameter's shape by path (read without evaluating anything)."""
    flat: dict[str, Any] = dict(tree_flatten(module.parameters()))
    return {name: tuple(p.shape) for name, p in flat.items()}


def _require_replaced(root: Path, name: str, before: dict[str, Any], module: Any) -> None:
    """Require every parameter of ``module`` to have been replaced by the base since ``before`` was taken.

    mflux's applier updates with ``strict=False``: a tensor the base lacks (a lost shard, a renamed key) leaves that
    parameter at its random float32 init without an error. ``before`` holds each parameter as built; one still the
    same object after the load was not in the base (``COMPUTED`` names the exceptions).

    Raises:
        DFloatFormatError: A parameter was not replaced by the base.
    """
    after: dict[str, Any] = dict(tree_flatten(module.parameters()))
    missing = [n for n, init in before.items() if n not in COMPUTED[name] and after.get(n) is init]
    if missing:
        more = len(missing) - _SHOWN
        shown = ", ".join(missing[:_SHOWN]) + (f" and {more} more" if more > 0 else "")
        raise DFloatFormatError(
            f"{root}: {name} {shown} not in the base: a missing shard or a renamed tensor?"
        )


def load_text_encoder(root: Path, *, dims: Mapping[str, Any] | None = None) -> Any:
    """A fresh ``ErnieMistralTextEncoder`` with the base's weights (lazy).

    ``dims`` are the encoder's constructor arguments (none: mflux's defaults, the published model). Only the language
    model's tensors are loaded; the vision tower and projector in the same file never are. mflux's applier assigns
    weights without a shape check and ignores a tensor the base lacks, so after the load every parameter's shape is
    compared with the freshly built module's (another model's base is refused here, before any encode) and every
    parameter must have been replaced by the base.

    Raises:
        DFloatFormatError: The directory lacks the text encoder, a loaded parameter's shape differs from the model's
            (another model's base), or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import (
        ErnieMistralTextEncoder,
    )

    encoder = ErnieMistralTextEncoder(**dict(dims or {}))
    want = _shapes(encoder)
    before: dict[str, Any] = dict(tree_flatten(encoder.parameters()))
    _hub.load_into(root, encoder_definition(), {"text_encoder": encoder}, what=_WHAT)
    for name, got in _shapes(encoder).items():
        if want.get(name) != got:
            raise DFloatFormatError(
                f"{root}: {name} has shape {got}, this model needs {want.get(name)}: another model's "
                "base repository?"
            )
    _require_replaced(root, "text_encoder", before, encoder)
    return encoder


def load_vae(root: Path) -> Any:
    """A fresh ``Flux2VAE`` (ERNIE-Image's VAE) with the base's weights (lazy).

    Raises:
        DFloatFormatError: The directory lacks the VAE, or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

    vae = Flux2VAE()
    before: dict[str, Any] = dict(tree_flatten(vae.parameters()))
    _hub.load_into(root, vae_definition(), {"vae": vae}, what=_WHAT)
    _require_replaced(root, "vae", before, vae)
    return vae


def load_tokenizers(root: Path) -> dict[str, Any]:
    """The ERNIE-Image tokenizer (one, keyed ``"ernie"``, read from ``tokenizer/``).

    Raises:
        DFloatFormatError: The tokenizer files are missing or incomplete.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.ernie_image.weights.ernie_weight_definition import ErnieWeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=ErnieWeightDefinition.get_tokenizers(), model_path=str(root)
            )
        )
    except (OSError, RuntimeError) as exc:  # OSError: transformers' error for a partial tokenizer/
        raise DFloatFormatError(f"{root}: no usable ERNIE-Image tokenizer: {exc}") from exc


def load_base(root: Path) -> ErnieComponents:
    """Text encoder, VAE and tokenizer of a base repository (weights lazy, every parameter from the base)."""
    return ErnieComponents(
        vae=load_vae(root),
        text_encoder=load_text_encoder(root),
        tokenizers=load_tokenizers(root),
    )
