"""The FLUX.2 Klein base repository without its transformer: text encoder, VAE and tokenizer, resolved and loaded.

mflux's ``Flux2Initializer.init`` wants every component under one path, so it would download and size the
transformer; this module loads the other components through mflux's own loader and applier with weight definitions
that name only them. The download patterns never name ``transformer/`` nor the repository's root checkpoint file.
"""

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
TOKENIZER_PATTERNS: tuple[str, ...] = ("tokenizer/*",)
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS

_WHAT = "FLUX.2 Klein base"
# Parameters no checkpoint holds: built at construction, so a base never replaces them. mflux 0.20.0:
# ``Qwen3TextEncoder.rotary_emb`` (qwen3_text_encoder.py:47-52) computes ``inv_freq`` in
# ``Qwen3TextRotaryEmbedding.__init__`` (qwen3_text_rotary_embedding.py:18). The VAE has none: its batch-norm
# statistics ``bn.running_mean`` / ``bn.running_var`` come from the file (flux2_weight_mapping.py:202-209).
COMPUTED: dict[str, frozenset[str]] = {
    "text_encoder": frozenset({"rotary_emb.inv_freq"}),
    "vae": frozenset(),
}
_SHOWN = 3  # missing names quoted in the error


@dataclass(slots=True, kw_only=True)
class Flux2Components:
    """The loaded base components: mflux modules (weights still lazy) and the tokenizers."""

    vae: Any
    text_encoder: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named FLUX.2 Klein components."""
    require_mflux()
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    return _hub.subset_definition(Flux2KleinWeightDefinition, names, patterns)


def encoder_definition() -> type:
    """The text encoder component of mflux's FLUX.2 Klein definition, with its download patterns."""
    return _definition(("text_encoder",), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's FLUX.2 Klein definition, with its download patterns."""
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


def load_text_encoder(root: Path, model_config: Any) -> Any:
    """A fresh ``Qwen3TextEncoder`` at the model's size with the base's weights (lazy).

    The encoder is built from ``model_config.text_encoder_overrides`` (Klein 4B and 9B differ in width). mflux's
    applier assigns weights without a shape check and ignores a tensor the base lacks, so after the load every
    parameter's shape is compared with the freshly built module's (a base repository of the other size is refused
    here, before any encode) and every parameter must have been replaced by the base.

    Raises:
        DFloatFormatError: The directory lacks the text encoder, a loaded parameter's shape differs from the
            model's (a base repository of another size), or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder

    encoder = Qwen3TextEncoder(**model_config.text_encoder_overrides)
    want = _shapes(encoder)
    before: dict[str, Any] = dict(tree_flatten(encoder.parameters()))
    _hub.load_into(root, encoder_definition(), {"text_encoder": encoder}, what=_WHAT)
    for name, got in _shapes(encoder).items():
        if want.get(name) != got:
            raise DFloatFormatError(
                f"{root}: {name} has shape {got}, this model needs {want.get(name)}: a base repository of "
                "another size?"
            )
    _require_replaced(root, "text_encoder", before, encoder)
    return encoder


def load_vae(root: Path) -> Any:
    """A fresh ``Flux2VAE`` module with the base's weights (lazy).

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
    """The FLUX.2 Klein tokenizers (one, keyed ``"qwen3"``).

    Raises:
        DFloatFormatError: The tokenizer files are missing.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=Flux2KleinWeightDefinition.get_tokenizers(), model_path=str(root)
            )
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise DFloatFormatError(f"{root}: no usable FLUX.2 tokenizers: {exc}") from exc


def load_base(root: Path, model_config: Any) -> Flux2Components:
    """Text encoder, VAE and tokenizers of a base repository (weights lazy, every parameter from the base)."""
    return Flux2Components(
        vae=load_vae(root),
        text_encoder=load_text_encoder(root, model_config),
        tokenizers=load_tokenizers(root),
    )
