"""The Qwen-Image 2.1 base repository without its transformer: text encoder, VAE and tokenizer, resolved and loaded.

mflux's ``Qwen21Initializer.init`` wants every component under one path, so it would download and size the
transformer; this module loads the other components through mflux's own loader and applier with weight definitions
that name only them. The download patterns never name ``transformer/``.
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

# The ComfyUI DF11 repository holds one weight file and no config; the config pattern is harmless there.
DF11_PATTERNS: tuple[str, ...] = ("*.safetensors", "config.json")
ENCODER_PATTERNS: tuple[str, ...] = ("text_encoder/*.safetensors", "text_encoder/*.json")
VAE_PATTERNS: tuple[str, ...] = ("vae/*.safetensors", "vae/*.json")
# The tokenizer's own files in processor/ (transformers' AutoTokenizer for Qwen2), named one by one: the directory
# also holds the image and video preprocessor configs, which mflux never builds, and anything a later revision adds.
TOKENIZER_PATTERNS: tuple[str, ...] = (
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/vocab.json",
    "processor/merges.txt",
    "processor/added_tokens.json",
    "processor/special_tokens_map.json",
    "processor/chat_template.jinja",
)
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS

_WHAT = "Qwen-Image 2.1 base"
# Parameters no checkpoint holds: built at construction, so a base never replaces them. mflux 0.20.0: the text
# encoder's ``rotary_emb.inv_freq`` is computed in ``Qwen3VLRotaryEmbedding.__init__`` (qwen3_vl_rope.py:21). The
# VAE has none: every one of its parameters is a target of mflux's VAE mapping, and its latent mean and standard
# deviation are NumPy class attributes, not parameters (qwen21_vae.py:12-32).
COMPUTED: dict[str, frozenset[str]] = {
    "text_encoder": frozenset({"rotary_emb.inv_freq"}),
    "vae": frozenset(),
}
_SHOWN = 3  # missing names quoted in the error


@dataclass(slots=True, kw_only=True)
class Qwen21Components:
    """The loaded base components: mflux modules (weights still lazy) and the tokenizers."""

    vae: Any
    text_encoder: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named Qwen-Image 2.1 components."""
    require_mflux()
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    return _hub.subset_definition(Qwen21WeightDefinition, names, patterns)


def encoder_definition() -> type:
    """The text encoder component of mflux's Qwen-Image 2.1 definition, with its download patterns."""
    return _definition(("text_encoder",), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's Qwen-Image 2.1 definition, with its download patterns."""
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
    """A fresh ``Qwen21TextEncoder`` with the base's weights (lazy).

    ``dims`` are the encoder's constructor arguments (none: mflux's defaults, the published model). Only the
    language model's tensors are mapped; the vision tower and ``lm_head`` in the same shards are never loaded.
    mflux's applier assigns weights without a shape check and ignores a tensor the base lacks, so after the load
    every parameter's shape is compared with the freshly built module's (another model's base is refused here,
    before any encode) and every parameter must have been replaced by the base.

    Raises:
        DFloatFormatError: The directory lacks the text encoder, a loaded parameter's shape differs from the
            model's (another model's base), or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import (
        Qwen21TextEncoder,
    )

    encoder = Qwen21TextEncoder(**dict(dims or {}))
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
    """A fresh ``Qwen21VAE`` module with the base's weights (lazy).

    Raises:
        DFloatFormatError: The directory lacks the VAE, or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE

    vae = Qwen21VAE()
    before: dict[str, Any] = dict(tree_flatten(vae.parameters()))
    _hub.load_into(root, vae_definition(), {"vae": vae}, what=_WHAT)
    _require_replaced(root, "vae", before, vae)
    return vae


def load_tokenizers(root: Path) -> dict[str, Any]:
    """The Qwen-Image 2.1 tokenizer (one, keyed ``"qwen21"``, read from ``processor/``).

    Raises:
        DFloatFormatError: The tokenizer files are missing or incomplete.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=Qwen21WeightDefinition.get_tokenizers(), model_path=str(root)
            )
        )
    except (OSError, RuntimeError) as exc:  # OSError: transformers' error for a partial processor/
        raise DFloatFormatError(f"{root}: no usable Qwen-Image 2.1 tokenizer: {exc}") from exc


def load_base(root: Path) -> Qwen21Components:
    """Text encoder, VAE and tokenizer of a base repository (weights lazy, every parameter from the base)."""
    return Qwen21Components(
        vae=load_vae(root),
        text_encoder=load_text_encoder(root),
        tokenizers=load_tokenizers(root),
    )
