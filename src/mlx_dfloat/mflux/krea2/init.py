"""The Krea 2 base repository without its transformer: text encoder, VAE and tokenizer, resolved and loaded.

mflux's ``Krea2Initializer.init`` wants every component under one path, so it would download and size the transformer
(for Raw, mflux's own download patterns fetch the 26 GB diffusers ``transformer/`` shards); this module loads the other
components through mflux's own loader and applier with weight definitions that name only them. The download patterns
never name ``transformer/``, the root ``raw.safetensors`` / ``turbo.safetensors`` or the sample ``images/``. Krea 2
Raw and Krea 2 Turbo share their text encoder, VAE and tokenizer byte for byte.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx.utils import tree_flatten

from mlx_dfloat.errors import DFloatAccessError, DFloatFormatError
from mlx_dfloat.mflux import _hub, require_mflux
from mlx_dfloat.mflux._hub import ResolvedRepo as ResolvedRepo

# The ComfyUI DF11 repository holds one weight file and no config.
DF11_PATTERNS: tuple[str, ...] = ("*.safetensors",)
ENCODER_PATTERNS: tuple[str, ...] = ("text_encoder/*.safetensors", "text_encoder/*.json")
VAE_PATTERNS: tuple[str, ...] = ("vae/*.safetensors", "vae/*.json")
# The tokenizer's files, named one by one (never mflux's "tokenizer/**", which would fetch anything a later revision
# drops in there). Both pinned base snapshots hold the first three (2026-10-08); the other two are files transformers'
# AutoTokenizer reads when present, named so a later revision that adds them still loads.
TOKENIZER_PATTERNS: tuple[str, ...] = (
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/special_tokens_map.json",
    "tokenizer/added_tokens.json",
)
BASE_PATTERNS: tuple[str, ...] = ENCODER_PATTERNS + VAE_PATTERNS + TOKENIZER_PATTERNS

_WHAT = "Krea 2 base"
# Parameters no checkpoint holds: built at construction, so a base never replaces them. mflux 0.20.0: the text
# encoder's ``rotary_emb.inv_freq`` is computed in ``Qwen3TextRotaryEmbedding.__init__`` (the base file holds only
# ``language_model.*`` and ``visual.*`` tensors). The VAE has none: every one of its parameters is a target of mflux's
# VAE mapping (Qwen-Image 1's).
COMPUTED: dict[str, frozenset[str]] = {
    "text_encoder": frozenset({"rotary_emb.inv_freq"}),
    "vae": frozenset(),
}
_SHOWN = 3  # missing names quoted in the error


def _licence_error(repo_id: str) -> DFloatAccessError:
    return DFloatAccessError(
        f"{repo_id}: this repository is gated or this account may not read it; accept the licence at "
        f"https://huggingface.co/{repo_id} and run `hf auth login`"
    )


def resolve(spec: str, *, patterns: tuple[str, ...], revision: str | None = None) -> ResolvedRepo:
    """``_hub.resolve``, with the licence page named when the Hub refuses access (both Krea 2 bases are gated).

    Raises:
        DFloatFormatError: Neither a directory nor a Hub id, or the Hub has no such repository.
        DFloatAccessError: The repository is gated or this account may not read it.
    """
    try:
        return _hub.resolve(spec, patterns=patterns, revision=revision)
    except DFloatAccessError as exc:
        raise _licence_error(spec) from exc


@dataclass(slots=True, kw_only=True)
class Krea2Components:
    """The loaded base components: mflux modules (weights still lazy) and the tokenizers."""

    vae: Any
    text_encoder: Any
    tokenizers: dict[str, Any]


def _definition(names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named Krea 2 components."""
    require_mflux()
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    return _hub.subset_definition(Krea2WeightDefinition, names, patterns)


def encoder_definition() -> type:
    """The text encoder component of mflux's Krea 2 definition (its prefix-stripping key transform kept)."""
    return _definition(("text_encoder",), ENCODER_PATTERNS)


def vae_definition() -> type:
    """The VAE component of mflux's Krea 2 definition, with its download patterns."""
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
    """A fresh ``Krea2TextEncoder`` with the base's weights (lazy).

    ``dims`` are the encoder's constructor arguments (none: mflux's defaults, the published model). Only the language
    model's tensors are loaded; the vision tower in the same file never is. mflux's applier assigns weights without a
    shape check and ignores a tensor the base lacks, so after the load every parameter's shape is compared with the
    freshly built module's (another model's base is refused here, before any encode) and every parameter must have
    been replaced by the base.

    Raises:
        DFloatFormatError: The directory lacks the text encoder, a loaded parameter's shape differs from the model's
            (another model's base), or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    encoder = Krea2TextEncoder(**dict(dims or {}))
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
    """A fresh ``QwenVAE`` (Qwen-Image 1's VAE, which Krea 2 uses) with the base's weights (lazy).

    Raises:
        DFloatFormatError: The directory lacks the VAE, or a parameter is not in the base.
    """
    require_mflux()
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

    vae = QwenVAE()
    before: dict[str, Any] = dict(tree_flatten(vae.parameters()))
    _hub.load_into(root, vae_definition(), {"vae": vae}, what=_WHAT)
    _require_replaced(root, "vae", before, vae)
    return vae


def load_tokenizers(root: Path) -> dict[str, Any]:
    """The Krea 2 tokenizer (one, keyed ``"qwen3vl"``, read from ``tokenizer/``).

    Raises:
        DFloatFormatError: The tokenizer files are missing or incomplete.
    """
    require_mflux()
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    try:
        return dict(
            TokenizerLoader.load_all(
                definitions=Krea2WeightDefinition.get_tokenizers(), model_path=str(root)
            )
        )
    except (OSError, RuntimeError) as exc:  # OSError: transformers' error for a partial tokenizer/
        raise DFloatFormatError(f"{root}: no usable Krea 2 tokenizer: {exc}") from exc


def load_base(root: Path) -> Krea2Components:
    """Text encoder, VAE and tokenizer of a base repository (weights lazy, every parameter from the base)."""
    return Krea2Components(
        vae=load_vae(root),
        text_encoder=load_text_encoder(root),
        tokenizers=load_tokenizers(root),
    )
