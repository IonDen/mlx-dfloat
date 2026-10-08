"""Hub helpers shared by the family adapters: resolving a repository, translating Hub failures, loading base components.

mflux's initializers want every component under one path, so they would download and size the transformer; the
adapters load the other components through mflux's own loader and applier with weight definitions that name only them.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_dfloat.errors import DFloatAccessError, DFloatFormatError, DFloatIntegrationError

_SHA_LENGTH = 40


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRepo:
    """Where a repository's files are, and which Hub repository and commit they came from (if any)."""

    root: Path
    repo_id: str | None
    revision: str | None


def is_hub_id(spec: str) -> bool:
    """Whether ``spec`` is an ``org/name`` Hub id rather than a path (an existing path always wins)."""
    if Path(spec).expanduser().exists():
        return False
    return "/" in spec and spec.count("/") == 1 and not spec.startswith(("./", "../", "~/", "/"))


def hub_revision(root: Path) -> str | None:
    """The commit SHA of a Hub cache snapshot directory (``…/snapshots/<sha>``); None for any other directory."""
    return root.name if root.parent.name == "snapshots" and len(root.name) == _SHA_LENGTH else None


def hub_error(exc: Exception, repo_id: str) -> Exception:
    """Translate a Hub failure: gated or unauthorised → access error, unknown repo → format error; else ``exc``.

    A cache miss whose cause is a Hub HTTP error is translated through that cause.
    """
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    if isinstance(exc, LocalEntryNotFoundError) and isinstance(exc.__cause__, HfHubHTTPError):
        # huggingface_hub reports a refused download as a cache miss, the refusal as its cause
        mapped = hub_error(exc.__cause__, repo_id)
        return exc if mapped is exc.__cause__ else mapped
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


def _snapshot_download(*, repo_id: str, allow_patterns: list[str], revision: str | None) -> str:
    """``huggingface_hub.snapshot_download`` behind one name a test can replace."""
    from huggingface_hub import snapshot_download

    return str(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, revision=revision))


def resolve(spec: str, *, patterns: tuple[str, ...], revision: str | None = None) -> ResolvedRepo:
    """A local directory as is, or a Hub id fetched (or found in the cache) with ``patterns`` only.

    ``revision`` pins a Hub id to one commit (None: the repository's default branch); a local directory ignores it.
    A spec taken as a Hub id gets one line on stderr, so a mistyped path does not start a download unannounced.

    Raises:
        DFloatFormatError: Neither a directory nor a Hub id, or the Hub has no such repository.
        DFloatAccessError: The repository is gated or this account may not read it.
    """
    if not is_hub_id(spec):
        root = Path(spec).expanduser()
        if not root.is_dir():
            raise DFloatFormatError(f"{spec}: not a directory or a Hub repository id")
        return ResolvedRepo(root=root, repo_id=None, revision=hub_revision(root))
    print(
        f"{spec} is not a local path; resolving it on the Hugging Face Hub (local cache or download)",
        file=sys.stderr,
    )
    try:
        root = Path(
            _snapshot_download(repo_id=spec, allow_patterns=list(patterns), revision=revision)
        )
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


def subset_definition(source: type, names: tuple[str, ...], patterns: tuple[str, ...]) -> type:
    """An mflux weight definition holding only the named components of ``source`` (mflux's own ``ComponentDefinition``s).

    Raises:
        DFloatIntegrationError: ``source`` does not define every name.
    """
    components = [c for c in source.get_components() if c.name in names]  # type: ignore[attr-defined]
    if [c.name for c in components] != list(names):
        raise DFloatIntegrationError(
            f"mflux's weight definition {source.__name__} does not define {names}"
        )
    return type(
        "DFloatBaseDefinition",
        (),
        {
            "get_components": staticmethod(lambda: list(components)),
            "get_tokenizers": staticmethod(source.get_tokenizers),  # type: ignore[attr-defined]
            "get_download_patterns": staticmethod(lambda: list(patterns)),
            "quantization_predicate": staticmethod(source.quantization_predicate),  # type: ignore[attr-defined]
        },
    )


def load_into(root: Path, definition: Any, models: dict[str, Any], *, what: str) -> None:
    """Mflux's loader and applier over ``definition`` into ``models``; the weights stay lazy until evaluated.

    The ``LoadedWeights`` object is not kept: it references every lazily loaded array, and the encoders must be
    droppable.

    Raises:
        DFloatFormatError: The directory lacks the components (a checkpoint or model repository without them), or
            holds an mflux-saved quantized model.
        DFloatIntegrationError: The applier quantized the components unasked.
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
            f"{root}: not a {what} with {names} (a DFloat11 checkpoint or model repository?): {exc}"
        ) from exc
    refuse_quantized(weights, root)
    bits = WeightApplier.apply_and_quantize(
        weights=weights, models=models, quantize_arg=None, weight_definition=definition
    )
    if bits is not None:
        raise DFloatIntegrationError(
            f"{root}: the applier quantized {names} to {bits} bits unasked"
        )
