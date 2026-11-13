"""Source fingerprint of the mflux modules the memory constants and timings were measured on.

Each file in scope is hashed as raw text (sha256 of its bytes with CRLF read as LF), keyed by its POSIX path relative
to the installed ``mflux`` package. ``tests/fixtures/mflux_fingerprint.json`` holds one row per tested mflux release;
``tests/test_mflux_fingerprint.py`` compares the installed mflux with the right row.
"""

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from mlx_dfloat.mflux._versions import release_tuple

# The mflux package directory of each supported family.
_FAMILIES: tuple[str, ...] = ("flux", "z_image", "flux2", "ernie_image", "qwen21", "krea2")

# Globs relative to the mflux package directory: each family's VAE, text encoder, transformer and initializer.
SCOPE: tuple[str, ...] = (
    # FLUX.1
    "models/flux/model/flux_vae/**/*.py",
    "models/flux/model/flux_text_encoder/**/*.py",
    "models/flux/model/flux_transformer/**/*.py",
    "models/flux/flux_initializer.py",
    # Z-Image
    "models/z_image/model/z_image_vae/**/*.py",
    "models/z_image/model/z_image_text_encoder/**/*.py",
    "models/z_image/model/z_image_transformer/**/*.py",
    "models/z_image/z_image_initializer.py",
    # FLUX.2 Klein
    "models/flux2/model/flux2_vae/**/*.py",
    "models/flux2/model/flux2_text_encoder/**/*.py",
    "models/flux2/model/flux2_transformer/**/*.py",
    "models/flux2/flux2_initializer.py",
    # ERNIE-Image (its VAE is FLUX.2's)
    "models/ernie_image/model/ernie_transformer/**/*.py",
    "models/ernie_image/model/ernie_text_encoder/**/*.py",
    "models/ernie_image/ernie_image_initializer.py",
    # Qwen-Image 2.1 (the text encoder's base class lives in common_models/qwen3_vl)
    "models/qwen21/model/qwen21_vae/**/*.py",
    "models/qwen21/model/qwen21_text_encoder/**/*.py",
    "models/qwen21/model/qwen21_transformer/**/*.py",
    "models/qwen21/qwen21_initializer.py",
    "models/common_models/qwen3_vl/**/*.py",
    # Krea 2 (its VAE is Qwen-Image's)
    "models/qwen/model/qwen_vae/**/*.py",
    "models/krea2/model/krea2_text_encoder/**/*.py",
    "models/krea2/model/krea2_transformer/**/*.py",
    "models/krea2/krea2_initializer.py",
    # Shared
    "models/common/vae/**/*.py",
    # Per family: the pipeline classes the adapters subclass (variants), the weight definitions they load, and the
    # latent creators that size the denoise input.
    *(f"models/{family}/variants/**/*.py" for family in _FAMILIES),
    *(f"models/{family}/weights/*_weight_definition.py" for family in _FAMILIES),
    *(f"models/{family}/latent_creator/**/*.py" for family in _FAMILIES),
    # Krea 2's sampler and scheduler drive its step loop
    "models/krea2/model/krea2_sampler.py",
    "models/krea2/model/krea2_scheduler/**/*.py",
    # The weight mappings each family's name map is derived from, and Qwen-Image's (Krea 2 loads its VAE through it)
    "models/flux/weights/flux_weight_mapping.py",
    "models/z_image/weights/z_image_weight_mapping.py",
    "models/flux2/weights/flux2_weight_mapping.py",
    "models/ernie_image/weights/ernie_weight_mapping.py",
    "models/qwen21/weights/qwen21_weight_mapping.py",
    "models/krea2/weights/krea2_weight_mapping.py",
    "models/qwen/weights/qwen_weight_mapping.py",
    # Shared code the variants call on every generate: the config, the latent creator, the schedulers and the
    # weight loader (the load-phase memory numbers depend on how it reads and assigns the weights)
    "models/common/config/**/*.py",
    "models/common/latent_creator/**/*.py",
    "models/common/schedulers/**/*.py",
    "models/common/weights/loading/**/*.py",
)


def _matches(root: Path, pattern: str) -> list[Path]:
    return [path for path in root.glob(pattern) if path.is_file()]


def fingerprint(root: Path, scope: Sequence[str]) -> dict[str, str]:
    r"""POSIX path relative to ``root`` -> sha256 of the file's bytes with ``\r\n`` read as ``\n``."""
    digests: dict[str, str] = {}
    for pattern in scope:
        for path in _matches(root, pattern):
            data = path.read_bytes().replace(b"\r\n", b"\n")
            digests[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return dict(sorted(digests.items()))


def unmatched_globs(root: Path, scope: Sequence[str]) -> list[str]:
    """The globs in ``scope`` that match no file under ``root``."""
    return [pattern for pattern in scope if not _matches(root, pattern)]


def row_to_compare(installed: str, rows: Iterable[str]) -> str | None:
    """The newest recorded release not after ``installed``; None when ``installed`` is older than every row."""
    mine = release_tuple(installed)
    if mine is None:
        return None
    candidates = [
        (tag, row) for row in rows if (tag := release_tuple(row)) is not None and tag <= mine
    ]
    return max(candidates)[1] if candidates else None


def compare(recorded: dict[str, str], measured: dict[str, str]) -> dict[str, list[str]]:
    """The paths only in ``measured`` (added), only in ``recorded`` (removed), and in both with another digest."""
    return {
        "added": sorted(measured.keys() - recorded.keys()),
        "removed": sorted(recorded.keys() - measured.keys()),
        "changed": sorted(
            p for p in recorded.keys() & measured.keys() if recorded[p] != measured[p]
        ),
    }
