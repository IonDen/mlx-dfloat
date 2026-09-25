"""DFloat11 checkpoint format: config, per-group arrays, validation, discovery."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlx_dfloat.errors import DFloatFormatError

SUPPORTED_VERSIONS: frozenset[str] = frozenset({"0.2.0", "0.3.1", "0.3.2", "0.5.0"})
THREADS_PER_BLOCK = 512
BYTES_PER_THREAD = 8
BLOCK_BYTES = THREADS_PER_BLOCK * BYTES_PER_THREAD
LUT_POINTER_MIN = 240
MAX_LUT_ROWS = (
    18  # row 0, up to 16 pointer-target rows (pointers 255..240 -> rows 1..16), lengths row
)
MAX_PATTERNS = 64
MAX_PATTERN_CHARS = 256
_QUANTIFIED_GROUP = re.compile(r"\)[*+{]")


@dataclass(frozen=True, slots=True, kw_only=True)
class DF11Config:
    """The ``dfloat11_config`` block of a checkpoint's ``config.json``."""

    version: str
    threads_per_block: int
    bytes_per_thread: int
    pattern_dict: Mapping[str, tuple[str, ...]]


def _check_pattern(pattern: str, *, source: str) -> None:
    if len(pattern) > MAX_PATTERN_CHARS:
        raise DFloatFormatError(
            f"{source}: pattern_dict pattern is too long ({len(pattern)} chars)"
        )
    if _QUANTIFIED_GROUP.search(pattern):
        raise DFloatFormatError(
            f"{source}: pattern {pattern!r} has a quantified group; refused to avoid catastrophic backtracking"
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise DFloatFormatError(
            f"{source}: pattern {pattern!r} is not a valid regular expression"
        ) from exc


def parse_df11_config(raw: object, *, source: str) -> DF11Config:
    """Validate a ``dfloat11_config`` mapping and return it typed.

    Raises:
        DFloatFormatError: The version is unsupported, a field is missing or malformed, or a
            pattern is unsafe.
    """
    if not isinstance(raw, dict):
        raise DFloatFormatError(f"{source}: dfloat11_config is not an object")
    version = raw.get("version")
    if not isinstance(version, str):
        raise DFloatFormatError(f"{source}: dfloat11_config has no version string")
    if version not in SUPPORTED_VERSIONS:
        raise DFloatFormatError(
            f"{source}: unsupported DF11 format version {version!r} "
            f"(supported: {', '.join(sorted(SUPPORTED_VERSIONS))})"
        )
    if raw.get("threads_per_block") != [THREADS_PER_BLOCK]:
        raise DFloatFormatError(
            f"{source}: threads_per_block must be [512], got {raw.get('threads_per_block')!r}"
        )
    if raw.get("bytes_per_thread") != BYTES_PER_THREAD:
        raise DFloatFormatError(
            f"{source}: bytes_per_thread must be 8, got {raw.get('bytes_per_thread')!r}"
        )
    patterns = raw.get("pattern_dict")
    if not isinstance(patterns, dict) or not patterns:
        raise DFloatFormatError(f"{source}: pattern_dict is missing or empty")
    if len(patterns) > MAX_PATTERNS:
        raise DFloatFormatError(f"{source}: pattern_dict has too many patterns ({len(patterns)})")
    parsed: dict[str, tuple[str, ...]] = {}
    for pattern, subpaths in patterns.items():
        if not isinstance(subpaths, list) or not all(isinstance(s, str) for s in subpaths):
            raise DFloatFormatError(
                f"{source}: pattern_dict entry {pattern!r} is not a list of names"
            )
        _check_pattern(pattern, source=source)
        parsed[pattern] = tuple(subpaths)
    return DF11Config(
        version=version,
        threads_per_block=THREADS_PER_BLOCK,
        bytes_per_thread=BYTES_PER_THREAD,
        pattern_dict=parsed,
    )


def read_df11_config(model_dir: Path) -> DF11Config:
    """Read ``config.json`` from a DF11 model directory.

    Raises:
        DFloatFormatError: The directory is a legacy pickle-format DF11 repo, or has no usable config.
    """
    config_path = model_dir / "config.json"
    has_legacy = any(model_dir.glob("*.pkl")) or any(model_dir.glob("*.ptx"))
    config: object = None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (ValueError, RecursionError) as exc:
            raise DFloatFormatError(f"{config_path}: not valid JSON") from exc
    has_df11 = isinstance(config, dict) and "dfloat11_config" in config
    if has_legacy and not has_df11:
        raise DFloatFormatError(
            f"{model_dir}: legacy pickle-format DF11 checkpoint (.pkl/.ptx) is not supported; "
            "mlx-dfloat never unpickles files"
        )
    if config is None:
        raise DFloatFormatError(f"{model_dir}: no config.json")
    if not has_df11:
        raise DFloatFormatError(f"{config_path}: no dfloat11_config block")
    assert isinstance(config, dict)  # narrowed by has_df11
    return parse_df11_config(config["dfloat11_config"], source=str(config_path))
