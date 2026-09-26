"""DFloat11 checkpoint format: config, per-group arrays, validation, discovery."""

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mlx_dfloat._safetensors import TensorInfo, read_array, read_header, short_repr
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
# pattern_dict keys come from a downloaded config.json and are matched with re.fullmatch, so they
# are held to the small grammar upstream DF11 configs use (literals, `\.` and a bare `.`, which
# the 0.2.0 configs of Qwen3-4B and FLUX.1-dev/schnell leave unescaped, `\d`, `\w`, classes such
# as `[0-9]`, `+`/`*`/`?`, `|` and plain groups), with a cap on every kind of backtracking choice
# point. Group names are at most 256 characters, so two unbounded quantifiers cost at most
# ~256^2 steps per match; optional parts and alternation branches each double the worst case.
_QUANTIFIED_GROUP = re.compile(r"\)[*+{]")
_PATTERN_TOKEN = re.compile(r"\\[.dw]|[A-Za-z0-9_.\-\[\]()|*+?]")
MAX_UNBOUNDED_QUANTIFIERS = 2
MAX_OPTIONAL_QUANTIFIERS = 2
MAX_ALTERNATIONS = 4


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
    pos = 0
    while pos < len(pattern):
        token = _PATTERN_TOKEN.match(pattern, pos)
        if token is None:
            raise DFloatFormatError(
                f"{source}: pattern {pattern!r}: {pattern[pos : pos + 2]!r} at offset {pos} is not "
                "allowed in a DF11 pattern"
            )
        pos = token.end()
    if "(?" in pattern:
        raise DFloatFormatError(
            f"{source}: pattern {pattern!r}: inline flags and extension groups '(?' are not allowed"
        )
    unbounded = pattern.count("*") + pattern.count("+")
    if unbounded > MAX_UNBOUNDED_QUANTIFIERS:
        raise DFloatFormatError(
            f"{source}: pattern {pattern!r} has {unbounded} unbounded quantifiers "
            f"(at most {MAX_UNBOUNDED_QUANTIFIERS})"
        )
    for symbol, limit, what in (
        ("?", MAX_OPTIONAL_QUANTIFIERS, "optional parts"),
        ("|", MAX_ALTERNATIONS, "alternations"),
    ):
        if pattern.count(symbol) > limit:
            raise DFloatFormatError(
                f"{source}: pattern {pattern!r} has too many {what} (at most {limit})"
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
            f"{source}: unsupported DF11 format version {short_repr(version)} "
            f"(supported: {', '.join(sorted(SUPPORTED_VERSIONS))})"
        )
    if raw.get("threads_per_block") != [THREADS_PER_BLOCK]:
        raise DFloatFormatError(
            f"{source}: threads_per_block must be [512], got {short_repr(raw.get('threads_per_block'))}"
        )
    if raw.get("bytes_per_thread") != BYTES_PER_THREAD:
        raise DFloatFormatError(
            f"{source}: bytes_per_thread must be 8, got {short_repr(raw.get('bytes_per_thread'))}"
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
                f"{source}: pattern_dict entry {short_repr(pattern)} is not a list of names"
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


def n_blocks_for(n_bytes: int) -> int:
    """Number of 4096-byte thread-blocks that cover ``n_bytes`` of encoded exponents."""
    return -(-n_bytes // BLOCK_BYTES)


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupArrays:
    """The six arrays stored for one compressed group, in decoder-ready dtypes."""

    encoded_exponent: npt.NDArray[np.uint8]
    sign_mantissa: npt.NDArray[np.uint8]
    luts: npt.NDArray[np.uint8]
    gaps: npt.NDArray[np.uint8]
    output_positions: npt.NDArray[np.uint32]
    split_positions: npt.NDArray[np.int64]

    @property
    def n_elements(self) -> int:
        """Number of BF16 values in the group."""
        return int(self.sign_mantissa.size)

    @property
    def n_bytes(self) -> int:
        """Length of the encoded exponent stream in bytes."""
        return int(self.encoded_exponent.size)

    @property
    def n_blocks(self) -> int:
        """Number of 512-thread blocks the upstream kernel launches for this group."""
        return n_blocks_for(self.n_bytes)

    @property
    def n_threads(self) -> int:
        """Total decode threads (512 per block)."""
        return self.n_blocks * THREADS_PER_BLOCK


def validate_group_arrays(arrays: GroupArrays, *, name: str) -> None:
    """Check the structural invariants a well-formed DF11 group satisfies.

    Raises:
        DFloatFormatError: Any invariant is violated; the message names the group.
    """
    n, n_bytes = arrays.n_elements, arrays.n_bytes
    if n == 0 or n_bytes == 0:
        raise DFloatFormatError(f"{name}: empty group")
    luts = arrays.luts
    if luts.ndim != 2 or luts.shape[1] != 256 or not 2 <= luts.shape[0] <= MAX_LUT_ROWS:
        raise DFloatFormatError(f"{name}: luts must be [2..{MAX_LUT_ROWS}, 256], got {luts.shape}")
    decode_rows = luts[:-1]
    pointers = decode_rows[decode_rows >= LUT_POINTER_MIN].astype(np.int64)
    targets = 256 - pointers
    if pointers.size and (targets.min() < 1 or targets.max() > luts.shape[0] - 2):
        raise DFloatFormatError(
            f"{name}: a LUT pointer targets a row outside 1..{luts.shape[0] - 2}"
        )
    positions = arrays.output_positions.astype(np.int64)
    n_blocks = arrays.n_blocks
    if positions.size < 2 or positions.size - 1 not in (n_blocks, n_blocks - 1):
        raise DFloatFormatError(
            f"{name}: output_positions has {positions.size} entries for {n_blocks} blocks"
        )
    if positions[0] != 0:
        raise DFloatFormatError(f"{name}: first output position is {positions[0]}, expected 0")
    if np.any(np.diff(positions) < 0):
        raise DFloatFormatError(f"{name}: output_positions is not monotonic")
    if positions[-1] != n:
        raise DFloatFormatError(f"{name}: last output position {positions[-1]} != {n} elements")
    need_gap_bytes = -(-5 * arrays.n_threads // 8)
    if arrays.gaps.size < need_gap_bytes:
        raise DFloatFormatError(
            f"{name}: gaps has {arrays.gaps.size} bytes, needs {need_gap_bytes}"
        )
    split = arrays.split_positions.astype(np.int64)
    if split.size and (split[0] <= 0 or split[-1] >= n or np.any(np.diff(split) <= 0)):
        raise DFloatFormatError(
            f"{name}: split_positions must be strictly increasing inside (0, {n})"
        )


GROUP_FIELDS: tuple[str, ...] = (
    "encoded_exponent",
    "sign_mantissa",
    "luts",
    "gaps",
    "output_positions",
    "split_positions",
)
GROUP_FIELD_TYPES: Mapping[str, tuple[str, int]] = {
    "encoded_exponent": ("U8", 1),
    "sign_mantissa": ("U8", 1),
    "luts": ("U8", 2),
    "gaps": ("U8", 1),
    "output_positions": ("U8", 1),
    "split_positions": ("I64", 1),
}
GROUP_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,256}")


def validate_group_name(name: str) -> None:
    """Refuse group names that could escape a directory or blow up regex matching."""
    if not GROUP_NAME.fullmatch(name) or ".." in name:
        raise DFloatFormatError(f"unsafe group name {name[:80]!r}")


def matrix_names_for(group: str, pattern_dict: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Names of the weight matrices a group decodes into, in concatenation order.

    Raises:
        DFloatFormatError: No pattern, or more than one pattern, fully matches the group name.
    """
    label = short_repr(group)
    for pattern in pattern_dict:  # a hand-built DF11Config never went through parse_df11_config
        _check_pattern(pattern, source=f"group {label}")
    matches = [subs for pattern, subs in pattern_dict.items() if re.fullmatch(pattern, group)]
    if not matches:
        raise DFloatFormatError(f"group {label}: no pattern in pattern_dict matches it")
    if len(matches) > 1:
        raise DFloatFormatError(f"group {label}: more than one pattern in pattern_dict matches it")
    subs = matches[0]
    if not subs:
        return (f"{group}.weight",)
    return tuple(f"{group}.{sub}.weight" for sub in subs)


@dataclass(frozen=True, slots=True, kw_only=True)
class DF11Group:
    """One compressed group: where its six tensors live and which matrices it decodes into."""

    name: str
    matrix_names: tuple[str, ...]
    path: Path
    tensors: Mapping[str, TensorInfo]

    def load(self) -> GroupArrays:
        """Memory-map the group's tensors (dtypes checked at discovery) and validate them."""
        raw = {field: read_array(self.path, self.tensors[field]) for field in GROUP_FIELDS}
        positions = np.ascontiguousarray(raw["output_positions"])
        if positions.size % 4:
            raise DFloatFormatError(
                f"{self.name}: output_positions byte length is not a multiple of 4"
            )
        arrays = GroupArrays(
            encoded_exponent=np.asarray(raw["encoded_exponent"]),
            sign_mantissa=np.asarray(raw["sign_mantissa"]),
            luts=np.asarray(raw["luts"]),
            gaps=np.asarray(raw["gaps"]),
            output_positions=positions.view("<u4").astype(np.uint32),
            split_positions=np.asarray(raw["split_positions"]),
        )
        validate_group_arrays(arrays, name=self.name)
        return arrays


@dataclass(frozen=True, slots=True, kw_only=True)
class DF11Checkpoint:
    """A DF11 model directory: its config, compressed groups, and uncompressed extra tensors."""

    root: Path
    config: DF11Config
    groups: Mapping[str, DF11Group]
    extras: Mapping[str, tuple[Path, TensorInfo]]


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)  # follows symlinks: HF blobs are regular files
    except OSError:
        return False


def open_checkpoint(path: str | os.PathLike[str]) -> DF11Checkpoint:
    """Discover the groups and extras of a DF11 checkpoint directory, reading headers only.

    Raises:
        DFloatFormatError: The config is unusable; a shard is not a regular file; a group name is
            unsafe, a group is incomplete, split across files, or has wrong dtypes/ranks; a tensor
            name appears in two files; or a group's matrix count disagrees with its pattern.
    """
    root = Path(path).expanduser()
    config = read_df11_config(root)
    owner: dict[str, Path] = {}
    headers: dict[Path, dict[str, TensorInfo]] = {}
    for file in sorted(root.glob("*.safetensors")):
        if not _regular_file(file):
            raise DFloatFormatError(f"{file.name}: not a regular file")
        header = read_header(file)
        headers[file] = header
        for name in header:
            if name in owner:
                raise DFloatFormatError(
                    f"tensor {short_repr(name)} appears in both {owner[name].name} and {file.name}"
                )
            owner[name] = file
    group_names = sorted({n.rsplit(".", 1)[0] for n in owner if n.endswith(".encoded_exponent")})
    groups: dict[str, DF11Group] = {}
    claimed: set[str] = set()
    for group in group_names:
        validate_group_name(group)
        home = owner[f"{group}.encoded_exponent"]
        tensors: dict[str, TensorInfo] = {}
        for field in GROUP_FIELDS:
            full = f"{group}.{field}"
            if full not in owner:
                raise DFloatFormatError(f"group {group!r}: missing {field}")
            if owner[full] != home:
                raise DFloatFormatError(f"group {group!r}: its tensors are split across files")
            info = headers[home][full]
            want_dtype, want_rank = GROUP_FIELD_TYPES[field]
            if info.dtype != want_dtype:
                raise DFloatFormatError(
                    f"group {group!r}: {field} has dtype {info.dtype}, expected {want_dtype}"
                )
            if len(info.shape) != want_rank:
                raise DFloatFormatError(
                    f"group {group!r}: {field} has rank {len(info.shape)}, expected {want_rank}"
                )
            tensors[field] = info
            claimed.add(full)
        names = matrix_names_for(group, config.pattern_dict)
        n_matrices = tensors["split_positions"].shape[0] + 1
        if n_matrices != len(names):
            raise DFloatFormatError(
                f"group {group!r}: holds {n_matrices} matrices but its pattern names {len(names)}"
            )
        groups[group] = DF11Group(name=group, matrix_names=names, path=home, tensors=tensors)
    extras = {n: (f, headers[f][n]) for n, f in owner.items() if n not in claimed}
    return DF11Checkpoint(root=root, config=config, groups=groups, extras=extras)
