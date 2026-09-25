"""Sampled DF11 parity via HTTP range reads: no full downloads, works on single-file repos.

Decodes selected groups (first, last, max-block, max-code, or names) with the reference decoder
and compares each matrix with the BF16 original read by range requests.

Usage:
    uv run python scripts/verify_remote_group.py --df11-repo DFloat11/FLUX.1-schnell-DF11 \
        --bf16-repo black-forest-labs/FLUX.1-schnell --bf16-subdir transformer \
        --groups first,last,max-block,max-code --out RESULT.json
Exit codes: 0 all sampled matrices equal, 1 a mismatch, 2 an error, 70/71 watchdog abort.
"""

import argparse
import json
import re
import struct
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import HfApi, HfFileSystem
from scripts._watchdog import Watchdog, default_ceiling
from scripts.verify_checkpoint import VerifyError, natural_key

from mlx_dfloat._memory_caps import install_memory_caps
from mlx_dfloat._safetensors import MAX_HEADER_BYTES, parse_header
from mlx_dfloat.errors import DFloatError
from mlx_dfloat.format import (
    GROUP_FIELD_TYPES,
    GROUP_FIELDS,
    DF11Config,
    GroupArrays,
    matrix_names_for,
    parse_df11_config,
    validate_group_name,
)
from mlx_dfloat.reference import (
    decode_group,
    max_code_length,
    max_elements_per_block,
    split_matrices,
)

_SHARD = re.compile(r"[A-Za-z0-9_.\-]+\.safetensors")


class RangeSource(Protocol):
    """A repository readable by byte ranges."""

    def read(self, path: str, start: int, n: int) -> bytes: ...  # noqa: D102

    def size(self, path: str) -> int: ...  # noqa: D102

    def text(self, path: str) -> str: ...  # noqa: D102

    def files(self) -> list[str]: ...  # noqa: D102


def _checked(data: bytes, n: int, path: str) -> bytes:
    if len(data) != n:
        raise VerifyError(f"{path}: short read ({len(data)} of {n} bytes)")
    return data


class HfRangeSource:
    """A Hugging Face repo at a pinned revision, read by HTTP range requests."""

    def __init__(self, repo: str, revision: str, subdir: str = "") -> None:
        """Pin the repo, revision and optional subdirectory."""
        self.repo, self.revision, self.subdir = repo, revision, subdir.strip("/")
        self._fs = HfFileSystem()

    def _full(self, path: str) -> str:
        if (
            not (_SHARD.fullmatch(path) or path.endswith((".json", ".txt")))
            or "/" in path
            or ".." in path
        ):
            raise VerifyError(f"unsafe path {path!r}")
        inner = f"{self.subdir}/{path}" if self.subdir else path
        return f"{self.repo}@{self.revision}/{inner}"

    def read(self, path: str, start: int, n: int) -> bytes:
        """Read exactly ``n`` bytes at ``start``."""
        # block_size=0 gives a streaming file whose seek() raises on huggingface_hub 2.0; use the
        # default (non-streaming, seekable) file object instead.
        with self._fs.open(self._full(path), "rb") as handle:
            handle.seek(start)
            return _checked(handle.read(n), n, path)

    def size(self, path: str) -> int:
        """Remote file size."""
        return int(self._fs.info(self._full(path))["size"])

    def text(self, path: str) -> str:
        """Read a small text file."""
        return self._fs.read_text(self._full(path))

    def files(self) -> list[str]:
        """List files directly in the (sub)directory."""
        prefix = f"{self.subdir}/" if self.subdir else ""
        names = HfApi().list_repo_files(self.repo, revision=self.revision)
        return [
            n[len(prefix) :] for n in names if n.startswith(prefix) and "/" not in n[len(prefix) :]
        ]


class LocalRangeSource:
    """A local directory with the same interface (tests, local caches)."""

    def __init__(self, root: Path) -> None:
        """Wrap a directory."""
        self.root = root

    def read(self, path: str, start: int, n: int) -> bytes:
        """Read exactly ``n`` bytes at ``start``."""
        with (self.root / path).open("rb") as handle:
            handle.seek(start)
            return _checked(handle.read(n), n, path)

    def size(self, path: str) -> int:
        """File size."""
        return (self.root / path).stat().st_size

    def text(self, path: str) -> str:
        """Read a small text file."""
        return (self.root / path).read_text()

    def files(self) -> list[str]:
        """List files."""
        return sorted(p.name for p in self.root.iterdir() if p.is_file())


def _header_length(src: RangeSource, path: str) -> int:
    (length,) = struct.unpack("<Q", src.read(path, 0, 8))
    if length == 0 or length > MAX_HEADER_BYTES:
        raise VerifyError(f"{path}: header length {length} is invalid")
    return length


def independent_header(src: RangeSource, path: str) -> tuple[dict, int]:
    """Parse a safetensors header without mlx_dfloat code (for the BF16 side)."""
    length = _header_length(src, path)
    return json.loads(src.read(path, 8, length)), 8 + length


def df11_index(src: RangeSource) -> dict[str, tuple[str, dict]]:
    """Group name -> (file, parsed header), for every DF11 group; built once.

    Fails closed: a shard that cannot be read or parsed raises rather than being silently
    dropped, because dropping a shard changes which groups ``first``/``last``/``max-block``/
    ``max-code`` resolve to and can turn a corrupt checkpoint into a false pass. When a
    ``*.safetensors.index.json`` is present, every shard it names must also be present and
    parse; a group name appearing in more than one shard is refused as ambiguous.

    Raises:
        DFloatError: A shard's header is malformed.
        VerifyError: A shard is missing, short, or a group name is duplicated across shards.
    """
    names = src.files()
    listed_shards: set[str] = set()
    for name in names:
        if name.endswith(".safetensors.index.json"):
            weight_map = json.loads(src.text(name))["weight_map"]
            shards = set(weight_map.values())
            missing = shards - set(names)
            if missing:
                raise VerifyError(f"{name}: shard(s) {sorted(missing)!r} not found")
            listed_shards |= shards
    shard_names = sorted({n for n in names if n.endswith(".safetensors")} | listed_shards)
    index: dict[str, tuple[str, dict]] = {}
    for name in shard_names:
        length = _header_length(src, name)
        header = parse_header(
            src.read(name, 8, length), data_start=8 + length, file_size=src.size(name), source=name
        )
        for tensor in header:
            if tensor.endswith(".encoded_exponent"):
                group = tensor.rsplit(".", 1)[0]
                validate_group_name(group)
                if group in index:
                    raise VerifyError(f"duplicate group {group!r} in {index[group][0]} and {name}")
                index[group] = (name, header)
    return index


def bf16_index(src: RangeSource) -> dict[str, str]:
    """Original tensor name -> shard file name (validated); built once."""
    for name in src.files():
        if name.endswith(".safetensors.index.json"):
            weight_map = json.loads(src.text(name))["weight_map"]
            for shard in set(weight_map.values()):
                if not _SHARD.fullmatch(shard):
                    raise VerifyError(f"unsafe shard name {shard!r} in {name}")
            return dict(weight_map)
    mapping: dict[str, str] = {}
    for name in src.files():
        if name.endswith(".safetensors"):
            header, _ = independent_header(src, name)
            mapping.update(dict.fromkeys((k for k in header if k != "__metadata__"), name))
    return mapping


def _group_arrays(src: RangeSource, file: str, header: dict, group: str) -> GroupArrays:
    raw: dict[str, bytes] = {}
    for field in GROUP_FIELDS:
        info = header[f"{group}.{field}"]
        want_dtype, want_rank = GROUP_FIELD_TYPES[field]
        if info.dtype != want_dtype or len(info.shape) != want_rank:
            raise VerifyError(
                f"{group}.{field}: dtype/rank {info.dtype}/{len(info.shape)} not {want_dtype}/{want_rank}"
            )
        raw[field] = src.read(file, info.offset, info.nbytes)
    if len(raw["output_positions"]) % 4:
        raise VerifyError(f"{group}: output_positions byte length is not a multiple of 4")
    return GroupArrays(
        encoded_exponent=np.frombuffer(raw["encoded_exponent"], np.uint8),
        sign_mantissa=np.frombuffer(raw["sign_mantissa"], np.uint8),
        luts=np.frombuffer(raw["luts"], np.uint8).reshape(-1, 256),
        gaps=np.frombuffer(raw["gaps"], np.uint8),
        output_positions=np.frombuffer(raw["output_positions"], "<u4").astype(np.uint32),
        split_positions=np.frombuffer(raw["split_positions"], "<i8").astype(np.int64),
    )


def pick_groups(stats: dict[str, dict], selector: str) -> list[str]:
    """Resolve a selector list to group names (natural order; KeyError for unknown names)."""
    ordered = sorted(stats, key=natural_key)
    picked: list[str] = []
    for token in selector.split(","):
        if token == "first":
            choice = ordered[0]
        elif token == "last":
            choice = ordered[-1]
        elif token == "max-block":
            choice = max(ordered, key=lambda g: stats[g]["elements_per_block"])
        elif token == "max-code":
            choice = max(ordered, key=lambda g: stats[g]["max_code_length"])
        elif token in stats:
            choice = token
        else:
            raise KeyError(token)
        if choice not in picked:
            picked.append(choice)
    return picked


def verify_group(
    df11: RangeSource,
    bf16: RangeSource | None,
    group: str,
    *,
    config: DF11Config,
    dindex: dict[str, tuple[str, dict]],
    bindex: dict[str, str],
) -> dict:
    """Decode one group and compare each matrix with its BF16 original (if a source is given)."""
    record: dict = {"group": group, "matrices": [], "max_code_length": None, "n_blocks": None}
    try:
        file, header = dindex[group]
        record["file"] = file
        arrays = _group_arrays(df11, file, header, group)
        record["max_code_length"] = max_code_length(arrays.luts)
        record["n_blocks"] = arrays.n_blocks
        names = matrix_names_for(group, config.pattern_dict)
        parts = split_matrices(decode_group(arrays, name=group), arrays.split_positions)
        if bf16 is None:
            record["status"] = "structural-ok"
            record["matrices"] = [
                {"name": n, "n": int(p.size), "equal": True}
                for n, p in zip(names, parts, strict=True)
            ]
            return record
        header_cache: dict[str, tuple[dict, int]] = {}
        for name, got in zip(names, parts, strict=True):
            shard = bindex.get(name)
            if shard is None:
                raise VerifyError(f"no original for {name}")
            if shard not in header_cache:
                header_cache[shard] = independent_header(bf16, shard)
            bheader, base = header_cache[shard]
            meta = bheader[name]
            if meta["dtype"] != "BF16":
                raise VerifyError(f"original {name} is {meta['dtype']}, not BF16")
            start, end = meta["data_offsets"]
            original = np.frombuffer(bf16.read(shard, base + start, end - start), "<u2")
            equal = original.size == got.size and bool(np.array_equal(got, original))
            record["matrices"].append({"name": name, "n": int(got.size), "equal": equal})
        record["status"] = "equal" if all(m["equal"] for m in record["matrices"]) else "mismatch"
    except (DFloatError, VerifyError, KeyError, ValueError) as exc:
        record["status"] = "error"
        record["error"] = str(exc)
    return record


def main(
    argv: list[str] | None = None,
    *,
    source_factory: Callable[[str, str, str], RangeSource] | None = None,
) -> int:
    """CLI entry point; any failure other than a real mismatch exits 2.

    ``source_factory(repo, revision, subdir)`` builds the ``RangeSource`` for the DF11 and BF16
    repos (subdir is always ``""`` for the DF11 side); it defaults to :class:`HfRangeSource`.
    Tests pass a factory that returns :class:`LocalRangeSource` over on-disk fixtures with
    explicit ``--df11-revision``/``--bf16-revision``, so no network is touched.
    """
    factory = source_factory or (
        lambda repo, revision, subdir: HfRangeSource(repo, revision, subdir)
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--df11-repo", required=True)
    parser.add_argument("--df11-revision")
    parser.add_argument("--bf16-repo")
    parser.add_argument("--bf16-revision")
    parser.add_argument("--bf16-subdir", default="")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="decode and validate structure only; no BF16 comparison",
    )
    parser.add_argument("--groups", default="first,last,max-block,max-code")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-budget", type=float, default=3 * 3600.0)
    args = parser.parse_args(argv)
    if not args.bf16_repo and not args.structural_only:
        print("error: either --bf16-repo or --structural-only is required", file=sys.stderr)
        return 2
    mode = "structural-only" if args.structural_only else "parity"
    install_memory_caps()
    watchdog = Watchdog(args.out.parent, ceiling=default_ceiling(), budget=args.wall_budget).start()
    try:
        api = HfApi()
        df11_rev = args.df11_revision or api.model_info(args.df11_repo).sha
        df11 = factory(args.df11_repo, df11_rev, "")
        config = parse_df11_config(
            json.loads(df11.text("config.json"))["dfloat11_config"], source="config.json"
        )
        bf16, bf16_rev, bindex = None, None, {}
        if mode == "parity":
            bf16_rev = args.bf16_revision or api.model_info(args.bf16_repo).sha
            bf16 = factory(args.bf16_repo, bf16_rev, args.bf16_subdir)
            bindex = bf16_index(bf16)
        dindex = df11_index(df11)
        stats = {}
        for group, (file, header) in dindex.items():
            pos_info, lut_info = header[f"{group}.output_positions"], header[f"{group}.luts"]
            positions = np.frombuffer(df11.read(file, pos_info.offset, pos_info.nbytes), "<u4")
            luts = np.frombuffer(
                df11.read(file, lut_info.offset, lut_info.nbytes), np.uint8
            ).reshape(-1, 256)
            stats[group] = {
                "elements_per_block": max_elements_per_block(positions)
                if positions.size > 1
                else 0,
                "max_code_length": max_code_length(luts),
            }
        records = [
            verify_group(df11, bf16, g, config=config, dindex=dindex, bindex=bindex)
            for g in pick_groups(stats, args.groups)
        ]
        args.out.write_text(
            json.dumps(
                {
                    "df11_repo": args.df11_repo,
                    "df11_revision": df11_rev,
                    "bf16_repo": args.bf16_repo,
                    "bf16_revision": bf16_rev,
                    "mode": mode,
                    "groups": records,
                },
                indent=1,
            )
        )
        if any(r["status"] == "error" for r in records):
            return 2
        return 1 if any(r["status"] == "mismatch" for r in records) else 0
    except Exception:
        traceback.print_exc()
        print("error: see traceback", file=sys.stderr)
        return 2
    finally:
        watchdog.stop()


if __name__ == "__main__":
    raise SystemExit(main())
