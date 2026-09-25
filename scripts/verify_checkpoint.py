"""Whole-checkpoint DF11 parity: decode every group, compare every matrix with the BF16 original.

Heavy-runs script (not a pytest lane). Results are written per group, atomically, as they finish,
and resumed only when the run key (revisions, mode, decoder, source hash, git, mlx) is unchanged.

Usage:
    uv run python scripts/verify_checkpoint.py --df11 DIR [--bf16 DIR] --out DIR \
        [--df11-revision SHA] [--bf16-revision SHA] [--groups a,b] [--ignore-original NAME ...]
Exit codes: 0 equal and complete, 1 mismatch, 2 error/coverage, 70/71 watchdog abort.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil
from scripts._watchdog import Watchdog, default_ceiling

from mlx_dfloat._memory_caps import install_memory_caps
from mlx_dfloat._safetensors import read_array
from mlx_dfloat.errors import DFloatError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.reference import decode_matrices

EXIT_OK, EXIT_MISMATCH, EXIT_ERROR = 0, 1, 2
_SNAPSHOT = re.compile(r"/snapshots/([0-9a-f]{40})(/|$)")
_SHARD = re.compile(r"[A-Za-z0-9_.\-]+\.safetensors")
_SRC = Path(__file__).resolve().parents[1] / "src" / "mlx_dfloat"


class VerifyError(Exception):
    """An input or tool problem (exit 2), distinct from a bit mismatch (exit 1)."""


def natural_key(name: str) -> list[tuple[int, int | str]]:
    """Sort key that orders layers.2 before layers.10 and never compares int with str."""
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", name)]


def revision_from_path(path: Path) -> str:
    """Hugging Face cache snapshot SHA from a path, or "unknown"."""
    match = _SNAPSHOT.search(str(path.resolve()))
    return match.group(1) if match else "unknown"


def source_hash() -> str:
    """sha256 over every package source file (reader, format, decoder)."""
    digest = hashlib.sha256()
    for file in sorted(_SRC.glob("*.py")):
        digest.update(file.name.encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()


def run_key(
    *, df11_revision: str, bf16_revision: str, mode: str, decoder: str = "reference"
) -> dict[str, str]:
    """Everything that must be unchanged for a stored result to be reused."""
    repo = _SRC.parents[1]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        git = sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        git = "unknown"
    return {
        "df11_revision": df11_revision,
        "bf16_revision": bf16_revision,
        "mode": mode,
        "decoder": decoder,
        "source": source_hash(),
        "git": git,
        "mlx": mx.__version__,
    }


def _shard(root: Path, name: str) -> Path:
    if not _SHARD.fullmatch(name):
        raise VerifyError(f"unsafe shard name {name!r} in the original's index")
    path = root / name
    if path.resolve().parent != root.resolve():
        raise VerifyError(f"shard {name!r} resolves outside {root}")
    return path


def index_original(root: Path) -> dict[str, Path]:
    """Map each original tensor name to its shard (validated to stay inside ``root``)."""
    for index in sorted(root.glob("*.safetensors.index.json")):
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: _shard(root, shard) for name, shard in weight_map.items()}
    mapping: dict[str, Path] = {}
    for shard in sorted(root.glob("*.safetensors")):
        for name in mx.load(str(shard)):
            mapping[name] = shard
    return mapping


def _load_bits(shard: Path, names: list[str]) -> dict[str, np.ndarray]:
    try:
        loaded = mx.load(str(shard))
        out: dict[str, np.ndarray] = {}
        for n in names:
            a = loaded.pop(n)
            if a.dtype != mx.bfloat16:
                raise VerifyError(f"original {n} is {a.dtype}, not bfloat16")
            out[n] = np.array(a.view(mx.uint16)).reshape(-1)
            del a
        del loaded
    except (RuntimeError, ValueError, KeyError) as exc:
        raise VerifyError(f"cannot load originals from {shard.name}: {exc}") from exc
    mx.clear_cache()
    return out


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name) + ".json"


def _write_atomic(path: Path, obj: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def _stored(path: Path, key: dict[str, str]) -> dict | None:
    if "unknown" in (key["df11_revision"], key["bf16_revision"]):
        return None
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return stored if isinstance(stored, dict) and stored.get("key") == key else None


def _compare(
    decoded: dict[str, np.ndarray], originals: dict[str, Path]
) -> tuple[list[dict], list[str]]:
    by_shard: dict[Path, list[str]] = {}
    missing: list[str] = []
    for matrix in decoded:
        if matrix in originals:
            by_shard.setdefault(originals[matrix], []).append(matrix)
        else:
            missing.append(matrix)
    records = []
    for shard, names in by_shard.items():
        for matrix, original in _load_bits(shard, names).items():
            got = decoded[matrix]
            equal = got.size == original.size and bool(np.array_equal(got, original))
            first = (
                int(np.flatnonzero(got != original)[0])
                if not equal and got.size == original.size
                else None
            )
            records.append(
                {"name": matrix, "n": int(got.size), "equal": equal, "first_mismatch_index": first}
            )
    return records, missing


def verify(
    df11_root: Path,
    bf16_root: Path | None,
    out_dir: Path,
    *,
    key: dict[str, str],
    groups: list[str] | None = None,
    ignore_originals: tuple[str, ...] = (),
) -> int:
    """Run parity (or structural-only decoding) and return the exit code."""
    install_memory_caps()
    mx.set_cache_limit(0)
    started = time.monotonic()
    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    try:
        ckpt = open_checkpoint(df11_root)
        originals = index_original(bf16_root) if bf16_root else {}
    except (DFloatError, VerifyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    selected = sorted(groups or ckpt.groups, key=natural_key)
    unknown = [g for g in selected if g not in ckpt.groups]
    if unknown:
        print(f"error: unknown groups {unknown[:5]}", file=sys.stderr)
        return EXIT_ERROR
    expected = sum(len(ckpt.groups[g].matrix_names) for g in selected)
    compared = mismatched = 0
    missing: list[str] = []
    consumed: set[str] = set()
    for name in selected:
        result_path = groups_dir / _safe_filename(name)
        stored = _stored(result_path, key)
        if stored is None:
            t0 = time.monotonic()
            try:
                decoded = decode_matrices(ckpt.groups[name])
                if bf16_root is None:
                    records = [
                        {"name": n, "n": int(v.size), "equal": True, "first_mismatch_index": None}
                        for n, v in decoded.items()
                    ]
                    status = "structural-ok"
                else:
                    records, group_missing = _compare(decoded, originals)
                    missing += group_missing
                    status = "equal" if all(r["equal"] for r in records) else "mismatch"
            except (DFloatError, VerifyError) as exc:
                print(f"error in {name}: {exc}", file=sys.stderr)
                return EXIT_ERROR
            stored = {
                "key": key,
                "status": status,
                "seconds": time.monotonic() - t0,
                "rss": int(psutil.Process().memory_info().rss),
                "matrices": records,
            }
            _write_atomic(result_path, stored)
            del decoded
        compared += len(stored["matrices"])
        mismatched += sum(not m["equal"] for m in stored["matrices"])
        consumed.update(m["name"] for m in stored["matrices"])
    extras_compared = 0
    if bf16_root is not None:
        try:
            for extra, (path, info) in sorted(ckpt.extras.items()):
                if extra not in originals or info.dtype != "BF16":
                    continue
                original = _load_bits(originals[extra], [extra])[extra]
                ours = np.asarray(read_array(path, info)).reshape(-1)
                extras_compared += 1
                consumed.add(extra)
                if not np.array_equal(ours, original):
                    mismatched += 1
                    print(f"mismatch in extra {extra}", file=sys.stderr)
        except VerifyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
    uncovered: list[str] = []
    if bf16_root is not None and groups is None:
        uncovered = sorted(set(originals) - set(ckpt.extras) - consumed - set(ignore_originals))
    complete = compared == expected and not missing and not uncovered
    code = EXIT_MISMATCH if mismatched else (EXIT_OK if complete else EXIT_ERROR)
    _write_atomic(
        out_dir / "summary.json",
        {
            "key": key,
            "exit_code": code,
            "groups": len(selected),
            "compared": compared,
            "expected": expected,
            "extras_compared": extras_compared,
            "mismatched": mismatched,
            "missing_originals": missing,
            "uncovered_originals": uncovered,
            "ignored_originals": list(ignore_originals),
            "seconds": time.monotonic() - started,
        },
    )
    return code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; any unexpected failure exits 2, never 1."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--df11", type=Path, required=True)
    parser.add_argument("--bf16", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--df11-revision")
    parser.add_argument("--bf16-revision")
    parser.add_argument("--groups", help="comma-separated group names (default: all)")
    parser.add_argument("--ignore-original", action="append", default=[])
    parser.add_argument("--decoder", choices=["reference"], default="reference")
    parser.add_argument("--wall-budget", type=float, default=6 * 3600.0)
    parser.add_argument("--no-watchdog", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    watchdog = None
    try:
        mode = "parity" if args.bf16 else "structural-only"
        key = run_key(
            df11_revision=args.df11_revision or revision_from_path(args.df11),
            bf16_revision=args.bf16_revision
            or (revision_from_path(args.bf16) if args.bf16 else "none"),
            mode=mode,
            decoder=args.decoder,
        )
        if not args.no_watchdog:
            watchdog = Watchdog(
                args.out, ceiling=default_ceiling(), budget=args.wall_budget
            ).start()
        groups = args.groups.split(",") if args.groups else None
        return verify(
            args.df11,
            args.bf16,
            args.out,
            key=key,
            groups=groups,
            ignore_originals=tuple(args.ignore_original),
        )
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    finally:
        if watchdog is not None:
            watchdog.stop()


if __name__ == "__main__":
    raise SystemExit(main())
