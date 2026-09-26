"""Whole-checkpoint DF11 parity: decode every group, compare every matrix with the BF16 original.

Operational script, not a test; run it directly. Results are written per group, atomically, as
they finish, and resumed only when the run key (revisions, mode, decoder, source hash, git, mlx)
is unchanged.

Usage (from the repository root of a synced checkout):
    uv run python -m scripts.verify_checkpoint --df11 DIR [--bf16 DIR] --out DIR \
        [--df11-revision SHA] [--bf16-revision SHA] [--groups a,b] [--ignore-original NAME ...]
``uv run python scripts/verify_checkpoint.py ...`` works too.
Exit codes: 0 equal and complete, 1 mismatch, 2 error/coverage, 70/71 watchdog abort.
"""

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Protocol

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path, and the
# `scripts.` imports below would fail.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx
    import numpy as np
    import psutil
    from scripts._watchdog import Watchdog, default_ceiling

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat._safetensors import read_array
    from mlx_dfloat.errors import DFloatError
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.reference import decode_matrices
except Exception as exc:  # a broken environment (no Metal, bad install) is 2, never a mismatch (1)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_MISMATCH, EXIT_ERROR = 0, 1, 2
_SNAPSHOT = re.compile(r"/snapshots/([0-9a-f]{40})(/|$)")
_SHARD = re.compile(r"[A-Za-z0-9_.\-]+\.safetensors")
_SRC = Path(__file__).resolve().parents[1] / "src" / "mlx_dfloat"
_REPO = _SRC.parents[1]
_SCRIPTS = Path(__file__).resolve().parent


class VerifyError(Exception):
    """An input or tool problem (exit 2), distinct from a bit mismatch (exit 1)."""


class Stoppable(Protocol):
    """What ``verify`` needs from a watchdog: a way to stop it before the verdict is written."""

    def stop(self) -> None: ...  # noqa: D102


def natural_key(name: str) -> list[tuple[int, int | str]]:
    """Sort key that orders layers.2 before layers.10 and never compares int with str."""
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", name)]


def revision_from_path(path: Path) -> str:
    """Hugging Face cache snapshot SHA from a path, or "unknown"."""
    match = _SNAPSHOT.search(str(path.resolve()))
    return match.group(1) if match else "unknown"


def source_hash() -> str:
    """sha256 over every package source file plus this script and the watchdog module.

    Recurses through ``src/mlx_dfloat`` (not just its top level) and also covers
    ``scripts/verify_checkpoint.py`` and ``scripts/_watchdog.py`` themselves, so an edit to the
    parity script or its watchdog invalidates a stored result too, not just an edit to the package.
    """
    files = sorted(_SRC.rglob("*.py")) + sorted(
        [_SCRIPTS / "_watchdog.py", _SCRIPTS / "verify_checkpoint.py"]
    )
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(_REPO)).encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()


def run_key(
    *, df11_revision: str, bf16_revision: str, mode: str, decoder: str = "reference"
) -> dict[str, str]:
    """Everything that must be unchanged for a stored result to be reused."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO), "status", "--porcelain", "--", "src", "scripts"],
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
    """A shard path directly inside ``root``, following symlinks (HF cache blobs are regular files).

    ``_SHARD`` already forbids ``/`` (and so ``..``) in ``name``, so ``root / name`` cannot escape
    ``root`` regardless of what the resulting path resolves to; a Hugging Face cache snapshot shard
    is itself a symlink into a sibling ``blobs/`` directory, so the check is "does this land on a
    regular file", not "does the resolved path stay under root".
    """
    if not _SHARD.fullmatch(name):
        raise VerifyError(f"unsafe shard name {name!r} in the original's index")
    path = root / name
    try:
        is_regular = stat.S_ISREG(path.stat().st_mode)
    except OSError as exc:
        raise VerifyError(f"shard {name!r} cannot be read: {exc}") from exc
    if not is_regular:
        raise VerifyError(f"shard {name!r} is not a regular file")
    return path


def index_original(root: Path) -> dict[str, Path]:
    """Map each original tensor name to its shard (validated to stay inside ``root``)."""
    for index in sorted(root.glob("*.safetensors.index.json")):
        data = json.loads(index.read_text())
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()
        ):
            raise VerifyError(f"{index.name}: no weight_map object of tensor name -> shard name")
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


def _stored(path: Path, key: dict[str, str], matrix_names: tuple[str, ...]) -> dict | None:
    """A previously-written group result, only if its key AND matrix names still match.

    A matrix with no matching original is recorded in ``missing``, not ``matrices``, so the check
    is against the union of the two (order-insensitive) rather than ``matrices`` alone — a real
    coverage gap must still resume, and only a name that appears in neither must force a recompute.
    """
    if "unknown" in (key["df11_revision"], key["bf16_revision"]):
        return None
    try:
        stored = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(stored, dict) or stored.get("key") != key:
        return None
    try:
        names = [str(m.get("name", "")) for m in stored.get("matrices", [])]
        names += [str(n) for n in stored.get("missing", [])]
    except AttributeError:  # a matrices entry isn't itself a dict: an unusably malformed record
        return None
    if sorted(names) != sorted(matrix_names):
        return None
    return stored


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
            if got.size != original.size:
                # A size mismatch is a mapping/format problem, not a bit mismatch: it means the
                # decoded matrix and its "original" don't describe the same tensor at all.
                raise VerifyError(
                    f"{matrix}: decoded {got.size} elements, original has {original.size}"
                )
            equal = bool(np.array_equal(got, original))
            first = int(np.flatnonzero(got != original)[0]) if not equal else None
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
    watchdog: Stoppable | None = None,
) -> int:
    """Run parity (or structural-only decoding) and return the exit code.

    ``watchdog`` is stopped before summary.json is written, so no abort can follow the verdict.
    """
    caps = list(install_memory_caps())
    mx.set_cache_limit(0)
    started = time.monotonic()
    compared = mismatched = 0

    def _finish(summary: dict) -> int:
        if watchdog is not None:
            watchdog.stop()
        _write_atomic(
            out_dir / "summary.json",
            {
                "key": key,
                **summary,
                "memory_caps_gb": caps,
                "seconds": time.monotonic() - started,
            },
        )
        return int(summary["exit_code"])

    def _error_summary(message: str) -> int:
        # "Error wins" (exit 2), but a mismatch found before the error stays visible here.
        print(f"error: {message}", file=sys.stderr)
        return _finish(
            {
                "exit_code": EXIT_ERROR,
                "error": message,
                "compared": compared,
                "mismatched": mismatched,
            }
        )

    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        # A stale summary from a previous (possibly differently-scoped) run must never be
        # mistaken for this run's result; move it aside rather than deleting it.
        summary_path.replace(out_dir / "summary.previous.json")
    if groups is not None and not groups:
        # `groups or ckpt.groups` would otherwise treat `[]` the same as `None` (run every group)
        # while `groups is None` (which gates the coverage check below) is False for `[]` -- an
        # empty list must never silently run the full checkpoint with coverage checking disabled.
        return _error_summary("groups is an empty list; pass None for all")
    try:
        ckpt = open_checkpoint(df11_root)
        originals = index_original(bf16_root) if bf16_root else {}
    except (DFloatError, VerifyError, OSError, ValueError) as exc:
        return _error_summary(str(exc))
    selected = sorted(groups or ckpt.groups, key=natural_key)
    unknown = [g for g in selected if g not in ckpt.groups]
    if unknown:
        return _error_summary(f"unknown groups {unknown[:5]}")
    expected = sum(len(ckpt.groups[g].matrix_names) for g in selected)
    if expected == 0:
        return _error_summary("checkpoint has zero DF11 groups to verify")
    missing: list[str] = []
    consumed: set[str] = set()
    for name in selected:
        matrix_names = ckpt.groups[name].matrix_names
        result_path = groups_dir / _safe_filename(name)
        stored = _stored(result_path, key, matrix_names)
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
                    group_missing: list[str] = []
                else:
                    records, group_missing = _compare(decoded, originals)
                    status = "equal" if all(r["equal"] for r in records) else "mismatch"
            except (DFloatError, VerifyError) as exc:
                return _error_summary(f"{name}: {exc}")
            stored = {
                "key": key,
                "status": status,
                "seconds": time.monotonic() - t0,
                "rss": int(psutil.Process().memory_info().rss),
                "matrices": records,
                "missing": group_missing,
            }
            _write_atomic(result_path, stored)
            del decoded
        compared += len(stored["matrices"])
        mismatched += sum(not m["equal"] for m in stored["matrices"])
        consumed.update(m["name"] for m in stored["matrices"])
        missing += stored.get("missing", [])
    extras_compared = 0
    if bf16_root is not None:
        try:
            for extra, (path, info) in sorted(ckpt.extras.items()):
                if extra not in originals:
                    continue
                if info.dtype != "BF16":
                    raise VerifyError(
                        f"extra {extra!r} is {info.dtype}, not BF16, but has a same-named original"
                    )
                original = _load_bits(originals[extra], [extra])[extra]
                ours = np.asarray(read_array(path, info)).reshape(-1)
                if ours.size != original.size:
                    # A size mismatch is a mapping/format problem, not a bit mismatch: same
                    # reasoning as `_compare`'s matrix-size check, applied to extras.
                    raise VerifyError(
                        f"extra {extra!r}: has {ours.size} elements, original has {original.size}"
                    )
                extras_compared += 1
                consumed.add(extra)
                if not np.array_equal(ours, original):
                    mismatched += 1
                    print(f"mismatch in extra {extra}", file=sys.stderr)
        except VerifyError as exc:
            return _error_summary(str(exc))
    uncovered: list[str] = []
    if bf16_root is not None and groups is None:
        uncovered = sorted(set(originals) - consumed - set(ignore_originals))
    complete = compared == expected and not missing and not uncovered
    code = EXIT_MISMATCH if mismatched else (EXIT_OK if complete else EXIT_ERROR)
    return _finish(
        {
            "exit_code": code,
            "groups": len(selected),
            "partial": groups is not None,
            "selected_groups": selected,
            "compared": compared,
            "expected": expected,
            "extras_compared": extras_compared,
            "mismatched": mismatched,
            "missing_originals": missing,
            "uncovered_originals": uncovered,
            "ignored_originals": list(ignore_originals),
        }
    )


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
            abort_path = args.out / "abort.json"
            if abort_path.exists():
                # A stale abort artifact from a previous run must never be mistaken for this
                # run's outcome; move it aside rather than deleting it.
                abort_path.replace(args.out / "abort.previous.json")
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
            watchdog=watchdog,
        )
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    finally:
        if watchdog is not None:
            watchdog.stop()


if __name__ == "__main__":
    raise SystemExit(main())
