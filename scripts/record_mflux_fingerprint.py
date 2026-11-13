"""Record the installed mflux's source fingerprint as a new row of ``tests/fixtures/mflux_fingerprint.json``.

    python -m scripts.record_mflux_fingerprint [--out tests/fixtures/mflux_fingerprint.json] [--replace]

Reads only the installed mflux files (its version from the distribution metadata, its directory from the import
system, without importing it). Rows are added, never edited: an existing row for the installed release is refused
unless ``--replace`` is given.

When ``test_the_measured_mflux_code_is_unchanged`` goes red on a new mflux: read the diff of the files it names,
decide whether the change has no effect, needs a code change, or needs the measurements repeated, and in the change
that raises ``NEWEST_TESTED_MFLUX`` run this script on the new release to add its row.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from scripts._mflux_fingerprint import SCOPE, fingerprint, unmatched_globs

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mflux_fingerprint.json"


def _installed() -> tuple[str, Path] | None:
    """The installed mflux's version and package directory, found without importing it; None when absent."""
    spec = importlib.util.find_spec("mflux")
    if spec is None or not spec.submodule_search_locations:
        return None
    return metadata.version("mflux"), Path(next(iter(spec.submodule_search_locations)))


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to a temp file next to ``path``, then rename it over ``path`` (os.replace): a failed write keeps the old file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600; the fixture is an ordinary tracked file
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Add the installed mflux's row; 0 when written, 1 when refused, 2 when mflux is not installed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--replace", action="store_true", help="overwrite the installed release's existing row"
    )
    args = parser.parse_args(argv)

    found = _installed()
    if found is None:
        print("mflux is not installed", file=sys.stderr)
        return 2
    version, root = found
    missing = unmatched_globs(root, SCOPE)
    if missing:
        print(f"mflux {version} at {root}: no file matches {missing}", file=sys.stderr)
        return 1

    data = json.loads(args.out.read_text()) if args.out.exists() else {"versions": {}}
    if not isinstance(data.get("versions"), dict):
        print(
            f"{args.out} has no 'versions' table of rows; fix the file by hand or write to a new --out",
            file=sys.stderr,
        )
        return 1
    if data.get("scope", list(SCOPE)) != list(SCOPE):
        print(
            f"{args.out} was recorded with another scope; rows must share one scope",
            file=sys.stderr,
        )
        return 1
    if version in data["versions"] and not args.replace:
        print(
            f"{args.out} already has a row for mflux {version}; pass --replace to overwrite it",
            file=sys.stderr,
        )
        return 1
    data["scope"] = list(SCOPE)
    data["versions"][version] = fingerprint(root, SCOPE)
    write_atomically(args.out, json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(
        f"recorded {len(data['versions'][version])} files of mflux {version} from {root} in {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
