"""The ``mlx-dfloat`` command: ``generate`` today; ``selftest`` and ``bench`` later."""

import argparse
from collections.abc import Sequence

from mlx_dfloat._version import __version__
from mlx_dfloat.mflux.flux1.cli import add_generate_parser


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser with its subcommands (no mflux import: each subcommand imports what it runs)."""
    parser = argparse.ArgumentParser(
        prog="mlx-dfloat",
        description="Run DFloat11 losslessly compressed BF16 models on Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"mlx-dfloat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    add_generate_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and run the chosen subcommand; the return value is the process exit code."""
    args = build_parser().parse_args(argv)
    run = args.run
    return int(run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
