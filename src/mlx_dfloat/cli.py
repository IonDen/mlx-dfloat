"""The ``mlx-dfloat`` command: ``generate`` and ``selftest``."""

import argparse
import json
import sys
from collections.abc import Sequence

from mlx_dfloat._version import __version__
from mlx_dfloat.errors import DFloatError
from mlx_dfloat.mflux.flux1.cli import add_generate_parser


def _run_selftest(args: argparse.Namespace) -> int:
    """Run the GPU decoder self-check: exit 0 all pass, 1 any check failed, 2 it could not run."""
    from mlx_dfloat.decode import selftest

    try:
        report = selftest()
    except (DFloatError, OSError) as exc:
        print(f"selftest: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # anything else is still a tool error (2), never a silent exit 1
        print(f"selftest: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            f"device: {report.device}  mlx {report.mlx_version}  mlx-dfloat {report.package_version}"
        )
        for c in report.checks:
            detail = f"  {c.detail}" if c.detail else ""
            status = "PASS" if c.ok else "FAIL"
            print(f"{status}  {c.group}  {c.path}  {c.elements} elements{detail}")
        if report.checks:
            failed = sum(not c.ok for c in report.checks)
            verdict = "ok" if report.ok else f"{failed} of {len(report.checks)} checks failed"
            print(f"selftest {verdict} in {report.seconds:.2f} s")
        else:
            print(f"selftest could not run: {report.reason or 'no checks ran'}")
    if not report.checks:
        print(report.reason or "the self-check ran no checks", file=sys.stderr)
        return 2
    return 0 if report.ok else 1


def add_selftest_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Register ``selftest``: decode the packaged canary groups on the GPU and with the CPU reference."""
    parser = sub.add_parser(
        "selftest",
        help="check the GPU decoder against groups with known output",
        description=(
            "Decode two packaged groups on both Metal write paths and with the CPU reference, "
            "and compare each with its known BF16 bits. Exit 0 when every check passes, 1 when "
            "a check fails, 2 when the GPU decoder cannot run here."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.set_defaults(run=_run_selftest)


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser with its subcommands (no mflux import: each subcommand imports what it runs)."""
    parser = argparse.ArgumentParser(
        prog="mlx-dfloat",
        description="Run DFloat11 losslessly compressed BF16 models on Apple Silicon.",
    )
    parser.add_argument("--version", action="version", version=f"mlx-dfloat {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    add_generate_parser(sub)
    add_selftest_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and run the chosen subcommand; the return value is the process exit code."""
    args = build_parser().parse_args(argv)
    run = args.run
    return int(run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
