"""Isolated decode throughput of the Metal DF11 kernel, per group, gated on bit-exact parity.

For each selected group: load it, decode it with the NumPy reference, and check every requested
Metal variant against those bits (``view(uint16)`` equality, never a tolerance). Before the first
full-group dispatch, the per-dispatch guard projects that dispatch's time from ``--rate-from``, or,
without it, from a calibration ramp: dispatches of the first k blocks for k = 1, 2, 4, 8, ..., each
projected at the previous step's rate through the same guard before it runs, until a step takes at
least 10 ms (long enough to measure throughput, not launch latency) or covers the whole group,
unless that step's rate fell below the previous step's (a transient slow dispatch). Each step runs
once untimed and then keeps the fastest of three timed runs per variant, so neither a pipeline
compile nor a one-off slow dispatch sets a rate. The last step's rate (slowest requested variant) is
the calibration rate. Each variant that passes parity then gets one warm-up and ``--reps`` timed
decodes, each ending in ``mx.eval``. A variant that fails parity gets no timing: a wrong kernel's
speed means nothing.

Variants: ``direct`` (every block writes straight to device memory), ``staged`` (the production
path: blocks that fit the threadgroup buffer are staged, the rest go direct) and ``reference``
(the NumPy decoder's own time, for scale).

Results go to ``--out`` atomically after every group, so an interrupted run resumes by skipping
groups already in the file's ``groups`` map. The file carries a run ``key`` (resolved checkpoint
path, variants, reps, source hash, mlx version); resuming into a file whose key differs, or that has
none, exits 2 and asks for a fresh ``--out``, so different runs never mix. The top-level ``gbps`` is
the slowest Metal median over every recorded group, the conservative rate
``verify_checkpoint --rate-from`` reads. ``--t-step S`` also reports the overhead-line throughput,
the decode rate at which decoding the whole checkpoint's bytes per step costs exactly 25 % of a
step, and whether each Metal variant's aggregate throughput clears it.

Usage (from the repository root of a synced checkout):
    uv run python -m scripts.bench_decode_kernel --df11 DIR --groups a,b --out FILE \
        [--reps 5] [--variants direct,staged,reference] [--rate-from JSON] \
        [--wall-budget S] [--t-step S]
Exit codes: 0 every variant equal, 1 a bit mismatch, 2 an input, format or tool error,
70/71 watchdog abort (footprint ceiling / wall budget).
"""

import argparse
import dataclasses
import json
import sys
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx
    import numpy as np
    from scripts._bench_common import (
        Timing,
        bench_exit_code,
        calibration_rate,
        gbps,
        kill_equivalent_throughput,
        move_stale_abort_aside,
        per_dispatch_guard,
        projected_bytes,
        provenance,
        ramp_next_k,
        ramp_should_stop,
        resume_key_diff,
        time_first_then_min,
        write_json_atomic,
    )
    from scripts._watchdog import Watchdog, default_ceiling, phys_footprint
    from scripts.verify_checkpoint import VerifyError, natural_key, rate_from, source_hash

    from mlx_dfloat import _metal_decode
    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat.decode import DecodeResult, available_backends, check, decode_group
    from mlx_dfloat.errors import DFloatError
    from mlx_dfloat.format import MxGroup, load_group_mx, open_checkpoint
except Exception as exc:  # a broken environment is 2, never a mismatch (1)
    print(f"error: cannot import the project modules ({exc})", file=sys.stderr)
    raise SystemExit(2) from exc

VARIANTS = ("direct", "staged", "reference")
METAL_VARIANTS = ("direct", "staged")
CACHE_LIMIT = int(1.4e9)


class BenchError(Exception):
    """An input or tool problem for one group (exit 2), distinct from a bit mismatch (exit 1)."""


def _metal(group: MxGroup, variant: str) -> DecodeResult:
    return _metal_decode.decode(group, force_direct=variant == "direct")


def _timed(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _run_metal(group: MxGroup, variant: str) -> None:
    res = _metal(group, variant)
    mx.eval(res.bits, res.status)


def _calibrate(group: MxGroup, variants: list[str]) -> tuple[float, list[dict[str, object]]]:
    """Decode rate (bytes/s) from a guarded ramp of block prefixes; returns it and the steps.

    Each step dispatches the first ``k`` blocks (``positions[:k+1]`` over the full arrays, so the
    kernel launches ``k`` threadgroups) once untimed and then three times per variant, and keeps
    the fastest of the three (``time_first_then_min``): the first run absorbs a pipeline compile
    for a new buffer binding, and the minimum filters a transient slow dispatch. Both timings are
    recorded per variant. A step's rate is its slowest variant's; every step after the first is
    projected at the previous step's rate through ``per_dispatch_guard`` before it runs. A step
    of 10 ms or more ends the ramp only if its rate did not fall below the previous step's
    (``ramp_should_stop``). The calibration rate is the last step's (``calibration_rate``).

    Raises:
        RuntimeError: The guard refuses a ramp step.
    """
    positions = np.array(group.positions).astype(np.int64)
    steps: list[dict[str, object]] = []
    rates: list[float] = []
    k: int | None = 1
    while k is not None:
        bytes_k = projected_bytes(positions, k)
        if rates:
            per_dispatch_guard(bytes_k, rates[-1])
        prefix = dataclasses.replace(
            group, positions=group.positions[: k + 1], intervals=group.intervals[:k], n_launch=k
        )
        timings = {v: time_first_then_min(partial(_run_metal, prefix, v)) for v in variants}
        seconds = max(timed for _, timed in timings.values())
        rates.append(bytes_k / seconds)
        steps.append(
            {
                "k": k,
                "bytes": bytes_k,
                "first_ms": {v: first * 1e3 for v, (first, _) in timings.items()},
                "timed_ms": {v: timed * 1e3 for v, (_, timed) in timings.items()},
                "rate_bps": rates[-1],
            }
        )
        firsts = ", ".join(f"{v} {f * 1e3:.3f}" for v, (f, _) in timings.items())
        print(
            f"  ramp k={k}: {bytes_k} bytes in {seconds * 1e3:.3f} ms (first: {firsts} ms) "
            f"-> {rates[-1] / 1e9:.3f} GB/s"
        )
        prev_rate = rates[-2] if len(rates) > 1 else None
        stop = ramp_should_stop(seconds, k, group.n_launch, rate=rates[-1], prev_rate=prev_rate)
        k = None if stop else ramp_next_k(k, group.n_launch)
    return calibration_rate(rates), steps


def bench_group(
    group: MxGroup, *, variants: list[str], reps: int, rate_bps: float | None
) -> dict[str, object]:
    """Parity-check, then time, every requested variant on one group.

    Raises:
        BenchError: The per-dispatch guard refuses a calibration step or the group.
        DFloatError: A block's status word reports an error, or the reference finds the group
            structurally invalid.
    """
    bytes_out = 2 * group.n_elements
    metal = [v for v in variants if v in METAL_VARIANTS]
    reference_bits = np.array(decode_group(group, backend="reference").bits)
    record: dict[str, object] = {
        "n_elements": group.n_elements,
        "n_bytes": group.n_bytes,
        "n_launch": group.n_launch,
        "bytes_out": bytes_out,
        "max_elements_per_block": group.max_elements_per_block,
        "direct_blocks": int(np.count_nonzero(group.intervals > _metal_decode.CAP)),
    }
    if metal:
        try:
            if rate_bps is not None:
                rate, guard = rate_bps, {"rate_bps": rate_bps, "source": "rate-from"}
            else:
                rate, ramp = _calibrate(group, metal)
                guard = {"rate_bps": rate, "source": "calibration", "ramp": ramp}
            record["guard"] = guard
            per_dispatch_guard(bytes_out, rate)
        except RuntimeError as exc:
            raise BenchError(str(exc)) from exc
    results: dict[str, object] = {}
    for variant in variants:
        if variant == "reference":
            fn: Callable[[], object] = partial(decode_group, group, backend="reference")
            entry: dict[str, object] = {}
        else:
            res = _metal(group, variant)
            check(res, name=f"{group.name} ({variant})")
            got = np.array(res.bits)
            equal = got.shape == reference_bits.shape and bool(np.array_equal(got, reference_bits))
            entry = {"parity": equal, "direct_blocks": res.direct_blocks}
            del res, got
            if not equal:
                results[variant] = entry
                continue
            fn = partial(_run_metal, group, variant)
        fn()  # warm-up
        timing = Timing(reps=tuple(_timed(fn) for _ in range(reps)))
        entry |= {
            "reps_s": list(timing.reps),
            "median_s": timing.median,
            "spread": timing.spread,
            "gbps": gbps(bytes_out, timing.median),
        }
        results[variant] = entry
    record["variants"] = results
    return record


def _load_existing(path: Path, key: dict[str, object]) -> dict[str, Any]:
    """The bench file to resume, or a fresh one carrying ``key``.

    Raises:
        VerifyError: The file is unreadable, has no groups map, or was written by a run whose
            key differs from ``key`` (or carries none).
    """
    if not path.exists():
        return {"key": key, "groups": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise VerifyError(f"--out {path}: cannot read the existing bench JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("groups"), dict):
        raise VerifyError(
            f"--out {path}: not a bench JSON with a groups map; refusing to overwrite"
        )
    diff = resume_key_diff(data.get("key"), key)
    if diff:
        raise VerifyError(
            f"--out {path} was written by a different run (differs in: {', '.join(diff)}); "
            "use a fresh --out"
        )
    return data


def _mismatches(groups: dict[str, dict]) -> int:
    return sum(
        1
        for record in groups.values()
        for entry in record.get("variants", {}).values()
        if entry.get("parity") is False
    )


def _slowest_gbps(groups: dict[str, dict]) -> float | None:
    rates = [
        entry["gbps"]
        for record in groups.values()
        for name, entry in record.get("variants", {}).items()
        if name in METAL_VARIANTS and "gbps" in entry
    ]
    return min(rates) if rates else None


def _kill_line(groups: dict[str, dict], *, bytes_per_step: int, t_step: float) -> dict[str, object]:
    need = kill_equivalent_throughput(bytes_per_step, t_step)
    per_variant: dict[str, object] = {}
    report: dict[str, object] = {
        "bytes_per_step": bytes_per_step,
        "t_step_s": t_step,
        "kill_equivalent_gbps": need / 1e9,
        "variants": per_variant,
    }
    print(
        f"overhead-line throughput: {need / 1e9:.3g} GB/s "
        f"({bytes_per_step} bytes per step within 25% of a {t_step} s step)"
    )
    for variant in METAL_VARIANTS:
        timed = [
            (record["bytes_out"], record["variants"][variant]["median_s"])
            for record in groups.values()
            if "median_s" in record.get("variants", {}).get(variant, {})
        ]
        if not timed:
            continue
        agg = gbps(sum(b for b, _ in timed), sum(s for _, s in timed))
        clears = agg * 1e9 >= need
        per_variant[variant] = {"aggregate_gbps": agg, "clears": clears}
        print(f"  {variant}: {agg:.2f} GB/s over {len(timed)} groups -> clears={clears}")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; any unexpected failure exits 2, never 1."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--df11", type=Path, required=True)
    parser.add_argument("--groups", required=True, help="comma-separated group names")
    parser.add_argument("--out", type=Path, required=True, help="bench JSON (resumed if present)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--rate-from", type=Path, help="bench JSON whose top-level gbps sets the guard"
    )
    parser.add_argument("--wall-budget", type=float, default=2 * 3600.0)
    parser.add_argument(
        "--t-step",
        type=float,
        help="seconds per denoising step; reports the throughput at which decoding costs 25 %% of it",
    )
    args = parser.parse_args(argv)
    requested = {v for v in args.variants.split(",") if v}
    variants = [v for v in VARIANTS if v in requested]  # canonical order, for the resume key
    unknown = sorted(requested - set(VARIANTS))
    if unknown or not variants or args.reps < 1:
        print(f"error: bad --variants {args.variants!r} or --reps {args.reps}", file=sys.stderr)
        return 2
    caps = list(install_memory_caps())
    mx.set_cache_limit(CACHE_LIMIT)
    watchdog = None
    errors: list[str] = []
    try:
        if any(v in METAL_VARIANTS for v in variants) and "metal" not in available_backends():
            print("error: the Metal backend cannot run here", file=sys.stderr)
            return 2
        rate_bps = rate_from(args.rate_from) if args.rate_from else None
        key: dict[str, object] = {
            "df11": str(args.df11.resolve()),
            "variants": variants,
            "reps": args.reps,
            "source_hash": source_hash(),
            "mlx": mx.__version__,
        }
        data = _load_existing(args.out, key)
        groups: dict[str, dict] = data["groups"]
        ckpt = open_checkpoint(args.df11)
        selected = sorted({g for g in args.groups.split(",") if g}, key=natural_key)
        missing = [g for g in selected if g not in ckpt.groups]
        if not selected or missing:
            print(f"error: unknown or empty --groups: {missing or selected}", file=sys.stderr)
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        move_stale_abort_aside(args.out.parent)
        watchdog = Watchdog(
            args.out.parent, ceiling=default_ceiling(), budget=args.wall_budget
        ).start()
        prov = provenance()
        data |= {"df11": str(args.df11), "memory_caps_gb": caps, "cache_limit": CACHE_LIMIT}
        for name in selected:
            if name in groups:
                print(f"{name}: already in {args.out.name}, skipped")
                continue
            watchdog.peak_footprint = 0  # per-group peak from here on
            t0 = time.monotonic()
            print(f"{name}:")
            try:
                group = load_group_mx(ckpt.groups[name])
                record = bench_group(group, variants=variants, reps=args.reps, rate_bps=rate_bps)
                del group
            except (DFloatError, BenchError) as exc:
                errors.append(f"{name}: {exc}")
                print(f"error: {name}: {exc}", file=sys.stderr)
                continue
            record |= {
                "seconds": time.monotonic() - t0,
                "footprint_peak": max(watchdog.peak_footprint, phys_footprint()),
                "provenance": prov,
            }
            groups[name] = record
            data |= {"provenance": prov, "gbps": _slowest_gbps(groups)}
            write_json_atomic(args.out, data)
            mx.clear_cache()
            summary = {v: (e.get("gbps"), e.get("parity")) for v, e in record["variants"].items()}
            print(f"{name}: {summary}")
        watchdog.stop()
        if args.t_step is not None:
            bytes_per_step = sum(
                2 * int(np.prod(g.tensors["sign_mantissa"].shape)) for g in ckpt.groups.values()
            )
            data["kill_line"] = _kill_line(
                groups, bytes_per_step=bytes_per_step, t_step=args.t_step
            )
        code = bench_exit_code(mismatched=_mismatches(groups), errors=len(errors))
        data |= {"errors": errors, "exit_code": code, "gbps": _slowest_gbps(groups)}
        write_json_atomic(args.out, data)
        return code
    except (VerifyError, DFloatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2
    finally:
        if watchdog is not None:
            watchdog.stop()


if __name__ == "__main__":
    raise SystemExit(main())
