"""Hardware proof that the Python API's watchdog stops a FLUX.2 Klein 4B call in each phase.

Seven units, each in its own process (1024², 4 steps, seed 42, one fixed prompt, every call with a wall
budget):

- U0 generates one image with the default watchdog and keeps ``model.report()["peaks"]``, with the
  watched peak sampled per phase;
- U1 builds a fresh model with the encode ceiling derived from U0;
- U2-U5 encode the prompt first and let memory settle, then set the set-load, denoise or VAE-decode line
  as ``model.watchdog`` and generate; U2's line uses its own settled reading. U1-U4 pass only on a raise
  at a checkpoint of their phase: U4 at ``call_end``, which discards the finished image;
- U5 is U4 with the ceiling only 64 MiB above the stop line, so the process ends by escalation (exit 70);
- U6 is U2, then a reading of the watched memory once the stopped call has cleaned up, which must be under
  the stop line (the set and encoder pages were released), then the same call again with its stop line
  1 GiB over U0's overall watched peak, which must complete: memory still held when the retry reaches its
  decode peak trips that line.

The proof must exit 0 before the release is tagged.

``--orchestrate`` runs the pending units in order, one subprocess each, and writes ``<out>/<unit>.json`` the
moment a unit ends (atomically): its exit code, the unit's own outcome, the abort record from
``<out>/<unit>-artifacts/``, the verdict, and the versions, macOS version, commit and dirty flag. A stopping
unit that fails (by the run-to-run spread of its peaks, say) is run once more in the same run: its first
``<unit>.json``, outcome and abort records move into ``<out>/earlier/<stamp>/``, the second attempt decides,
and its record names the first under ``first_attempt``. U0 and a unit found not isolable are never run
twice. A re-run skips every unit whose JSON exists. A failed U0 is kept as ``U0-failed-<stamp>.json`` and
the run stops, so the next run repeats it. A U0 whose peaks place no ceiling for some unit is kept as
``U0-not-isolable-<stamp>.json`` with the reason, and the run stops the same way before any other unit.
``--unit`` runs one unit in this process; the model is built only there.

Commit only the ``<unit>.json`` files: they embed the abort record with the home directory written as
``~``. The raw ``<unit>-artifacts/`` and ``earlier/`` folders are not scrubbed and stay out of a commit.

Exit codes: 0 every unit passed; 2 the baseline failed or left a unit not isolable, or a unit is recorded
not isolable; else 1 a unit failed. A unit process exits 0 when its outcome is written, and 70 or 71 when
the watchdog ended it.

    python -m scripts.prove_api_watchdog --orchestrate --out DIR
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

try:
    from scripts._api_watchdog_proof import (
        PHASE_OF,
        UNITS,
        PhaseSampler,
        ceilings_for,
        pending_units,
        retry_line,
        unit_verdict,
    )

    from mlx_dfloat._scrub import scrub_home
    from mlx_dfloat.mflux import WatchdogConfig
except Exception as exc:  # a broken environment is a tool error (2), never a unit verdict (1)
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

UnitRunner = Callable[..., "subprocess.CompletedProcess[Any]"]
"""Runs one unit command, as ``subprocess.run(cmd, cwd=..., check=False)`` does."""
MODEL = "flux2-klein-4b"
PROMPT = "A lighthouse on a rocky coast at dusk, waves breaking below, warm light in the windows"
SEED = 42
STEPS = 4
SIZE = 1024
WALL_S = 600.0
ESCALATION_MARGIN = 64 * 1024**2  # U5: the VAE decode cannot stop inside, so it reaches the ceiling


def _write_json(path: Path, data: object) -> None:
    """Write ``data`` as JSON through a temporary file and a rename, so the file is whole or absent.

    The home directory is written as ``~`` (the records are committed); values JSON cannot hold, such as
    a ``Path``, become text first, so they are scrubbed too.
    """
    plain = json.loads(json.dumps(data, default=repr))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(scrub_home(plain, home=str(Path.home())), indent=1))
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _git(*args: str) -> str | None:
    try:
        run = subprocess.run(
            ["git", "-C", str(_REPO), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return run.stdout.strip()


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _sysctl(name: str) -> str | None:
    try:
        run = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return run.stdout.strip()


def provenance() -> dict[str, object]:
    """Versions, hardware, date, commit and dirty flag (with the dirty files), without importing MLX or mflux."""
    status = _git("status", "--porcelain")
    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "mlx": _version("mlx"),
        "mflux": _version("mflux"),
        "mlx_dfloat": _version("mlx-dfloat"),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "hardware": {
            "model": _sysctl("hw.model"),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "memsize": _sysctl("hw.memsize"),
        },
        "commit": _git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": None if status is None else status.splitlines(),
    }


def _latest_artifact(folder: Path) -> Any:
    records = sorted(folder.glob("abort-*.json")) if folder.is_dir() else []
    return _read_json(records[-1]) if records else None


def _aside_folder(out: Path) -> Path:
    """A new, empty ``<out>/earlier/<stamp>[-n]/`` folder: two set-asides never share one."""
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    (out / "earlier").mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        folder = out / "earlier" / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise RuntimeError(f"no free folder for {stamp} under {out / 'earlier'}")


def _set_aside(out: Path, unit: str, *, record: bool = False) -> Path | None:
    """Move an attempt's outcome and abort records (and, with ``record``, its ``<unit>.json``) out of the way.

    Never deleted: they go into a new folder under ``<out>/earlier/``, which is returned (None if nothing
    was there to move).
    """
    names = [f"{unit}.outcome.json", f"{unit}-artifacts"]
    if record:
        names.insert(0, f"{unit}.json")
    leftovers = [out / name for name in names if (out / name).exists()]
    if not leftovers:
        return None
    aside = _aside_folder(out)
    for leftover in leftovers:
        leftover.rename(aside / leftover.name)
    return aside


def _not_isolable(peaks: Any) -> list[str]:
    """The stopping units whose ceiling the baseline's ``peaks`` cannot place, derived as ``run_unit`` does."""
    peaks = peaks if isinstance(peaks, dict) else {}
    default_margin = WatchdogConfig().stop_margin
    blocked = []
    for unit, phase in PHASE_OF.items():
        margin = ESCALATION_MARGIN if unit == "U5" else default_margin
        if ceilings_for(peaks, stop_margin=margin)[phase] is None:
            blocked.append(unit)
    return blocked


def _attempt(
    out: Path, unit: str, *, model: str, wall: float, run: UnitRunner, facts: dict[str, object]
) -> dict[str, Any]:
    """Run one unit process and return its record with the verdict (not yet written)."""
    _set_aside(out, unit)
    cmd = [sys.executable, "-m", "scripts.prove_api_watchdog", "--unit", unit]
    cmd += ["--out", str(out), "--model", model, "--wall", str(wall)]
    start = time.monotonic()
    code = run(cmd, cwd=_REPO, check=False).returncode
    record: dict[str, Any] = {
        "unit": unit,
        "model": model,
        "exit_code": code,
        "unit_s": time.monotonic() - start,
        "result": _read_json(out / f"{unit}.outcome.json"),
        "artifact": _latest_artifact(out / f"{unit}-artifacts"),
        **facts,
    }
    record["verdict"] = unit_verdict(unit, record)
    print(f"{unit}: {record['verdict']} (exit {code})", flush=True)
    return record


def orchestrate(out: Path, *, model: str, wall: float, run: UnitRunner = subprocess.run) -> int:
    """Run the pending units as subprocesses in order; write each unit's record as it ends.

    ``run`` starts one unit process (``subprocess.run`` by default). A stopping unit that fails is run once
    more: its first record, outcome and abort records move into a new ``earlier/<stamp>/`` folder, the second
    attempt's verdict decides, and its record names the first under ``first_attempt``. U0 and a unit found
    not isolable are never run twice.

    Returns 0 when every unit passed, 2 when any unit is recorded not isolable (whenever it was found), else
    1 when one failed. A failed baseline, or one that leaves a unit not isolable, returns 2 at once: no other
    unit runs and the baseline is kept as ``U0-failed-<stamp>.json`` or ``U0-not-isolable-<stamp>.json``, so
    the next run repeats it.
    """
    out.mkdir(parents=True, exist_ok=True)
    facts = provenance()
    for unit in pending_units(out):
        record = _attempt(out, unit, model=model, wall=wall, run=run, facts=facts)
        if unit != "U0" and record["verdict"] == "fail":
            _write_json(out / f"{unit}.json", record)  # on record before the second attempt
            aside = _set_aside(out, unit, record=True)
            assert aside is not None  # the record was just written
            record = _attempt(out, unit, model=model, wall=wall, run=run, facts=facts)
            record["first_attempt"] = {
                "record": (aside / f"{unit}.json").relative_to(out).as_posix(),
                "verdict": "fail",
            }
        if unit == "U0" and record["verdict"] != "pass":
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            _write_json(out / f"U0-failed-{stamp}.json", record)
            return 2
        if unit == "U0" and (blocked := _not_isolable((record["result"] or {}).get("peaks"))):
            record["not_isolable"] = blocked
            record["reason"] = (
                f"the baseline's peaks place no ceiling for {', '.join(blocked)}; "
                "no other unit was run"
            )
            print(f"U0: {record['reason']}", flush=True)
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            _write_json(out / f"U0-not-isolable-{stamp}.json", record)
            return 2
        _write_json(out / f"{unit}.json", record)
    verdicts = {u: (_read_json(out / f"{u}.json") or {}).get("verdict") for u in UNITS}
    print(json.dumps(verdicts), flush=True)
    if "not isolable" in verdicts.values():
        return 2
    return 0 if all(v == "pass" for v in verdicts.values()) else 1


def run_unit(unit: str, out: Path, *, model: str, wall: float) -> int:
    """Run one unit in this process; the only place a model is built. Never called by a test."""
    outcome_path = out / f"{unit}.outcome.json"
    result: dict[str, Any] = {"unit": unit, "model": model, "raised": None}
    result.update(completed=False, image_saved=False)
    stop_margin = ESCALATION_MARGIN if unit == "U5" else WatchdogConfig().stop_margin
    peaks: Any = None
    ceiling: tuple[int, int] | None = None
    if unit != "U0":
        baseline = _read_json(out / "U0.json") or {}
        if baseline.get("verdict") != "pass":
            raise SystemExit(f"{unit}: no passed baseline at {out / 'U0.json'}; run U0 first")
        peaks = baseline["result"]["peaks"]
        # U2 and U6 place their final line after the settle below; this one is the plan.
        ceiling = ceilings_for(peaks, stop_margin=stop_margin)[PHASE_OF[unit]]
        result["ceiling"] = None if ceiling is None else list(ceiling)
    # On record before the call, in case the call ends the process.
    _write_json(outcome_path, result)
    if unit != "U0" and ceiling is None:
        return 0

    from mlx_dfloat._memory_caps import install_memory_caps
    from mlx_dfloat._watchdog import phys_footprint, read_counters, watched_memory
    from mlx_dfloat.errors import DFloatWatchdogError
    from mlx_dfloat.mflux._drain import drain
    from mlx_dfloat.mflux.families import family_of

    def watched() -> int:
        """The number the watchdog compares with its lines: max(footprint, MLX active + cache)."""
        counters = read_counters()
        return watched_memory(
            footprint=counters.footprint,
            mlx_active=counters.mlx_active,
            mlx_cache=counters.mlx_cache,
        )[0]

    def stopping(line: tuple[int, int]) -> WatchdogConfig:
        ceiling_bytes, margin = line
        return WatchdogConfig(
            ceiling=ceiling_bytes, stop_margin=margin, wall_budget=wall, artifact_dir=artifacts
        )

    def raised(exc: DFloatWatchdogError) -> dict[str, Any]:
        return {
            "type": type(exc).__name__,
            "reason": exc.reason,
            "phase": exc.phase,
            "site": exc.site,
            "escalation": exc.record.get("escalation"),
            "elapsed": exc.record.get("elapsed"),
            "checkpoint_latency_s": exc.record.get("checkpoint_latency_s"),
            "artifact": None if exc.artifact is None else str(exc.artifact),
        }

    def generate() -> Any:
        return pipe.generate_image(
            seed=SEED, prompt=PROMPT, num_inference_steps=STEPS, height=SIZE, width=SIZE
        )

    result["caps_gb"] = list(install_memory_caps())
    artifacts = out / f"{unit}-artifacts"
    default = WatchdogConfig(wall_budget=wall, artifact_dir=artifacts)
    factory = family_of(model).load_model_class()
    # U1: the text encoder loads lazily, so its whole climb lands in the one evaluation between
    # `encoder_loaded` and `prompt_encoded`; the line sits under the encode's top and the call raises at
    # `prompt_encoded`, its ceiling above the top.
    pipe = factory(model, watchdog=stopping(ceiling) if unit == "U1" and ceiling else default)
    if unit not in ("U0", "U1"):
        pipe.encode(PROMPT)  # pre-encoded: these units must not stop in encode
        drain(phys_footprint)  # settle: synchronize, clear, a kick, a 0.25 s poll of the footprint
        result["settled"] = watched()
        if unit in ("U2", "U6"):
            line = ceilings_for(peaks, stop_margin=stop_margin, settled=result["settled"])
            ceiling = line[PHASE_OF[unit]]
            result["ceiling"] = None if ceiling is None else list(ceiling)
        _write_json(outcome_path, result)
        if ceiling is None:
            return 0
        pipe.watchdog = stopping(ceiling)
    sampler = None
    if unit == "U0":  # the watched peak per phase, which the boundary readings can miss
        sampler = PhaseSampler(watched, lambda: getattr(pipe, "open_phase", None)).start()
    start = time.monotonic()
    try:
        image = generate()
    except DFloatWatchdogError as exc:
        result["raised"] = raised(exc)
        result["record"] = exc.record
        if unit == "U6":  # the stopped call has cleaned up; its set and encoder pages must be gone
            result["watched_after_abort"] = watched()
    else:
        result["completed"] = True
        if unit == "U0":
            png = out / "U0.png"
            image.save(str(png), export_json_metadata=False, overwrite=True)
            result["image_saved"] = png.is_file()
    finally:
        if sampler is not None:
            sampler.stop()
    result["call_s"] = time.monotonic() - start
    if sampler is not None and result["completed"]:
        result["peaks"] = sampler.sampled.into(pipe.report()["peaks"])
    if unit == "U6" and result["raised"] is not None:
        _write_json(outcome_path, result)  # the abort on record before the retry
        # the baseline call again, in this process, under a line 1 GiB over U0's overall peak: memory still
        # held when the retry reaches its decode peak trips it, so U6 fails
        again = retry_line(peaks, stop_margin=WatchdogConfig().stop_margin)
        result["retry_ceiling"] = None if again is None else list(again)
        pipe.watchdog = default if again is None else stopping(again)
        retry: dict[str, Any] = {"completed": False, "raised": None}
        start = time.monotonic()
        try:
            generate()
        except DFloatWatchdogError as exc:
            retry["raised"] = raised(exc)
        else:
            retry["completed"] = True
        retry["call_s"] = time.monotonic() - start
        result["retry"] = retry
    result["pid"] = os.getpid()
    _write_json(outcome_path, result)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line: one ``--unit`` or ``--orchestrate``, with ``--out``."""
    p = argparse.ArgumentParser(
        prog="python -m scripts.prove_api_watchdog",
        description="Prove on this Mac that the API watchdog stops a FLUX.2 Klein 4B call in each phase.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unit", choices=UNITS, help="run one unit in this process")
    mode.add_argument(
        "--orchestrate", action="store_true", help="run the pending units, one subprocess each"
    )
    p.add_argument("--out", type=Path, required=True, help="folder for the unit records")
    p.add_argument(
        "--model", choices=(MODEL,), default=MODEL, help="the model (default %(default)s)"
    )
    p.add_argument(
        "--wall",
        type=float,
        default=WALL_S,
        help="wall budget per call, seconds (default %(default)s)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, run: UnitRunner = subprocess.run) -> int:
    """Run one unit or orchestrate them all."""
    args = parse_args(argv)
    args.out = args.out.resolve()  # the unit processes run from the repository root
    if args.orchestrate:
        return orchestrate(args.out, model=args.model, wall=args.wall, run=run)
    args.out.mkdir(parents=True, exist_ok=True)
    return run_unit(args.unit, args.out, model=args.model, wall=args.wall)


if __name__ == "__main__":
    sys.exit(main())
