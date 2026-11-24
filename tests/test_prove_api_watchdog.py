"""The proof orchestrator, driven through a fake unit runner: no test here builds a model or runs a unit.

The fake stands in for the unit subprocess: it writes the unit's ``<unit>.outcome.json`` and abort records
where a real unit would, and returns a chosen exit code. The peaks are synthetic, with multi-GiB gaps, so
every phase is isolable unless a test removes a phase record.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import scripts.prove_api_watchdog as prove

GIB = 1_073_741_824


def _phase(start: float, peak: float, end: float) -> dict[str, int]:
    return {
        "footprint_start": int(start * GIB),
        "mlx_peak": int(peak * GIB),
        "footprint_end": int(end * GIB),
    }


PEAKS: dict[str, Any] = {
    "label": "sampled at phase boundaries",
    "encode": _phase(1, 6, 7),
    "set_load": _phase(4, 5, 8),
    "denoise": _phase(8, 12, 12),
    "vae": _phase(12, 16, 13),
}

U0_DONE = {"unit": "U0", "completed": True, "image_saved": True, "peaks": PEAKS}
SITE = {"U1": "prompt_encoded", "U2": "set_group", "U3": "block_start", "U4": "call_end"}
PHASE = {"U1": "encode", "U2": "set_load", "U3": "denoise", "U4": "vae"}
U5_ABORT = {"reason": "memory", "context": {"phase": "vae"}, "escalation": "ceiling"}


def _raised(unit: str, *, site: str | None = None, artifact: str | None = None) -> dict[str, Any]:
    return {
        "unit": unit,
        "ceiling": [9 * GIB, GIB],
        "raised": {
            "type": "DFloatWatchdogError",
            "reason": "memory",
            "phase": PHASE[unit],
            "site": site or SITE[unit],
            "artifact": artifact,
        },
    }


def _passing_plan() -> dict[str, tuple[int, Any, list[tuple[str, Any]]]]:
    plan: dict[str, tuple[int, Any, list[tuple[str, Any]]]] = {"U0": (0, U0_DONE, [])}
    for unit in ("U1", "U2", "U3", "U4"):
        plan[unit] = (0, _raised(unit), [])
    plan["U5"] = (
        70,
        {"unit": "U5", "ceiling": [9 * GIB, 64 << 20]},
        [("abort-1Z-9.json", U5_ABORT)],
    )
    # U6: U2's abort, the watched memory back at 1 GiB after it (under the 8 GiB stop line), then the baseline
    # call again in the same process, completed.
    plan["U6"] = (
        0,
        {**_raised("U2"), "unit": "U6", "watched_after_abort": GIB, "retry": {"completed": True}},
        [],
    )
    return plan


class FakeUnits:
    """Plays each unit process: writes its outcome and abort records, returns the planned exit code.

    A plan entry is one attempt, used for every run of the unit, or a list of attempts, one per run in order.
    """

    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[str] = []

    def __call__(self, cmd: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        unit = cmd[cmd.index("--unit") + 1]
        out = Path(cmd[cmd.index("--out") + 1])
        attempt = self.calls.count(unit)
        self.calls.append(unit)
        entry = self.plan[unit]
        code, outcome, aborts = entry[attempt] if isinstance(entry, list) else entry
        if outcome is not None:
            (out / f"{unit}.outcome.json").write_text(json.dumps(outcome))
        for name, record in aborts:
            (out / f"{unit}-artifacts").mkdir(exist_ok=True)
            (out / f"{unit}-artifacts" / name).write_text(json.dumps(record))
        return subprocess.CompletedProcess(cmd, code)


def _record(out: Path, unit: str) -> Any:
    return json.loads((out / f"{unit}.json").read_text())


def test_every_unit_passing_returns_0_with_one_record_per_unit(tmp_path):
    # Bug caught: the return code inverted, or a passing unit's record not written.
    fake = FakeUnits(_passing_plan())
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 0
    assert fake.calls == ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
    for unit in fake.calls:
        assert _record(tmp_path, unit)["verdict"] == "pass", unit
    assert _record(tmp_path, "U5")["exit_code"] == 70


def test_one_failed_unit_returns_1_and_the_later_units_still_run(tmp_path):
    # Bug caught: a failed stopping unit ends the run early, or the run returns 0 on a failed unit.
    plan = _passing_plan()
    # stopped, but at the wrong checkpoint, on both attempts
    plan["U3"] = (0, _raised("U3", site="call_end"), [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 1
    assert fake.calls == ["U0", "U1", "U2", "U3", "U3", "U4", "U5", "U6"]
    assert _record(tmp_path, "U3")["verdict"] == "fail"


def _first_attempt(out: Path, unit: str) -> Any:
    pointer = _record(out, unit)["first_attempt"]
    return pointer, json.loads((out / pointer["record"]).read_text())


def test_a_unit_that_fails_then_passes_returns_0_and_both_attempts_are_kept(tmp_path):
    # Bug caught: no second attempt (exit 1 on a spread fail), the first attempt overwritten or deleted, its
    # outcome or abort records left where the second attempt reads them, or the second record not naming it.
    plan = _passing_plan()
    first = (70, {"unit": "U4", "ceiling": [9 * GIB, GIB]}, [("abort-1Z-9.json", U5_ABORT)])
    plan["U4"] = [first, (0, _raised("U4"), [])]
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 0
    assert fake.calls == ["U0", "U1", "U2", "U3", "U4", "U4", "U5", "U6"]
    second = _record(tmp_path, "U4")
    assert second["verdict"] == "pass"
    assert second["artifact"] is None  # the first attempt's abort record was moved, not read again
    pointer, kept = _first_attempt(tmp_path, "U4")
    assert pointer["verdict"] == "fail"
    assert pointer["record"].startswith("earlier/")
    assert (kept["exit_code"], kept["verdict"]) == (70, "fail")
    folder = (tmp_path / pointer["record"]).parent
    assert json.loads((folder / "U4.outcome.json").read_text())["ceiling"] == [9 * GIB, GIB]
    assert (folder / "U4-artifacts" / "abort-1Z-9.json").is_file()


def test_a_unit_that_fails_twice_returns_1_and_the_second_attempt_decides(tmp_path):
    # Bug caught: the first attempt's verdict reported, a third attempt run, or the retry's fail turned into a pass.
    plan = _passing_plan()
    plan["U2"] = [
        (0, _raised("U2", site="block_start"), []),
        (0, _raised("U2", site="call_end"), []),
    ]
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 1
    assert fake.calls.count("U2") == 2
    assert _record(tmp_path, "U2")["result"]["raised"]["site"] == "call_end"
    _, kept = _first_attempt(tmp_path, "U2")
    assert kept["result"]["raised"]["site"] == "block_start"


def test_a_failed_baseline_is_never_run_twice(tmp_path):
    # Bug caught: the second attempt also given to U0, which then passes and runs the stopping units on peaks
    # from a run that failed once.
    plan = _passing_plan()
    plan["U0"] = [(1, None, []), (0, U0_DONE, [])]
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert fake.calls == ["U0"]


def test_a_unit_not_isolable_is_not_rerun_and_the_run_returns_2(tmp_path):
    # Bug caught: "not isolable" (U2's settled reading too close to its line) re-run as a fail, or counted as a
    # plain fail (1) and not as a run that could not test the unit (2).
    plan = _passing_plan()
    plan["U2"] = (0, {"unit": "U2", "ceiling": None, "settled": 5 * GIB}, [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert fake.calls == ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
    assert _record(tmp_path, "U2")["verdict"] == "not isolable"


def test_not_isolable_outranks_a_failed_unit(tmp_path):
    # Bug caught: the exit code checks for a fail first, so a run with both reads 1.
    plan = _passing_plan()
    plan["U2"] = (0, {"unit": "U2", "ceiling": None}, [])
    plan["U3"] = (0, _raised("U3", site="call_end"), [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2


def test_a_not_isolable_record_found_on_resume_returns_2(tmp_path):
    # Bug caught: only the units run now are checked for "not isolable", so a resumed run reads 1 or 0.
    (tmp_path / "U0.json").write_text(json.dumps({"verdict": "pass", "result": U0_DONE}))
    (tmp_path / "U2.json").write_text(json.dumps({"unit": "U2", "verdict": "not isolable"}))
    fake = FakeUnits(_passing_plan())
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert "U2" not in fake.calls


def test_stale_leftovers_and_a_failed_first_attempt_go_to_separate_folders(tmp_path):
    # Bug caught: both set-asides in the same second share one folder, so the first attempt's outcome replaces the
    # killed attempt's (or its artifacts folder cannot be moved onto the one already there).
    (tmp_path / "U0.json").write_text(json.dumps({"verdict": "pass", "result": U0_DONE}))
    (tmp_path / "U1.outcome.json").write_text(json.dumps({"stale": True}))
    (tmp_path / "U1-artifacts").mkdir()
    (tmp_path / "U1-artifacts" / "abort-old.json").write_text(json.dumps({"stale": True}))
    plan = _passing_plan()
    first = (70, {"unit": "U1", "attempt": 1}, [("abort-new.json", U5_ABORT)])
    plan["U1"] = [first, (0, _raised("U1"), [])]
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 0
    outcomes = [json.loads(p.read_text()) for p in (tmp_path / "earlier").glob("*/U1.outcome.json")]
    assert sorted(outcomes, key=str) == sorted(
        [{"stale": True}, {"unit": "U1", "attempt": 1}], key=str
    )
    assert len(list((tmp_path / "earlier").glob("*/U1-artifacts/abort-old.json"))) == 1
    assert len(list((tmp_path / "earlier").glob("*/U1-artifacts/abort-new.json"))) == 1


def test_a_failed_baseline_returns_2_keeps_a_u0_failed_file_and_runs_nothing_else(tmp_path):
    # Bug caught: a failed U0 written as U0.json (a re-run would skip it) or the stopping units run anyway.
    plan = _passing_plan()
    plan["U0"] = (1, None, [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert fake.calls == ["U0"]
    assert not (tmp_path / "U0.json").exists()
    failed = list(tmp_path.glob("U0-failed-*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text())["exit_code"] == 1
    again = FakeUnits(_passing_plan())
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=again) == 0
    assert again.calls[0] == "U0"


def test_a_unit_that_is_not_isolable_stops_the_run_with_2_right_after_u0(tmp_path):
    # Bug caught: the orchestrator runs U1-U6 for hours before reporting that U4 and U5 could never stop.
    plan = _passing_plan()
    no_vae = {key: value for key, value in PEAKS.items() if key != "vae"}
    plan["U0"] = (0, {**U0_DONE, "peaks": no_vae}, [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert fake.calls == ["U0"]
    assert not (tmp_path / "U0.json").exists()  # the next run repeats the baseline
    stopped = list(tmp_path.glob("U0-not-isolable-*.json"))
    assert len(stopped) == 1
    record = json.loads(stopped[0].read_text())
    assert record["not_isolable"] == ["U4", "U5"]
    assert "U4" in record["reason"]
    assert "U5" in record["reason"]


def test_units_with_a_record_are_skipped_and_their_record_is_untouched(tmp_path):
    # Bug caught: resume ignored, so a finished unit is run again and its record overwritten.
    (tmp_path / "U0.json").write_text(json.dumps({"verdict": "pass", "result": U0_DONE}))
    kept = json.dumps({"unit": "U1", "verdict": "pass", "kept": True})
    (tmp_path / "U1.json").write_text(kept)
    fake = FakeUnits(_passing_plan())
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 0
    assert fake.calls == ["U2", "U3", "U4", "U5", "U6"]
    assert (tmp_path / "U1.json").read_text() == kept


def test_an_earlier_attempt_is_set_aside_and_not_read_as_this_run(tmp_path):
    # Bug caught: a stale outcome or abort record from a killed attempt counted as this run's result.
    (tmp_path / "U0.json").write_text(json.dumps({"verdict": "pass", "result": U0_DONE}))
    (tmp_path / "U1.outcome.json").write_text(json.dumps(_raised("U1")))
    (tmp_path / "U1-artifacts").mkdir()
    (tmp_path / "U1-artifacts" / "abort-old.json").write_text(json.dumps({"stale": True}))
    plan = _passing_plan()
    plan["U1"] = (70, None, [])  # this attempt wrote nothing
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 1
    record = _record(tmp_path, "U1")
    assert record["result"] is None
    assert record["artifact"] is None
    assert record["verdict"] == "fail"
    aside = list((tmp_path / "earlier").glob("*/U1.outcome.json"))
    assert len(aside) == 1
    assert json.loads(aside[0].read_text())["raised"]["site"] == "prompt_encoded"
    assert list((tmp_path / "earlier").glob("*/U1-artifacts/abort-old.json"))


def test_the_newest_abort_record_is_the_one_kept(tmp_path):
    # Bug caught: the first-written, the last-written or the first-listed abort record read instead of the newest
    # by its stamp.
    plan = _passing_plan()
    middle = ("abort-20261125T100002.000000Z-7.json", {**U5_ABORT, "escalation": "grace"})
    newest = ("abort-20261125T100003.000000Z-7.json", U5_ABORT)
    oldest = ("abort-20261125T100001.000000Z-7.json", {**U5_ABORT, "escalation": None})
    # written out of name order: the newest is neither the first nor the last written
    plan["U5"] = (70, plan["U5"][1], [middle, newest, oldest])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 0
    assert _record(tmp_path, "U5")["artifact"] == U5_ABORT


def test_records_under_a_fake_home_name_no_home_directory(tmp_path, monkeypatch):
    # Bug caught: the records (committed with the proof) carry the user's home path, as the abort artifact
    # path and the resolved --out do.
    home = tmp_path.resolve() / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    out = home / "proof"
    plan = _passing_plan()
    artifact = str(out / "U1-artifacts" / "abort-1Z-9.json")
    plan["U1"] = (0, _raised("U1", artifact=artifact), [])
    fake = FakeUnits(plan)
    assert prove.main(["--orchestrate", "--out", "proof"], run=fake) == 0
    assert str(home) in (out / "U1.outcome.json").read_text()  # the input does carry the home
    written = [out / f"{unit}.json" for unit in fake.calls]
    for path in written:
        assert str(home) not in path.read_text(), path.name
    assert (
        _record(out, "U1")["result"]["raised"]["artifact"] == "~/proof/U1-artifacts/abort-1Z-9.json"
    )


def test_a_path_value_under_the_home_is_written_with_a_tilde(tmp_path, monkeypatch):
    # Bug caught: the scrub runs before JSON turns a Path into text, so a Path under the home escapes it.
    home = tmp_path.resolve() / "home"
    monkeypatch.setenv("HOME", str(home))
    target = tmp_path / "r.json"
    prove._write_json(target, {"folder": home / "proof", "n": 3})
    text = target.read_text()
    assert str(home) not in text
    assert json.loads(text)["n"] == 3
    assert "~/proof" in text


def test_u5_is_placed_with_its_own_64_mib_margin():
    # Bug caught: `_not_isolable` places U5 with the default 1 GiB margin, so a VAE that only an exit line can
    # isolate stops the whole run at U0. The VAE top is 900 MiB above the denoise top (12 GiB):
    # - U4, 1 GiB margin: stop 12 GiB + 900 - 512 = 12 GiB + 388 MiB, not above 12 GiB + 410 MiB headroom: blocked;
    # - U5, 64 MiB margin: stop (12 GiB + 12 GiB + 900 - 64) / 2 = 12 GiB + 418 MiB, above it: placed.
    mib = 1024**2
    peaks = {
        **PEAKS,
        "vae": {"footprint_start": 12 * GIB, "mlx_peak": 12 * GIB + 900 * mib, "footprint_end": 0},
    }
    assert prove._not_isolable(peaks) == ["U4"]


def test_a_baseline_without_peaks_stops_with_every_stopping_unit_not_isolable(tmp_path):
    # Bug caught: a passing U0 whose outcome lacks peaks crashes the orchestrator (no reason written) instead of
    # stopping with exit 2 and naming the six units it cannot place.
    plan = _passing_plan()
    plan["U0"] = (0, {key: value for key, value in U0_DONE.items() if key != "peaks"}, [])
    fake = FakeUnits(plan)
    assert prove.orchestrate(tmp_path, model=prove.MODEL, wall=60.0, run=fake) == 2
    assert fake.calls == ["U0"]
    (stopped,) = tmp_path.glob("U0-not-isolable-*.json")
    assert json.loads(stopped.read_text())["not_isolable"] == ["U1", "U2", "U3", "U4", "U5", "U6"]


def test_help_works_when_mlx_cannot_be_imported(tmp_path):
    # Bug caught: the orchestrator's imports load MLX (the guard would turn that into exit 2), so the
    # parent process of a proof run holds Metal state and --help fails on a host without MLX.
    shadow = tmp_path / "mlx"
    shadow.mkdir()
    (shadow / "__init__.py").write_text('raise ImportError("no mlx here")\n')
    result = subprocess.run(
        [sys.executable, "-m", "scripts.prove_api_watchdog", "--help"],
        cwd=Path(prove.__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage:" in result.stdout
