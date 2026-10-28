"""The tier table names the mflux version each row was measured on, from the row's own record."""

import json
from importlib import metadata
from pathlib import Path

from mlx_dfloat.bench.table import (
    render_tier_table,
    tier_row_from_abort_artifact,
    tier_row_from_generate_report,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "generate_report_schnell_1024.json"


def _report(mflux: str | None) -> dict:
    """A report ``mlx-dfloat generate`` wrote, its ``versions.mflux`` set (None: no ``versions`` key)."""
    report = json.loads(_FIXTURE.read_text())
    if mflux is None:
        del report["versions"]
    else:
        report["versions"] = {**report["versions"], "mflux": mflux}
    return report


def _abort() -> dict:
    # An abort artifact records no versions (the watchdog writes it, not generate).
    return {
        "reason": "memory",
        "elapsed": 5.6,
        "ceiling": 9_842_633_386,
        "peak_watched": 9_900_000_000,
        "peak_footprint": 9_800_000_000,
        "peak_mlx": 9_700_000_000,
        "context": {"model": "z-image-turbo", "tier_gb": 16, "label": "CAPPED"},
    }


def _last_cell(line: str) -> str:
    return line.rstrip(" |").rsplit("|", 1)[1].strip()


def _cells(rows) -> list[str]:
    lines = render_tier_table(rows).splitlines()
    return [_last_cell(line) for line in lines[2:]]


def test_the_header_ends_with_an_mflux_column():
    # Red when: the table has no column saying which mflux the row ran on.
    header = render_tier_table([tier_row_from_generate_report(_report("0.23.0"), source="a")])
    assert header.splitlines()[0].endswith("| Result | mflux |")


def test_a_generate_row_carries_the_reports_mflux_version():
    # Red when: the row ignores versions.mflux, or takes it from somewhere other than the report.
    assert tier_row_from_generate_report(_report("0.23.0"), source="a").mflux == "0.23.0"
    assert tier_row_from_generate_report(_report("0.20.0"), source="a").mflux == "0.20.0"


def test_a_report_without_versions_has_no_mflux_version():
    # Red when: a report written before versions existed is refused, or given a made-up version.
    assert tier_row_from_generate_report(_report(None), source="a").mflux is None


def test_a_row_older_than_the_newest_says_it_was_not_re_run():
    # Red when: an older row reads like a current one (the label hard-coded, or taken from the newest row), or the
    # newest row is also marked as not re-run.
    rows = [
        tier_row_from_generate_report(_report("0.23.0"), source="new.json"),
        tier_row_from_generate_report(_report("0.20.0"), source="old.json"),
    ]
    assert _cells(rows) == ["0.23", "0.20, not re-run on 0.23"]


def test_when_every_row_ran_on_one_version_no_row_says_not_re_run():
    # Red when: the "newest" version is a constant (0.23) rather than the newest among the rows.
    rows = [tier_row_from_generate_report(_report("0.20.0"), source=s) for s in ("a", "b")]
    assert _cells(rows) == ["0.20", "0.20"]


def test_the_newest_version_compares_numerically():
    # Red when: versions compare as strings ("0.9.1" > "0.23.0"), so the 0.23 row is called the old one.
    rows = [
        tier_row_from_generate_report(_report("0.9.1"), source="a"),
        tier_row_from_generate_report(_report("0.23.0"), source="b"),
    ]
    assert _cells(rows) == ["0.9, not re-run on 0.23", "0.23"]


def test_a_row_without_a_recorded_version_says_so():
    # Red when: an abort row (no versions in the artifact) is labelled with a guessed version, or a report without
    # versions breaks the render.
    rows = [
        tier_row_from_generate_report(_report("0.23.0"), source="a"),
        tier_row_from_abort_artifact(_abort(), source="b"),
        tier_row_from_generate_report(_report(None), source="c"),
    ]
    assert _cells(rows) == ["0.23", "not recorded", "not recorded"]


def _abort_on(versions: dict | None) -> dict:
    """The abort artifact, its run context recording ``versions`` (None: an older artifact without them)."""
    artifact = _abort()
    if versions is not None:
        artifact["context"] = {**artifact["context"], "versions": versions, "macos": "27.0.1"}
    return artifact


def test_an_abort_row_carries_the_mflux_version_its_run_context_records():
    # Red when: an abort row ignores context.versions, so every stopped run reads "not recorded" even when the
    # artifact says which mflux it ran on, or the row takes the newest version instead of its own.
    rows = [
        tier_row_from_generate_report(_report("0.23.0"), source="a"),
        tier_row_from_abort_artifact(_abort_on({"mlx": "0.32.2", "mflux": "0.20.0"}), source="b"),
        tier_row_from_abort_artifact(_abort_on(None), source="c"),
    ]
    assert rows[1].mflux == "0.20.0"
    assert _cells(rows) == ["0.23", "0.20, not re-run on 0.23", "not recorded"]


def test_a_stopped_runs_row_names_the_mflux_generate_ran_on():
    # Red when: generate's watchdog context (the abort artifact's) records no versions, so the row of a run the
    # watchdog stopped cannot name its mflux.
    from mlx_dfloat.mflux import generate as gen

    args = gen.build_parser().parse_args(["--prompt", "p"])
    limits = {"memory": 1, "cache": 2, "wired": 3}
    context = gen._run_context(args, tier_gb=16, label="CAPPED", limits=limits)
    row = tier_row_from_abort_artifact({**_abort(), "context": context}, source="b")
    assert row.mflux == _installed("mflux")


def _installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def test_a_pre_release_does_not_sort_above_its_release():
    # Red when: the version key reads the 1 of "rc1" as a fourth number, so 0.23.0rc1 counts as newer than 0.23.0 and
    # the 0.23.0 row is marked "not re-run on 0.23". Both orders: the newest is a max over the rows.
    for order in (("0.23.0", "0.23.0rc1"), ("0.23.0rc1", "0.23.0")):
        rows = [tier_row_from_generate_report(_report(v), source=v) for v in order]
        assert _cells(rows) == ["0.23", "0.23"], order


def test_a_later_release_still_sorts_above_an_earlier_pre_release():
    # Red when: the fix drops the numbers that do count (a constant or a one-number key ties 0.24.1 with 0.23.0rc1).
    rows = [tier_row_from_generate_report(_report(v), source=v) for v in ("0.23.0rc1", "0.24.1")]
    assert _cells(rows) == ["0.23, not re-run on 0.24", "0.24"]
