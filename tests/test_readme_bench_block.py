"""The README's generated numbers equal what the generator renders from the committed results."""

import re
from pathlib import Path

import pytest
from scripts import bench_table as bt

REPO = Path(__file__).resolve().parents[1]


def test_readme_blocks_equal_the_render_of_the_committed_results():
    # Bug caught: a hand-edited number in a bench block (or a result file added without
    # regenerating the README): the render differs from the text.
    readme = (REPO / "README.md").read_text()
    collected = bt.collect(REPO / "bench" / "results")
    assert bt.render_readme(readme, collected, date=collected.date) == readme


def test_hand_edit_is_caught(tmp_path):
    # Red when: render_readme stops rewriting a block, or --check stops reporting a stale one (a
    # hand-edited number would survive). The edit is to a real rendered number, the dev q8 ratio.
    readme = (REPO / "README.md").read_text()
    edited = readme.replace("same cache limit): 1.26×", "same cache limit): 1.20×", 1)  # noqa: RUF001
    assert edited != readme
    collected = bt.collect(REPO / "bench" / "results")
    assert bt.render_readme(edited, collected, date=collected.date) == readme
    path = tmp_path / "README.md"
    path.write_text(edited)
    assert bt.main(["--check", "--readme", str(path)]) == 1
    assert path.read_text() == edited  # --check never writes


# The "Does this help me?" table summarises the measured numbers in hand-written cells. They are read here and checked
# against the committed results, so a re-run that moves a number turns this file red until the cell is updated.
_GIB = 1024**3
README_PATH = REPO / "README.md"


def _help_row(starts_with: str) -> str:
    rows = [line for line in README_PATH.read_text().splitlines() if line.startswith(starts_with)]
    assert len(rows) == 1, (
        f"expected one README table row starting {starts_with!r}, found {len(rows)}"
    )
    return rows[0]


def _flux1_row() -> str:
    return _help_row("| A 32 GB Mac, FLUX.1-schnell, FLUX.1-dev or FLUX.1-Krea-dev,")


def _q8_row() -> str:
    return _help_row("| Less memory or a faster step,")


def _number(pattern: str, row: str) -> float:
    match = re.search(pattern, row)
    assert match is not None, f"{pattern!r} not found in the README row: {row}"
    return float(match.group(1))


def _summaries() -> dict:
    return bt.collect(REPO / "bench" / "results").summaries


def test_the_flux1_rows_step_overheads_are_the_committed_per_block_medians():
    # Bug caught: a scenario re-run moves the dev or schnell per-block overhead (now +3.5 % and +4.0 %) and the
    # hand-written "slower" cell keeps the old figure, or the two models' figures are swapped in the cell.
    row = _flux1_row()
    summaries = _summaries()
    for model in ("dev", "schnell"):
        stated = _number(rf"(\d+\.\d) % slower on {model}\b", row)
        measured = summaries[f"flux1-{model}-1024"].overhead["per-block"]
        assert measured > 0, f"{model}: the step is faster, not slower"
        assert f"{stated:.1f}" == f"{measured * 100:.1f}", model


def test_the_flux1_rows_schnell_peak_is_its_tier_files_watched_peak():
    # Bug caught: a schnell tier re-run moves its watched peak (now 19.77 GiB, mflux 0.23.0) and the hand-written cell
    # keeps the old figure, or the cell's number is taken from dev or Krea-dev's file.
    stated = _number(r"Measured peak at 1024²: (\d+\.\d\d) GiB for schnell \(mflux", _flux1_row())
    measured = _tier_row("schnell-1024.json").watched_peak_bytes / _GIB
    assert f"{stated:.2f}" == f"{measured:.2f}"


def test_the_flux1_rows_dev_and_krea_dev_peak_is_each_tier_rows_watched_peak_rounded_to_a_gib():
    # Bug caught: a dev or Krea-dev tier re-run peaks at, say, 21 GiB while the cell still says "about 20 GiB for dev
    # and Krea-dev", or one of the two drops out of the tier files the claim covers.
    stated = _number(r"about (\d+) GiB for dev and Krea-dev", _flux1_row())
    for file in ("dev-1024.json", "krea-dev-1024.json"):
        assert round(_tier_row(file).watched_peak_bytes / _GIB) == stated, file


def test_the_q8_rows_ratio_lies_within_the_measured_q8_ratios():
    # Bug caught: a re-run moves the DF11-over-q8 ratios (now 1.26 on dev, 1.35 on schnell) away from the cell's
    # "roughly 1.3 times faster"; "roughly" covers the measured span, so the stated figure must lie inside it.
    stated = _number(r"roughly (\d+\.\d) times faster", _q8_row())
    ratios = [s.q8_ratio for s in _summaries().values()]
    assert ratios
    assert None not in ratios
    assert min(ratios) <= stated <= max(ratios), ratios


def test_the_q8_rows_memory_saving_is_the_step_peak_gap_rounded_to_a_gib():
    # Bug caught: the q8 or DF11 step peak moves on a re-run (the gap is now 5.16 GiB on dev, 4.88 on schnell) and the
    # cell's "about 5 GiB lower" stays, or the gap is taken against the wrong condition.
    stated = _number(r"peaked about (\d+) GiB lower", _q8_row())
    for key, s in _summaries().items():
        gap = s.conditions["df11"]["step_watched_peak"] - s.conditions["q8"]["step_watched_peak"]
        assert round(gap / _GIB) == stated, key


def _tier_row(file: str):
    rows = [
        r for r in bt.collect(REPO / "bench" / "results").tier_rows if Path(r.source).name == file
    ]
    assert len(rows) == 1, f"expected one tier row from {file}, found {len(rows)}"
    return rows[0]


@pytest.mark.parametrize(
    ("row_start", "pattern", "tier_file"),
    [
        pytest.param(
            "| A 32 GB Mac, Z-Image-Turbo or Z-Image,",
            r"(\d+\.\d\d) GiB for Z-Image \(mflux 0\.23\)",
            "z-image-1024.json",
            id="z-image",
        ),
        pytest.param(
            "| A 32 GB Mac, FLUX.2 Klein (4B or 9B,",
            r"Peak at 1024²: (\d+\.\d\d) to \d+\.\d\d GiB for the 4B models",
            "flux2-klein-4b-1024.json",
            id="klein-4b-distilled",
        ),
        pytest.param(
            "| A 32 GB Mac, Qwen-Image-2.1,",
            r"Peak at 1024²: (\d+\.\d\d) GiB \(mflux 0\.23\)",
            "qwen-image-2.1-1024.json",
            id="qwen-image-2.1",
        ),
    ],
)
def test_a_hand_written_peak_cell_is_its_tier_files_watched_peak(row_start, pattern, tier_file):
    # Bug caught: a re-run of Z-Image (14.40 GiB), the distilled Klein 4B (11.85, the low end of the 4B span) or
    # Qwen-Image-2.1 (20.37) moves its watched peak in bench/results/tiers and the hand-written cell keeps the old one.
    stated = _number(pattern, _help_row(row_start))
    measured = _tier_row(tier_file).watched_peak_bytes / _GIB
    assert f"{stated:.2f}" == f"{measured:.2f}", tier_file
