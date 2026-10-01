"""The README's generated numbers equal what the generator renders from the committed results."""

from pathlib import Path

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
