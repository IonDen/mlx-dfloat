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
    # Red when: render_readme stops rewriting the block (an edited number would survive).
    readme = (REPO / "README.md").read_text()
    edited = readme.replace("No result files yet.", "Overhead is 0 %.", 1)
    collected = bt.collect(tmp_path)
    assert bt.render_readme(edited, collected, date="") != edited
