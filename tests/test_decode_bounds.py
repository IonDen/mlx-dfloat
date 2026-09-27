"""Memory safety under Metal shader validation: a positive control, our own over-read mutant, the real kernel."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.metal
REPO = Path(__file__).parents[1]
ENV = {
    **os.environ,
    "MTL_SHADER_VALIDATION": "1",
    "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1",
    "PYTHONPATH": str(REPO),
}


def _run(mode):
    proc = subprocess.run(
        [sys.executable, str(REPO / "tests" / "_shader_validation_repro.py"), mode],
        env=ENV,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok", f"stdout: {proc.stdout!r}\nstderr: {proc.stderr}"
    return proc.stderr


def test_validator_reports_the_positive_control():
    # Bug caught: a silent validator (wrong env var, wrong macOS) making the clean result below meaningless.
    assert "Invalid" in _run("control")


def test_validator_reports_our_unguarded_gap_mutant():
    # Bug caught: our own over-read being invisible (buffer slack), which would make the clean run prove nothing.
    # Offset 640 is the two-block fixture's one-byte read past its 640-byte `gaps`; the warm-up's own dispatch
    # over-reads at offset 2240 (7 blocks x 320 bytes), so it cannot satisfy this assertion.
    assert "at offset 640" in _run("mutant")


def test_our_kernel_is_clean_under_validation():
    # No out-of-bounds access reported on these small groups (every checked buffer is allocated at its exact
    # size); larger groups' page slack is not covered. Bug caught: the gap guard removed, a lookahead byte read
    # past `encoded`, a staged copy past `out`, on either path.
    assert "Invalid" not in _run("kernel")
