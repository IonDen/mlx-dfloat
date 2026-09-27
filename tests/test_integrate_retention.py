"""Pins the per-block eval policy's bounded decoded footprint against the real Metal decoder.

The fake transformer's default hidden width (`D=4`) makes a block's decoded footprint under 1 KB,
far below MLX's own allocation granularity and the bound's slack -- an injected run-ahead bug
(the exact regression this test exists to catch) is invisible at that scale (measured peak_above
under 11 KB above baseline, comfortably under the 1 MB-dominated bound; see the task report for
the full numbers). This module scales the fake's width up before measuring, so the bound reflects
real retention instead of noise.
"""

import gc

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests import _flux_fakes
from tests._flux_fakes import FLUX_TABLE, FakeSeamTransformer, block_lists, df11_groups, inputs

from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.integrate.providers import DF11Provider

_WIDTH = 128  # D; the fakes keep FF = 2*D, so this also scales FF to 256


class _DiscardingSeen:
    """Stands in for `Recorder.seen`, but keeps nothing.

    `FakeDoubleBlock`/`FakeSingleBlock` append the block's `attn.to_q.weight` to `seen` on every
    run, for other tests' identity assertions. That weight is a view into the block's decoded DF11
    group, and MLX views pin their whole parent buffer resident (confirmed directly: dropping every
    other reference to a 20 MB source array while keeping one 2 KB split-off view still reports the
    full 20 MB active, mlx 0.32.2). A real `Recorder` would therefore pin every block's whole
    decoded buffer resident for the run's lifetime regardless of eval policy, swamping the
    measurement below with an artifact of test bookkeeping rather than eval-policy retention.
    """

    def append(self, _item):
        pass


class _MeasuringRecorder:
    """Same shape as `tests._flux_fakes.Recorder`, but `seen` does not pin decoded buffers."""

    def __init__(self):
        self.seen = _DiscardingSeen()
        self.events = []
        self.keep = []


def _scale_fake_width(monkeypatch, width):
    """Widen the fake blocks so a block's decoded footprint is several MB, not a few hundred bytes.

    `FakeDoubleBlock`/`FakeSingleBlock`/`inputs()` all read `_flux_fakes.D`/`.FF` from the module
    at call time (verified: assigning the module attributes before constructing changes the built
    layers' shapes), so patching them here reaches every fake built after this call.
    """
    monkeypatch.setattr(_flux_fakes, "D", width)
    monkeypatch.setattr(_flux_fakes, "FF", 2 * width)


@pytest.mark.metal
def test_per_block_evaluation_keeps_the_decoded_footprint_under_three_blocks(monkeypatch):
    # Bug caught: one lazy eval over every block's decode (every decoded group allocated at once,
    # the run-ahead the per-block eval exists to prevent), or a decoded array retained on a module
    # attribute after the step.
    _scale_fake_width(monkeypatch, _WIDTH)
    tf = FakeSeamTransformer(_MeasuringRecorder(), n_double=6, n_single=4)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(31))
    per_block = sum(2 * g.n_elements for g in groups.values()) / len(groups)
    provider = DF11Provider(groups, names, FLUX_TABLE)  # the Metal backend
    tf.attach(provider, shapes, eval_policy="per-block")
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = int(mx.get_active_memory())
    mx.eval(tf(*inputs()))
    tf.verify_step()
    peak_above = int(mx.get_peak_memory()) - base
    # Measured (mlx 0.32.2, this Mac, D=128/FF=256, 6 double + 4 single blocks, seed 31, warm
    # kernel): per_block == 694_681.6, peak_above == 997_564 (~1.44x per_block, well under the
    # 3-block bound below). An injected run-ahead bug on this same fixture (the seam's per-block
    # `mx.eval` replaced by a single eval at the very end -- exactly this test's target regression)
    # measured peak_above == 7_643_666, ~2.5x the bound. Both numbers reproduced exactly across
    # repeated runs.
    assert peak_above < 3 * per_block + 1_000_000  # three blocks of decoded output plus activations
    assert provider.launches == 10
    gc.collect()
    for name, param in tree_flatten(tf.parameters()):
        assert param.dtype != mx.bfloat16 or param.size == 0 or "norm" in name, name
