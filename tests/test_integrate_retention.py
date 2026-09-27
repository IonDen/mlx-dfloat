"""Pins the bounded decoded footprint of the seam's eval policies against the real Metal decoder.

The fake transformer's default hidden width (`D=4`) makes a block's decoded footprint under 1 KB,
far below MLX's own allocation granularity, so an injected run-ahead bug (the regression this module
exists to catch) would be invisible at that scale: the measured peak stayed under 11 KB above
baseline either way. This module widens the fake to `D=128` before measuring, where one block's
decoded group is about 0.7 MB and the bound below reflects real retention instead of noise.
"""

import gc
from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from tests import _flux_fakes
from tests._flux_fakes import (
    FLUX_TABLE,
    FakeSeamTransformer,
    all_block_weights,
    block_lists,
    df11_groups,
    inputs,
)

from mlx_dfloat.decode import decode_group
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
@pytest.mark.parametrize("policy", ["per-block", "depth2"])
def test_block_evaluation_keeps_the_decoded_footprint_bounded_and_drops_it_after_the_step(
    monkeypatch, policy
):
    # Bug caught: one lazy eval over every block's decode (every decoded group allocated at once,
    # the run-ahead the per-block and depth-2 evals exist to prevent), or a decoded array retained
    # on a module attribute or in the seam's state after the step.
    _scale_fake_width(monkeypatch, _WIDTH)
    tf = FakeSeamTransformer(_MeasuringRecorder(), n_double=6, n_single=4)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(31))
    per_block = sum(2 * g.n_elements for g in groups.values()) / len(groups)
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=partial(decode_group, backend="metal")
    )
    tf.attach(provider, shapes, eval_policy=policy)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = int(mx.get_active_memory())
    mx.eval(tf(*inputs()))
    tf.verify_step()
    peak_above = int(mx.get_peak_memory()) - base
    # Measured (mlx 0.32.2, M1 Max, D=128/FF=256, 6 double + 4 single blocks, seed 31, warm kernel):
    # per_block == 694_681.6; per-block peak_above == 997_564 (~1.44x per_block), depth-2 peaked at
    # ~2.85x per_block. The bound is 3 blocks plus 1 MB of activations, about 4.4 blocks here. An
    # injected run-ahead bug on this fixture (the seam's per-block `mx.eval` replaced by a single
    # eval at the very end) measured peak_above == 7_643_666, ~2.5x the bound.
    assert peak_above < 3 * per_block + 1_000_000
    assert provider.launches == 10
    gc.collect()
    # After the step nothing decoded stays resident: every block matrix is a placeholder again and
    # the active memory is back near the base (measured ~17 KB above it).
    assert all(w.size == 0 for w in all_block_weights(tf))
    assert int(mx.get_active_memory()) - base < per_block // 4
