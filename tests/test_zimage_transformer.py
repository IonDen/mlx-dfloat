"""The Z-Image transformer with the class-swap seam: the pure parts offline, one real forward in the mflux lane."""

import importlib.util
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_dfloat.errors import DFloatDependencyError
from mlx_dfloat.integrate.placeholders import get_attr_path, install_placeholders
from mlx_dfloat.mflux.zimage.transformer import block_lists, seam_transformer_class


class NameTracer:
    """A trace hook that records the order blocks ran in."""

    def __init__(self):
        self.blocks = []

    def begin_step(self):
        pass

    def end_step(self):
        pass

    def record(self, block, **_times):
        self.blocks.append(block)


def test_block_lists_follow_the_run_order_of_the_transformer_call():
    # Bug caught: the lists in attribute order instead of call order (layers before the refiners), which
    # would run the status/eval bookkeeping and the tracer's block order against the wrong sequence.
    tf = SimpleNamespace(noise_refiner=[1], context_refiner=[2, 3], layers=[4])
    assert block_lists(tf) == [("noise_refiner", [1]), ("context_refiner", [2, 3]), ("layers", [4])]


def test_the_seamed_class_needs_the_mflux_extra(monkeypatch):
    # Bug caught: the adapter importing mflux without the guard (a bare ImportError instead of the
    # package's own dependency error naming the extra).
    # seam_transformer_class is functools.cache'd: an earlier test's class would hide the guard, so the
    # cache is emptied before the call and again after it (later tests rebuild the real class).
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "mflux" else real(name, *a, **k),
    )
    seam_transformer_class.cache_clear()
    try:
        with pytest.raises(DFloatDependencyError, match=r"mlx-dfloat\[mflux\]"):
            seam_transformer_class()
    finally:
        seam_transformer_class.cache_clear()


@pytest.mark.mflux
def test_a_real_zimage_transformer_runs_one_forward_through_the_class_swap_seam():
    # Bug caught: an mflux point release calling blocks some other way (the swapped __call__ never runs and the
    # blocks matmul zero-size placeholders), a block Linear set that differs from the map (install_placeholders
    # refuses), or a swap that moves parameters. mflux 0.20.0: ZImageTransformer(n_layers, n_refiner_layers, ...)
    # transformer.py:14-57, __call__(x, timestep, sigmas, cap_feats) :59-154; FeedForward hidden
    # int(3840 / 3 * 8) = 10240 (transformer_block.py:12); adaLN Linear(min(3840, 256), 4 * 3840) (:17).
    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.integrate.providers import ReuseProvider
    from mlx_dfloat.mflux.zimage.names import zimage_name_map

    tf = seam_transformer_class()(n_layers=1, n_refiner_layers=1)
    names = zimage_name_map()
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    shapes = install_placeholders(block_lists(tf), names)
    assert seam_blocks(block_lists(tf), tf.seam_cell) == (
        "noise_refiner.0",
        "context_refiner.0",
        "layers.0",
    )
    assert sorted(k for k, _v in tree_flatten(tf.parameters())) == before
    assert [len(per) for per in shapes.values()] == [8, 7, 8]
    assert shapes["layers.0"]["feed_forward.w1"] == (10240, 3840)
    assert shapes["noise_refiner.0"]["adaLN_modulation.0"] == (15360, 256)
    zeros = {
        kind: {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes[f"{kind}.0"].items()}
        for kind in ("noise_refiner", "context_refiner", "layers")
    }
    tracer = NameTracer()
    tf.attach(ReuseProvider(zeros, names), shapes, eval_policy="per-block", tracer=tracer)
    keys = mx.random.split(mx.random.key(0), 2)
    out = tf(
        x=mx.random.normal((16, 1, 4, 4), key=keys[0]),
        timestep=mx.array([0.5]),
        sigmas=mx.array([1.0, 0.0]),
        cap_feats=mx.random.normal((8, 2560), key=keys[1]),
    )
    mx.eval(out)
    tf.verify_step()
    assert out.shape == (16, 1, 4, 4)
    assert bool(mx.isfinite(out).all().item())
    assert tracer.blocks == ["noise_refiner.0", "context_refiner.0", "layers.0"]
    for block, per in shapes.items():
        kind, _dot, idx = block.partition(".")
        for attr in per:
            assert get_attr_path(getattr(tf, kind)[int(idx)], attr).weight.size == 0
