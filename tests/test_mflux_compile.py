"""The compile bypass: the override mechanics offline (a stand-in chip check), the real effect in the mflux lane."""

import sys
import types

import mlx.core as mx
import numpy as np
import pytest
from tests._zimage_tiny import tiny_groups, tiny_inputs, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider
from mlx_dfloat.mflux import _compile
from mlx_dfloat.mflux._compile import eager_factories, uncompiled

PLAIN = types.FunctionType
COMPILED = type(
    mx.compile(lambda x: x)
)  # mlx.gc_func: not a FunctionType, so `type(f) is` is the only honest check


class FakeChip:
    """mflux's ``AppleSiliconUtil`` stand-in: a classmethod answering "not M1/M2" (a Max chip)."""

    @classmethod
    def is_m1_or_m2(cls):
        return False


@pytest.fixture
def fake_chip(monkeypatch):
    module = types.ModuleType("mflux.utils.apple_silicon")
    module.AppleSiliconUtil = FakeChip
    for name in ("mflux", "mflux.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mflux.utils.apple_silicon", module)
    monkeypatch.setattr(_compile, "require_mflux", lambda: None)
    return FakeChip


def test_inside_the_context_the_chip_check_says_m1_and_afterwards_the_original_is_back(fake_chip):
    # Bug caught: the override never installed (factories still compile), or left installed after the block.
    original = fake_chip.__dict__["is_m1_or_m2"]
    with eager_factories():
        assert fake_chip.is_m1_or_m2() is True
    assert fake_chip.is_m1_or_m2() is False
    assert fake_chip.__dict__["is_m1_or_m2"] is original


def test_uncompiled_returns_what_the_factory_built_while_the_override_was_on(fake_chip):
    # Bug caught: the factory called outside the context (it reads the chip at call time, so it would compile), or
    # its arguments not passed through.
    seen = []

    def factory(a, b):
        seen.append((fake_chip.is_m1_or_m2(), a, b))
        return lambda: None

    assert type(uncompiled(factory, 1, 2)) is PLAIN
    assert seen == [(True, 1, 2)]


def test_a_factory_that_still_compiles_is_refused_and_named(fake_chip):
    # Bug caught: a future mflux reading the chip some other way (the override no longer reaches it), so the
    # factory returns mx.compile's gc_func and the seam's per-block eval fails deep in the first step, or (worse)
    # a FunctionType check by isinstance letting the gc_func subclass through.
    from mlx_dfloat.errors import DFloatIntegrationError

    def stubborn():
        return mx.compile(lambda x: x)

    with pytest.raises(DFloatIntegrationError, match=r"stubborn.*compiled"):
        uncompiled(stubborn)
    assert fake_chip.is_m1_or_m2() is False  # restored on the refusal too


def test_predict_mode_reports_what_the_factory_returns_under_the_override(fake_chip):
    # Bug caught: report()["predict"] a constant string ("uncompiled") that stays true when the bypass is lost.
    from mlx_dfloat.mflux._compile import predict_mode

    assert predict_mode(lambda: lambda: None) == "uncompiled"
    assert predict_mode(lambda: mx.compile(lambda x: x)) == "compiled"
    assert predict_mode(
        lambda: mx.compile(lambda x: x) if not fake_chip.is_m1_or_m2() else (lambda: 0)
    ) == ("uncompiled")


def test_the_fake_override_is_restored_when_the_factory_raises(fake_chip):
    # Bug caught: a raising factory leaving every later mflux model in the process on the eager path.
    def boom():
        raise RuntimeError("factory")

    with pytest.raises(RuntimeError, match="factory"):
        uncompiled(boom)
    assert fake_chip.is_m1_or_m2() is False


def _as_max_chip(monkeypatch):
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    return AppleSiliconUtil


@pytest.mark.mflux
def test_on_a_max_chip_stock_predict_compiles_and_the_bypass_returns_the_plain_function(
    monkeypatch,
):
    # Bug caught: the bypass not reaching the factory (mflux reads is_m1_or_m2 at call time, z_image.py:238), so the
    # seam's per-block mx.eval would run inside mx.compile. The first assert is the control that this test can fail.
    from mflux.models.z_image.variants.z_image import ZImage

    util = _as_max_chip(monkeypatch)
    assert type(ZImage._predict(object())) is COMPILED
    assert type(uncompiled(ZImage._predict, object())) is PLAIN
    assert util.is_m1_or_m2() is False  # restored
    assert type(ZImage._predict(object())) is COMPILED  # stock mflux still compiles


@pytest.mark.mflux
def test_the_override_is_restored_when_a_real_factory_raises(monkeypatch):
    # Bug caught: a raising factory leaving every later mflux model in the process on the eager path.
    util = _as_max_chip(monkeypatch)

    def boom(_t):
        raise RuntimeError("factory")

    with pytest.raises(RuntimeError, match="factory"):
        uncompiled(boom, object())
    assert util.is_m1_or_m2() is False


@pytest.mark.mflux
def test_an_uncompiled_cfg_step_decodes_once_per_block_per_transformer_call(monkeypatch):
    # Bug caught: predict compiled (decodes traced once, or the eval raises), the negative branch skipping the seam,
    # or a block decoded twice in one call. Two calls (CFG, z_image.py:222-236) x 4 blocks = 8 decodes.
    from mflux.models.z_image.variants.z_image import ZImage

    from mlx_dfloat.mflux.zimage.names import zimage_name_map

    _as_max_chip(monkeypatch)
    tf, shapes = tiny_seamed(n_layers=2)
    groups, names, _src = tiny_groups(shapes, np.random.default_rng(9))
    calls = []
    count = lambda g: calls.append(g.name) or decode_group(g, backend="reference")  # noqa: E731
    tf.attach(
        DF11Provider(groups, names, zimage_name_map(), decode=count), shapes, verify_in_call=True
    )
    predict = uncompiled(ZImage._predict, tf)
    noise = predict(guidance=4.0, **tiny_inputs())
    mx.eval(noise)
    assert calls == ["noise_refiner.0", "context_refiner.0", "layers.0", "layers.1"] * 2
    assert bool(mx.isfinite(noise).all().item())


@pytest.mark.mflux
def test_the_stock_compiled_predict_cannot_run_the_seam(monkeypatch):
    # Bug caught (the reason for the bypass, as a control): if mflux's compiled predict ever ran the per-block eval,
    # the bypass would be unnecessary and its cost on M3+ unjustified.
    from mflux.models.z_image.variants.z_image import ZImage

    _as_max_chip(monkeypatch)
    tf, shapes = tiny_seamed(n_layers=1)
    zeros = {
        b: {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in per.items()} for b, per in shapes.items()
    }
    tf.attach(ResidentProvider(zeros), shapes)
    with pytest.raises(ValueError, match=r"\[eval\] Attempting to eval"):
        mx.eval(ZImage._predict(tf)(guidance=0.0, **{**tiny_inputs(), "negative_encodings": None}))
