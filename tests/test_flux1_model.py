"""Tests for `DFloatFlux1`: construction, refusals, memory planning, and the generate prelude.

Every test is `@pytest.mark.mflux`; the module imports mflux only inside helpers, so it collects
without it.
"""

import inspect
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tests._flux_fakes import (
    FLUX_TABLE,
    FakeSeamTransformer,
    Recorder,
    StubEncoder,
    StubTokenizer,
    block_lists,
    write_flux_checkpoint,
)

from mlx_dfloat.errors import (
    DFloatFormatError,
    DFloatIntegrationError,
    DFloatResourceError,
    DFloatUnsupportedError,
)
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.mflux.flux1 import init as base_init
from mlx_dfloat.mflux.flux1.init import BaseComponents, ResolvedRepo
from mlx_dfloat.mflux.flux1.memory import FluxSizes

GIB = 1024**3
BUDGET_32_GB = (
    24_653_119_488  # 22.96 GiB: budget_bytes() on the 32 GB M1 Max of the 2026-09-28 runs
)
# The schnell DF11 checkpoint and base as measured on 2026-09-28 (DF11 `51a428b9`, base `741f7c3c`).
SCHNELL_SIZES = FluxSizes(
    compressed=16_195_141_095, extras=113_899_648, encoders=9_770_792_936, vae=167_666_902
)
pytestmark = pytest.mark.mflux


def _fake_model(tmp_path, monkeypatch, **overrides):
    """A DFloatFlux1 over the fake-width transformer and stub base parts; no weights, no network."""
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=2)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    write_flux_checkpoint(tmp_path / "df11", shapes, np.random.default_rng(3))
    ckpt = open_checkpoint(tmp_path / "df11")
    (tmp_path / "base").mkdir(exist_ok=True)
    components = BaseComponents(
        vae=nn.Linear(1, 1),
        t5=StubEncoder(),
        clip=StubEncoder(),
        tokenizers={"t5": StubTokenizer(256), "clip": StubTokenizer(77)},
    )
    # Reloads after a drop build the same stubs (the real path loads from the base repo).
    monkeypatch.setattr(base_init, "load_encoders", lambda root: (StubEncoder(), StubEncoder()))
    # A fixed budget: the CI runner's device reports a 7 GiB working set, and the measured VAE and
    # overhead constants alone come close to it; the tests are about the arithmetic, not the host.
    from mlx_dfloat.mflux.flux1 import model as model_module

    monkeypatch.setattr(model_module, "budget_bytes", lambda: 23 * GIB)
    parts = {
        "model": "schnell",
        "model_config": ModelConfig.schnell(),
        "ckpt": ckpt,
        "df11": ResolvedRepo(
            root=tmp_path / "df11", repo_id="DFloat11/FLUX.1-schnell-DF11", revision="a" * 40
        ),
        "base": ResolvedRepo(
            root=tmp_path / "base", repo_id="black-forest-labs/FLUX.1-schnell", revision="b" * 40
        ),
        "components": components,
        "transformer": tf,
        "shapes": shapes,
        "name_map": FLUX_TABLE,
        "sizes": FluxSizes(compressed=1_000, extras=100, encoders=1_000, vae=10),
    }
    parts.update(overrides)
    return DFloatFlux1._from_parts(**parts)


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new `self.<name>` in Flux1.generate_image that
    # our init path never sets (an AttributeError deep inside the loop, after a 24 s set load).
    from mflux.models.flux.variants.txt2img.flux import Flux1

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(Flux1.generate_image)))
    assert names >= {
        "prompt_cache",
        "tokenizers",
        "t5_text_encoder",
        "clip_text_encoder",
        "transformer",
        "vae",
        "callbacks",
        "tiling_config",
        "bits",
        "lora_paths",
        "lora_scales",
        "model_config",
    }
    model = _fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, Flux1)
    assert model.bits is None
    assert model.lora_paths == []
    assert model.lora_scales == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantize": 8},
        {"lora_paths": ["a.safetensors"]},
        {"lora_scales": [1.0]},
        {"bake_lora": False},
    ],
)
def test_constructor_refusals_come_before_any_resolution(monkeypatch, kwargs):
    # Bug caught: resolving (and downloading) the checkpoint before refusing an option.
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    monkeypatch.setattr(
        base_init, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        DFloatFlux1("schnell", **kwargs)


@pytest.mark.metal
def test_a_failed_gpu_canary_refuses_the_model_before_any_resolution_or_load(monkeypatch):
    # Bug caught: the canary first running when the decode provider is built, which the model does only after the
    # encoders loaded, the prompt was encoded and the compressed set was read (minutes in on a broken GPU).
    import mlx.core as mx
    import numpy as np

    from mlx_dfloat import _metal_decode
    from mlx_dfloat.errors import DFloatBackendError
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    real = _metal_decode._dispatch

    def faulty(group, **kwargs):
        out, status = real(group, **kwargs)
        if group.name.startswith("canary-"):
            bits = np.array(out)
            bits[0] ^= 1
            out = mx.array(bits)
        return out, status

    monkeypatch.setattr(_metal_decode, "_dispatch", faulty)
    monkeypatch.setattr(_metal_decode, "_CANARY", {})
    monkeypatch.setattr(
        base_init, "resolve", lambda *a, **k: pytest.fail("resolved before the canary")
    )
    with pytest.raises(DFloatBackendError, match="canary"):
        DFloatFlux1("schnell")


def test_an_unknown_model_and_the_none_policy_are_refused(monkeypatch, tmp_path):
    # Bug caught: mflux's from_name accepting "dev-fill" (another class upstream) and this path
    # building a FLUX.1 model for it; or policy "none" (every decode resident) accepted, whether
    # the check runs early (__init__, before any resolution) or only late (_assemble, for a
    # _from_parts build).
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    monkeypatch.setattr(base_init, "resolve", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(DFloatUnsupportedError, match="dev-fill"):
        DFloatFlux1("dev-fill")
    with pytest.raises(DFloatUnsupportedError, match="none"):
        DFloatFlux1("schnell", eval_policy="none")
    with pytest.raises(DFloatUnsupportedError, match="none"):
        _fake_model(tmp_path, monkeypatch, eval_policy="none")


@pytest.mark.parametrize(
    "kwargs", [{"image_path": "in.png"}, {"image_strength": 0.5}, {"pid_decode": True}]
)
def test_generate_refusals_come_before_encoding_or_loading(tmp_path, monkeypatch, kwargs):
    # Bug caught: encoding the prompt (a 10 s encoder load) before refusing img2img.
    model = _fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        model.generate_image(seed=1, prompt="p", **kwargs)


def test_save_model_is_refused_and_freeze_skips_dropped_encoders(tmp_path, monkeypatch):
    # Bug caught: Flux1.freeze calling .freeze() on a dropped (None) encoder; save_model writing
    # a checkpoint with zero-size placeholders as the transformer.
    model = _fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_embeddings("p")  # drops the encoders
    assert model.t5_text_encoder is None
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))


def test_plan_call_rounds_to_multiples_of_16_and_warns_below_the_derived_minimum(
    tmp_path, monkeypatch, caplog
):
    # Bug caught: planning at the requested 1000² while mflux runs 992², or an
    # override below the two largest groups accepted silently.
    model = _fake_model(tmp_path, monkeypatch)
    assert (
        model.plan_call(height=1000, width=1000).cache_limit
        == model.plan_call(height=992, width=992).cache_limit
    )
    tiny = _fake_model(tmp_path, monkeypatch, cache_limit=1)
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.flux1"):
        plan = tiny.plan_call(height=256, width=256)
    assert plan.cache_limit == 1
    assert "below the derived minimum" in caplog.text


def _budget(monkeypatch, value):
    from mlx_dfloat.mflux.flux1 import model as model_module

    monkeypatch.setattr(model_module, "budget_bytes", lambda: value)


@pytest.mark.parametrize("size", [512, 768, 1024])
def test_on_a_32_gb_budget_every_schnell_call_drops_the_set_before_the_vae(
    tmp_path, monkeypatch, size
):
    # Bug caught (the 2026-09-28 measurement: 23.29 GiB with the set resident at 1024², and the
    # decode's 7.82 GiB transient is a floor below it): a smaller image planned with the set
    # resident through the decode, on the claim that a smaller decode would fit next to it.
    model = _fake_model(tmp_path, monkeypatch, sizes=SCHNELL_SIZES)
    _budget(monkeypatch, BUDGET_32_GB)
    plan = model.plan_call(height=size, width=size)
    assert plan.drop_set_before_vae
    assert plan.estimate.fits


def test_a_64_gb_budget_keeps_the_set_through_the_vae_at_1024(tmp_path, monkeypatch):
    # Bug caught: the set dropped (a 26 s reload on every call) where the budget holds the
    # resident set plus the decode, i.e. the drop made unconditional.
    model = _fake_model(tmp_path, monkeypatch, sizes=SCHNELL_SIZES)
    _budget(monkeypatch, 40 * GIB)
    assert not model.plan_call(height=1024, width=1024).drop_set_before_vae


def test_a_budget_passed_to_the_model_replaces_the_device_budget_in_the_fit_refusal(
    tmp_path, monkeypatch
):
    # Bug caught: plan_call ignoring the constructor's budget_bytes (a --tier 16 run planned and
    # refused against the host's 23 GiB instead of the tier's). The fixture patches the device
    # budget to 23 GiB, where schnell at 1024² fits; only the 10 GiB override refuses it.
    model = _fake_model(tmp_path, monkeypatch, sizes=SCHNELL_SIZES, budget_bytes=10 * GIB)
    with pytest.raises(DFloatResourceError, match=r"10\.0 GiB"):
        model.plan_call(height=1024, width=1024)


def test_a_32_gb_ceiling_passed_as_the_budget_drops_the_set_before_the_vae(tmp_path, monkeypatch):
    # Bug caught (the rev 1 bug): the override ignored in the VAE decision. The device budget is
    # patched to 40 GiB (the set kept through the decode); the 22.96 GiB override must drop it, or
    # a capped run keeps the set resident and aborts in the VAE decode.
    model = _fake_model(tmp_path, monkeypatch, sizes=SCHNELL_SIZES, budget_bytes=int(22.96 * GIB))
    _budget(monkeypatch, 40 * GIB)
    assert model.plan_call(height=1024, width=1024).drop_set_before_vae is True


def test_the_report_carries_the_component_sizes(tmp_path, monkeypatch):
    # Bug caught: the sizes missing from the report (the README's weights column reads
    # sizes.compressed + sizes.extras), or one component dropped or mislabelled.
    model = _fake_model(tmp_path, monkeypatch)
    # The fixture's own FluxSizes(compressed=1_000, extras=100, encoders=1_000, vae=10).
    assert model.report()["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "encoders": 1_000,
        "vae": 10,
    }


def test_a_call_planned_to_drop_the_set_runs_the_loop_with_it_and_the_guard_drops_it(
    tmp_path, monkeypatch
):
    # Bug caught: the after-loop guard ignoring the plan (the set resident through a decode
    # measured over the budget), or dropping it before the loop.
    big = _fake_model(
        tmp_path, monkeypatch, sizes=FluxSizes(compressed=15 * GIB, extras=0, encoders=0, vae=0)
    )
    plan = big.plan_call(height=1024, width=1024)
    assert plan.drop_set_before_vae
    assert plan.estimate.fits
    assert plan.estimate.peak_phase == "denoise"
    seen = _patch_upstream_generate(monkeypatch, big)
    big.generate_image(seed=1, prompt="p", height=1024, width=1024)
    assert seen["set_resident"]  # resident for the loop
    assert not big._lifecycle.set_resident  # dropped by the guard
    assert big.report()["drop_set_before_vae"] is True


def test_plan_call_refuses_sizes_above_the_measured_1024_ceiling_unless_fit_check_is_off(
    tmp_path, monkeypatch, caplog
):
    # Bug caught: a call above 1024² planned on an estimate no run has checked (the VAE transient
    # and the activation term are extrapolations there), or fit_check=False refusing instead of
    # warning.
    model = _fake_model(tmp_path, monkeypatch)
    model.plan_call(height=1024, width=1024)  # the measured size itself
    with pytest.raises(DFloatResourceError, match="fit_check=False") as info:
        model.plan_call(height=1024, width=1040)
    assert "1024" in str(info.value)
    lax = _fake_model(tmp_path, monkeypatch, fit_check=False)
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.flux1"):
        lax.plan_call(height=1024, width=1040)
    assert "extrapolation" in caplog.text


def test_plan_call_refuses_an_over_budget_call_and_fit_check_false_warns_instead(
    tmp_path, monkeypatch, caplog
):
    # Bug caught: the fit rule not enforced (a paging storm instead of an error), or fit_check=False not logging.
    model = _fake_model(
        tmp_path, monkeypatch, sizes=FluxSizes(compressed=40 * GIB, extras=0, encoders=0, vae=0)
    )
    with pytest.raises(DFloatResourceError, match="denoise"):
        model.plan_call(height=1024, width=1024)
    lax = _fake_model(
        tmp_path,
        monkeypatch,
        fit_check=False,
        sizes=FluxSizes(compressed=40 * GIB, extras=0, encoders=0, vae=0),
    )
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.flux1"):
        lax.plan_call(height=1024, width=1024)
    assert "fit_check=False" in caplog.text


def _patch_upstream_generate(monkeypatch, model, *, raise_after_loop=None, raise_before_loop=None):
    """Replace Flux1.generate_image with a probe: records the cache limit in force, fires our
    after-loop subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from tests._mlx_limits import current_limits

    seen = {}

    def fake(self, **kwargs):
        seen["limit_at_entry"] = _cache_limit_in_force()
        seen["limits_at_entry"] = current_limits()
        seen["set_resident"] = self._lifecycle.set_resident
        seen["kwargs"] = kwargs
        if raise_before_loop is not None:
            raise raise_before_loop
        for subscriber in self.callbacks.after_loop_callbacks():
            subscriber.call_after_loop(
                seed=kwargs["seed"], prompt=kwargs["prompt"], latents=mx.zeros(1), config=None
            )
        seen["limit_after_loop"] = _cache_limit_in_force()
        if raise_after_loop is not None:
            raise raise_after_loop
        return "image"

    monkeypatch.setattr(Flux1, "generate_image", fake)
    return seen


def test_generate_encodes_then_loads_the_set_then_runs_upstream_under_the_derived_limit_and_restores_it(
    tmp_path, monkeypatch
):
    # Bug caught: the set loaded before the prompt is encoded (T5 next to the set), the cache
    # limit not in force during the loop, the VAE guard not zeroing it, or the process limit left changed.
    model = _fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    seen = _patch_upstream_generate(monkeypatch, model)
    plan = model.plan_call(height=256, width=256)
    image = model.generate_image(seed=7, prompt="p", num_inference_steps=1, height=256, width=256)
    assert image == "image"
    assert seen["set_resident"]
    assert seen["limit_at_entry"] == plan.cache_limit
    assert seen["limit_after_loop"] == 0
    assert _cache_limit_in_force() == before
    assert model._lifecycle.set_resident  # the fake's 1 kB set fits next to the decode
    assert model.t5_text_encoder is None
    assert seen["kwargs"]["image_path"] is None
    assert seen["kwargs"]["pid_decode"] is False
    report = model.report()
    assert report["cache_limit_in_force"] == plan.cache_limit
    assert report["fit"]["label"] == "predicted"
    assert report["drop_set_before_vae"] is False
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert report["lifecycle"]["set_loads"] == 1
    assert report["model"] == "schnell"


def test_a_new_prompt_with_the_set_resident_drops_it_reloads_the_encoders_and_reloads_the_set(
    tmp_path, monkeypatch
):
    # Bug caught: a new prompt encoded next to the resident compressed set (the encoders and the
    # set sharing memory that the lifecycle exists to keep apart), or the set not reloaded before
    # the second call's denoise step.
    model = _fake_model(tmp_path, monkeypatch)
    calls = []

    def recording_load_encoders(root):
        calls.append(root)
        return StubEncoder(), StubEncoder()

    monkeypatch.setattr(base_init, "load_encoders", recording_load_encoders)
    _patch_upstream_generate(monkeypatch, model)
    model.generate_image(seed=1, prompt="p1", height=256, width=256)
    model.generate_image(seed=2, prompt="p2", height=256, width=256)
    lifecycle = model.report()["lifecycle"]
    assert lifecycle["forced_set_drops"] == 1
    assert lifecycle["encoder_loads"] == 1
    assert lifecycle["set_loads"] == 2
    assert calls == [model._base.root]


@pytest.mark.parametrize("where", ["raise_before_loop", "raise_after_loop"])
def test_a_format_error_mid_step_drops_the_set_and_restores_the_limit(tmp_path, monkeypatch, where):
    # Bug caught: a corrupt block leaving the set attached with stale status words,
    # so the retry is refused by begin_step, or the cache limit left at the call's value; checked
    # both before the after-loop guard ran (inside the loop) and after it (during the decode).
    model = _fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(
        monkeypatch, model, **{where: DFloatFormatError("block 3: invalid code")}
    )
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    with pytest.raises(DFloatIntegrationError, match="attach"):
        model.transformer._state()


def test_a_format_errors_traceback_does_not_pin_the_set_past_the_drop(tmp_path, monkeypatch):
    # Bug caught: `except DFloatFormatError: self._lifecycle.drop_set(); raise` used to run while
    # the exception's own traceback still held its raising frame's locals alive — in production
    # that frame is inside SeamMixin.__call__ / DF11Provider.verify() and references the whole
    # resident set, so drop_set's own gc.collect() could not free it and _reclaim's active-memory
    # check turned a clean DFloatFormatError into a DFloatResourceError (the real error demoted to
    # __context__).
    import dataclasses

    from mflux.models.flux.variants.txt2img.flux import Flux1

    model = _fake_model(tmp_path, monkeypatch)

    def fake(self, **kwargs):
        # A frame that still references the resident set (an 8 MiB stand-in for a real block's
        # decoded weights) at the moment the format error is raised, like a real seam frame would.
        name, group = next(iter(self._provider._resident.items()))
        big = mx.zeros(2 * 1024 * 1024, dtype=mx.float32)  # 8 MiB
        mx.eval(big)
        self._provider._resident[name] = dataclasses.replace(group, sign_mantissa=big)
        held = next(iter(self._provider._resident.values())).sign_mantissa
        mx.eval(held)
        raise DFloatFormatError("block 3: invalid code")

    monkeypatch.setattr(Flux1, "generate_image", fake)
    bound = int(mx.get_active_memory()) + 1 * 1024**2
    model._lifecycle.retained_bound = lambda: bound
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident


def test_an_interrupt_keeps_the_set_resident_and_restores_the_limit(tmp_path, monkeypatch):
    # Bug caught: Ctrl-C (mflux's StopImageGenerationException) dropping the set
    # (a 24 s reload on the retry) or leaving the limit changed.
    from mflux.utils.exceptions import StopImageGenerationException

    model = _fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(
        monkeypatch, model, raise_before_loop=StopImageGenerationException("stop")
    )
    with pytest.raises(StopImageGenerationException):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert model._lifecycle.set_resident
    assert _cache_limit_in_force() == before


def test_from_name_builds_this_class_and_a_repeated_prompt_reuses_the_cache(tmp_path, monkeypatch):
    # Bug caught: from_name returning a stock Flux1 (mflux's, over the 24 GB base); a second call
    # with the same prompt reloading the encoders.
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    built = {}
    monkeypatch.setattr(
        DFloatFlux1, "__init__", lambda self, model, **kw: built.update(model=model, **kw)
    )
    DFloatFlux1.from_name("dev")
    assert built == {"model": "dev", "quantize": None}
    model = _fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch, model)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    model.generate_image(seed=2, prompt="p", height=256, width=256)
    assert (
        model.report()["lifecycle"]["encoder_loads"] == 0
    )  # the construction-time encoders served the first
    assert model.report()["lifecycle"]["set_loads"] == 1


def test_the_default_repositories_per_model_are_the_published_ones():
    # Bug caught: a typo in a default repo id (a 404 at the user's first run).
    from mlx_dfloat.mflux.flux1.model import MODELS

    assert MODELS == {
        "schnell": ("DFloat11/FLUX.1-schnell-DF11", "black-forest-labs/FLUX.1-schnell"),
        "dev": ("DFloat11/FLUX.1-dev-DF11", "black-forest-labs/FLUX.1-dev"),
        "krea-dev": ("DFloat11/FLUX.1-Krea-dev-DF11", "black-forest-labs/FLUX.1-Krea-dev"),
    }


def test_a_phase_peak_is_never_below_what_was_active_when_it_began(tmp_path, monkeypatch):
    # Bug caught: `mx.reset_peak_memory()` resets the counter to zero, not to the active memory,
    # so a phase that allocates nothing new reports an MLX peak below what it held throughout.
    model = _fake_model(tmp_path, monkeypatch)
    held = mx.zeros(4 * 1024 * 1024, dtype=mx.float32)  # 16 MiB alive through the phase
    mx.eval(held)
    model._phase_begin("probe")
    model._phase_end("probe")
    assert model._peaks["probe"]["mlx_peak"] >= 16 * 1024 * 1024
    del held


def test_the_retained_bound_grows_by_the_cached_embeddings(tmp_path, monkeypatch):
    # Bug caught: the bound fixed at construction, so every embedding cached afterwards counts as
    # a leak against the drop check (and enough prompts turn a clean drop into an error).
    model = _fake_model(tmp_path, monkeypatch)
    before = model._lifecycle.retained_bound()
    model.encode("a", "b")
    grown = sum(int(a.nbytes) for pair in model.prompt_cache.values() for a in pair)
    assert grown > 0
    assert model._lifecycle.retained_bound() - before == grown


class _ShapedEncoder(nn.Module):
    """An encoder stub whose output has a real FLUX.1 embedding shape and depends on its weight."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.bias = mx.zeros((shape[-1],))

    def __call__(self, ids):
        return mx.broadcast_to(self.bias, self.shape) + ids.astype(mx.float32).sum() * 0


class _StubVAE(nn.Module):
    """``decode(latents)`` → a 64x64 RGB image in mflux's 5-D decoder layout."""

    def __init__(self):
        super().__init__()
        self.scale = mx.zeros((1,))

    def decode(self, latents):
        return mx.zeros((1, 3, 1, 64, 64)) + self.scale


def test_upstream_generate_image_runs_through_the_model_on_the_cached_embeddings(
    tmp_path, monkeypatch
):
    # Bug caught: a lifecycle caching into a copy of `prompt_cache` (upstream would then miss the
    # prompt and call the dropped encoder, None), or the after-loop guard not registered where
    # mflux's own loop fires it. Runs mflux 0.20's real `Flux1.generate_image` (latents, scheduler,
    # the seamed Transformer at 1+1 blocks, callbacks, VAEUtil.decode, ImageUtil.to_image).
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
    from mflux.utils.generated_image import GeneratedImage

    from mlx_dfloat.integrate.providers import ResidentProvider
    from mlx_dfloat.mflux.flux1.names import flux_name_map
    from mlx_dfloat.mflux.flux1.transformer import seam_transformer_class

    tf = seam_transformer_class()(
        ModelConfig.schnell(), num_transformer_blocks=1, num_single_transformer_blocks=1
    )
    shapes = install_placeholders(
        [
            ("transformer_blocks", tf.transformer_blocks),
            ("single_transformer_blocks", tf.single_transformer_blocks),
        ],
        flux_name_map(),
    )
    components = BaseComponents(
        vae=_StubVAE(),
        t5=_ShapedEncoder((1, 256, 4096)),
        clip=_ShapedEncoder((1, 768)),
        tokenizers={"t5": StubTokenizer(256), "clip": StubTokenizer(77)},
    )
    model = _fake_model(tmp_path, monkeypatch, transformer=tf, components=components)
    model._shapes = shapes

    def load_resident_zeros():
        zeros = {
            block: {attr: mx.zeros(shape, dtype=mx.bfloat16) for attr, shape in per.items()}
            for block, per in shapes.items()
        }
        model.transformer.attach(
            ResidentProvider(zeros), shapes, eval_policy=model._policy, verify_in_call=True
        )

    monkeypatch.setattr(model._lifecycle, "_load_set", load_resident_zeros)
    upstream_calls = []
    real_encode = PromptEncoder.encode_prompt

    def recording_encode(prompt, prompt_cache, **kwargs):
        if prompt_cache is model.prompt_cache:
            upstream_calls.append((prompt in prompt_cache, kwargs["t5_text_encoder"]))
        return real_encode(prompt, prompt_cache, **kwargs)

    monkeypatch.setattr(PromptEncoder, "encode_prompt", staticmethod(recording_encode))
    image = model.generate_image(
        seed=1, prompt="p", num_inference_steps=2, height=64, width=64, guidance=0.0
    )
    assert isinstance(image, GeneratedImage)
    assert image.image.size == (64, 64)
    assert upstream_calls == [(True, None)]
    assert model.t5_text_encoder is None
    report = model.report()
    assert "vae" in report["peaks"]
    assert report["decode_launches"] == 0


def test_open_phase_names_the_phase_running_now_and_none_outside_a_call(tmp_path, monkeypatch):
    # Bug caught: the watchdog's abort context naming no phase, or a stale one: the phase must follow the call
    # through encode, set load, the loop and the VAE decode, and be None again once the call returns.
    from mflux.models.flux.variants.txt2img.flux import Flux1

    model = _fake_model(tmp_path, monkeypatch)
    assert model.open_phase is None
    seen = []
    life = model._lifecycle
    for attr in ("_encode", "_load_set"):
        real = getattr(life, attr)

        def wrapped(*args, _real=real, **kwargs):
            seen.append(model.open_phase)
            return _real(*args, **kwargs)

        monkeypatch.setattr(life, attr, wrapped)

    def fake(self, **kwargs):
        seen.append(self.open_phase)
        for subscriber in self.callbacks.after_loop_callbacks():
            subscriber.call_after_loop(seed=1, prompt="p", latents=mx.zeros(1), config=None)
        seen.append(self.open_phase)
        return "image"

    monkeypatch.setattr(Flux1, "generate_image", fake)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert seen == ["encode", "set_load", "denoise", "vae"]
    assert model.open_phase is None


@pytest.mark.parametrize("raised", [None, RuntimeError("in the loop")], ids=["returns", "raises"])
def test_a_python_api_call_runs_under_the_commands_caps_and_restores_mlxs_defaults(
    tmp_path, monkeypatch, raised
):
    # Bug caught: generate_image not run under the per-call caps (a Python-API process sits at MLX's default wired
    # limit 0, while every VAE term was measured under the command's caps), or the caps left installed after a call
    # that returns or raises.
    from tests._mlx_limits import command_caps, current_limits, mlx_without_wired_cap

    model = _fake_model(tmp_path, monkeypatch)
    seen = _patch_upstream_generate(monkeypatch, model, raise_before_loop=raised)
    wired, memory = command_caps()
    assert wired > 0
    with mlx_without_wired_cap() as start:
        if raised is None:
            model.generate_image(seed=1, prompt="p", height=256, width=256)
        else:
            with pytest.raises(RuntimeError, match="in the loop"):
                model.generate_image(seed=1, prompt="p", height=256, width=256)
        after = current_limits()
    assert (seen["limits_at_entry"]["wired"], seen["limits_at_entry"]["memory"]) == (wired, memory)
    assert after == start


def test_encode_runs_under_the_commands_caps_and_restores_mlxs_defaults(tmp_path, monkeypatch):
    # Bug caught: encode() from Python run at MLX's default wired limit 0 (a prompt encode is one of the measured
    # phases, sized under the command's caps), or the caps left installed after it returns.
    from tests._mlx_limits import command_caps, current_limits, mlx_without_wired_cap

    model = _fake_model(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(
        model._lifecycle, "ensure_embeddings", lambda *prompts: seen.append(current_limits())
    )
    wired, memory = command_caps()
    assert wired > 0
    with mlx_without_wired_cap() as start:
        model.encode("p")
        after = current_limits()
    assert (seen[0]["wired"], seen[0]["memory"]) == (wired, memory)
    assert after == start
