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


def test_an_unknown_model_and_the_none_policy_are_refused(monkeypatch, tmp_path):
    # Bug caught: mflux's from_name accepting "dev-fill" (another class upstream) and this path
    # building a FLUX.1 model for it; or policy "none" (every decode resident) accepted.
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    monkeypatch.setattr(base_init, "resolve", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(DFloatUnsupportedError, match="dev-fill"):
        DFloatFlux1("dev-fill")
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
    # Bug caught: planning at the requested 1000² while mflux runs 992² (Review Focus 1), or an
    # override below the two largest groups accepted silently (Review Focus 4).
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


def test_plan_drops_the_set_before_the_vae_only_when_the_resident_decode_would_not_fit(
    tmp_path, monkeypatch
):
    # Bug caught (the 2026-09-28 measurement: 23.29 GiB with the set resident at 1024²): a call planned with the
    # set resident through the VAE decode when that phase exceeds the budget (a paging storm), or the set dropped
    # for a small image where it would have fitted (a needless 28 s reload).
    big = _fake_model(
        tmp_path, monkeypatch, sizes=FluxSizes(compressed=15 * GIB, extras=0, encoders=0, vae=0)
    )
    plan = big.plan_call(height=1024, width=1024)
    assert plan.drop_set_before_vae
    assert plan.estimate.fits
    assert plan.estimate.peak_phase == "denoise"
    small = _fake_model(
        tmp_path, monkeypatch, sizes=FluxSizes(compressed=1 * GIB, extras=0, encoders=0, vae=0)
    )
    assert not small.plan_call(height=1024, width=1024).drop_set_before_vae
    seen = _patch_upstream_generate(monkeypatch, big)
    big.generate_image(seed=1, prompt="p", height=1024, width=1024)
    assert seen["set_resident"]  # resident for the loop
    assert not big._lifecycle.set_resident  # dropped by the guard
    assert big.report()["drop_set_before_vae"] is True


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

    seen = {}

    def fake(self, **kwargs):
        seen["limit_at_entry"] = _cache_limit_in_force()
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
    assert model._lifecycle.set_resident  # small sizes keep the set
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


def test_a_format_error_mid_step_drops_the_set_and_restores_the_limit(tmp_path, monkeypatch):
    # Bug caught (Review Focus 2): a corrupt block leaving the set attached with stale status words,
    # so the retry is refused by begin_step, or the cache limit left at the call's value.
    model = _fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(
        monkeypatch, model, raise_after_loop=DFloatFormatError("block 3: invalid code")
    )
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    with pytest.raises(DFloatIntegrationError, match="attach"):
        model.transformer._state()


def test_an_interrupt_keeps_the_set_resident_and_restores_the_limit(tmp_path, monkeypatch):
    # Bug caught (Review Focus 3): Ctrl-C (mflux's StopImageGenerationException) dropping the set
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
