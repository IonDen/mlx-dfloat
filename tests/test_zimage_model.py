"""Tests for `DFloatZImage`: construction, refusals, the lifecycle and the set. Every test is `@pytest.mark.mflux`;
the module imports mflux only inside helpers, so it collects without it."""

import inspect
import re
import types

import mlx.core as mx
import numpy as np
import pytest
from tests._zimage_tiny import TINY_SEED, StubTextEncoder, fake_model, write_tiny_checkpoint

from mlx_dfloat.errors import DFloatUnsupportedError
from mlx_dfloat.mflux.zimage import init as zinit

pytestmark = pytest.mark.mflux


def _source(tmp_path):
    """The matrices ``fake_model`` compressed: the same seed gives the same arrays."""
    groups, _constants = write_tiny_checkpoint(tmp_path / "again", np.random.default_rng(TINY_SEED))
    return groups


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new self.<name> in ZImage.generate_image our init never sets.
    from mflux.models.z_image.variants.z_image import ZImage

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(ZImage.generate_image)))
    assert names >= {"vae", "transformer", "callbacks", "tiling_config", "model_config", "bits"}
    model = fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, ZImage)
    assert (model.bits, model.lora_paths, model.lora_scales) == (None, [], [])


@pytest.mark.parametrize(
    "kwargs", [{"quantize": 8}, {"lora_paths": ["a"]}, {"lora_scales": [1.0]}, {"bake_lora": False}]
)
def test_constructor_refusals_come_before_any_resolution(monkeypatch, kwargs):
    # Bug caught: resolving (downloading 8 GB) before refusing an option this path cannot honour.
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    monkeypatch.setattr(
        zinit, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        DFloatZImage("z-image-turbo", **kwargs)


@pytest.mark.metal
def test_a_failed_gpu_canary_refuses_the_model_before_any_resolution(monkeypatch):
    # Bug caught: the canary first running at set load, minutes in (after the encoder loaded and the prompt encoded).
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.errors import DFloatBackendError
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

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
    monkeypatch.setattr(zinit, "resolve", lambda *a, **k: pytest.fail("resolved before the canary"))
    with pytest.raises(DFloatBackendError, match="canary"):
        DFloatZImage("z-image-turbo")


def test_the_default_checkpoint_resolves_at_its_pin_and_a_user_checkpoint_unpinned(monkeypatch):
    # Bug caught: the pin not reaching the resolver (the default follows main), or applied to a user's --df11.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    class StopError(Exception):
        pass

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        calls.append((spec, revision))
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(zinit, "resolve", fake_resolve)
    for kwargs in (
        {"model": "z-image-turbo"},
        {"model": "z-image"},
        {"model": "z-image", "df11_path": "me/fork"},
    ):
        with pytest.raises(StopError):
            DFloatZImage(**kwargs)
    assert calls == [
        ("mingyi456/Z-Image-Turbo-DF11", "422db601eac214d0f1e2cc99e2cf34906e8a7267"),
        ("mingyi456/Z-Image-DF11", "36d582560630bfe2f5b78183c4164c66d6919e5b"),
        ("me/fork", None),
    ]


def test_controlnet_unknown_models_and_the_none_policy_are_refused(monkeypatch, tmp_path):
    # Bug caught: "z-image-turbo-controlnet-union-2.1" (another class upstream) built as plain Turbo; policy "none"
    # (every decode resident) accepted, whether the check runs early (__init__) or only late (_assemble).
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    monkeypatch.setattr(zinit, "resolve", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(DFloatUnsupportedError, match="ControlNet is another model class"):
        DFloatZImage("z-image-turbo-controlnet-union-2.1")
    with pytest.raises(DFloatUnsupportedError, match="schnell"):
        DFloatZImage("schnell")
    with pytest.raises(DFloatUnsupportedError, match="none"):
        DFloatZImage("z-image-turbo", eval_policy="none")
    with pytest.raises(DFloatUnsupportedError, match="none"):
        fake_model(tmp_path, monkeypatch, eval_policy="none")


def test_the_set_load_installs_the_nonblock_weight_and_attaches_with_verify_in_call(
    tmp_path, monkeypatch
):
    # Bug caught: cap_embedder.1 left on its placeholder (zero-size matmul at the first step), or the set attached
    # without verify_in_call (base with CFG would refuse its second transformer call).
    model = fake_model(tmp_path, monkeypatch, model="z-image")
    model._lifecycle.ensure_set()
    w = model.transformer.cap_embedder[1].weight
    expected = _source(tmp_path)["cap_embedder"][0]
    assert w.size > 0
    assert np.array_equal(np.array(w.view(mx.uint16)), expected)
    assert model.transformer.seam_cell.state.verify_in_call is True


def test_a_drop_restores_every_placeholder_and_a_reload_reinstalls_the_nonblock_weight(
    tmp_path, monkeypatch
):
    # Bug caught (Review Focus 2): the second load after a new prompt finding cap_embedder.1 still a placeholder,
    # or the dropped set still referenced by the module (the active-memory check would refuse the reload).
    model = fake_model(tmp_path, monkeypatch)
    expected = _source(tmp_path)["cap_embedder"][0]
    model._lifecycle.ensure_set()
    model._lifecycle.drop_set()
    assert model.transformer.cap_embedder[1].weight.size == 0
    assert model.transformer.seam_cell.state is None
    model._lifecycle.ensure_set()
    again = np.array(model.transformer.cap_embedder[1].weight.view(mx.uint16))
    assert np.array_equal(again, expected)


def test_a_new_prompt_encodes_to_one_array_and_reloads_the_encoder_from_the_base(
    tmp_path, monkeypatch
):
    # Bug caught: the cached embeddings not a tuple (the lifecycle's cache is shared with FLUX.1's pairs), a prompt
    # evaluated lazily (the encoder weights stay alive past the drop), or the reload not reading the base.
    model = fake_model(tmp_path, monkeypatch)
    loads = []
    monkeypatch.setattr(
        zinit,
        "load_text_encoder",
        lambda root: (loads.append(root), StubTextEncoder())[1],
    )
    model._lifecycle.ensure_embeddings("a")  # encoders pre-loaded by the build: no reload
    assert model.text_encoder is None
    assert loads == []
    model._lifecycle.ensure_embeddings("b")
    assert loads == [model._base.root]
    assert sorted(model._embeddings) == ["a", "b"]
    assert len(model._embeddings["a"]) == 1
    assert model._embeddings["a"][0].shape == (8, 32)


def test_predict_is_the_plain_function_on_a_max_chip(tmp_path, monkeypatch):
    # Bug caught: our class not overriding _predict (mflux's compiled closure would run the seam under compile).
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch)
    compiled = type(
        mx.compile(lambda x: x)
    )  # mlx.gc_func: a FunctionType subclass, so never isinstance
    assert (
        type(ZImage._predict(model.transformer)) is compiled
    )  # the control: upstream compiles here
    ours = model._predict(model.transformer)
    assert type(ours) is types.FunctionType


def test_a_lost_bypass_is_refused_at_predict_and_the_report_says_compiled(tmp_path, monkeypatch):
    # Bug caught: the chip-check override no longer reaching mflux's factory (here: the override made a no-op on a
    # Max chip) while _predict hands the compiled closure to the loop and report() still prints "uncompiled".
    from contextlib import nullcontext

    from mflux.utils.apple_silicon import AppleSiliconUtil

    from mlx_dfloat.errors import DFloatIntegrationError
    from mlx_dfloat.mflux import _compile

    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    assert model.report()["predict"] == "uncompiled"
    monkeypatch.setattr(_compile, "eager_factories", nullcontext)
    with pytest.raises(DFloatIntegrationError, match="compiled"):
        model._predict(model.transformer)
    assert model.report()["predict"] == "compiled"


def test_save_model_is_refused_and_freeze_skips_a_dropped_encoder(tmp_path, monkeypatch):
    # Bug caught: freeze() calling .freeze() on a dropped (None) encoder; save_model writing a checkpoint with
    # zero-size placeholders as the transformer.
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_embeddings("p")  # drops the encoder
    assert model.text_encoder is None
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))


# --- the generate prelude, the call plan ----------------------------------------------------------------------

GIB = 1024**3


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def _patch_upstream_generate(monkeypatch, model, *, raise_after_loop=None, raise_before_loop=None):
    """Replace ZImage.generate_image with a probe: records the cache limit in force, fires our after-loop
    subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.z_image.variants.z_image import ZImage
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
        seen["set_resident_after_loop"] = self._lifecycle.set_resident
        if raise_after_loop is not None:
            raise raise_after_loop
        return "image"

    monkeypatch.setattr(ZImage, "generate_image", fake)
    return seen


@pytest.mark.parametrize(
    ("model", "guidance", "negative", "expected"),
    [
        ("z-image", None, None, ("p",)),
        ("z-image", 1.0, None, ("p",)),
        ("z-image", 4.0, None, ("p", " ")),
        ("z-image", 4.0, "  ", ("p", " ")),
        ("z-image", 4.0, "blurry", ("p", "blurry")),
        ("z-image-turbo", 4.0, "blurry", ("p",)),
    ],
)
def test_cfg_prompts_follow_mflux_rules(tmp_path, monkeypatch, model, guidance, negative, expected):
    # Bug caught (Review Focus 1): a negative prompt encoded at guidance <= 1 (a wasted encoder reload), not encoded
    # at > 1 (the CFG call would find nothing cached), or Turbo running CFG although mflux forces guidance 0
    # (z_image.py:65-67, 177-179).
    instance = fake_model(tmp_path, monkeypatch, model=model)
    assert instance.cfg_prompts("p", negative_prompt=negative, guidance=guidance) == expected


def test_encode_prompts_reads_the_cache_with_the_same_rule_and_refuses_a_missing_prompt(
    tmp_path, monkeypatch
):
    # Bug caught: mflux's own _encode_prompts running (the encoder is dropped: a None call), a negative looked up at
    # guidance <= 1, or a missing prompt surfacing as a KeyError deep in the loop.
    from mlx_dfloat.errors import DFloatIntegrationError

    model = fake_model(tmp_path, monkeypatch, model="z-image")
    model.encode("p", " ")
    text, negative = model._encode_prompts(prompt="p", negative_prompt=None, guidance=4.0)
    assert text is model._embeddings["p"][0]
    assert negative is model._embeddings[" "][0]
    assert model._encode_prompts(prompt="p", negative_prompt="blurry", guidance=1.0) == (text, None)
    with pytest.raises(DFloatIntegrationError, match="blurry"):
        model._encode_prompts(prompt="p", negative_prompt="blurry", guidance=4.0)


def test_the_prelude_encodes_then_drops_the_encoder_then_loads_the_set_then_runs_mflux(
    tmp_path, monkeypatch
):
    # Bug caught: the set loaded while the encoder is resident (the 16 GB rung's whole point), or mflux's loop
    # entered before the embeddings exist. ZImage.generate_image patched to record; lifecycle events in order.
    model = fake_model(tmp_path, monkeypatch, model="z-image")
    events = []
    life = model._lifecycle
    for attr in ("_encode", "_unload_encoders", "_load_set"):
        real = getattr(life, attr)

        def wrapped(*args, _real=real, _name=attr, **kwargs):
            events.append(_name + (f":{args[0]!r}" if args else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(life, attr, wrapped)
    seen = _patch_upstream_generate(monkeypatch, model)
    from mflux.models.z_image.variants.z_image import ZImage

    upstream = ZImage.generate_image

    def recording(self, **kwargs):
        events.append("upstream")
        return upstream(self, **kwargs)

    monkeypatch.setattr(ZImage, "generate_image", recording)
    image = model.generate_image(
        seed=7, prompt="p", num_inference_steps=1, height=256, width=256, guidance=4.0
    )
    assert image == "image"
    assert events == [
        "_encode:'p'",
        "_encode:' '",
        "_unload_encoders",
        "_load_set",
        "upstream",
    ]
    assert seen["kwargs"]["image_path"] is None
    assert seen["kwargs"]["pid_decode"] is False
    assert seen["kwargs"]["guidance"] == 4.0
    assert model._cfg_calls == 2
    assert model.text_encoder is None


@pytest.mark.parametrize(
    "kwargs", [{"image_path": "in.png"}, {"image_strength": 0.5}, {"pid_decode": True}]
)
def test_generate_refusals_come_before_encoding_or_loading(tmp_path, monkeypatch, kwargs):
    # Bug caught: encoding the prompt (an encoder load) before refusing img2img or the PiD decoder.
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        model.generate_image(seed=1, prompt="p", **kwargs)


def test_plan_call_uses_the_rounded_size_and_refuses_above_1024_squared(
    tmp_path, monkeypatch, caplog
):
    # Bug caught (Review Focus 3): planning 1000x1000 while mflux runs 992x992; a 1040x1040 request accepted without
    # a measurement. 1040x1040 with fit_check=True -> DFloatResourceError; fit_check=False -> a warning.
    from mlx_dfloat.errors import DFloatResourceError

    model = fake_model(tmp_path, monkeypatch)
    assert (
        model.plan_call(height=1000, width=1000).cache_limit
        == model.plan_call(height=992, width=992).cache_limit
    )
    model.plan_call(height=1024, width=1024)  # the measured size itself
    with pytest.raises(DFloatResourceError, match="fit_check=False"):
        model.plan_call(height=1040, width=1040)
    lax = fake_model(tmp_path / "lax", monkeypatch, fit_check=False)
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.zimage"):
        lax.plan_call(height=1040, width=1040)
    assert "extrapolation" in caplog.text


def test_plan_call_uses_the_two_largest_groups_for_the_derived_minimum(
    tmp_path, monkeypatch, caplog
):
    # Bug caught: an override below the two largest decoded groups accepted silently (every block's decode output
    # would be allocated fresh), or the minimum computed from one kind only (one byte under the two-group sum
    # would then pass without a warning).
    two = sum(sorted(unforced_largest(_written(tmp_path, monkeypatch)).values(), reverse=True)[:2])
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.zimage"):
        fake_model(tmp_path / "ok", monkeypatch, cache_limit=two).plan_call(height=256, width=256)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.zimage"):
        plan = fake_model(tmp_path / "low", monkeypatch, cache_limit=two - 1).plan_call(
            height=256, width=256
        )
    assert plan.cache_limit == two - 1
    assert "below the derived minimum" in caplog.text
    unforced = fake_model(tmp_path / "u", monkeypatch).plan_call(height=256, width=256)
    assert unforced.cache_limit == two + 500_000_000  # 256²: the allowance floor


def _written(root, monkeypatch):
    """The root holding a tiny checkpoint (a model built there writes ``root/df11``)."""
    fake_model(root, monkeypatch)
    return root


def unforced_largest(root):
    """Largest decoded group per kind of the tiny checkpoint written under ``root/df11``, from its headers."""
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.integrate.memory import largest_decoded_bytes
    from mlx_dfloat.mflux.zimage.names import NONBLOCK_GROUPS, zimage_name_map

    ckpt = open_checkpoint(root / "df11")
    return largest_decoded_bytes(ckpt, zimage_name_map(), skip=frozenset(NONBLOCK_GROUPS))


def test_the_vae_decision_follows_the_budget_and_an_over_budget_call_is_refused(
    tmp_path, monkeypatch, caplog
):
    # Bug caught: the set kept through a decode that does not fit (VAE phase 12e9 + 5_042_526_170 + 510_030_936 =
    # 17_552_557_106 against a 16e9 budget), the drop made unconditional (a reload per call on a big Mac), the
    # constructor's budget ignored, or the fit rule not enforced.
    from mlx_dfloat.errors import DFloatResourceError
    from mlx_dfloat.mflux.zimage.memory import ZImageSizes

    big = ZImageSizes(compressed=12_000_000_000, extras=0, nonblock=0, encoders=0, vae=0)
    tight = fake_model(tmp_path, monkeypatch, sizes=big, budget_bytes=16_000_000_000)
    plan = tight.plan_call(height=1024, width=1024)
    assert plan.drop_set_before_vae
    assert plan.estimate.fits
    assert plan.estimate.peak_phase == "denoise"
    assert plan.estimate.phases["vae"] == 5_042_526_170 + 510_030_936  # the set left the phase
    roomy = fake_model(tmp_path / "r", monkeypatch, sizes=big, budget_bytes=40 * GIB)
    assert not roomy.plan_call(height=1024, width=1024).drop_set_before_vae
    refused = fake_model(tmp_path / "x", monkeypatch, sizes=big, budget_bytes=10 * GIB)
    with pytest.raises(DFloatResourceError, match=r"10\.0 GiB"):
        refused.plan_call(height=1024, width=1024)
    lax = fake_model(tmp_path / "y", monkeypatch, sizes=big, budget_bytes=10 * GIB, fit_check=False)
    with caplog.at_level("WARNING", logger="mlx_dfloat.mflux.zimage"):
        lax.plan_call(height=1024, width=1024)
    assert "fit_check=False" in caplog.text


def test_the_vae_guard_drops_the_set_only_when_the_plan_says_so_and_zeroes_the_cache_limit(
    tmp_path, monkeypatch
):
    # Bug caught: the after-loop guard not acting on the plan (the set kept through a decode the 16 GB budget
    # cannot hold, or dropped on a roomy Mac, a ~26 s reload per call), the derived cache limit not in force
    # during the loop, or a transformer-sized cache limit left for the VAE decode.
    from mlx_dfloat.mflux.zimage.memory import ZImageSizes

    big = ZImageSizes(compressed=12_000_000_000, extras=0, nonblock=0, encoders=0, vae=0)
    two = sum(
        sorted(unforced_largest(_written(tmp_path / "w", monkeypatch)).values(), reverse=True)[:2]
    )
    tight = fake_model(tmp_path / "t", monkeypatch, sizes=big, budget_bytes=16_000_000_000)
    seen = _patch_upstream_generate(monkeypatch, tight)
    assert tight.generate_image(seed=1, prompt="p", height=1024, width=1024) == "image"
    assert (
        seen["limit_at_entry"] == two + 1_588_235_294
    )  # 1024², 512 text tokens: 1.5e9 * 4608 / 4352
    assert seen["set_resident"]
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is False
    roomy = fake_model(tmp_path / "r", monkeypatch, sizes=big, budget_bytes=40 * GIB)
    seen = _patch_upstream_generate(monkeypatch, roomy)
    roomy.generate_image(seed=1, prompt="p", height=1024, width=1024)
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is True


def test_the_device_budget_is_read_when_the_model_has_no_override(tmp_path, monkeypatch):
    # Bug caught: plan_call always using the constructor's value (None) or never the device's.
    from mlx_dfloat.errors import DFloatResourceError
    from mlx_dfloat.mflux.zimage import model as model_module
    from mlx_dfloat.mflux.zimage.memory import ZImageSizes

    big = ZImageSizes(compressed=12_000_000_000, extras=0, nonblock=0, encoders=0, vae=0)
    model = fake_model(tmp_path, monkeypatch, sizes=big)
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 10 * GIB)
    with pytest.raises(DFloatResourceError, match=r"10\.0 GiB"):
        model.plan_call(height=1024, width=1024)


@pytest.mark.parametrize("where", ["raise_before_loop", "raise_after_loop"])
def test_a_format_error_mid_step_drops_the_set_and_restores_the_limit(tmp_path, monkeypatch, where):
    # Bug caught: a corrupt block's retry reusing stale resident state, or the process cache limit left at the
    # call's; checked both before the after-loop guard ran (inside the loop) and after it (during the decode).
    from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(
        monkeypatch, model, **{where: DFloatFormatError("block 3: invalid code")}
    )
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    assert (
        model.transformer.cap_embedder[1].weight.size == 0
    )  # the non-block weight left with the set
    with pytest.raises(DFloatIntegrationError, match="attach"):
        model.transformer._state()


def test_a_keyboard_interrupt_keeps_the_set_restores_the_limit_and_the_next_call_runs(
    tmp_path, monkeypatch
):
    # Bug caught (Review Focus 4): StopImageGenerationException leaving status words pending (begin_step refuses
    # the next call) or the cache limit not restored; or the interrupt dropping the set (a reload per Ctrl-C).
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.utils.exceptions import StopImageGenerationException
    from tests._zimage_tiny import tiny_inputs

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    interrupt = {"on": True}

    def fake(self, **kwargs):
        provider = self._provider
        real = provider.weights_for
        calls = []

        def interrupting(name, shapes):
            calls.append(name)
            if interrupt["on"] and len(calls) == 3:
                raise KeyboardInterrupt
            return real(name, shapes)

        monkeypatch.setattr(provider, "weights_for", interrupting)
        inputs = tiny_inputs()
        predict = self._predict(self.transformer)
        try:
            noise = predict(
                latents=inputs["latents"],
                timestep=inputs["timestep"],
                sigmas=inputs["sigmas"],
                text_encodings=inputs["text_encodings"],
                negative_encodings=None,
                guidance=0.0,
            )
            mx.eval(noise)
        except KeyboardInterrupt:
            raise StopImageGenerationException("stopped") from None
        return "image"

    monkeypatch.setattr(ZImage, "generate_image", fake)
    with pytest.raises(StopImageGenerationException):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    assert model._provider.pending == []
    interrupt["on"] = False
    assert model.generate_image(seed=1, prompt="p", height=256, width=256) == "image"
    assert model._lifecycle.counters.set_loads == 1


def test_a_format_errors_traceback_does_not_pin_the_set_past_the_drop(tmp_path, monkeypatch):
    # Bug caught: `except DFloatFormatError: drop_set(); raise` running while the exception's own traceback still
    # holds the raising frame's locals (in production a frame inside the seam referencing the whole resident set),
    # so drop_set's gc.collect cannot free it and the active-memory check turns this clean format error into a
    # DFloatResourceError with the real error demoted to __context__.
    import dataclasses

    from mflux.models.z_image.variants.z_image import ZImage

    from mlx_dfloat.errors import DFloatFormatError

    model = fake_model(tmp_path, monkeypatch)

    def fake(self, **kwargs):
        name, group = next(iter(self._provider._resident.items()))
        big = mx.zeros(
            2 * 1024 * 1024, dtype=mx.float32
        )  # 8 MiB stand-in for a block's decoded weights
        mx.eval(big)
        self._provider._resident[name] = dataclasses.replace(group, sign_mantissa=big)
        held = next(iter(self._provider._resident.values())).sign_mantissa
        mx.eval(held)
        raise DFloatFormatError("block 3: invalid code")

    monkeypatch.setattr(ZImage, "generate_image", fake)
    bound = int(mx.get_active_memory()) + 1 * 1024**2
    model._lifecycle.retained_bound = lambda: bound
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident


# --- the report and one real tiny generation ----------------------------------------------------------------


def test_the_report_carries_the_sizes_and_the_family_keys_before_any_call(tmp_path, monkeypatch):
    # Bug caught: the sizes missing from the report (the README's weights column reads compressed + extras), the
    # non-block bytes dropped from them, or the Z-Image keys absent (the CLI prints them).
    model = fake_model(tmp_path, monkeypatch)
    report = model.report()
    # The fixture's own ZImageSizes(compressed=1_000, extras=100, nonblock=10, encoders=1_000, vae=10).
    assert report["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "nonblock": 10,
        "encoders": 1_000,
        "vae": 10,
    }
    assert report["family"] == "zimage"
    assert report["model"] == "z-image-turbo"
    assert report["predict"] == "uncompiled"
    assert report["nonblock_groups"] == ["cap_embedder"]
    assert report["cfg_calls_per_step"] == 1
    assert report["fit"] is None
    assert report["df11"]["repo_id"] == "mingyi456/tiny"
    assert report["base"]["revision"] == "b" * 40
    assert report["eval_policy"] == "per-block"


@pytest.mark.parametrize(
    ("model_name", "guidance", "calls"), [("z-image", 4.0, 2), ("z-image-turbo", 4.0, 1)]
)
def test_one_real_tiny_generation_decodes_once_per_block_per_call(
    tmp_path, monkeypatch, model_name, guidance, calls
):
    # Bug caught: anything between our prelude and mflux's loop (the uncompiled predict, cached embeddings, the VAE
    # guard) not composing. Tiny transformer (2 layers: 4 blocks), 2 steps: launches == 2 steps * calls * 4 blocks
    # == 16 for z-image at guidance 4.0 and 8 for turbo (mflux forces its guidance to 0); ZImage._decode_latents
    # is patched to a zero image.
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.utils.apple_silicon import AppleSiliconUtil

    # A Max chip: stock mflux compiles predict here, so a lost bypass fails this run end to end.
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch, model=model_name)
    monkeypatch.setattr(ZImage, "_decode_latents", lambda self, **kw: mx.zeros((1, 3, 32, 32)))
    in_loop = []

    class LimitProbe:
        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            in_loop.append(_cache_limit_in_force())

    model.callbacks.register(LimitProbe())
    before = _cache_limit_in_force()
    image = model.generate_image(
        seed=3, prompt="p", num_inference_steps=2, height=32, width=32, guidance=guidance
    )
    assert image is not None
    report = model.report()
    assert report["decode_launches"] == 2 * calls * 4
    assert report["cfg_calls_per_step"] == calls
    assert report["predict"] == "uncompiled"
    assert report["fit"]["label"] == "predicted"
    assert len(in_loop) == 2
    assert (
        in_loop == [report["cache_limit_in_force"]] * 2
    )  # what was in force in the loop, not just planned
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert report["lifecycle"]["set_loads"] == 1
    assert _cache_limit_in_force() == before


def test_open_phase_names_the_phase_running_now_and_none_outside_a_call(tmp_path, monkeypatch):
    # Bug caught: the watchdog's abort context naming no phase, or a stale one: the phase must follow the call
    # through encode, set load, the loop and the VAE decode, and be None again once the call returns.
    from mflux.models.z_image.variants.z_image import ZImage

    model = fake_model(tmp_path, monkeypatch)
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

    monkeypatch.setattr(ZImage, "generate_image", fake)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert seen == ["encode", "set_load", "denoise", "vae"]
    assert model.open_phase is None


def _innermost_locals(tb):
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_locals


@pytest.mark.parametrize("chained", [False, True])
def test_any_exception_from_the_loop_releases_its_frames_locals(tmp_path, monkeypatch, chained):
    # Bug caught: only DFloatFormatError clearing the traceback's frames, so a RuntimeError (or an error raised
    # while handling another, through __context__) keeps the raising frame's locals, in production the seam's
    # decoded weights, alive for as long as the caller holds the exception.
    from mflux.models.z_image.variants.z_image import ZImage

    model = fake_model(tmp_path, monkeypatch)

    def failing_step():
        big = mx.zeros(1024)  # noqa: F841  # stands in for a block's decoded weights
        raise ValueError("inner")

    def fake(self, **kwargs):
        if not chained:
            failing_step()
        try:
            failing_step()
        except ValueError:
            raise RuntimeError("outer") from None  # __context__ is kept even with `from None`

    monkeypatch.setattr(ZImage, "generate_image", fake)
    with pytest.raises((ValueError, RuntimeError)) as info:
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    exc = info.value
    assert "big" not in _innermost_locals(exc.__traceback__)
    if chained:
        assert isinstance(exc.__context__, ValueError)
        assert "big" not in _innermost_locals(exc.__context__.__traceback__)
    assert model._lifecycle.set_resident  # only a format error drops the set


def test_a_failed_set_load_leaves_nothing_installed(tmp_path, monkeypatch):
    # Bug caught: a raise late in _load_set (here at attach) leaving the decoded caption-embedder weight installed
    # and the provider holding the compressed set, while the lifecycle believes no set is resident; the retry's
    # active-memory check would then refuse to load a second copy.
    model = fake_model(tmp_path, monkeypatch)

    def broken_attach(*args, **kwargs):
        raise RuntimeError("attach")

    monkeypatch.setattr(model.transformer, "attach", broken_attach)
    with pytest.raises(RuntimeError, match="attach"):
        model._lifecycle.ensure_set()
    assert model._provider is None
    assert model.transformer.cap_embedder[1].weight.size == 0
    assert not model._lifecycle.set_resident


@pytest.mark.parametrize("raised", [None, RuntimeError("in the loop")], ids=["returns", "raises"])
def test_a_python_api_call_runs_under_the_commands_caps_and_restores_mlxs_defaults(
    tmp_path, monkeypatch, raised
):
    # Bug caught: generate_image not run under the per-call caps (a Python-API process sits at MLX's default wired
    # limit 0, while every VAE term was measured under the command's caps), or the caps left installed after a call
    # that returns or raises.
    from tests._mlx_limits import command_caps, current_limits, mlx_without_wired_cap

    model = fake_model(tmp_path, monkeypatch)
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
