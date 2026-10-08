"""Tests for `DFloatQwenImage21`: construction, refusals, the lifecycle and the set, the prelude and the report.

Every test is `@pytest.mark.mflux`; the module imports mflux only inside helpers, so it collects without it.
"""

import gc
import inspect
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tests._qwen21_tiny import (
    PREFIX_IDS,
    TINY_SEED,
    StubTextEncoder,
    fake_model,
    write_tiny_checkpoint,
)

from mlx_dfloat.errors import DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.mflux.qwen21 import init as qinit

pytestmark = pytest.mark.mflux

MODULATION = "modulation.layers.1.weight"  # mflux's path for the checkpoint's modulation.1 (qwen21_weight_mapping.py:18)


def _modulation_source(tmp_path):
    """The modulation matrix ``fake_model`` compressed: the same seed gives the same arrays."""
    matrices, _c, _layout = write_tiny_checkpoint(
        tmp_path / "again", np.random.default_rng(TINY_SEED)
    )
    return matrices["modulation.1"]["weight"]


def _modulation(model):
    return model.transformer.modulation.layers[1].weight


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new self.<name> in QwenImage21.generate_image our assembly never
    # sets (mflux's own initializer is not run).
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(QwenImage21.generate_image)))
    assert names >= {"vae", "transformer", "text_encoder", "prompt_cache", "tokenizers", "bits"}
    model = fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, QwenImage21)
    assert (model.bits, model.prompt_cache, model.tiling_config) == (None, {}, None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantize": 8}, "quantize"),
        ({"lora_paths": ["a"]}, "lora_paths"),
        ({"lora_scales": [1.0]}, "lora_scales"),
        ({"bake_lora": False}, "bake_lora"),
        ({"eval_policy": "none"}, "none"),
    ],
)
def test_construction_refusals_come_before_any_resolution(monkeypatch, kwargs, match):
    # Bug caught: a quantize / LoRA argument or the "none" policy silently honoured (mflux's applier would quantize the
    # decoded weights), or resolved (downloading 9.7 GB) before the refusal.
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    monkeypatch.setattr(
        qinit, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=match):
        DFloatQwenImage21(**{"model": "qwen-image-2.1", **kwargs})


def test_an_unknown_name_is_refused_listing_the_names_this_path_runs(monkeypatch):
    # Bug caught: Qwen-Image 1 ("qwen-image", another mflux class and another checkpoint layout) accepted by this
    # class, or the refusal not telling the user which names it runs.
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    monkeypatch.setattr(qinit, "resolve", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(DFloatUnsupportedError, match=r"'qwen-image'.*qwen-image-2\.1"):
        DFloatQwenImage21("qwen-image")


def test_the_set_load_installs_modulation_and_attaches_with_verify_in_call(tmp_path, monkeypatch):
    # Bug caught: modulation.1 decoded but not installed (a zero-size matmul before the first block), or the set
    # attached without verify_in_call (a CFG call's second transformer call would be refused).
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_set()
    weight = _modulation(model)
    assert weight.shape == (128, 32)
    assert np.array_equal(np.array(weight.view(mx.uint16)), _modulation_source(tmp_path))
    assert model.transformer.seam_cell.state.verify_in_call is True


def test_a_new_prompt_drops_the_set_and_reinstalls_modulation(tmp_path, monkeypatch):
    # Bug caught (Review Focus 5): modulation.layers.1.weight left on its placeholder after the reload a new prompt
    # forces, or the encoder reload reading another base than the model was built from.
    model = fake_model(tmp_path, monkeypatch)
    loads = []
    monkeypatch.setattr(
        qinit,
        "load_text_encoder",
        lambda root, **kwargs: (loads.append(root), StubTextEncoder())[1],
    )
    model.encode("a")  # the encoder came with the assembly: no reload
    model._lifecycle.ensure_set()
    model.encode("b")  # the set must go before the encoder comes back
    counters = model._lifecycle.counters
    assert (counters.forced_set_drops, counters.encoder_loads) == (1, 1)
    assert loads == [model._base.root]
    assert not model._lifecycle.set_resident
    assert _modulation(model).size == 0
    model._lifecycle.ensure_set()
    assert counters.set_loads == 2
    again = np.array(_modulation(model).view(mx.uint16))
    assert again.shape == (128, 32)
    assert np.array_equal(again, _modulation_source(tmp_path))


def test_encoded_pairs_live_in_mflux_prompt_cache_under_the_raw_prompt(tmp_path, monkeypatch):
    # Bug caught: the empty prompt stored under mflux's normalised " " (mflux's own lookup of "" would miss and call
    # the dropped encoder: qwen21_prompt_encoder.py:22-25), or a cache of our own that mflux never reads.
    model = fake_model(tmp_path, monkeypatch)
    model.encode("a", "")
    assert set(model.prompt_cache) == {"a", ""}
    for embeds, mask in model.prompt_cache.values():
        # 8 stub ids minus the 3-id system prefix mflux drops; TINY's context_in_dim 32.
        assert embeds.shape == (1, 8 - PREFIX_IDS, 32)
        assert mask.shape == (1, 8 - PREFIX_IDS)


def _bits(array):
    return np.array(array.view(mx.uint16)) if array.dtype == mx.bfloat16 else np.array(array)


def test_a_cfg_call_caches_each_prompt_with_its_own_embeddings_as_mflux_encodes_it(
    tmp_path, monkeypatch
):
    # Bug caught: the negative prompt's embeddings stored under the prompt (or the reverse), one prompt encoded for
    # both, or a fixed text encoded instead of the caller's. The stub encoder's output depends on the stub
    # tokenizer's ids, which depend on the prompt, so each pair must equal mflux's own encode of that prompt.
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
        Qwen21PromptEncoder,
    )
    from tests._qwen21_tiny import StubTokenizer

    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, guidance=4.0, negative_prompt="n")
    assert not np.array_equal(_bits(model.prompt_cache["p"][0]), _bits(model.prompt_cache["n"][0]))
    for prompt in ("p", "n"):
        embeds, mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache={},
            tokenizer=StubTokenizer(),
            text_encoder=StubTextEncoder(),
        )
        got_embeds, got_mask = model.prompt_cache[prompt]
        assert np.array_equal(_bits(got_embeds), _bits(embeds)), prompt
        assert np.array_equal(_bits(got_mask), _bits(mask)), prompt


def test_a_dropped_encoder_refuses_with_a_named_error(tmp_path, monkeypatch):
    # Bug caught: None left in the encoder's place (mflux would call it on a cache miss and fail with a TypeError far
    # from the cause).
    model = fake_model(tmp_path, monkeypatch)
    model.encode("a")
    with pytest.raises(DFloatIntegrationError, match="not encoded"):
        model.text_encoder(input_ids=mx.ones((1, 4), dtype=mx.int32))


def test_a_failed_set_load_leaves_nothing_installed(tmp_path, monkeypatch):
    # Bug caught: a raise late in the set load (here at attach) leaving the decoded modulation installed and the
    # provider holding the compressed set while the lifecycle believes no set is resident; the retry's active-memory
    # check would then refuse to load a second copy.
    model = fake_model(tmp_path, monkeypatch)

    def broken_attach(*args, **kwargs):
        raise RuntimeError("attach")

    monkeypatch.setattr(model.transformer, "attach", broken_attach)
    with pytest.raises(RuntimeError, match="attach"):
        model._lifecycle.ensure_set()
    assert model._provider is None
    assert _modulation(model).size == 0
    assert not model._lifecycle.set_resident


def _corrupting(group_name):
    """A reference decode whose status words report an invalid code for ``group_name`` while ``on`` is set."""
    from dataclasses import replace
    from functools import partial

    from mlx_dfloat.decode import STATUS_INVALID_CODE, decode_group

    reference = partial(decode_group, backend="reference")
    state = {"on": True}

    def decode(group):
        result = reference(group)
        if state["on"] and group.name == group_name:
            return replace(result, status=result.status | STATUS_INVALID_CODE)
        return result

    return decode, state


def _frames(tb):
    while tb is not None:
        yield tb.tb_frame
        tb = tb.tb_next


def test_a_corrupt_modulation_group_at_set_load_releases_the_set(tmp_path, monkeypatch):
    # Bug caught: a status failure while the set loads (modulation.1 decoded once at the load) leaving the compressed
    # set reachable from the exception's frames (_install_set's `resident`) while the caller holds the exception, so
    # the retry refuses with "a previous set is still resident".
    from mlx_dfloat.errors import DFloatFormatError

    decode, state = _corrupting("modulation.1")
    model = fake_model(tmp_path, monkeypatch, decode=decode)
    model.encode("p")
    gc.collect()
    mx.clear_cache()
    before = int(mx.get_active_memory())
    with pytest.raises(DFloatFormatError, match=r"modulation\.1") as info:
        model._lifecycle.ensure_set()
    assert not model._lifecycle.set_resident
    install = [f for f in _frames(info.value.__traceback__) if f.f_code.co_name == "_install_set"]
    assert len(install) == 1
    assert "resident" not in install[0].f_locals
    monkeypatch.setattr(model._lifecycle, "retained_bound", lambda: before)
    state["on"] = False
    model._lifecycle.ensure_set()  # the retry, with the exception still held
    assert model._lifecycle.counters.set_loads == 1
    del info


def test_the_retained_bound_counts_the_cached_pairs_and_the_vae(tmp_path, monkeypatch):
    # Bug caught: the bound reading a cache of our own (empty: every drop after an encode would refuse) instead of
    # mflux's prompt_cache, where the pairs live.
    model = fake_model(tmp_path, monkeypatch)
    empty = model._retained_bound()
    model.encode("a")
    # One pair: embeds (1, 5, 32) bf16 = 320 B, mask (1, 5) int32 = 20 B.
    assert model._retained_bound() - empty == 320 + 20


def test_save_model_is_refused_and_freeze_skips_a_dropped_encoder(tmp_path, monkeypatch):
    # Bug caught: freeze() calling .freeze() on the encoder stand-in (it is not a module), or save_model writing a
    # checkpoint with zero-size placeholders as the transformer.
    model = fake_model(tmp_path, monkeypatch)
    model.encode("p")  # drops the encoder
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))


# --- the generate prelude, the call plan ----------------------------------------------------------------------

GIB = 1024**3


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def _patch_upstream_generate(monkeypatch, *, raise_after_loop=None, raise_before_loop=None):
    """Replace QwenImage21.generate_image with a probe: records the cache limit in force and the kwargs, fires our
    after-loop subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
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

    monkeypatch.setattr(QwenImage21, "generate_image", fake)
    return seen


def _record_encodes(monkeypatch, model):
    encodes = []
    real = model._lifecycle._encode
    monkeypatch.setattr(model._lifecycle, "_encode", lambda p: (encodes.append(p), real(p))[1])
    return encodes


@pytest.mark.parametrize(
    ("guidance", "negative", "expected"),
    [
        (1.0, "n", ("p",)),
        (4.0, None, ("p",)),
        (4.0, "", ("p",)),
        (4.0, " ", ("p", " ")),
        (1.0001, "n", ("p", "n")),
        (None, "n", ("p",)),
    ],
)
def test_cfg_prompts_follow_mflux(tmp_path, monkeypatch, guidance, negative, expected):
    # Bug caught: `>=` for `>` (guidance 1.0 encoding the negative: a wasted encoder reload), an empty negative running
    # CFG (mflux disables it: `config.guidance > 1.0 and bool(negative_prompt)`, qwen_image_21.py:89), or a blank-but-
    # non-empty negative dropped (bool(" ") is True: mflux encodes it).
    model = fake_model(tmp_path, monkeypatch)
    assert model.cfg_prompts("p", negative_prompt=negative, guidance=guidance) == expected


def test_the_generate_defaults_are_mfluxs(tmp_path, monkeypatch):
    # Bug caught: our signature drifting from QwenImage21.generate_image's (qwen_image_21.py:42-54: 40 steps, 1024^2,
    # guidance 1.0, scheduler "linear", no negative prompt), so the Python API ran another recipe than mflux's.
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    defaults = {
        n: p.default
        for n, p in inspect.signature(DFloatQwenImage21.generate_image).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "num_inference_steps": 40,
        "height": 1024,
        "width": 1024,
        "guidance": 1.0,
        "image_path": None,
        "image_strength": None,
        "scheduler": "linear",
        "negative_prompt": None,
    }


def test_a_missing_guidance_and_scheduler_become_mflux_defaults(tmp_path, monkeypatch):
    # Bug caught: the CLI's None (no --guidance, no --scheduler) reaching mflux's Config, which builds the scheduler by
    # name, instead of mflux's defaults (guidance 1.0, "linear").
    model = fake_model(tmp_path, monkeypatch)
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, guidance=None, scheduler=None)
    assert seen["kwargs"]["guidance"] == 1.0
    assert seen["kwargs"]["scheduler"] == "linear"
    assert model._cfg_calls == 1


def test_an_empty_prompt_and_a_negative_equal_to_the_prompt_run_from_the_cache(
    tmp_path, monkeypatch
):
    # Bug caught (Review Focus 3): the empty prompt cached under " " (mflux's loop looks it up as "" and calls the
    # dropped encoder, which raises), an empty negative encoded or run as CFG, or a negative equal to the prompt
    # encoded twice. Real mflux loop, one step.
    model = fake_model(tmp_path, monkeypatch)
    encodes = _record_encodes(monkeypatch, model)
    model.generate_image(
        seed=1,
        prompt="",
        num_inference_steps=1,
        height=64,
        width=64,
        guidance=4.0,
        negative_prompt="",
    )
    assert (encodes, model._cfg_calls) == ([""], 1)
    model.generate_image(
        seed=1,
        prompt="x",
        num_inference_steps=1,
        height=64,
        width=64,
        guidance=4.0,
        negative_prompt="x",
    )
    assert (encodes, model._cfg_calls) == (["", "x"], 2)
    assert set(model.prompt_cache) == {"", "x"}
    # 1 step x 1 call x 2 blocks, then 1 step x 2 calls x 2 blocks: the second call ran the negative branch.
    assert model.report()["decode_launches"] == 2 + 4


def test_text_tokens_follow_the_longest_prompt_of_the_call(tmp_path, monkeypatch):
    # Bug caught: the plan sized by the positive prompt only (a longer negative prompt's activations unplanned), or the
    # system prefix counted. Stub lengths 7 and 12 ids, a 3-id prefix: 9 text tokens.
    from tests._qwen21_tiny import StubTokenizer

    model = fake_model(tmp_path, monkeypatch, tokenizer=StubTokenizer({"p": 7, "n": 12}))
    planned = []
    real = model.plan_call
    monkeypatch.setattr(
        model, "plan_call", lambda **kw: (planned.append(kw["text_tokens"]), real(**kw))[1]
    )
    _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, guidance=4.0, negative_prompt="n")
    assert planned == [12 - PREFIX_IDS]
    assert model.report()["text_tokens"] == 12 - PREFIX_IDS


@pytest.mark.parametrize("kwargs", [{"image_path": "in.png"}, {"image_strength": 0.5}])
def test_img2img_is_refused_before_anything_loads(tmp_path, monkeypatch, kwargs):
    # Bug caught: encoding the prompt (an encoder load) or loading the set before refusing img2img.
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        model.generate_image(seed=1, prompt="p", **kwargs)
    assert model._lifecycle.counters.set_loads == 0


def test_the_prelude_encodes_then_drops_the_encoder_then_loads_the_set_then_runs_mflux(
    tmp_path, monkeypatch
):
    # Bug caught: the set loaded while the encoder is resident, mflux's loop entered before the pairs exist, img2img
    # reaching QwenImage21.generate_image, or the negative prompt not passed through (mflux reads it for the CFG rule
    # and the image metadata).
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    model = fake_model(tmp_path, monkeypatch)
    events = []
    life = model._lifecycle
    for attr in ("_encode", "_unload_encoders", "_load_set"):
        real = getattr(life, attr)

        def wrapped(*args, _real=real, _name=attr, **kwargs):
            events.append(_name + (f":{args[0]!r}" if args else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(life, attr, wrapped)
    seen = _patch_upstream_generate(monkeypatch)
    upstream = QwenImage21.generate_image

    def recording(self, **kwargs):
        events.append("upstream")
        return upstream(self, **kwargs)

    monkeypatch.setattr(QwenImage21, "generate_image", recording)
    image = model.generate_image(
        seed=7,
        prompt="p",
        num_inference_steps=1,
        height=256,
        width=256,
        guidance=4.0,
        negative_prompt="n",
    )
    assert image == "image"
    assert events == ["_encode:'p'", "_encode:'n'", "_unload_encoders", "_load_set", "upstream"]
    kwargs = seen["kwargs"]
    assert (kwargs["image_path"], kwargs["image_strength"]) == (None, None)
    assert (kwargs["guidance"], kwargs["negative_prompt"], kwargs["num_inference_steps"]) == (
        4.0,
        "n",
        1,
    )
    assert model._cfg_calls == 2


def test_plan_call_reads_the_familys_constants_sizes_and_groups(tmp_path, monkeypatch):
    # Bug caught: plan_call not wired to the shared planner with this model's own largest group and the call's text
    # tokens (the cache limit would miss the allowance or the decoded group), or the plan not kept for the VAE guard.
    from mlx_dfloat.errors import DFloatResourceError

    model = fake_model(tmp_path, monkeypatch)
    plan = model.plan_call(height=1030, width=1030, text_tokens=5)  # rounded to 1024^2 first
    # One block kind: the tiny block group, 13_312 elements (26_624 B), next to Qwen's allowance at 4096 image + 5
    # text tokens: 2_295_554_018 x 4101 / 4129 = 2_279_987_170 (truncated).
    assert plan.cache_limit == 26_624 + 2_279_987_170
    assert model._plan is plan
    with pytest.raises(DFloatResourceError, match="1024x1024"):
        model.plan_call(height=1040, width=1040, text_tokens=5)


def test_the_vae_guard_follows_the_plan_and_zeroes_the_cache_limit(tmp_path, monkeypatch):
    # Bug caught: the guard not reading this call's plan (the set kept through a decode the budget cannot hold, or
    # dropped on a roomy Mac: a reload per call), or a transformer-sized cache limit left for the VAE decode.
    # Tight: VAE phase 12e9 + 11_119_436_651 + 459_445_098 = 23_578_881_749 > 16e9; denoise ~15.2e9 fits.
    from mlx_dfloat.mflux._phases import FamilySizes

    big = FamilySizes(compressed=12_000_000_000, extras=0, nonblock=0, encoders=0, vae=0)
    tight = fake_model(tmp_path / "t", monkeypatch, sizes=big, budget_bytes=16_000_000_000)
    seen = _patch_upstream_generate(monkeypatch)
    assert tight.generate_image(seed=1, prompt="p", height=1024, width=1024) == "image"
    assert seen["set_resident"]
    assert seen["limit_at_entry"] == tight._plan.cache_limit
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is False
    roomy = fake_model(tmp_path / "r", monkeypatch, sizes=big, budget_bytes=40 * GIB)
    seen = _patch_upstream_generate(monkeypatch)
    roomy.generate_image(seed=1, prompt="p", height=1024, width=1024)
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is True


@pytest.mark.parametrize("where", ["raise_before_loop", "raise_after_loop"])
def test_a_format_error_mid_step_drops_the_set_and_restores_the_limit(tmp_path, monkeypatch, where):
    # Bug caught: a corrupt block's retry reusing stale resident state, or the process cache limit left at the call's;
    # checked before the after-loop guard ran (inside the loop) and after it (during the decode).
    from mlx_dfloat.errors import DFloatFormatError

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(monkeypatch, **{where: DFloatFormatError("block 1: invalid code")})
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    assert _modulation(model).size == 0
    assert model.open_phase is None


def test_keyboard_interrupt_in_the_loop_restores_the_cache_limit_and_leaves_no_pending_words(
    tmp_path, monkeypatch
):
    # Bug caught: mflux's StopImageGenerationException (raised for a Ctrl-C in its loop, qwen_image_21.py:125-129)
    # leaving status words pending (begin_step would refuse the next call), the cache limit not restored, or the
    # interrupt dropping the set (a reload per Ctrl-C). Real mflux loop; the interrupt fires at the second block of the
    # step's second transformer call (CFG: 2 blocks per call).
    from mflux.utils.exceptions import StopImageGenerationException

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    model.encode("p", "n")
    model._lifecycle.ensure_set()
    provider = model._provider
    real = provider.weights_for
    calls = []
    interrupt = {"on": True}

    def interrupting(name, shapes):
        calls.append(name)
        if interrupt["on"] and len(calls) == 4:
            raise KeyboardInterrupt
        return real(name, shapes)

    monkeypatch.setattr(provider, "weights_for", interrupting)
    run = {
        "seed": 1,
        "prompt": "p",
        "num_inference_steps": 1,
        "height": 64,
        "width": 64,
        "guidance": 4.0,
        "negative_prompt": "n",
    }
    with pytest.raises(StopImageGenerationException):
        model.generate_image(**run)
    assert len(calls) == 4
    assert _cache_limit_in_force() == before
    assert provider.pending == []
    assert model._lifecycle.set_resident
    interrupt["on"] = False
    assert model.generate_image(**run) is not None
    assert model._lifecycle.counters.set_loads == 1


def _innermost_locals(tb):
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_locals


@pytest.mark.parametrize("chained", [False, True])
def test_any_exception_from_the_loop_releases_its_frames_locals(tmp_path, monkeypatch, chained):
    # Bug caught: only DFloatFormatError clearing the traceback's frames, so a RuntimeError (or an error raised while
    # handling another, through __context__) keeps the raising frame's locals, in production the seam's decoded
    # weights, alive for as long as the caller holds the exception.
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

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

    monkeypatch.setattr(QwenImage21, "generate_image", fake)
    with pytest.raises((ValueError, RuntimeError)) as info:
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    exc = info.value
    assert "big" not in _innermost_locals(exc.__traceback__)
    if chained:
        assert isinstance(exc.__context__, ValueError)
        assert "big" not in _innermost_locals(exc.__context__.__traceback__)
    assert model._lifecycle.set_resident  # only a format error drops the set


def test_open_phase_names_the_phase_running_now_and_none_outside_a_call(tmp_path, monkeypatch):
    # Bug caught: the watchdog's abort context naming no phase, or a stale one: the phase must follow the call through
    # encode, set load, the loop and the VAE decode, and be None again once the call returns.
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

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

    monkeypatch.setattr(QwenImage21, "generate_image", fake)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert seen == ["encode", "set_load", "denoise", "vae"]
    assert model.open_phase is None


def test_a_failed_set_load_in_a_call_closes_the_phase(tmp_path, monkeypatch):
    # Bug caught: the tracker left at "set_load" after the set load raised (the watchdog's abort context would name a
    # phase that is not running).
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)

    def broken_attach(*args, **kwargs):
        raise RuntimeError("attach")

    monkeypatch.setattr(model.transformer, "attach", broken_attach)
    with pytest.raises(RuntimeError, match="attach"):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert model.open_phase is None


def _fake_encode_peak(monkeypatch, model, above_start):
    """Make the tracker record an encode phase whose MLX peak is ``above_start`` bytes over the phase's start."""
    real_end = model._tracker.end

    def end(name):
        real_end(name)
        if name == "encode":
            record = model._tracker.peaks["encode"]
            record["mlx_peak"] = record["active_at_start"] + above_start

    monkeypatch.setattr(model._tracker, "end", end)


# fake_model's encoders (1_000) + Qwen's encode term (66_567_800, in place of the allowance) + the 256 MiB slack
# (268_435_456).
ENCODE_WARN_ABOVE = 335_004_256


def test_an_encode_peak_over_the_estimate_warns_that_mflux_may_load_more_of_the_encoder(
    tmp_path, monkeypatch
):
    # Bug caught: an mflux change that loads the vision tower or the lm_head with the language model going unnoticed:
    # the encode phase grows by ~2.4 GB while the fit estimate, sized by the language model, keeps passing calls.
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, ENCODE_WARN_ABOVE + 1)
    # The stub tokenizer gives 8 ids, 3 of them the system prefix mflux drops: a 5-token prompt.
    with pytest.warns(
        UserWarning, match=r"for a 5-token prompt.*more of the encoder than its language model"
    ):
        model.generate_image(seed=1, prompt="p", height=256, width=256)


def test_an_encode_peak_at_the_estimate_does_not_warn(tmp_path, monkeypatch):
    # Bug caught: the slack or the encode term left out of the bound, or `>=` for `>` at the bound. Pytest turns a
    # warning into an error here.
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, ENCODE_WARN_ABOVE)
    assert model.generate_image(seed=1, prompt="p", height=256, width=256) == "image"


def test_the_budget_is_the_constructors_else_the_devices_read_at_each_call(tmp_path, monkeypatch):
    # Bug caught: the constructor's budget ignored (a --tier run planned against the whole Mac), or the device's read
    # once at assembly instead of at each call.
    from mlx_dfloat.mflux.qwen21 import model as model_module

    assert fake_model(tmp_path / "a", monkeypatch, budget_bytes=9 * GIB)._budget() == 9 * GIB
    model = fake_model(tmp_path / "b", monkeypatch)
    assert model._budget() == 23 * GIB  # fake_model's fixed device budget
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 10 * GIB)
    assert model._budget() == 10 * GIB


# --- the report -----------------------------------------------------------------------------------------------


def test_the_report_carries_the_family_keys_and_the_sizes_before_any_call(tmp_path, monkeypatch):
    # Bug caught: the family, sizes or the checkpoint's config source missing from the report (the CLI and the table
    # read them; a config-less checkpoint's provenance is the layout that matched it), or a Klein-only key.
    model = fake_model(tmp_path, monkeypatch)
    report = model.report()
    assert report["family"] == "qwen21"
    assert report["model"] == "qwen-image-2.1"
    assert "size" not in report
    assert report["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "nonblock": 10,
        "encoders": 1_000,
        "vae": 10,
    }  # fake_model's own FamilySizes
    assert report["predict"] == "uncompiled"
    assert report["nonblock_groups"] == ["modulation.1"]
    assert (report["cfg_calls_per_step"], report["text_tokens"], report["fit"]) == (1, None, None)
    assert report["df11"]["repo_id"] == "mingyi456/tiny"
    assert "layout tiny-qwen" in report["df11"]["config_source"]
    assert report["base"]["revision"] == "b" * 40
    assert report["eval_policy"] == "per-block"


@pytest.mark.parametrize(
    ("policy", "status"), [("per-block", "measured"), ("depth2", "unmeasured")]
)
def test_the_report_labels_a_depth2_plan_unmeasured(tmp_path, monkeypatch, policy, status):
    # Bug caught: a depth2 run reported like the measured per-block path (no Qwen-Image 2.1 run used depth2: its
    # look-ahead buffer and step time are unmeasured), or depth2 refused instead of labelled.
    report = fake_model(tmp_path, monkeypatch, eval_policy=policy).report()
    assert (report["eval_policy"], report["eval_policy_status"]) == (policy, status)


def test_the_report_after_a_cfg_call_names_two_calls_per_step_and_the_estimate(
    tmp_path, monkeypatch
):
    # Bug caught: cfg_calls_per_step not following the call (the bench table would mislabel a CFG run), or the
    # estimate missing or labelled as a measurement.
    model = fake_model(tmp_path, monkeypatch)
    model.generate_image(
        seed=1,
        prompt="a",
        num_inference_steps=1,
        height=64,
        width=64,
        guidance=4.0,
        negative_prompt="n",
    )
    report = model.report()
    assert report["cfg_calls_per_step"] == 2
    assert report["predict"] == "uncompiled"
    assert report["fit"]["label"] == "predicted"
    assert report["text_tokens"] == 8 - PREFIX_IDS


def test_a_compiled_forward_is_refused_in_the_loop_and_the_report_says_compiled(
    tmp_path, monkeypatch
):
    # Bug caught: the eager guard not reached through mflux's loop (the seam would evaluate inside mx.compile), or
    # report() still printing "uncompiled" after mflux compiled the forward pass.
    model = fake_model(tmp_path, monkeypatch)
    assert model.report()["predict"] == "uncompiled"
    model.transformer._step_fn = mx.compile(model.transformer._forward)
    assert model.report()["predict"] == "compiled"
    with pytest.raises(DFloatIntegrationError, match="compiled"):
        model.generate_image(seed=1, prompt="a", num_inference_steps=1, height=64, width=64)
    assert model.open_phase is None


def test_the_default_checkpoint_resolves_at_its_pin_and_a_user_checkpoint_unpinned(monkeypatch):
    # Bug caught: the pin not reaching the resolver (the default, a single file on one person's Hub account, following
    # whatever lands on main instead of the revision whose header and bytes were verified), or a pin applied to a
    # user's own --df11.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    class StopError(Exception):
        pass

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        calls.append((spec, revision))
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(qinit, "resolve", fake_resolve)
    for kwargs in ({}, {"df11_path": "me/fork"}):
        with pytest.raises(StopError):
            DFloatQwenImage21("qwen-image-2.1", **kwargs)
    assert calls == [
        ("mingyi456/Qwen-Image-2.1-DF11-ComfyUI", "1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc"),
        ("me/fork", None),
    ]


def test_the_default_base_resolves_at_its_pin_and_a_user_base_unpinned(tmp_path, monkeypatch):
    # Bug caught: the base's pin not reaching the resolver (the text encoder, VAE and tokenizer following whatever
    # lands on Qwen/Qwen-Image-2.1's main instead of the snapshot the recorded runs used), or a pin applied to a user's
    # own --base.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux.qwen21 import model as model_module
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    class StopError(Exception):
        pass

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        if patterns == qinit.DF11_PATTERNS:
            return ResolvedRepo(root=tmp_path, repo_id=spec, revision=revision)
        calls.append((spec, revision))
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(model_module, "open_checkpoint", lambda root: None)
    monkeypatch.setattr(qinit, "resolve", fake_resolve)
    for kwargs in ({}, {"base_path": "me/base"}):
        with pytest.raises(StopError):
            DFloatQwenImage21("qwen-image-2.1", **kwargs)
    assert calls == [
        ("Qwen/Qwen-Image-2.1", "d26bb61231c349cf6b7896fa83353113880e1ba3"),
        ("me/base", None),
    ]


# --- the constructor, end to end past the resolver ------------------------------------------------------------


def test_the_constructor_builds_from_local_dirs_with_mfluxs_defaults_and_the_language_model_bytes(
    tmp_path, monkeypatch
):
    # Bug caught (S3 review fix 9 for Klein): __init__ wiring past the resolver untested: the config-less checkpoint
    # not opened through the layout table, the transformer built with arguments mflux's initializer does not use
    # (Qwen21Initializer._init_models builds Qwen21Transformer() bare, qwen21_initializer.py:52), the base components
    # read from another directory, or the sizes taken from the shared rule (the encoder shards' file size, vision
    # tower and lm_head included) instead of the language model's tensor bytes.
    from tests._df11_fixtures import write_bf16_original
    from tests._qwen21_tiny import TINY, stub_components

    from mlx_dfloat import _layouts, _metal_decode
    from mlx_dfloat.mflux.qwen21 import model as model_module

    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    monkeypatch.setattr(_layouts, "KNOWN_LAYOUTS", (layout,))
    base = tmp_path / "base"
    # The language model: embed_tokens (10, 4) 80 B, one layer's up_proj (4, 4) 32 B, final norm (4,) 8 B. Never
    # loaded: a vision tensor (8, 4) 64 B and an untied lm_head (10, 4) 80 B. 80 + 32 + 8 = 120.
    write_bf16_original(
        base / "text_encoder",
        {
            "model.language_model.embed_tokens.weight": np.zeros((10, 4), np.uint16),
            "model.language_model.layers.0.mlp.up_proj.weight": np.zeros((4, 4), np.uint16),
            "model.language_model.norm.weight": np.zeros((4,), np.uint16),
            "model.visual.blocks.0.attn.qkv.weight": np.zeros((8, 4), np.uint16),
            "lm_head.weight": np.zeros((10, 4), np.uint16),
        },
    )
    (base / "vae").mkdir()
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    loaded = []
    monkeypatch.setattr(
        qinit, "load_base", lambda root: (loaded.append(root), stub_components())[1]
    )
    built = []
    real_build = model_module.build_transformer

    def tiny_build(ckpt, **kwargs):
        built.append(kwargs)
        return real_build(ckpt, transformer_kwargs=TINY)

    monkeypatch.setattr(model_module, "build_transformer", tiny_build)
    model = model_module.DFloatQwenImage21(
        "qwen-image-2.1", df11_path=str(tmp_path / "df11"), base_path=str(base)
    )
    assert built == [{}]
    assert loaded == [base]
    report = model.report()
    assert report["sizes"]["encoders"] == 120
    assert report["sizes"]["vae"] == 100
    assert report["sizes"]["nonblock"] == 128 * 32 * 2  # modulation.1 decoded, (128, 32) BF16
    assert report["df11"]["config_source"].startswith(
        "header and spot checks match layout tiny-qwen"
    )
    assert (report["df11"]["root"], report["df11"]["repo_id"]) == (str(tmp_path / "df11"), None)
    assert report["base"]["root"] == str(base)


@pytest.mark.parametrize("raised", [None, RuntimeError("in the loop")], ids=["returns", "raises"])
def test_a_python_api_call_runs_under_the_commands_caps_and_restores_mlxs_defaults(
    tmp_path, monkeypatch, raised
):
    # Bug caught: generate_image not run under the per-call caps (a Python-API process sits at MLX's default wired
    # limit 0, while the VAE term, kept with the set on a 32 GB Mac with 1.74 GiB to spare, was measured under the
    # command's caps), or the caps left installed after a call that returns or raises.
    from tests._mlx_limits import command_caps, current_limits, mlx_without_wired_cap

    model = fake_model(tmp_path, monkeypatch)
    seen = _patch_upstream_generate(monkeypatch, raise_before_loop=raised)
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


class _RecordingVAE(nn.Module):
    """The stub VAE's decode, keeping the latents mflux hands it (the denoised result, unpacked)."""

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)
        self.seen = []

    def decode(self, latents):
        self.seen.append(latents)
        return mx.zeros((1, 3, 16 * latents.shape[-2], 16 * latents.shape[-1]))


class _VaryingTextEncoder(nn.Module):
    """A text stack whose hidden states differ in direction from token to token (the stub's are one vector scaled by
    the id, which the transformer's RMSNorm on the text stream maps to the same vector for every prompt)."""

    def __init__(self):
        super().__init__()
        self.freq = mx.arange(1, 33, dtype=mx.float32) / 7

    def __call__(self, input_ids, attention_mask=None):
        return mx.sin(input_ids[..., None].astype(mx.float32) * self.freq).astype(mx.bfloat16)


def _stock_qwen(tokenizer, vae, matrices, extras):
    """Stock mflux ``QwenImage21`` over a plain tiny ``Qwen21Transformer`` holding the BF16 source of a tiny checkpoint
    (``write_tiny_checkpoint``'s matrices and extras), uncompiled like our side."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
    from mlx.utils import tree_flatten, tree_unflatten
    from tests._qwen21_tiny import TINY

    transformer = Qwen21Transformer(**TINY)
    shapes = {name: p.shape for name, p in tree_flatten(transformer.parameters())}
    weights = [
        (f"{group}.{sub}.weight", mx.array(matrix).view(mx.bfloat16))
        for group, subs in matrices.items()
        if group != "modulation.1"
        for sub, matrix in subs.items()
    ]
    weights.append(
        (
            "modulation.layers.1.weight",
            mx.array(matrices["modulation.1"]["weight"]).view(mx.bfloat16),
        )
    )
    weights += [(name, mx.array(bits).view(mx.bfloat16)) for name, bits in extras.items()]
    assert all(tuple(shapes[name]) == bits.shape for name, bits in extras.items())
    transformer.update(tree_unflatten(weights))
    transformer._step_fn = (
        transformer._forward
    )  # mflux compiles it otherwise; our side runs it plain
    stock = QwenImage21.__new__(QwenImage21)
    nn.Module.__init__(stock)
    Qwen21Initializer._init_config(stock, ModelConfig.qwen_image_21())
    stock.vae = vae
    stock.text_encoder = _VaryingTextEncoder()
    stock.tokenizers = {"qwen21": tokenizer}
    stock.transformer = transformer
    stock.bits = None
    return stock


@pytest.mark.parametrize(
    ("guidance", "negative"), [(1.0, None), (4.0, "n")], ids=["guidance-1", "cfg"]
)
def test_a_tiny_generation_gives_stock_mfluxs_latents_bit_for_bit(
    tmp_path, monkeypatch, guidance, negative
):
    # Bug caught: anything between the checkpoint and mflux's loop changing the result: a decoded matrix on the wrong
    # parameter, a gate_up half swapped, an extra mis-routed, or the CFG branch fed the other prompt's embeddings or
    # mask (the negative prompt is 12 ids here, the prompt 8, so a swap changes the latents or fails). The reference is
    # stock mflux over the same BF16 weights, uncompiled as ours; equality is bit for bit (an integer view of the
    # latents). The extras are random: constant ones make the forward pass blind to its inputs.
    from tests._qwen21_tiny import TINY, StubTokenizer, stub_components

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.qwen21.transformer import build_transformer

    rng = np.random.default_rng(11)
    matrices, extras, layout = write_tiny_checkpoint(tmp_path / "random", rng, random_extras=rng)
    ckpt = open_checkpoint(tmp_path / "random", layouts=(layout,))
    tokenizer = StubTokenizer({"n": 12})
    ours_vae, stock_vae = _RecordingVAE(), _RecordingVAE()
    components = stub_components(tokenizer)
    components.vae, components.text_encoder = ours_vae, _VaryingTextEncoder()
    model = fake_model(
        tmp_path,
        monkeypatch,
        ckpt=ckpt,
        build=build_transformer(ckpt, transformer_kwargs=TINY),
        components=components,
    )
    stock = _stock_qwen(tokenizer, stock_vae, matrices, extras)
    kwargs = {
        "seed": 3,
        "prompt": "p",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": guidance,
        "negative_prompt": negative,
    }
    model.generate_image(**kwargs)
    stock.generate_image(**kwargs, scheduler="linear")
    (ours,), (theirs,) = ours_vae.seen, stock_vae.seen
    assert model.report()["cfg_calls_per_step"] == (2 if negative else 1)
    assert bool(mx.all(mx.isfinite(theirs)))
    assert ours.dtype == theirs.dtype
    assert ours.shape == theirs.shape
    view = mx.uint32 if ours.dtype == mx.float32 else mx.uint16
    assert np.array_equal(np.array(ours.view(view)), np.array(theirs.view(view)))


def test_encode_runs_under_the_commands_caps_and_restores_mlxs_defaults(tmp_path, monkeypatch):
    # Bug caught (X4): encode() from Python run at MLX's default wired limit 0 (a prompt encode is one of the measured
    # phases, sized under the command's caps), or the caps left installed after it returns.
    from tests._mlx_limits import command_caps, current_limits, mlx_without_wired_cap

    model = fake_model(tmp_path, monkeypatch)
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
