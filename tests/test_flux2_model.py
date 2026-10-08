"""Tests for `DFloatFlux2Klein`: construction, refusals, the lifecycle and the set, the prelude and the report.

Every test is `@pytest.mark.mflux`; the module imports mflux only inside helpers, so it collects without it.
"""

import inspect
import re
import types

import mlx.core as mx
import numpy as np
import pytest
from tests._flux2_tiny import TINY_SEED, StubTextEncoder, fake_model, write_tiny_checkpoint

from mlx_dfloat.errors import DFloatUnsupportedError
from mlx_dfloat.mflux.flux2 import init as finit

pytestmark = pytest.mark.mflux

# The five non-block groups of the FLUX.2 Klein DF11 checkpoints (their pattern_dict, read 2026-10-08).
NONBLOCK = [
    "context_embedder",
    "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.linear",
    "norm_out.linear",
    "single_stream_modulation.linear",
]


def _source(tmp_path):
    """The matrices ``fake_model`` compressed: the same seed gives the same arrays."""
    groups, _constants = write_tiny_checkpoint(tmp_path / "again", np.random.default_rng(TINY_SEED))
    return groups


def _nonblock_weight(model, group):
    from mlx_dfloat.integrate.placeholders import get_attr_path

    return get_attr_path(model.transformer, group).weight


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new self.<name> in Flux2Klein.generate_image our init never sets.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(Flux2Klein.generate_image)))
    assert names >= {"vae", "transformer", "callbacks", "tiling_config", "model_config", "bits"}
    model = fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, Flux2Klein)
    assert (model.bits, model.lora_paths, model.lora_scales) == (None, [], [])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantize": 8}, "quantize"),
        ({"lora_paths": ["a"]}, "lora_paths"),
        ({"lora_scales": [1.0]}, "lora_scales"),
        ({"bake_lora": False}, "bake_lora"),
        ({"eval_policy": "none"}, "none"),
        ({"model": "flux2-klein-9b-kv"}, "KV"),
    ],
)
def test_construction_refusals_come_before_any_resolution(monkeypatch, kwargs, match):
    # Bug caught: a quantize / LoRA argument or the "none" policy silently honoured, the KV-cache model built on the
    # txt2img class, or any of them resolved (downloading 5-12 GB) before the refusal.
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    monkeypatch.setattr(
        finit, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=match):
        DFloatFlux2Klein(**{"model": "flux2-klein-4b", **kwargs})


def test_an_unknown_name_is_refused_listing_the_four_klein_models(monkeypatch):
    # Bug caught: "flux2-klein-edit" (another class upstream) or a FLUX.1 name accepted by the Klein class, or the
    # refusal not telling the user which names this path runs.
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    monkeypatch.setattr(finit, "resolve", lambda *a, **k: pytest.fail("resolved"))
    with pytest.raises(DFloatUnsupportedError) as info:
        DFloatFlux2Klein("flux2-klein-edit")
    for name in ("flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b"):
        assert name in str(info.value)


def test_the_default_checkpoints_resolve_at_their_pins_and_a_user_checkpoint_unpinned(monkeypatch):
    # Bug caught: the pin not reaching the resolver (a default following whatever lands on main), a pin taken from
    # another model's entry, or a pin applied to a user's own --df11.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    class StopError(Exception):
        pass

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        calls.append((spec, revision))
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(finit, "resolve", fake_resolve)
    for kwargs in (
        {"model": "flux2-klein-base-4b"},
        {"model": "flux2-klein-4b"},
        {"model": "flux2-klein-base-9b"},
        {"model": "flux2-klein-9b"},
        {"model": "flux2-klein-9b", "df11_path": "me/fork"},
    ):
        with pytest.raises(StopError):
            DFloatFlux2Klein(**kwargs)
    assert calls == [
        ("mingyi456/FLUX.2-klein-base-4B-DF11", "b887c73c5cbc3f4d50887a04c41509fb25a0a0f0"),
        ("mingyi456/FLUX.2-klein-4B-DF11", "d29a2c249ff0afeb678da101e707bd134596c96f"),
        ("mingyi456/FLUX.2-klein-base-9B-DF11", "50dc8e7cba7a41eeaf9e9c4fcec557d1a6888ded"),
        ("mingyi456/FLUX.2-klein-9B-DF11", "45a202a0bd19ec01ce8db2d5236890586cbac6a6"),
        ("me/fork", None),
    ]


def test_the_constructor_builds_from_local_dirs_with_the_models_overrides_and_the_used_encoder_bytes(
    tmp_path, monkeypatch
):
    # Bug caught (review 2026-10-08): __init__ wiring past the resolver untested: the transformer built without the
    # model config's overrides (a 9B checkpoint against the 4B default depth), the base components loaded with another
    # config, or the sizes taken from the shared rule (the encoder's file size, every one of its 36 layers and the
    # untied head) instead of the layers Klein runs.
    from tests._df11_fixtures import write_bf16_original
    from tests._flux2_tiny import TINY_OVERRIDES, stub_components

    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux.flux2 import model as model_module

    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    base = tmp_path / "base"
    # A Qwen3-shaped encoder in two shards: embeddings (8, 4) 64 B, final norm (4,) 8 B, an untied lm_head (8, 4)
    # 64 B, 36 layers of two (4, 4) matrices, 64 B per layer. Klein runs layers 0-26: 64 + 8 + 27 x 64 = 1_800.
    tensors = {
        "model.embed_tokens.weight": np.zeros((8, 4), np.uint16),
        "model.norm.weight": np.zeros((4,), np.uint16),
        "lm_head.weight": np.zeros((8, 4), np.uint16),
    }
    for i in range(36):
        for proj in ("self_attn.q_proj", "mlp.down_proj"):
            tensors[f"model.layers.{i}.{proj}.weight"] = np.zeros((4, 4), np.uint16)
    write_bf16_original(base / "text_encoder", tensors)
    (base / "vae").mkdir()
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    loaded = []
    monkeypatch.setattr(
        finit,
        "load_base",
        lambda root, model_config: (loaded.append((root, model_config)), stub_components())[1],
    )
    built = []
    real_build = model_module.build_transformer

    def tiny_build(ckpt, *, transformer_overrides):
        built.append(dict(transformer_overrides))
        return real_build(ckpt, transformer_overrides=TINY_OVERRIDES)

    monkeypatch.setattr(model_module, "build_transformer", tiny_build)
    model = model_module.DFloatFlux2Klein(
        "flux2-klein-base-9b", df11_path=str(tmp_path / "df11"), base_path=str(base)
    )
    # mflux 0.20.0 model_config.py:537-542, FLUX.2-klein-base-9B.
    assert built == [
        {
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        }
    ]
    assert loaded == [(base, model.model_config)]
    report = model.report()
    assert report["sizes"]["encoders"] == 1_800
    assert report["sizes"]["vae"] == 100
    assert report["size"] == "9b"
    assert (report["df11"]["root"], report["df11"]["repo_id"]) == (str(tmp_path / "df11"), None)
    assert report["base"]["root"] == str(base)


def test_the_set_load_installs_the_five_nonblock_weights_and_attaches_with_verify_in_call(
    tmp_path, monkeypatch
):
    # Bug caught: a non-block group decoded but not installed (a zero-size matmul at the first step), or the set
    # attached without verify_in_call (base with guidance would refuse its second transformer call).
    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    model._lifecycle.ensure_set()
    source = _source(tmp_path)
    for group in NONBLOCK:
        w = _nonblock_weight(model, group)
        assert w.size > 0, group
        assert np.array_equal(np.array(w.view(mx.uint16)), source[group][0]), group
    assert model.transformer.seam_cell.state.verify_in_call is True


def test_a_new_prompt_drops_the_set_reloads_the_encoder_and_reinstalls_the_nonblock_weights(
    tmp_path, monkeypatch
):
    # Bug caught (Review Focus 5): context_embedder (or any non-block weight) left on its placeholder after the
    # reload a new prompt forces, or the reload not reading the base the model was built from.
    model = fake_model(tmp_path, monkeypatch)
    loads = []
    monkeypatch.setattr(
        finit,
        "load_text_encoder",
        lambda root, model_config: (loads.append((root, model_config)), StubTextEncoder())[1],
    )
    model.encode("a")  # the encoder came with the assembly: no reload
    model._lifecycle.ensure_set()
    model.encode("b")  # the set must go before the encoder comes back
    counters = model._lifecycle.counters
    assert counters.forced_set_drops == 1
    assert counters.encoder_loads == 1
    assert loads == [(model._base.root, model.model_config)]
    assert not model._lifecycle.set_resident
    assert all(_nonblock_weight(model, g).size == 0 for g in NONBLOCK)
    model._lifecycle.ensure_set()
    assert counters.set_loads == 2
    source = _source(tmp_path)
    for group in NONBLOCK:
        again = np.array(_nonblock_weight(model, group).view(mx.uint16))
        assert np.array_equal(again, source[group][0]), group


def test_encode_caches_the_embeddings_and_the_text_ids_pair(tmp_path, monkeypatch):
    # Bug caught: only the embeddings cached (mflux's predict needs text_ids too), or the ids built with the wrong
    # dtype or layout (Flux2PromptEncoder.prepare_text_ids: (t, h, w, token) per token, int32).
    model = fake_model(tmp_path, monkeypatch)
    model.encode("a")
    pair = model._embeddings["a"]
    assert len(pair) == 2
    embeds, ids = pair
    assert embeds.shape == (1, 8, 32)
    assert ids.shape == (1, 8, 4)
    assert ids.dtype == mx.int32
    assert np.array_equal(np.array(ids[0, :, 3]), np.arange(8))
    assert model.text_encoder is None  # dropped after the encode


KLEIN_MODELS = ("flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b")


def test_our_encode_uses_mfluxs_prompt_literals():
    # Bug caught (review 2026-10-08): an mflux bump moving the token count or the stacked hidden states inside
    # Flux2Klein._encode_prompt_pair (flux2_klein.py:153-170 in 0.20.0: max_sequence_length=512,
    # text_encoder_out_layers=(9, 18, 27)) while _encode keeps ours: the embeddings diverge from stock mflux, the
    # used-layer encoder estimate goes wrong, and verify_image stays green (both sides read our saved embeddings).
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_dfloat.mflux.flux2 import memory as kmem

    source = inspect.getsource(Flux2Klein._encode_prompt_pair)
    layers = re.findall(r"text_encoder_out_layers=\(([\d, ]+)\)", source)
    lengths = re.findall(r"max_sequence_length=(\d+)", source)
    assert len(layers) == len(lengths) == 2  # the prompt and the negative
    assert {tuple(int(v) for v in found.split(",")) for found in layers} == {
        kmem.TEXT_ENCODER_OUT_LAYERS
    }
    configs = [ModelConfig.from_name(model_name=name, base_model=None) for name in KLEIN_MODELS]
    assert {int(n) for n in lengths} == {kmem.text_tokens(c) for c in configs}


def test_encode_takes_the_token_count_from_the_model_config(tmp_path, monkeypatch):
    # Bug caught: _encode passing a module constant instead of the model config's max_sequence_length (the count the
    # plan's text tokens already read), so the two could disagree after an mflux change.
    from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model.model_config, "max_sequence_length", 300)
    seen = {}

    def recording(**kwargs):
        seen.update(kwargs)
        return mx.zeros((1, 8, 32)), mx.zeros((1, 8, 4), dtype=mx.int32)

    monkeypatch.setattr(Flux2PromptEncoder, "encode_prompt", staticmethod(recording))
    model._encode("p")
    assert seen["max_sequence_length"] == 300
    assert seen["text_encoder_out_layers"] == (9, 18, 27)


def test_predict_is_the_plain_function_on_a_max_chip(tmp_path, monkeypatch):
    # Bug caught: our class not overriding _predict (mflux's compiled closure would run the seam under compile).
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch)
    compiled = type(
        mx.compile(lambda x: x)
    )  # mlx.gc_func: a FunctionType subclass, so never isinstance
    assert (
        type(Flux2Klein._predict(model.transformer)) is compiled
    )  # the control: upstream compiles here
    assert type(model._predict(model.transformer)) is types.FunctionType


def test_a_failed_set_load_leaves_nothing_installed(tmp_path, monkeypatch):
    # Bug caught: a raise late in _load_set (here at attach) leaving the decoded non-block weights installed and the
    # provider holding the compressed set while the lifecycle believes no set is resident; the retry's active-memory
    # check would then refuse to load a second copy.
    model = fake_model(tmp_path, monkeypatch)

    def broken_attach(*args, **kwargs):
        raise RuntimeError("attach")

    monkeypatch.setattr(model.transformer, "attach", broken_attach)
    with pytest.raises(RuntimeError, match="attach"):
        model._lifecycle.ensure_set()
    assert model._provider is None
    assert all(_nonblock_weight(model, g).size == 0 for g in NONBLOCK)
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


def test_a_corrupt_nonblock_group_at_set_load_releases_the_set_and_closes_the_phase(
    tmp_path, monkeypatch
):
    # Bug caught (review 2026-10-08): a status failure while the set loads (a corrupt non-block group, the last one
    # decoded) leaving the set reachable from the exception's frames (_install_set's `resident`: the whole compressed
    # set) while the caller holds it, so the retry refuses with "a previous set is still resident"; or the phase
    # tracker left at "set_load" (the watchdog's abort context would name a phase that is not running).
    from mlx_dfloat.errors import DFloatFormatError

    decode, state = _corrupting("norm_out.linear")
    model = fake_model(tmp_path, monkeypatch, decode=decode)
    model.encode("p")  # the encoder's cache entry is not what this test measures
    _patch_upstream_generate(monkeypatch)
    import gc

    gc.collect()
    mx.clear_cache()
    before = int(mx.get_active_memory())
    with pytest.raises(DFloatFormatError, match=r"norm_out\.linear") as info:
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert model.open_phase is None
    assert not model._lifecycle.set_resident
    install = [f for f in _frames(info.value.__traceback__) if f.f_code.co_name == "_install_set"]
    assert len(install) == 1
    assert "resident" not in install[0].f_locals
    # The retry, with the exception still held: nothing of the failed load may still be active.
    monkeypatch.setattr(model._lifecycle, "retained_bound", lambda: before)
    state["on"] = False
    assert model.generate_image(seed=1, prompt="p", height=256, width=256) == "image"
    assert model._lifecycle.counters.set_loads == 1
    del info


def _fake_encode_peak(monkeypatch, model, above_start):
    """Make the tracker record an encode phase whose MLX peak is ``above_start`` bytes over the phase's start."""
    real_end = model._tracker.end

    def end(name):
        real_end(name)
        if name == "encode":
            record = model._tracker.peaks["encode"]
            record["mlx_peak"] = record["active_at_start"] + above_start

    monkeypatch.setattr(model._tracker, "end", end)


# fake_model's encoders (1_000) + the allowance at 256^2 (its 500_000_000 floor) + the 256 MiB slack (268_435_456).
ENCODE_WARN_ABOVE = 768_436_456


def test_an_encode_peak_over_the_estimate_warns_that_mflux_may_run_every_encoder_layer(
    tmp_path, monkeypatch
):
    # Bug caught (review 2026-10-08): an mflux change that evaluates all 36 encoder layers (Klein reads hidden states
    # 9, 18 and 27 only) going unnoticed: the encode phase grows by 9 layers (4B +1.8 GB, 9B +3.5 GB) while the fit
    # estimate, sized by the 27 used layers, keeps passing calls it no longer covers.
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, ENCODE_WARN_ABOVE + 1)
    with pytest.warns(UserWarning, match="every encoder layer"):
        model.generate_image(seed=1, prompt="p", height=256, width=256)


def test_an_encode_peak_at_the_estimate_does_not_warn(tmp_path, monkeypatch):
    # Bug caught: the slack or the allowance left out of the bound (every real encode would warn: the measured 4B encode
    # holds 0.41 GB over its used layers, the 9B 0.33 GB), or `>=` for `>` at the bound. Pytest turns a warning into an
    # error here.
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, ENCODE_WARN_ABOVE)
    assert model.generate_image(seed=1, prompt="p", height=256, width=256) == "image"


def test_the_constants_follow_the_klein_size_of_the_model_config(tmp_path, monkeypatch):
    # Bug caught: every Klein planned with the 4B constants (the 9B's wider activations under-predicted), or the size
    # read from the tiny transformer instead of the model's config.
    from mlx_dfloat.mflux.flux2.memory import CONSTANTS

    assert (
        fake_model(tmp_path / "a", monkeypatch, model="flux2-klein-4b")._constants
        is CONSTANTS["4b"]
    )
    nine = fake_model(tmp_path / "b", monkeypatch, model="flux2-klein-base-9b")
    assert nine._constants is CONSTANTS["9b"]


def test_the_budget_is_the_constructors_else_the_devices_read_at_each_call(tmp_path, monkeypatch):
    # Bug caught: the constructor's budget ignored (a --tier run planned against the whole Mac), or the device's read
    # once at assembly instead of at each call.
    from mlx_dfloat.mflux.flux2 import model as model_module

    assert (
        fake_model(tmp_path / "a", monkeypatch, budget_bytes=9 * 1024**3)._budget() == 9 * 1024**3
    )
    model = fake_model(tmp_path / "b", monkeypatch)
    assert model._budget() == 23 * 1024**3  # fake_model's fixed device budget
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 10 * 1024**3)
    assert model._budget() == 10 * 1024**3


# --- the generate prelude, the call plan ----------------------------------------------------------------------

GIB = 1024**3


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def _patch_upstream_generate(monkeypatch, *, raise_after_loop=None, raise_before_loop=None):
    """Replace Flux2Klein.generate_image with a probe: records the cache limit in force and the kwargs, fires our
    after-loop subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

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
        seen["set_resident_after_loop"] = self._lifecycle.set_resident
        if raise_after_loop is not None:
            raise raise_after_loop
        return "image"

    monkeypatch.setattr(Flux2Klein, "generate_image", fake)
    return seen


@pytest.mark.parametrize(
    ("guidance", "expected"),
    [(None, ("p",)), (1.0, ("p",)), (1.0001, ("p", " ")), (4.0, ("p", " "))],
)
def test_cfg_prompts_follow_mflux(tmp_path, monkeypatch, guidance, expected):
    # Bug caught (Review Focus 4): `>=` for `>` (guidance 1.0 encoding the negative: a wasted encoder reload and a
    # second transformer call), or guidance None encoding one. mflux 0.20.0 flux2_klein.py:78-80, 163: the negative is
    # " " and is encoded only when guidance > 1.0.
    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    assert model.cfg_prompts("p", guidance=guidance) == expected


def test_encode_prompt_pair_answers_from_the_cache_with_mflux_four_tuple(tmp_path, monkeypatch):
    # Bug caught: mflux's own _encode_prompt_pair running (the encoder is dropped: a None call), the negative looked up
    # at guidance <= 1 or when mflux passes no negative, or a prompt not encoded first surfacing as a KeyError deep in
    # the loop.
    from mlx_dfloat.errors import DFloatIntegrationError

    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    model.encode("p", " ")
    (pe, pi), (ne, ni) = model._embeddings["p"], model._embeddings[" "]
    four = model._encode_prompt_pair(prompt="p", negative_prompt=" ", guidance=4.0)
    assert [a is b for a, b in zip(four, (pe, pi, ne, ni), strict=True)] == [True] * 4
    assert model._encode_prompt_pair(prompt="p", negative_prompt=" ", guidance=1.0)[2:] == (
        None,
        None,
    )
    assert model._encode_prompt_pair(prompt="p", negative_prompt=None, guidance=4.0)[2:] == (
        None,
        None,
    )
    with pytest.raises(DFloatIntegrationError, match="not encoded"):
        model._encode_prompt_pair(prompt="q", negative_prompt=" ", guidance=1.0)


def test_a_cached_prompt_equal_to_the_negative_is_used_for_both_branches(tmp_path, monkeypatch):
    # Bug caught: the prompt " " at guidance 4 encoded twice (the lifecycle sees two missing prompts) or looked up
    # under a key it was not cached under.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    encodes = []
    real = model._lifecycle._encode
    monkeypatch.setattr(model._lifecycle, "_encode", lambda p: (encodes.append(p), real(p))[1])
    seen = {}

    def fake(self, **kwargs):
        seen["four"] = self._encode_prompt_pair(prompt=" ", negative_prompt=" ", guidance=4.0)
        return "image"

    monkeypatch.setattr(Flux2Klein, "generate_image", fake)
    model.generate_image(seed=1, prompt=" ", height=64, width=64, guidance=4.0)
    assert encodes == [" "]
    assert list(model._embeddings) == [" "]
    embeds, ids, neg, neg_ids = seen["four"]
    assert neg is embeds
    assert neg_ids is ids
    assert model._cfg_calls == 2  # still two transformer calls per step


@pytest.mark.parametrize(
    "kwargs", [{"image_path": "in.png"}, {"image_strength": 0.5}, {"pid_decode": True}]
)
def test_img2img_and_pid_decode_are_refused_before_anything_loads(tmp_path, monkeypatch, kwargs):
    # Bug caught: encoding the prompt (an encoder load) or loading the set before refusing img2img or the PiD decoder.
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        model.generate_image(seed=1, prompt="p", **kwargs)
    assert model._lifecycle.counters.set_loads == 0


def test_the_prelude_encodes_then_drops_the_encoder_then_loads_the_set_then_runs_mflux(
    tmp_path, monkeypatch
):
    # Bug caught: the set loaded while the encoder is resident, mflux's loop entered before the embeddings exist, or
    # img2img / PiD / a negative_prompt keyword reaching Flux2Klein.generate_image (it has no negative_prompt).
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    events = []
    life = model._lifecycle
    for attr in ("_encode", "_unload_encoders", "_load_set"):
        real = getattr(life, attr)

        def wrapped(*args, _real=real, _name=attr, **kwargs):
            events.append(_name + (f":{args[0]!r}" if args else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(life, attr, wrapped)
    seen = _patch_upstream_generate(monkeypatch)
    upstream = Flux2Klein.generate_image

    def recording(self, **kwargs):
        events.append("upstream")
        return upstream(self, **kwargs)

    monkeypatch.setattr(Flux2Klein, "generate_image", recording)
    image = model.generate_image(
        seed=7, prompt="p", num_inference_steps=1, height=256, width=256, guidance=4.0
    )
    assert image == "image"
    assert events == ["_encode:'p'", "_encode:' '", "_unload_encoders", "_load_set", "upstream"]
    kwargs = seen["kwargs"]
    assert (kwargs["image_path"], kwargs["image_strength"], kwargs["pid_decode"]) == (
        None,
        None,
        False,
    )
    assert kwargs["guidance"] == 4.0
    assert "negative_prompt" not in kwargs
    assert model._cfg_calls == 2
    assert model.text_encoder is None


def test_a_missing_guidance_and_scheduler_become_mflux_defaults(tmp_path, monkeypatch):
    # Bug caught: the CLI's None (no --guidance, no --scheduler for a Klein entry) reaching mflux's Config, which
    # builds the scheduler by name, instead of mflux's defaults (guidance 1.0, flow_match_euler_discrete).
    model = fake_model(tmp_path, monkeypatch)
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, guidance=None, scheduler=None)
    assert seen["kwargs"]["guidance"] == 1.0
    assert seen["kwargs"]["scheduler"] == "flow_match_euler_discrete"
    assert model._cfg_calls == 1


def test_plan_call_reads_the_familys_constants_sizes_and_groups(tmp_path, monkeypatch):
    # Bug caught: plan_call not wired to the shared planner with this model's own largest groups and 512 text tokens
    # (the cache limit would miss the allowance or a decoded group), or the plan not kept for the VAE guard.
    from mlx_dfloat.errors import DFloatResourceError

    model = fake_model(tmp_path, monkeypatch)
    plan = model.plan_call(height=1030, width=1030)  # rounded to 1024^2 first
    # The largest decoded group of each kind, from the TINY dims (inner 2 x 16 = 32, MLP hidden 3 x 32 = 96, gated
    # linear_in 192 wide): double block 8 x 32 x 32 + 2 x (192 x 32 + 32 x 96) = 26_624 elements, 53_248 B; single
    # block 288 x 32 + 32 x 128 = 13_312 elements, 26_624 B. Allowance at 1024^2 + 512 text tokens: 1.5e9 * 4608 / 4352.
    assert plan.cache_limit == 53_248 + 26_624 + 1_588_235_294
    assert model._plan is plan
    with pytest.raises(DFloatResourceError, match="1024x1024"):
        model.plan_call(height=1040, width=1040)


def test_the_vae_guard_follows_the_plan_and_zeroes_the_cache_limit(tmp_path, monkeypatch):
    # Bug caught: the guard not reading this call's plan (the set kept through a decode the budget cannot hold, or
    # dropped on a roomy Mac: a reload per call), or a transformer-sized cache limit left for the VAE decode.
    # Tight: VAE phase 12e9 + 7_443_018_794 + 381_164_752 = 19_824_183_546 > 16e9, denoise ~14.9e9 fits.
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
    _patch_upstream_generate(monkeypatch, **{where: DFloatFormatError("block 3: invalid code")})
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    assert all(_nonblock_weight(model, g).size == 0 for g in NONBLOCK)


def test_keyboard_interrupt_in_the_loop_restores_the_cache_limit_and_leaves_no_pending_words(
    tmp_path, monkeypatch
):
    # Bug caught: mflux's StopImageGenerationException (raised for a Ctrl-C in its loop, flux2_klein.py:113-117)
    # leaving status words pending (begin_step would refuse the next call), the cache limit not restored, or the
    # interrupt dropping the set (a reload per Ctrl-C). Real mflux loop; the interrupt fires at the second block of
    # the step's second transformer call (guidance 4: 3 blocks per call).
    from mflux.utils.exceptions import StopImageGenerationException

    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    before = _cache_limit_in_force()
    model.encode("p", " ")
    model._lifecycle.ensure_set()
    provider = model._provider
    real = provider.weights_for
    calls = []
    interrupt = {"on": True}

    def interrupting(name, shapes):
        calls.append(name)
        if interrupt["on"] and len(calls) == 5:
            raise KeyboardInterrupt
        return real(name, shapes)

    monkeypatch.setattr(provider, "weights_for", interrupting)
    with pytest.raises(StopImageGenerationException):
        model.generate_image(
            seed=1, prompt="p", num_inference_steps=1, height=64, width=64, guidance=4.0
        )
    assert len(calls) == 5
    assert _cache_limit_in_force() == before
    assert provider.pending == []
    assert model._lifecycle.set_resident
    interrupt["on"] = False
    image = model.generate_image(
        seed=1, prompt="p", num_inference_steps=1, height=64, width=64, guidance=4.0
    )
    assert image is not None
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
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

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

    monkeypatch.setattr(Flux2Klein, "generate_image", fake)
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
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

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

    monkeypatch.setattr(Flux2Klein, "generate_image", fake)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert seen == ["encode", "set_load", "denoise", "vae"]
    assert model.open_phase is None


# --- the report -----------------------------------------------------------------------------------------------


def test_the_report_carries_the_family_keys_and_the_sizes_before_any_call(tmp_path, monkeypatch):
    # Bug caught: the family, size or sizes missing from the report (the CLI and the table read them), or the
    # non-block groups listed in another order than the one other families use (sorted).
    model = fake_model(tmp_path, monkeypatch)
    report = model.report()
    assert report["family"] == "flux2"
    assert report["model"] == "flux2-klein-4b"
    assert report["size"] == "4b"
    assert report["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "nonblock": 10,
        "encoders": 1_000,
        "vae": 10,
    }  # fake_model's own FamilySizes
    assert report["predict"] == "uncompiled"
    assert report["nonblock_groups"] == NONBLOCK
    assert report["cfg_calls_per_step"] == 1
    assert report["fit"] is None
    assert report["df11"]["repo_id"] == "mingyi456/tiny"
    assert report["base"]["revision"] == "b" * 40
    assert report["eval_policy"] == "per-block"


def test_a_9b_model_plans_with_the_9b_constants_and_reports_its_size(tmp_path, monkeypatch):
    # Bug caught: a 9B call planned with the 4B denoise activation (402_128_193 vs 343_390_778 at 1024^2 + 512 text
    # tokens), or the report naming the wrong size.
    from mlx_dfloat.mflux._phases import FamilySizes

    zero = FamilySizes(compressed=0, extras=0, nonblock=0, encoders=0, vae=0)
    nine = fake_model(tmp_path / "n", monkeypatch, model="flux2-klein-base-9b", sizes=zero)
    four = fake_model(tmp_path / "f", monkeypatch, model="flux2-klein-base-4b", sizes=zero)
    d9 = nine.plan_call(height=1024, width=1024).estimate.phases["denoise"]
    d4 = four.plan_call(height=1024, width=1024).estimate.phases["denoise"]
    # Same tiny groups and cache limit: the phases differ only by activation and overhead.
    assert d9 - d4 == (402_128_193 - 343_390_778) + (378_553_792 - 381_164_752)
    report = nine.report()
    assert report["size"] == "9b"
    assert report["fit"]["label"] == "predicted"


def test_a_lost_bypass_is_refused_at_predict_and_the_report_says_compiled(tmp_path, monkeypatch):
    # Bug caught: the chip-check override no longer reaching mflux's factory while _predict hands the compiled
    # closure to the loop and report() still prints "uncompiled".
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
    model.encode("p")  # drops the encoder
    assert model.text_encoder is None
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))
