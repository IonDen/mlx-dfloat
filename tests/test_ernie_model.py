"""Tests for `DFloatErnieImage`: construction, refusals, the lifecycle and the set, the prelude and the report.

Every test is `@pytest.mark.mflux`; the module imports mflux only inside helpers, so it collects without it.
"""

import inspect
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tests._ernie_tiny import TINY_SEED, StubTextEncoder, fake_model, write_tiny_checkpoint

from mlx_dfloat.errors import DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.mflux.ernie import init as einit

pytestmark = pytest.mark.mflux

# mflux's parameter for each of the four non-block matrices (ernie_weight_mapping.py:12-19, 40-63) and its shape at
# TINY size (hidden 32): Linear(32, 32) x 2, Linear(32, 6 x 32), Linear(32, 2 x 32).
NONBLOCK = {
    "time_embedding.linear_1.weight": ("time_embedding", (32, 32)),
    "time_embedding.linear_2.weight": ("time_embedding", (32, 32)),
    "adaln_modulation.weight": ("adaLN_modulation.1", (192, 32)),
    "final_norm.linear.weight": ("final_norm.linear", (64, 32)),
}


def _nonblock_sources(tmp_path):
    """The non-block matrices ``fake_model`` compressed: the same seed gives the same arrays."""
    matrices, _c, _conv = write_tiny_checkpoint(
        tmp_path / "again", np.random.default_rng(TINY_SEED)
    )
    return {param: matrices[group][param] for param, (group, _shape) in NONBLOCK.items()}


def _param(model, name):
    module = model.transformer
    for part in name.split(".")[:-1]:
        module = getattr(module, part)
    return module.weight


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new self.<name> in ErnieImage.generate_image our assembly never sets
    # (mflux's own initializer is not run).
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(ErnieImage.generate_image)))
    assert names >= {"model_config", "callbacks", "transformer", "bits", "lora_paths"}
    model = fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, ErnieImage)
    assert (model.bits, model.prompt_cache, model.lora_paths, model.lora_scales) == (
        None,
        {},
        [],
        [],
    )
    assert model.tiling_config.vae_decode_tiles_per_dim is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantize": 8}, "quantize"),
        ({"lora_paths": ["a"]}, "lora_paths"),
        ({"lora_scales": [1.0]}, "lora_scales"),
        ({"bake_lora": False}, "bake_lora"),
        ({"eval_policy": "none"}, "none"),
        ({"model": "ernie"}, r"'ernie'.*ernie-image.*ernie-image-turbo"),
    ],
)
def test_construction_refuses_quantize_lora_and_unknown_names(monkeypatch, kwargs, match):
    # Bug caught: quantize=8 silently honoured by mflux's applier (it would quantize the decoded weights), a LoRA or
    # the "none" policy accepted, an unknown name not listing what this path runs, or any of them resolved (a 10.9 GB
    # download) before the refusal.
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    monkeypatch.setattr(
        einit, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=match):
        DFloatErnieImage(**{"model": "ernie-image-turbo", **kwargs})


@pytest.mark.parametrize(
    ("name", "df11", "base"),
    [
        (
            "ernie-image-turbo",
            ("mingyi456/ERNIE-Image-Turbo-DF11", "27f84b44a3b78fcfaaadbaeeeae7cc7f7d75153b"),
            ("baidu/ERNIE-Image-Turbo", "bc68c81e2a1730a394d5fc9fae70713dee940140"),
        ),
        (
            "ernie-image",
            ("mingyi456/ERNIE-Image-DF11", "c2dd30ad7dd5a928df2309581b282337f7cdb41f"),
            ("baidu/ERNIE-Image", "5346b31d68c9c23758ba56ef8be5e9dc174c7f99"),
        ),
    ],
)
def test_the_default_checkpoint_and_base_resolve_at_their_pins_and_user_paths_unpinned(
    tmp_path, monkeypatch, name, df11, base
):
    # Bug caught: a pin not reaching the resolver (the checkpoint, encoder, VAE and tokenizer following whatever lands
    # on main instead of the snapshot the recorded runs used), or a pin applied to a user's own path.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux.ernie import model as model_module
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    class StopError(Exception):
        pass

    df11_calls, base_calls = [], []

    def stop_at_df11(spec, *, patterns, revision=None):
        df11_calls.append((spec, revision))
        raise StopError

    def stop_at_base(spec, *, patterns, revision=None):
        if patterns == einit.DF11_PATTERNS:
            return ResolvedRepo(root=tmp_path, repo_id=spec, revision=revision)
        base_calls.append((spec, revision))
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(model_module, "open_checkpoint", lambda root: None)
    monkeypatch.setattr(einit, "resolve", stop_at_df11)
    for kwargs in ({}, {"df11_path": "me/fork"}):
        with pytest.raises(StopError):
            DFloatErnieImage(name, **kwargs)
    monkeypatch.setattr(einit, "resolve", stop_at_base)
    for kwargs in ({}, {"base_path": "me/base"}):
        with pytest.raises(StopError):
            DFloatErnieImage(name, **kwargs)
    assert df11_calls == [df11, ("me/fork", None)]
    assert base_calls == [base, ("me/base", None)]


def test_the_set_load_installs_the_three_nonblock_groups_and_attaches_with_verify_in_call(
    tmp_path, monkeypatch
):
    # Bug caught: a non-block group decoded but not installed (a zero-size matmul in the timestep MLP, the modulation
    # or the final norm), a matrix installed from another group, or the set attached without verify_in_call.
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_set()
    for param, want in _nonblock_sources(tmp_path).items():
        got = _param(model, param)
        assert got.shape == NONBLOCK[param][1], param
        assert np.array_equal(np.array(got.view(mx.uint16)), want), param
    assert model.transformer.seam_cell.state.verify_in_call is True


def test_a_new_batch_after_a_generation_drops_the_set_and_reinstalls_the_nonblock_weights(
    tmp_path, monkeypatch
):
    # Bug caught (Review Focus 4): a non-block weight left on its placeholder after the reload a new batch forces (a
    # zero-size matmul on adaln_modulation before the first block), or the encoder reload reading another base.
    model = fake_model(tmp_path, monkeypatch)
    loads = []
    monkeypatch.setattr(
        einit,
        "load_text_encoder",
        lambda root, **kwargs: (loads.append(root), StubTextEncoder())[1],
    )
    model._lifecycle.ensure_embeddings(
        model._batch_key(("a",))
    )  # the encoder came with the assembly
    model._lifecycle.ensure_set()
    model._lifecycle.ensure_embeddings(model._batch_key((" ", "a")))  # CFG: a new batch
    counters = model._lifecycle.counters
    assert (counters.forced_set_drops, counters.encoder_loads) == (1, 1)
    assert loads == [model._base.root]
    assert not model._lifecycle.set_resident
    assert [_param(model, p).size for p in NONBLOCK] == [0, 0, 0, 0]
    model._lifecycle.ensure_set()
    assert counters.set_loads == 2
    for param, want in _nonblock_sources(tmp_path).items():
        assert np.array_equal(np.array(_param(model, param).view(mx.uint16)), want), param


def test_batches_are_cached_under_their_prompt_list(tmp_path, monkeypatch):
    # Bug caught: mflux's own prompt_cache used (its lookup would bypass the lifecycle), or a batch stored under a key
    # other than its prompt list (a CFG call would miss its batch).
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_embeddings(model._batch_key(("a",)), model._batch_key((" ", "a")))
    assert set(model._batches) == {'["a"]', '[" ", "a"]'}
    assert model.prompt_cache == {}
    for key, batch in (('["a"]', 1), ('[" ", "a"]', 2)):
        text_bth, text_lens = model._batches[key]
        # The stub tokenizer gives "a" and " " one token each; build_text_batch's width 3072.
        assert (text_bth.shape, text_bth.dtype) == ((batch, 1, 3072), mx.bfloat16)
        assert (text_lens.shape, text_lens.dtype) == ((batch,), mx.int32)


def test_a_dropped_encoder_refuses_with_a_named_error(tmp_path, monkeypatch):
    # Bug caught: None left in the encoder's place (mflux would call it on a cache miss and fail with a TypeError far
    # from the cause).
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_embeddings(model._batch_key(("a",)))
    with pytest.raises(DFloatIntegrationError, match="not resident"):
        model.text_encoder(mx.zeros((1, 1), mx.int32))


def test_the_retained_bound_counts_the_cached_batches_and_the_vae(tmp_path, monkeypatch):
    # Bug caught: the bound reading mflux's prompt_cache (empty here: every drop after an encode would refuse when the
    # batches are large) instead of the batch cache.
    model = fake_model(tmp_path, monkeypatch)
    empty = model._retained_bound()
    model._lifecycle.ensure_embeddings(model._batch_key((" ", "a")))
    # text_bth (2, 1, 3072) bf16 = 12_288 B, text_lens (2,) int32 = 8 B.
    assert model._retained_bound() - empty == 12_288 + 8
    # The absolute bound before any batch (T4): the assembly baseline + the stub VAE's one weight, 4 bf16 = 8 B, + the
    # 2 GiB slack. Bug caught: the VAE left out of the bound (a drop after the VAE decode would refuse).
    from mlx_dfloat.mflux.ernie.model import RETAINED_SLACK_BYTES

    assert empty == model._baseline_active + 8 + RETAINED_SLACK_BYTES


def test_the_pos_cache_worst_case_fits_inside_the_retained_slack():
    # Bug caught: a slack under what mflux's _pos_cache may legitimately keep active across calls (64 entries,
    # transformer.py:120-122), so a drop after many prompt lengths would refuse. One 1024² batch-2 entry at the
    # 2048-token maximum, by hand: cos + sin 2 x (2 x 6144 x 1 x 128) bf16 = 6_291_456, mask 2 x 6144 bf16 = 24_576.
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer

    from mlx_dfloat.mflux.ernie.model import RETAINED_SLACK_BYTES

    tf = ErnieTransformer(num_layers=0, rope_axes_dim=[32, 48, 48])  # the published head dim 128
    entry = tf.get_pos_encoding(
        B=2, H=64, W=64, T=2048, text_lens=mx.array([2048, 2048], dtype=mx.int32)
    )
    mx.eval(*entry)
    assert sum(int(a.nbytes) for a in entry) == 6_316_032
    assert RETAINED_SLACK_BYTES > 64 * 6_316_032


# --- the CFG batch, the generate prelude, the plan and the report ------------------------------------------------

GIB = 1024**3


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def _patch_upstream_generate(monkeypatch, *, raise_before_loop=None):
    """Replace ErnieImage.generate_image with a probe: records the limits in force and the kwargs, fires our
    after-loop subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
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
        return "image"

    monkeypatch.setattr(ErnieImage, "generate_image", fake)
    return seen


def _no_encoder(monkeypatch):
    """Any encoder reload fails the test (the batch must come from the cache)."""
    monkeypatch.setattr(
        einit, "load_text_encoder", lambda *a, **k: pytest.fail("the encoder was reloaded")
    )


@pytest.mark.parametrize(
    ("guidance", "negative", "expected"),
    [
        (1.0, "n", ("p",)),
        (None, "n", ("p",)),
        (0.5, "n", ("p",)),
        (1.0001, "n", ("n", "p")),
        (4.0, None, (" ", "p")),
        (4.0, "", (" ", "p")),
        (4.0, "   ", (" ", "p")),
        (4.0, " x ", (" x ", "p")),
    ],
)
def test_cfg_prompts_follow_mflux(guidance, negative, expected):
    # Bug caught: `>=` for `>` (ernie_image.py:164: guidance <= 1.0 runs one prompt; Config stores None as 0.0,
    # config.py:48), the positive first (mflux takes pred[:1] as the unconditional branch, :249: guidance would push
    # toward the negative), or a blank negative not replaced by " " (:167).
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    assert (
        DFloatErnieImage.cfg_prompts("p", negative_prompt=negative, guidance=guidance) == expected
    )


def _bits(array):
    return np.array(array.view(mx.uint16)) if array.dtype == mx.bfloat16 else np.array(array)


@pytest.mark.parametrize(("negative", "guidance"), [("n", 4.0), (None, 1.0)])
def test_the_override_returns_mfluxs_own_batch(tmp_path, monkeypatch, negative, guidance):
    # Bug caught: the copied CFG rule drifting from mflux's own _encode_prompts (order, blank handling, threshold), or
    # a batch stored under another call's key. The oracle is mflux's method on a throwaway object.
    from types import SimpleNamespace

    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

    model = fake_model(tmp_path, monkeypatch)
    model.encode("p", negative_prompt=negative, guidance=guidance)
    got = model._encode_prompts(prompt="p", negative_prompt=negative, guidance=guidance)
    oracle = SimpleNamespace(
        prompt_cache={}, tokenizers=model.tokenizers, text_encoder=StubTextEncoder()
    )
    want = ErnieImage._encode_prompts(
        oracle, prompt="p", negative_prompt=negative, guidance=guidance
    )
    assert np.array_equal(_bits(got[0]), _bits(want[0]))
    assert np.array_equal(np.array(got[1].view(mx.uint32)), np.array(want[1].view(mx.uint32)))


def test_a_call_whose_batch_was_not_encoded_is_refused(tmp_path, monkeypatch):
    # Bug caught: a cache miss falling through to mflux's encoder (dropped: a TypeError, or an encoder reload inside
    # the loop next to the set).
    model = fake_model(tmp_path, monkeypatch)
    with pytest.raises(DFloatIntegrationError, match=r'\["p"\] were not encoded'):
        model._encode_prompts(prompt="p", negative_prompt=None, guidance=1.0)


def test_blank_and_missing_negatives_share_one_batch(tmp_path, monkeypatch):
    # Bug caught (Review Focus 2): None, "" and "   " keyed apart (three encodes and set reloads for one mflux batch
    # [" ", prompt]), or a blank negative encoded as itself. Real mflux loop, one step each.
    model = fake_model(tmp_path, monkeypatch)
    run = {"seed": 1, "prompt": "a", "num_inference_steps": 1, "height": 64, "width": 64}
    model.generate_image(**run, guidance=4.0, negative_prompt=None)
    loads = model._lifecycle.counters.encoder_loads
    _no_encoder(monkeypatch)
    for negative in ("", "   "):
        model.generate_image(**run, guidance=4.0, negative_prompt=negative)
    assert list(model._batches) == ['[" ", "a"]']
    assert model._lifecycle.counters.encoder_loads == loads
    assert model._lifecycle.counters.set_loads == 1


def test_guidance_4_then_5_reuses_the_batch_and_1_does_not(tmp_path, monkeypatch):
    # Bug caught: the key built from the raw guidance (a re-encode and a set reload on every guidance change), or a
    # guidance-1 call reusing the CFG batch (a batch-2 call where mflux runs one prompt).
    model = fake_model(tmp_path, monkeypatch)
    run = {"seed": 1, "prompt": "a", "num_inference_steps": 1, "height": 64, "width": 64}
    model.generate_image(**run, guidance=4.0)
    counters = model._lifecycle.counters
    model.generate_image(**run, guidance=5.0)
    assert (counters.forced_set_drops, len(model._batches)) == (0, 1)
    model.generate_image(**run, guidance=1.0)
    assert counters.forced_set_drops == 1
    assert set(model._batches) == {'[" ", "a"]', '["a"]'}


def test_text_tokens_follow_the_longest_prompt_of_the_call(tmp_path, monkeypatch):
    # Bug caught: the plan sized by the positive prompt only (build_text_batch pads every prompt to the longest, a
    # longer negative's activations unplanned).
    from tests._ernie_tiny import StubTokenizer

    model = fake_model(
        tmp_path, monkeypatch, model="ernie-image", tokenizer=StubTokenizer({"p": 7, "n": 12})
    )
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, negative_prompt="n")
    assert model.report()["text_tokens"] == 12
    # T3: the negative prompt reaches mflux's generate_image (its own CFG rule and the image metadata read it).
    assert seen["kwargs"]["negative_prompt"] == "n"


def _double(allowance):
    from mlx_dfloat.mflux._phases import PhaseConstants

    return PhaseConstants(
        overhead_bytes=0,
        vae_transient_bytes=0,
        denoise_activation_at_reference=0,
        reference_tokens=4097,
        allowance_at_reference=allowance,
        allowance_reference_tokens=4097,
    )


@pytest.mark.parametrize(
    ("guidance", "batch", "limit"), [(4.0, 2, 3_000_017_408), (1.0, 1, 1_000_017_408)]
)
def test_a_cfg_call_plans_with_the_batch_two_constants(
    tmp_path, monkeypatch, guidance, batch, limit
):
    # Bug caught (Review Focus 1, Python-API half): the constants looked up by model name (Turbo planned with batch-1
    # activations for a batch-2 call: 1_000_017_408 at guidance 4), or the batch counted from the model instead of the
    # prompts. The doubles' reference is 4096 + 1 tokens ("a" and " " are one stub token each), so no scaling, no
    # floor: the tiny decoded block 17_408 B + the allowance, by hand.
    model = fake_model(tmp_path, monkeypatch)
    model._constants = {1: _double(1_000_000_000), 2: _double(3_000_000_000)}
    model.generate_image(
        seed=1, prompt="a", num_inference_steps=1, height=1024, width=1024, guidance=guidance
    )
    report = model.report()
    assert (report["cfg_batch"], report["cache_limit_in_force"]) == (batch, limit)


@pytest.mark.parametrize(
    ("name", "guidance", "steps", "batch"),
    [("ernie-image", 4.0, 50, 2), ("ernie-image-turbo", 1.0, 8, 1)],
)
def test_the_python_api_defaults_follow_mflux_per_variant(
    tmp_path, monkeypatch, name, guidance, steps, batch
):
    # Bug caught: the base's Python default left at mflux's class default 1.0 (no CFG: another image than the command
    # gives), or the defaults swapped between variants. Literals: ernie_image_generate.py:22, 30-31 (4.0),
    # ernie_image_turbo_generate.py:42-43 (1.0), defaults.py:35-36 (50, 8), ernie_image.py:64-65 ("linear").
    model = fake_model(tmp_path, monkeypatch, model=name)
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="a", height=64, width=64)
    kwargs = seen["kwargs"]
    assert (kwargs["guidance"], kwargs["num_inference_steps"], kwargs["scheduler"]) == (
        guidance,
        steps,
        "linear",
    )
    assert model.report()["cfg_batch"] == batch


def test_an_explicit_guidance_on_turbo_is_passed_through(tmp_path, monkeypatch):
    # Bug caught: the Python API forcing Turbo's guidance to 1.0 (mflux's API allows CFG on Turbo; only the command
    # refuses it).
    model = fake_model(tmp_path, monkeypatch, model="ernie-image-turbo")
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="a", height=64, width=64, guidance=2.0)
    assert seen["kwargs"]["guidance"] == 2.0
    assert model.report()["cfg_batch"] == 2


@pytest.mark.parametrize("raised", [None, RuntimeError("in the loop")], ids=["returns", "raises"])
def test_a_python_api_call_runs_under_the_commands_caps_and_restores_mlxs_defaults(
    tmp_path, monkeypatch, raised
):
    # Bug caught: generate_image not decorated (the Python API at MLX's default wired limit 0, while the VAE term is
    # measured under the command's caps), or the caps left installed after a call that returns or raises.
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


def _fake_encode_peak(monkeypatch, model, above_start):
    """Make the tracker record an encode phase whose MLX peak is ``above_start`` bytes over the phase's start."""
    real_end = model._tracker.end

    def end(name):
        real_end(name)
        if name == "encode":
            record = model._tracker.peaks["encode"]
            record["mlx_peak"] = record["active_at_start"] + above_start

    monkeypatch.setattr(model._tracker, "end", end)


@pytest.mark.parametrize(("over", "warns"), [(1, True), (0, False)])
def test_an_encode_over_the_bound_warns_through_the_pure_helper(tmp_path, monkeypatch, over, warns):
    # Bug caught: _check_encode_peak never called after the encode phase, measuring the wrong phase record, or a bound
    # of its own that drifts from the helper's (`>` at the bound: silent). The bound comes from the plan through the
    # helper Task 7 pins with literals.
    import warnings

    from mlx_dfloat.mflux.ernie import memory as emem

    model = fake_model(tmp_path, monkeypatch)
    plan = model.plan_call(height=256, width=256, text_tokens=1, batch=1)
    bound = emem.encode_peak_bound(plan.estimate.phases["encode"], model._constants[1])
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, bound + over)
    if warns:
        with pytest.warns(UserWarning, match=r"for a 1-token prompt"):
            model.generate_image(seed=1, prompt="a", height=256, width=256)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert model.generate_image(seed=1, prompt="a", height=256, width=256) == "image"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_path": "in.png"}, "image_path"),
        ({"image_strength": 0.5}, "image_path"),
        ({"pid_decode": True}, "pid_decode"),
    ],
)
def test_img2img_and_pid_decode_are_refused_before_anything_loads(
    tmp_path, monkeypatch, kwargs, match
):
    # Bug caught: encoding the prompt (an encoder load) or loading the set before refusing img2img or mflux's PiD
    # decoder (which loads its own caption encoder next to the set).
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=match):
        model.generate_image(seed=1, prompt="p", **kwargs)
    assert model._lifecycle.counters.set_loads == 0
    assert model._lifecycle.counters.encoder_loads == 0


@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_empty_or_blank_prompt_is_refused_before_anything_loads(tmp_path, monkeypatch, prompt):
    # Bug caught: an empty positive prompt reaching mflux (ERNIE-Image's tokenizer gives no tokens for "" — 0 tokens,
    # ids [] on both base snapshots, 2026-10-08 — so the call would run a zero-length text row through the transformer),
    # or the refusal coming after an encode or a set load. A blank negative stays mflux's (" ").
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(
        DFloatUnsupportedError, match="an empty or blank prompt is refused as a user error"
    ):
        model.generate_image(seed=1, prompt=prompt, height=64, width=64)
    assert model._lifecycle.counters.set_loads == 0


def test_keyboard_interrupt_restores_the_cache_limit_and_leaves_no_pending_words(
    tmp_path, monkeypatch
):
    # Bug caught: mflux's StopImageGenerationException (raised for a Ctrl-C in its loop, ernie_image.py:104-108)
    # leaving status words pending (begin_step would refuse the next call), the cache limit not restored, or the
    # interrupt dropping the set (a reload per Ctrl-C). Real mflux loop; the interrupt fires at the second block of
    # the CFG step's one batch-2 call.
    from mflux.utils.exceptions import StopImageGenerationException

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    model.encode("p", negative_prompt=None, guidance=4.0)
    model._lifecycle.ensure_set()
    provider = model._provider
    real = provider.weights_for
    calls = []
    interrupt = {"on": True}

    def interrupting(name, shapes):
        calls.append(name)
        if interrupt["on"] and len(calls) == 2:
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
    }
    with pytest.raises(StopImageGenerationException):
        model.generate_image(**run)
    assert len(calls) == 2
    assert _cache_limit_in_force() == before
    assert provider.pending == []
    assert model._lifecycle.set_resident
    interrupt["on"] = False
    assert model.generate_image(**run) is not None
    assert model._lifecycle.counters.set_loads == 1


@pytest.mark.parametrize(
    ("policy", "status"), [("per-block", "measured"), ("depth2", "unmeasured")]
)
def test_the_report_names_the_family_the_batch_and_the_uncompiled_predict(
    tmp_path, monkeypatch, policy, status
):
    # Bug caught: the report naming no predict mode, or "uncompiled" before any call ran the bypass; cfg_batch not
    # following the call (the bench table would mislabel a CFG run); a non-block group missing; the estimate labelled
    # as a measurement; depth2 reported like the measured path.
    model = fake_model(tmp_path, monkeypatch, eval_policy=policy)
    before = model.report()
    assert (before["family"], before["model"], before["predict"], before["fit"]) == (
        "ernie",
        "ernie-image-turbo",
        None,
        None,
    )
    assert before["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "nonblock": 10,
        "encoders": 1_000,
        "vae": 10,
    }  # fake_model's own FamilySizes
    model.generate_image(
        seed=1, prompt="a", num_inference_steps=1, height=64, width=64, guidance=4.0
    )
    report = model.report()
    assert (report["cfg_batch"], report["predict"], report["text_tokens"]) == (2, "uncompiled", 1)
    assert report["nonblock_groups"] == [
        "adaLN_modulation.1",
        "final_norm.linear",
        "time_embedding",
    ]
    assert report["fit"]["label"] == "predicted"
    assert (report["eval_policy"], report["eval_policy_status"]) == (policy, status)
    assert report["base"]["revision"] == "b" * 40


def test_save_model_is_refused_and_freeze_skips_a_dropped_encoder(tmp_path, monkeypatch):
    # Bug caught: freeze() calling .freeze() on the encoder stand-in (it is not a module), or save_model writing a
    # checkpoint with zero-size placeholders as the transformer.
    model = fake_model(tmp_path, monkeypatch)
    model.encode("p")  # drops the encoder
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))


def test_the_constructor_builds_with_the_models_overrides_and_the_language_model_bytes(
    tmp_path, monkeypatch
):
    # Bug caught (S3 review fix 9): __init__ wiring past the resolver untested: the transformer built without the
    # model's overrides (mflux builds ErnieTransformer(**model_config.transformer_overrides),
    # ernie_image_initializer.py:63; ModelConfig.ernie_image_turbo() sets rope_axes_dim [32, 48, 48]), the base
    # components read from another directory, or the sizes taken from the encoder file (vision tower included) instead
    # of the language model's tensor bytes.
    from tests._df11_fixtures import write_bf16_original
    from tests._ernie_tiny import TINY, stub_components

    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux.ernie import model as model_module

    write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    base = tmp_path / "base"
    # The language model: embed_tokens (10, 4) 80 B, two layers' up_proj (4, 4) 32 B each, final norm (4,) 8 B; the
    # last layer never runs (mflux returns the second-to-last hidden state). Never loaded: a vision tower tensor
    # (8, 4) 64 B. 80 + 32 + 8 = 120.
    write_bf16_original(
        base / "text_encoder",
        {
            "language_model.model.embed_tokens.weight": np.zeros((10, 4), np.uint16),
            "language_model.model.layers.0.mlp.up_proj.weight": np.zeros((4, 4), np.uint16),
            "language_model.model.layers.1.mlp.up_proj.weight": np.zeros((4, 4), np.uint16),
            "language_model.model.norm.weight": np.zeros((4,), np.uint16),
            "vision_tower.transformer.layers.0.attention.q_proj.weight": np.zeros(
                (8, 4), np.uint16
            ),
        },
    )
    (base / "vae").mkdir()
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    loaded = []
    monkeypatch.setattr(
        einit, "load_base", lambda root: (loaded.append(root), stub_components())[1]
    )
    built = []
    real_build = model_module.build_transformer

    def tiny_build(ckpt, **kwargs):
        built.append(kwargs)
        return real_build(ckpt, transformer_overrides=TINY)

    monkeypatch.setattr(model_module, "build_transformer", tiny_build)
    model = model_module.DFloatErnieImage(
        "ernie-image-turbo", df11_path=str(tmp_path / "df11"), base_path=str(base)
    )
    assert built == [{"transformer_overrides": {"rope_axes_dim": [32, 48, 48]}}]
    assert loaded == [base]
    report = model.report()
    assert report["sizes"]["encoders"] == 120
    assert report["sizes"]["vae"] == 100
    # The three non-block groups decoded at TINY size: (32, 32) x 2 + (192, 32) + (64, 32) BF16.
    assert report["sizes"]["nonblock"] == (2 * 32 * 32 + 192 * 32 + 64 * 32) * 2
    assert report["df11"]["config_source"] == "config.json"
    assert report["base"]["root"] == str(base)


def _innermost_locals(tb):
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_locals


def test_clear_frames_walks_the_whole_chain_of_causes_and_contexts():
    # Bug caught: only the raised exception and its __context__ cleared, so a __cause__ one link further (raise ... from
    # an earlier error) keeps its frame's locals, in production a block's decoded weights, alive while the caller holds
    # the exception; or a chain that loops back on itself never ending.
    from mlx_dfloat.mflux.ernie.model import _clear_frames

    def fail(kind, message):
        big = mx.zeros(1024)  # noqa: F841  # stands in for a block's decoded weights
        raise kind(message)

    try:
        try:
            try:
                fail(ValueError, "first")
            except ValueError as first:
                raise KeyError("second") from first
        except KeyError:
            fail(RuntimeError, "third")
    except RuntimeError as third:
        caught = third
    second = caught.__context__
    first = second.__cause__
    first.__context__ = caught  # a loop: the walk must still end
    _clear_frames(caught)
    for exc in (caught, second, first):
        assert "big" not in _innermost_locals(exc.__traceback__), exc


class _RecordingVAE(nn.Module):
    """The stub VAE's decode, keeping the packed latents mflux hands it (the denoised result)."""

    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((4,), dtype=mx.bfloat16)
        self.seen = []

    def decode_packed_latents(self, latents, tiling_config=None):
        del tiling_config
        self.seen.append(latents)
        return mx.zeros((1, 3, 16 * latents.shape[-2], 16 * latents.shape[-1]))


class _VaryingTextEncoder(nn.Module):
    """A text stack whose hidden states differ in direction from token to token (the stub's are one vector scaled by
    the id, which a norm maps to the same vector for every prompt); width 3072, as build_text_batch pads."""

    def __init__(self):
        super().__init__()
        self.freq = mx.arange(1, 3073, dtype=mx.float32) / 7

    def __call__(self, input_ids, attention_mask=None):
        return mx.sin(input_ids[..., None].astype(mx.float32) * self.freq).astype(mx.bfloat16)


def _stock_ernie(tokenizer, vae, matrices, extras):
    """Stock mflux ``ErnieImage`` over a plain tiny ``ErnieTransformer`` holding the BF16 source of a tiny checkpoint
    (``write_tiny_checkpoint``'s matrices and extras in mflux's layout)."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer
    from mflux.models.ernie_image.model.ernie_transformer.transformer import ErnieTransformer
    from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
    from mlx.utils import tree_flatten, tree_unflatten
    from tests._ernie_tiny import TINY

    transformer = ErnieTransformer(**TINY)
    shapes = {name: tuple(p.shape) for name, p in tree_flatten(transformer.parameters())}
    weights = []
    for group, subs in matrices.items():
        for sub, matrix in subs.items():
            # Block groups key their matrices by sub-path; the non-block groups by mflux's parameter name.
            name = sub if sub.endswith(".weight") else f"{group}.{sub}.weight"
            weights.append((name, mx.array(matrix).view(mx.bfloat16)))
    weights += [(name, mx.array(bits).view(mx.bfloat16)) for name, bits in extras.items()]
    assert sorted(name for name, _w in weights) == sorted(shapes)
    assert all(shapes[name] == tuple(w.shape) for name, w in weights)
    transformer.update(tree_unflatten(weights))
    stock = ErnieImage.__new__(ErnieImage)
    nn.Module.__init__(stock)
    ErnieImageInitializer._init_config(
        stock, ModelConfig.from_name(model_name="ernie-image", base_model=None)
    )
    stock.vae = vae
    stock.text_encoder = _VaryingTextEncoder()
    stock.tokenizers = {"ernie": tokenizer}
    stock.transformer = transformer
    stock.bits = None
    stock.lora_paths, stock.lora_scales = [], []
    return stock


@pytest.mark.parametrize("guidance", [1.0, 4.0], ids=["guidance-1", "cfg"])
def test_a_tiny_generation_gives_stock_mfluxs_latents_bit_for_bit(tmp_path, monkeypatch, guidance):
    # Bug caught (T1-T3): anything between the checkpoint and mflux's loop changing the result: a decoded matrix on the
    # wrong parameter, a non-block group's matrices swapped, an extra mis-routed, the negative prompt dropped from the
    # CFG batch or the batch built in the other order (the negative prompt is 12 ids here, the prompt 8, so the text
    # lengths differ too). The reference is stock mflux over the same BF16 weights, its step uncompiled as ours;
    # equality is bit for bit (an integer view of the latents). The extras are random: constant ones make the forward
    # pass blind to its inputs.
    from mflux.utils.apple_silicon import AppleSiliconUtil
    from tests._ernie_tiny import TINY, StubTokenizer, stub_components

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.ernie.transformer import build_transformer

    rng = np.random.default_rng(11)
    matrices, extras, _conv = write_tiny_checkpoint(tmp_path / "random", rng, random_extras=rng)
    ckpt = open_checkpoint(tmp_path / "random")
    tokenizer = StubTokenizer({"n": 12, "p": 8})
    ours_vae, stock_vae = _RecordingVAE(), _RecordingVAE()
    components = stub_components(tokenizer)
    components.vae, components.text_encoder = ours_vae, _VaryingTextEncoder()
    model = fake_model(
        tmp_path,
        monkeypatch,
        model="ernie-image",
        ckpt=ckpt,
        build=build_transformer(ckpt, transformer_overrides=TINY),
        components=components,
    )
    stock = _stock_ernie(tokenizer, stock_vae, matrices, extras)
    kwargs = {
        "seed": 3,
        "prompt": "p",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": guidance,
        "negative_prompt": "n",
        "scheduler": "linear",
    }
    model.generate_image(**kwargs)
    # Stock mflux runs its step uncompiled only on a base or Pro M1/M2: answer that, so both sides run the same graph.
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    stock.generate_image(**kwargs)
    (ours,), (theirs,) = ours_vae.seen, stock_vae.seen
    assert model.report()["cfg_batch"] == (2 if guidance > 1.0 else 1)
    assert bool(mx.all(mx.isfinite(theirs)))
    assert (ours.dtype, ours.shape) == (theirs.dtype, theirs.shape)
    view = mx.uint32 if ours.dtype == mx.float32 else mx.uint16
    assert np.array_equal(np.array(ours.view(view)), np.array(theirs.view(view)))


def test_encode_then_generate_on_the_base_reuses_the_batch(tmp_path, monkeypatch):
    # Bug caught (C1): encode("p") keyed at guidance 1.0 while generate_image(seed, "p") runs the base's default 4.0
    # (CFG: the batch [" ", "p"]), so the call drops the set and reloads the encoder for a batch encode() was meant to
    # prepare.
    model = fake_model(tmp_path, monkeypatch, model="ernie-image")
    _patch_upstream_generate(monkeypatch)
    model.encode("p")
    _no_encoder(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64)
    counters = model._lifecycle.counters
    assert list(model._batches) == ['[" ", "p"]']
    assert (counters.encoder_loads, counters.forced_set_drops) == (0, 0)


@pytest.mark.parametrize("prompt", ["", "   "])
def test_encode_refuses_an_empty_or_blank_prompt(tmp_path, monkeypatch, prompt):
    # Bug caught (C3): encode() taking a prompt generate_image refuses (an empty batch cached, an encoder load spent).
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(
        DFloatUnsupportedError, match="an empty or blank prompt is refused as a user error"
    ):
        model.encode(prompt)


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
