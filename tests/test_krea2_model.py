"""Tests for `DFloatKrea2`: construction, refusals, the lifecycle and the set, the prelude, CFG and the report.

Every test is `@pytest.mark.mflux`; the module imports mflux only inside helpers, so it collects without it.
"""

import gc
import inspect
import re
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from tests._krea2_tiny import (
    TINY_SEED,
    StubTextEncoder,
    StubTokenizer,
    fake_model,
    write_tiny_checkpoint,
)

from mlx_dfloat.errors import DFloatIntegrationError, DFloatUnsupportedError
from mlx_dfloat.mflux.krea2 import init as kinit

pytestmark = pytest.mark.mflux

# The non-block matrices as mflux names them (krea2_weight_mapping.py: the six timestep / text-MLP renames, the text
# fusion by concrete index): 2 + 1 + 2 + 4 x 8 = 37 matrices in the seven groups.
_FUSION = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)
NONBLOCK = {
    "tmlp": ("tmlp.linear_in.weight", "tmlp.linear_out.weight"),
    "tproj": ("tproj.linear.weight",),
    "txtmlp": ("txtmlp.linear_in.weight", "txtmlp.linear_out.weight"),
    **{
        f"txtfusion.{kind}.{i}": tuple(f"txtfusion.{kind}.{i}.{s}.weight" for s in _FUSION)
        for kind in ("layerwise_blocks", "refiner_blocks")
        for i in (0, 1)
    },
}


def _nonblock_sources(tmp_path):
    """The non-block matrices ``fake_model`` compressed: the same seed gives the same arrays."""
    matrices, _c, _layout = write_tiny_checkpoint(
        tmp_path / "again", np.random.default_rng(TINY_SEED)
    )
    return {p: matrices[group][p] for group, params in NONBLOCK.items() for p in params}


def _param(model, name):
    module = model.transformer
    for part in name.split(".")[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module.weight


def test_every_attribute_upstreams_generate_image_reads_exists_on_the_model(tmp_path, monkeypatch):
    # Bug caught: an mflux point release reading a new self.<name> in Krea2.generate_image our assembly never sets
    # (mflux's own initializer is not run).
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    names = set(re.findall(r"self\.(\w+)", inspect.getsource(Krea2.generate_image)))
    assert names >= {
        "model_config",
        "callbacks",
        "transformer",
        "bits",
        "lora_paths",
        "lora_scales",
    }
    model = fake_model(tmp_path, monkeypatch)
    assert [n for n in sorted(names) if not hasattr(model, n)] == []
    assert isinstance(model, Krea2)
    assert (model.bits, model.prompt_cache, model.lora_paths, model.lora_scales) == (
        None,
        {},
        [],
        [],
    )
    assert model.tiling_config is None  # krea2_initializer.py:47-54: the VAE is never tiled


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantize": 8}, "quantize"),
        ({"lora_paths": ["a"]}, "lora_paths"),
        ({"lora_scales": [1.0]}, "lora_scales"),
        ({"bake_lora": False}, "bake_lora"),
        ({"eval_policy": "none"}, "none"),
        ({"model": "krea-dev"}, r"'krea-dev'.*\['krea-2', 'krea-2-raw'\]"),
    ],
)
def test_construction_refuses_quantize_lora_and_unknown_names(monkeypatch, kwargs, match):
    # Bug caught: a quantize / LoRA argument or the "none" policy silently honoured (mflux's applier would quantize the
    # decoded weights), FLUX.1 Krea-dev (another family) accepted by this class, or any of them resolved (a 17.5 GB
    # download) before the refusal.
    from mlx_dfloat.mflux.krea2.model import DFloatKrea2

    monkeypatch.setattr(
        kinit, "resolve", lambda *a, **k: pytest.fail("resolved before the refusal")
    )
    with pytest.raises(DFloatUnsupportedError, match=match):
        DFloatKrea2(**{"model": "krea-2", **kwargs})


@pytest.mark.parametrize(
    ("model", "df11", "base"),
    [
        (
            "krea-2",
            ("mingyi456/Krea-2-Turbo-DF11-ComfyUI", "978da5fb7647bd222d33125993abd8fdc2840cfc"),
            ("krea/Krea-2-Turbo", "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"),
        ),
        (
            "krea-2-raw",
            ("mingyi456/Krea-2-Raw-DF11-ComfyUI", "8320616b25ac9340a830a7fb21f1b0237e160e66"),
            ("krea/Krea-2-Raw", "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"),
        ),
    ],
)
def test_the_default_checkpoint_and_base_resolve_at_their_pins_and_user_paths_unpinned(
    tmp_path, monkeypatch, model, df11, base
):
    # Bug caught: a pin not reaching the resolver (the default checkpoint, one file on one person's Hub account, or the
    # base following whatever lands on main instead of the revisions whose bytes and runs were checked), the Raw and
    # Turbo pins swapped, or a pin applied to a user's own --df11 / --base.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux.krea2 import model as model_module
    from mlx_dfloat.mflux.krea2.model import DFloatKrea2

    class StopError(Exception):
        pass

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        calls.append((spec, revision))
        if patterns == kinit.DF11_PATTERNS:
            return ResolvedRepo(root=tmp_path, repo_id=spec, revision=revision)
        raise StopError

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(
        model_module,
        "open_checkpoint",
        lambda root: SimpleNamespace(root=root, config_source="config.json"),
    )
    monkeypatch.setattr(kinit, "resolve", fake_resolve)
    for kwargs in ({}, {"df11_path": "me/fork", "base_path": "me/base"}):
        with pytest.raises(StopError):
            DFloatKrea2(model, **kwargs)
    assert calls == [df11, base, ("me/fork", None), ("me/base", None)]


def test_the_other_models_checkpoint_is_refused_before_the_base_resolves(tmp_path, monkeypatch):
    # Bug caught: the variant check run only at the build, after the base is resolved, so asking for Krea 2 Raw
    # with Turbo's DF11 file downloads the whole Raw base (about 16 GB of encoder and VAE) before the refusal.
    from mlx_dfloat import _metal_decode
    from mlx_dfloat.errors import DFloatFormatError
    from mlx_dfloat.mflux._hub import ResolvedRepo
    from mlx_dfloat.mflux.krea2 import model as model_module

    calls = []

    def fake_resolve(spec, *, patterns, revision=None):
        calls.append(patterns == kinit.DF11_PATTERNS)
        return ResolvedRepo(root=tmp_path, repo_id=spec, revision=revision)

    turbo = "header and spot checks match layout krea-2-turbo-comfyui (mingyi456/Krea-2-Turbo-DF11-ComfyUI@978da5f)"
    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(
        model_module,
        "open_checkpoint",
        lambda root: SimpleNamespace(root=root, config_source=turbo),
    )
    monkeypatch.setattr(kinit, "resolve", fake_resolve)
    with pytest.raises(DFloatFormatError, match=r"Krea 2 Turbo DF11 checkpoint.*--model krea-2$"):
        model_module.DFloatKrea2("krea-2-raw", df11_path="me/turbo")
    assert calls == [True]  # the checkpoint only


@pytest.mark.parametrize(("nonblock", "bytes_"), [("per-call", 10), ("resident", 0)])
def test_sizes_that_disagree_with_the_builds_nonblock_mode_are_refused(
    tmp_path, monkeypatch, nonblock, bytes_
):
    # Bug caught: a per-call build planned with the resident non-block bytes (every call over-estimated by 1.23
    # GiB on the published model, refused on a 32 GB Mac) or a resident build planned without them (the fit check
    # 1.23 GiB short). The checkpoint holds all seven non-block groups.
    from mlx_dfloat.errors import DFloatIntegrationError
    from mlx_dfloat.mflux._phases import FamilySizes

    sizes = FamilySizes(compressed=1_000, extras=100, nonblock=bytes_, encoders=1_000, vae=10)
    with pytest.raises(
        DFloatIntegrationError, match=f"{nonblock} non-block decode.*nonblock={bytes_}"
    ):
        fake_model(tmp_path, monkeypatch, nonblock=nonblock, sizes=sizes)


def test_the_report_lists_only_the_nonblock_groups_the_checkpoint_compresses(tmp_path, monkeypatch):
    # Bug caught: the report naming all seven non-block groups whatever the checkpoint holds, so a checkpoint
    # storing them as plain BF16 extras reads as if it compressed them.
    from tests._krea2_tiny import TINY

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.krea2.transformer import build_transformer, seam_transformer_class

    _m, _c, layout = write_tiny_checkpoint(
        tmp_path / "plain", np.random.default_rng(TINY_SEED), nonblock_as_groups=False
    )
    ckpt = open_checkpoint(tmp_path / "plain", layouts=(layout,))
    build = build_transformer(
        ckpt, transformer_kwargs=TINY, transformer_class=seam_transformer_class(per_call=True)
    )
    model = fake_model(tmp_path, monkeypatch, ckpt=ckpt, build=build)
    assert model.report()["nonblock_groups"] == []


def _weights_at_first_block(model, monkeypatch):
    """Run one uncompiled guided call and return every non-block weight as it stood when the first release ran (at
    block 0, before its weights are requested: the weights the pre-block graph read)."""
    from tests._krea2_tiny import tiny_inputs, tiny_predict

    from mlx_dfloat.mflux.krea2 import transformer as ktf

    seen = {}
    real = ktf.clear_nonblock

    def recording(module, shapes):
        if not seen:
            seen.update(
                {p: np.array(_param(model, p).view(mx.uint16)) for p in _nonblock_sources_names()}
            )
        real(module, shapes)

    monkeypatch.setattr(ktf, "clear_nonblock", recording)
    mx.eval(tiny_predict(model.transformer, tiny_inputs(), guidance=3.5))
    monkeypatch.setattr(ktf, "clear_nonblock", real)
    return seen


def _nonblock_sources_names():
    return [p for params in NONBLOCK.values() for p in params]


def test_the_model_decodes_the_nonblock_groups_per_call_by_default(tmp_path, monkeypatch):
    # Bug caught: the shipped model keeping the seven groups decoded with the set (the layout whose one-step
    # calibration run peaked 0.095 GiB over the 32 GB line on Krea 2 Raw), a group bound but not installed for the call (a placeholder on
    # tproj.linear is a zero-size matmul before the first block), a group's matrices on each other's parameters, or
    # the set attached without verify_in_call (a guided step's second call would be refused).
    model = fake_model(tmp_path, monkeypatch)
    model._lifecycle.ensure_set()
    assert model.transformer.nonblock_bound() is True
    assert _param(model, "tproj.linear.weight").size == 0  # nothing decoded at set load
    assert model.transformer.seam_cell.state.verify_in_call is True
    assert model.report()["nonblock"] == "per-call"
    seen = _weights_at_first_block(model, monkeypatch)
    sources = _nonblock_sources(tmp_path)
    assert len(sources) == 37
    for name, source in sources.items():
        assert np.array_equal(seen[name], source), name
    assert _param(model, "tproj.linear.weight").size == 0  # released after the call


def test_the_resident_set_load_installs_all_seven_nonblock_groups(tmp_path, monkeypatch):
    # Bug caught: on the resident build, a non-block group decoded but not installed, or installed on another
    # parameter.
    model = fake_model(tmp_path, monkeypatch, nonblock="resident")
    model._lifecycle.ensure_set()
    assert model.report()["nonblock"] == "resident"
    sources = _nonblock_sources(tmp_path)
    assert len(sources) == 37
    for name, source in sources.items():
        got = _param(model, name)
        assert np.array_equal(np.array(got.view(mx.uint16)), source), name
    assert model.transformer.seam_cell.state.verify_in_call is True


@pytest.mark.parametrize("nonblock", ["per-call", "resident"])
def test_a_new_prompt_after_a_generation_drops_the_set_and_reinstalls_the_nonblock_weights(
    tmp_path, monkeypatch, nonblock
):
    # Bug caught: a non-block weight left on its placeholder after the reload a new prompt forces
    # (tproj.linear before the first block, a refiner block's wo in the text fusion), the per-call groups not bound
    # again, or the encoder reload reading another base than the model was built from.
    model = fake_model(tmp_path, monkeypatch, nonblock=nonblock)
    loads = []
    monkeypatch.setattr(
        kinit,
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
    tproj, wo = "tproj.linear.weight", "txtfusion.refiner_blocks.1.attn.wo.weight"
    assert (_param(model, tproj).size, _param(model, wo).size) == (0, 0)
    if nonblock == "per-call":
        assert model.transformer.nonblock_bound() is False  # the drop forgot the compressed groups
    model._lifecycle.ensure_set()
    assert counters.set_loads == 2
    sources = _nonblock_sources(tmp_path)
    if nonblock == "per-call":
        seen = _weights_at_first_block(model, monkeypatch)
        for name in (tproj, wo):
            assert np.array_equal(seen[name], sources[name]), name
    else:
        assert (_param(model, tproj).shape, _param(model, wo).shape) == ((192, 32), (32, 32))
        for name in (tproj, wo):
            assert np.array_equal(np.array(_param(model, name).view(mx.uint16)), sources[name]), (
                name
            )


def test_embeddings_are_cached_per_prompt_and_mfluxs_cache_stays_empty(tmp_path, monkeypatch):
    # Bug caught: the embeddings put into mflux's own prompt_cache (keyed by (prompt, negative, guidance) there:
    # prompt_encoder.py:25, so ours would never match and mflux would call the dropped encoder), or one prompt encoded
    # for another. The stub tokenizer gives 1 id per character after its 5-id template prefix mflux strips.
    model = fake_model(tmp_path, monkeypatch)
    model.encode("a", " ", "abc")
    assert set(model._embeds) == {"a", " ", "abc"}
    assert model.prompt_cache == {}
    assert [model._embeds[p][0].shape for p in ("a", " ", "abc")] == [
        (1, 1, 384),
        (1, 1, 384),
        (1, 3, 384),
    ]
    a, blank = (np.array(model._embeds[p][0].view(mx.uint16)) for p in ("a", " "))
    assert not np.array_equal(a, blank)


def test_the_empty_prompt_is_refused_by_encode_as_it_gives_no_tokens(tmp_path, monkeypatch):
    # Bug caught: "" encoded (Krea 2's tokenizer gives it no ids at all, 2026-10-08: the transformer would get an
    # empty context), while " " (mflux's own blank negative) must still encode.
    model = fake_model(tmp_path, monkeypatch)
    with pytest.raises(DFloatUnsupportedError, match="empty prompt"):
        model.encode("a", "")
    assert model._embeds == {}
    model.encode(" ")
    assert set(model._embeds) == {" "}


def test_a_dropped_encoder_refuses_with_a_named_error(tmp_path, monkeypatch):
    # Bug caught: None left in the encoder's place (mflux would call it on a cache miss and fail with a TypeError far
    # from the cause).
    model = fake_model(tmp_path, monkeypatch)
    model.encode("a")
    with pytest.raises(DFloatIntegrationError, match="not encoded"):
        model.text_encoder.get_prompt_embeds(mx.ones((1, 4), dtype=mx.int32))
    with pytest.raises(DFloatIntegrationError, match="not encoded"):
        model.text_encoder(mx.ones((1, 4), dtype=mx.int32))


def test_a_failed_set_load_leaves_nothing_installed(tmp_path, monkeypatch):
    # Bug caught: a raise late in the set load (here at attach) leaving the decoded non-block weights installed and
    # the provider holding the compressed set while the lifecycle believes no set is resident; the retry's
    # active-memory check would then refuse to load a second copy.
    model = fake_model(tmp_path, monkeypatch)

    def broken_attach(*args, **kwargs):
        raise RuntimeError("attach")

    monkeypatch.setattr(model.transformer, "attach", broken_attach)
    with pytest.raises(RuntimeError, match="attach"):
        model._lifecycle.ensure_set()
    assert model._provider is None
    assert _param(model, "tproj.linear.weight").size == 0
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


def test_a_corrupt_nonblock_group_at_set_load_releases_the_set(tmp_path, monkeypatch):
    # Bug caught: a status failure while the set loads (the non-block groups decode once at the load) leaving the
    # compressed set reachable from the exception's frames (_install_set's `resident`) while the caller holds the
    # exception, so the retry refuses with "a previous set is still resident".
    from mlx_dfloat.errors import DFloatFormatError

    decode, state = _corrupting("tproj")
    model = fake_model(tmp_path, monkeypatch, decode=decode, nonblock="resident")
    model.encode("p")
    gc.collect()
    mx.clear_cache()
    before = int(mx.get_active_memory())
    with pytest.raises(DFloatFormatError, match="tproj") as info:
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


def test_a_corrupt_nonblock_group_under_per_call_decode_fails_the_call_and_drops_the_set(
    tmp_path, monkeypatch
):
    # Bug caught: a status failure in the per-call decode (at the call, not at set load) surfacing as anything but the
    # format error naming the group, leaving decoded weights installed, or keeping the set for a retry that would hit
    # the same corrupt group again without a clean reload.
    from mlx_dfloat.errors import DFloatFormatError

    decode, _state = _corrupting("tproj")
    model = fake_model(tmp_path, monkeypatch, decode=decode)
    with pytest.raises(DFloatFormatError, match="tproj"):
        model.generate_image(seed=1, prompt="p", num_inference_steps=1, height=64, width=64)
    assert not model._lifecycle.set_resident
    for name in _nonblock_sources_names():
        assert _param(model, name).size == 0, name


def test_the_retained_bound_counts_the_cached_embeddings_and_the_vae(tmp_path, monkeypatch):
    # Bug caught: the bound reading mflux's prompt_cache (empty: every drop after an encode would refuse) instead of
    # our per-prompt cache, where the embeddings live.
    model = fake_model(tmp_path, monkeypatch)
    empty = model._retained_bound()
    model.encode("abc")
    # One prompt: 3 rows of 384 bf16 = 2_304 B.
    assert model._retained_bound() - empty == 3 * 384 * 2


def test_save_model_is_refused_and_freeze_skips_a_dropped_encoder(tmp_path, monkeypatch):
    # Bug caught: freeze() calling .freeze() on the encoder stand-in (it is not a module), or save_model writing a
    # checkpoint with zero-size placeholders as the transformer.
    model = fake_model(tmp_path, monkeypatch)
    model.encode("p")  # drops the encoder
    model.freeze()
    with pytest.raises(DFloatUnsupportedError, match="save"):
        model.save_model(str(tmp_path))


def _innermost_locals(tb):
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_locals


def test_clear_frames_walks_the_whole_chain_of_causes_and_contexts():
    # Bug caught: only the raised exception and its __context__ cleared, so a __cause__ one link further (raise ... from
    # an earlier error) keeps its frame's locals, in production a block's decoded weights, alive while the caller holds
    # the exception; or a chain that loops back on itself never ending.
    from mlx_dfloat.mflux.krea2.model import _clear_frames

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


def _users_own_error():
    """An exception a caller is handling, raised in a finished frame that holds a local."""
    mine = "the caller's state"  # noqa: F841  # must survive our cleanup
    raise LookupError("the caller's own")


def test_clear_frames_stops_at_the_exception_the_caller_was_already_handling():
    # Bug caught: the walk following __context__ out of our call into the exception the caller was handling when
    # it called us, and clearing the caller's own frames (their locals gone from a traceback they still use).
    from mlx_dfloat.mflux.krea2.model import _clear_frames

    def ours():
        big = mx.zeros(1024)  # noqa: F841
        raise RuntimeError("ours")

    users = caught = None
    try:
        _users_own_error()
    except LookupError as exc:
        users = exc
        try:
            ours()
        except RuntimeError as inner:
            caught = inner
        _clear_frames(caught, outer=users)
    assert caught.__context__ is users
    assert "big" not in _innermost_locals(caught.__traceback__)
    assert _innermost_locals(users.__traceback__)["mine"] == "the caller's state"


def test_a_failed_generation_inside_a_callers_except_leaves_the_callers_frames(
    tmp_path, monkeypatch
):
    # Bug caught: generate_image's cleanup clearing the frames of the exception its caller was handling (the
    # failure's __context__), not only its own.
    from mlx_dfloat.errors import DFloatFormatError

    decode, _state = _corrupting("tproj")
    model = fake_model(tmp_path, monkeypatch, decode=decode)
    users = failure = None
    try:
        _users_own_error()
    except LookupError as exc:
        users = exc
        with pytest.raises(DFloatFormatError, match="tproj") as info:
            model.generate_image(seed=1, prompt="p", num_inference_steps=1, height=64, width=64)
        failure = info.value
    chain, link = [], failure
    while link is not None:
        chain.append(link)
        link = link.__context__
    assert users in chain
    assert _innermost_locals(users.__traceback__)["mine"] == "the caller's state"


# --- the generate prelude, CFG, the plan ------------------------------------------------------------------------

GIB = 1024**3


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def _patch_upstream_generate(monkeypatch, *, raise_after_loop=None, raise_before_loop=None):
    """Replace Krea2.generate_image with a probe: records the cache limit and MLX limits in force and the kwargs,
    fires our after-loop subscriber the way mflux's GenerationContext would, returns a sentinel."""
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2
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

    monkeypatch.setattr(Krea2, "generate_image", fake)
    return seen


@pytest.mark.parametrize(
    ("guidance", "negative", "expected"),
    [
        (1.0, "n", ("p",)),
        (3.5, "n", ("p", "n")),
        (0.5, None, ("p", " ")),
        (1.0001, "", ("p", " ")),
        (3.5, "   ", ("p", " ")),
        (3.5, " x ", ("p", " x ")),
    ],
)
def test_cfg_prompts_follow_mflux(guidance, negative, expected):
    # Bug caught: `> 1.0` copied from the other families (guidance 0.5 would skip mflux's negative branch: a different
    # image), or a blank negative not replaced by mflux's " ". Literal table from prompt_encoder.py:33-35.
    from mlx_dfloat.mflux.krea2.model import DFloatKrea2

    assert DFloatKrea2.cfg_prompts("p", negative_prompt=negative, guidance=guidance) == expected


@pytest.mark.parametrize(("guidance", "negative"), [(3.5, "n"), (1.0, "n"), (0.5, None)])
def test_the_override_returns_mfluxs_own_pair(tmp_path, monkeypatch, guidance, negative):
    # Bug caught: the copied rule drifting from mflux's (the override answering another pair than mflux's own
    # _encode_prompts would build), or the pair swapped. The oracle is mflux's method on a throwaway object.
    from types import SimpleNamespace

    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    model = fake_model(tmp_path, monkeypatch, tokenizer=StubTokenizer({"n": 12}))
    oracle = SimpleNamespace(
        prompt_cache={}, tokenizers=model.tokenizers, text_encoder=StubTextEncoder()
    )
    want = Krea2._encode_prompts(oracle, prompt="p", negative_prompt=negative, guidance=guidance)
    model.encode(*model.cfg_prompts("p", negative_prompt=negative, guidance=guidance))
    got = model._encode_prompts(prompt="p", negative_prompt=negative, guidance=guidance)
    assert np.array_equal(np.array(got[0].view(mx.uint16)), np.array(want[0].view(mx.uint16)))
    if guidance == 1.0:
        assert got[1] is None
        assert want[1] is None
    else:
        assert np.array_equal(np.array(got[1].view(mx.uint16)), np.array(want[1].view(mx.uint16)))


def test_a_prompt_not_encoded_first_is_refused_by_the_override(tmp_path, monkeypatch):
    # Bug caught: a cache miss falling through to mflux's encoder (dropped: a refusal deep in the loop, or an encoder
    # reload next to the set).
    model = fake_model(tmp_path, monkeypatch)
    model.encode("p")
    with pytest.raises(DFloatIntegrationError, match=r"\[' '\] were not encoded"):
        model._encode_prompts(prompt="p", negative_prompt=None, guidance=3.5)


def test_an_unknown_scheduler_is_refused_before_anything_loads(tmp_path, monkeypatch):
    # Bug caught: mflux raising only inside its own generate_image, after the prelude spent the
    # encoder and the set load (about 45 s on the real model); or "linear" refused although mflux maps it to er_sde.
    from mlx_dfloat.errors import DFloatUnsupportedError as Unsupported

    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(Unsupported, match=r"'ddim'.*er_sde, euler, linear"):
        model.generate_image(seed=1, prompt="p", height=64, width=64, scheduler="ddim")
    counters = model._lifecycle.counters
    assert (counters.encoder_loads, counters.set_loads) == (0, 0)
    monkeypatch.undo()
    model = fake_model(tmp_path / "again", monkeypatch)
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, scheduler="linear")
    assert seen["kwargs"]["scheduler"] == "linear"


@pytest.mark.parametrize(
    ("model_name", "kwargs", "want"),
    [
        # mflux: Raw has no step-table entry, so the fallback 25 (defaults.py:19,111-118); guidance 1.0 (krea2.py:58).
        ("krea-2-raw", {}, (1.0, 25, "er_sde", 1)),
        # mflux: Turbo 8 steps (defaults.py:47), guidance 1.0 (krea2_generate.py:17).
        ("krea-2", {}, (1.0, 8, "er_sde", 1)),
        # The model card's recipe for Raw passes through: two calls per step.
        ("krea-2-raw", {"num_inference_steps": 52, "guidance": 3.5}, (3.5, 52, "er_sde", 2)),
        ("krea-2", {"guidance": 2.0}, (2.0, 8, "er_sde", 2)),
    ],
)
def test_the_python_api_defaults_follow_the_registry_per_model(
    tmp_path, monkeypatch, model_name, kwargs, want
):
    # Bug caught: Raw's entry still carrying the card's 52 / 3.5 (a CFG run nobody asked for, about 4x the time), the
    # per-model lookup taking Turbo's 8 for Raw, or the scheduler left None for mflux (it resolves None itself, but the
    # run would not record it).
    model = fake_model(tmp_path, monkeypatch, model=model_name)
    seen = _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, **kwargs)
    got = seen["kwargs"]
    assert (got["guidance"], got["num_inference_steps"], got["scheduler"]) == want[:3]
    assert model.report()["cfg_calls_per_step"] == want[3]


def test_the_generate_signature_leaves_the_recipe_to_the_registry():
    # Bug caught: a default baked into the signature (mflux's class default of 8 steps would then run Raw at 8).
    from mlx_dfloat.mflux.krea2.model import DFloatKrea2

    defaults = {
        n: p.default
        for n, p in inspect.signature(DFloatKrea2.generate_image).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert defaults == {
        "num_inference_steps": None,
        "height": 1024,
        "width": 1024,
        "guidance": None,
        "negative_prompt": None,
        "image_path": None,
        "image_strength": None,
        "scheduler": None,
        "pid_decode": False,
        "pid_degrade_sigma": 0.0,
    }


@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_empty_or_blank_prompt_is_refused_before_anything_loads(tmp_path, monkeypatch, prompt):
    # Bug caught: "" sent on (Krea 2's tokenizer gives it no ids: the transformer gets an empty context), or a blank
    # prompt treated as a real one; either way after an encoder load.
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match="empty or blank prompt"):
        model.generate_image(seed=1, prompt=prompt, height=64, width=64)


@pytest.mark.parametrize(
    "kwargs", [{"image_path": "in.png"}, {"image_strength": 0.5}, {"pid_decode": True}]
)
def test_img2img_and_pid_decode_are_refused_before_anything_loads(tmp_path, monkeypatch, kwargs):
    # Bug caught: encoding the prompt (an encoder load) or loading the set before refusing img2img or mflux's PiD
    # decoder (which loads its own caption encoder next to the set).
    model = fake_model(tmp_path, monkeypatch)
    monkeypatch.setattr(model._lifecycle, "ensure_embeddings", lambda *p: pytest.fail("encoded"))
    with pytest.raises(DFloatUnsupportedError, match=next(iter(kwargs))):
        model.generate_image(seed=1, prompt="p", **kwargs)
    assert model._lifecycle.counters.set_loads == 0


def test_text_tokens_follow_the_longer_prompt_of_the_call(tmp_path, monkeypatch):
    # Bug caught: the plan sized by the prompt only (a longer negative prompt's activations unplanned), or the
    # template prefix counted. Stub lengths: the prompt 7 ids, the negative 12, after the 5-id prefix mflux strips.
    model = fake_model(tmp_path, monkeypatch, tokenizer=StubTokenizer({"p": 7, "n": 12}))
    planned = []
    real = model.plan_call
    monkeypatch.setattr(
        model, "plan_call", lambda **kw: (planned.append(kw["text_tokens"]), real(**kw))[1]
    )
    _patch_upstream_generate(monkeypatch)
    model.generate_image(seed=1, prompt="p", height=64, width=64, guidance=3.5, negative_prompt="n")
    assert planned == [12]
    assert model.report()["text_tokens"] == 12


def test_plan_call_reads_the_familys_sizes_and_the_one_block_kind(tmp_path, monkeypatch):
    # Bug caught: plan_call not wired to the shared planner with this model's own largest block group (the cache limit
    # would miss the decoded group), the non-block groups counted as the largest block, or the plan not kept for the
    # VAE guard. Injected constants (literals): allowance 3e9 at 4096 + 30 tokens.
    from mlx_dfloat.mflux._phases import PhaseConstants

    model = fake_model(tmp_path, monkeypatch)
    model._constants = PhaseConstants(
        overhead_bytes=1,
        vae_transient_bytes=1,
        denoise_activation_at_reference=1,
        reference_tokens=4096 + 30,
        allowance_at_reference=3_000_000_000,
        allowance_reference_tokens=4096 + 30,
    )
    plan = model.plan_call(height=1030, width=1030, text_tokens=30)  # rounded to 1024^2 first
    # The tiny block group: 1024 + 512 + 512 + 1024 + 1024 + 3 x 4096 = 16_384 elements, 32_768 B (TINY's shapes).
    assert model._largest == {"blocks": 32_768}
    assert plan.cache_limit == 32_768 + 3_000_000_000
    assert model._plan is plan


def test_a_dropped_set_vae_phase_keeps_the_extras(tmp_path, monkeypatch):
    # Bug caught: the VAE phase planned after the set drop leaving out the extras, which stay loaded (they are
    # the transformer's own BF16 parameters, not part of the set): the fit estimate would under-predict the VAE phase.
    # Literals: VAE phase with the set 12e9 + 777 + 100 + 11e9 + 400 > the 16e9 budget (per-call: no resident
    # non-block bytes), so the set drops; without it the phase is 777 + 100 + 11e9 + 400.
    from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants

    sizes = FamilySizes(compressed=12_000_000_000, extras=777, nonblock=0, encoders=0, vae=100)
    model = fake_model(tmp_path, monkeypatch, sizes=sizes, budget_bytes=16_000_000_000)
    model._constants = PhaseConstants(
        overhead_bytes=400,
        vae_transient_bytes=11_000_000_000,
        denoise_activation_at_reference=0,
        reference_tokens=4096,
        allowance_at_reference=0,
        allowance_reference_tokens=4096,
        allowance_floor=0,
    )
    plan = model.plan_call(height=1024, width=1024, text_tokens=0)
    assert plan.drop_set_before_vae is True
    assert plan.estimate.phases["vae"] == 777 + 100 + 11_000_000_000 + 400


def test_the_vae_guard_follows_the_plan_and_zeroes_the_cache_limit(tmp_path, monkeypatch):
    # Bug caught: the guard not reading this call's plan (the set kept through a decode the budget cannot hold), or a
    # transformer-sized cache limit left for the VAE decode.
    # Injected constants (literals): the VAE phase with the set is 12e9 + 11e9 + 400 > 16e9, without it 11e9 + 400;
    # the denoise phase stays near 12e9 (no allowance, no activation term).
    from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants

    constants = PhaseConstants(
        overhead_bytes=400,
        vae_transient_bytes=11_000_000_000,
        denoise_activation_at_reference=0,
        reference_tokens=4096,
        allowance_at_reference=0,
        allowance_reference_tokens=4096,
        allowance_floor=0,
    )
    big = FamilySizes(compressed=12_000_000_000, extras=0, nonblock=0, encoders=0, vae=0)
    tight = fake_model(tmp_path / "t", monkeypatch, sizes=big, budget_bytes=16_000_000_000)
    tight._constants = constants
    seen = _patch_upstream_generate(monkeypatch)
    assert tight.generate_image(seed=1, prompt="p", height=1024, width=1024) == "image"
    assert seen["set_resident"]
    assert seen["limit_at_entry"] == tight._plan.cache_limit
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is False
    roomy = fake_model(tmp_path / "r", monkeypatch, sizes=big, budget_bytes=40 * GIB)
    roomy._constants = constants
    seen = _patch_upstream_generate(monkeypatch)
    roomy.generate_image(seed=1, prompt="p", height=1024, width=1024)
    assert seen["limit_after_loop"] == 0
    assert seen["set_resident_after_loop"] is True


def test_the_prelude_encodes_then_drops_the_encoder_then_loads_the_set_then_runs_mflux(
    tmp_path, monkeypatch
):
    # Bug caught: the set loaded while the encoder is resident, mflux's loop entered before the embeddings exist, or
    # the negative prompt not passed through (mflux reads it for the image metadata).
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

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
    upstream = Krea2.generate_image

    def recording(self, **kwargs):
        events.append("upstream")
        return upstream(self, **kwargs)

    monkeypatch.setattr(Krea2, "generate_image", recording)
    image = model.generate_image(
        seed=7,
        prompt="p",
        num_inference_steps=1,
        height=256,
        width=256,
        guidance=3.5,
        negative_prompt="n",
    )
    assert image == "image"
    assert events == ["_encode:'p'", "_encode:'n'", "_unload_encoders", "_load_set", "upstream"]
    kwargs = seen["kwargs"]
    assert (kwargs["image_path"], kwargs["image_strength"], kwargs["pid_decode"]) == (
        None,
        None,
        False,
    )
    assert (kwargs["guidance"], kwargs["negative_prompt"], kwargs["num_inference_steps"]) == (
        3.5,
        "n",
        1,
    )


@pytest.mark.parametrize("where", ["raise_before_loop", "raise_after_loop"])
def test_a_format_error_mid_step_drops_the_set_and_restores_the_limit(tmp_path, monkeypatch, where):
    # Bug caught: a corrupt block's retry reusing stale resident state, or the process cache limit left at the call's.
    from mlx_dfloat.errors import DFloatFormatError

    model = fake_model(tmp_path, monkeypatch)
    before = _cache_limit_in_force()
    _patch_upstream_generate(monkeypatch, **{where: DFloatFormatError("block 1: invalid code")})
    with pytest.raises(DFloatFormatError):
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert not model._lifecycle.set_resident
    assert _cache_limit_in_force() == before
    assert _param(model, "tproj.linear.weight").size == 0
    assert model.open_phase is None


def test_keyboard_interrupt_restores_the_cache_limit_and_leaves_no_pending_words(
    tmp_path, monkeypatch
):
    # Bug caught: mflux's StopImageGenerationException (raised for a Ctrl-C in its loop, krea2.py:103-107) leaving
    # status words pending (begin_step would refuse the next call), the cache limit not restored, or the interrupt
    # dropping the set (a reload per Ctrl-C). Real mflux loop; the interrupt fires at the second block of the step's
    # second transformer call (CFG: 2 blocks per call).
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
        "guidance": 3.5,
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


@pytest.mark.parametrize("chained", [False, True])
def test_any_exception_from_the_loop_releases_its_frames_locals(tmp_path, monkeypatch, chained):
    # Bug caught: only DFloatFormatError clearing the traceback's frames, so a RuntimeError (or an error raised while
    # handling another) keeps the raising frame's locals, in production the seam's decoded weights, alive for as long
    # as the caller holds the exception.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    model = fake_model(tmp_path, monkeypatch)

    def failing_step():
        big = mx.zeros(1024)  # noqa: F841  # stands in for a block's decoded weights
        raise ValueError("inner")

    def fake(self, **kwargs):
        if not chained:
            failing_step()
        try:
            failing_step()
        except ValueError as exc:
            raise RuntimeError("outer") from exc

    monkeypatch.setattr(Krea2, "generate_image", fake)
    with pytest.raises((ValueError, RuntimeError)) as info:
        model.generate_image(seed=1, prompt="p", height=256, width=256)
    exc = info.value
    assert "big" not in _innermost_locals(exc.__traceback__)
    if chained:
        assert isinstance(exc.__cause__, ValueError)
        assert "big" not in _innermost_locals(exc.__cause__.__traceback__)
    assert model._lifecycle.set_resident  # only a format error drops the set


def test_open_phase_names_the_phase_running_now_and_none_outside_a_call(tmp_path, monkeypatch):
    # Bug caught: the watchdog's abort context naming no phase, or a stale one.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

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

    monkeypatch.setattr(Krea2, "generate_image", fake)
    model.generate_image(seed=1, prompt="p", height=256, width=256)
    assert seen == ["encode", "set_load", "denoise", "vae"]
    assert model.open_phase is None


def _fake_encode_peak(monkeypatch, model, above_bound):
    """Make the tracker record an encode whose MLX hold is ``above_bound`` bytes over the call's warning bound."""
    from mlx_dfloat.mflux.krea2 import memory as kmem

    real_end = model._tracker.end

    def end(name):
        real_end(name)
        if name == "encode":
            record = model._tracker.peaks["encode"]
            bound = kmem.encode_peak_bound(model._plan.estimate.phases["encode"], model._constants)
            record["mlx_peak"] = record["active_at_start"] + bound + above_bound

    monkeypatch.setattr(model._tracker, "end", end)


def test_an_encode_over_the_bound_warns_through_the_pure_helper(tmp_path, monkeypatch):
    # Bug caught: the encode hold never compared with the bound (an mflux change that loads the vision tower, +0.77
    # GiB on the real file, going unnoticed), `>=` for `>` at the bound, or the warning missing the call's prompt
    # length. The bound itself is pinned by the helper's own tests; this pins the wiring. The stub gives "p" 1 token.
    model = fake_model(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, 0)
    assert (
        model.generate_image(seed=1, prompt="p", height=256, width=256) == "image"
    )  # at the bound: silent
    model = fake_model(tmp_path / "again", monkeypatch)
    _patch_upstream_generate(monkeypatch)
    _fake_encode_peak(monkeypatch, model, 1)
    with pytest.warns(UserWarning, match=r"for a 1-token prompt.*vision tower"):
        model.generate_image(seed=1, prompt="p", height=256, width=256)


def test_the_budget_is_the_constructors_else_the_devices_read_at_each_call(tmp_path, monkeypatch):
    # Bug caught: the constructor's budget ignored (a --tier run planned against the whole Mac), or the device's read
    # once at assembly instead of at each call.
    from mlx_dfloat.mflux.krea2 import model as model_module

    assert fake_model(tmp_path / "a", monkeypatch, budget_bytes=9 * GIB)._budget() == 9 * GIB
    model = fake_model(tmp_path / "b", monkeypatch)
    assert model._budget() == 23 * GIB  # fake_model's fixed device budget
    monkeypatch.setattr(model_module, "budget_bytes", lambda: 10 * GIB)
    assert model._budget() == 10 * GIB


# --- the report -----------------------------------------------------------------------------------------------


def test_the_report_names_the_family_the_calls_the_layout_and_the_uncompiled_predict(
    tmp_path, monkeypatch
):
    # Bug caught: the family, the call count, the predict mode or the checkpoint's layout missing from the report (the
    # CLI and the bench table read them), a non-block group missing from the list, or the estimate labelled as a
    # measurement.
    model = fake_model(tmp_path, monkeypatch)
    report = model.report()
    assert (report["family"], report["model"]) == ("krea2", "krea-2")
    assert (
        report["cfg_calls_per_step"],
        report["text_tokens"],
        report["predict"],
        report["fit"],
    ) == (
        1,
        None,
        None,
        None,
    )
    assert report["nonblock_groups"] == [
        "tmlp",
        "tproj",
        "txtfusion.layerwise_blocks.0",
        "txtfusion.layerwise_blocks.1",
        "txtfusion.refiner_blocks.0",
        "txtfusion.refiner_blocks.1",
        "txtmlp",
    ]
    assert report["layout"].startswith("header and spot checks match layout tiny-krea")
    assert report["nonblock"] == "per-call"
    assert report["sizes"] == {
        "compressed": 1_000,
        "extras": 100,
        "nonblock": 0,
        "encoders": 1_000,
        "vae": 10,
    }
    assert report["eval_policy_status"] == "measured"
    model.generate_image(
        seed=1,
        prompt="a",
        num_inference_steps=1,
        height=64,
        width=64,
        guidance=3.5,
        negative_prompt="n",
    )
    report = model.report()
    assert (report["cfg_calls_per_step"], report["text_tokens"], report["predict"]) == (
        2,
        1,
        "uncompiled",
    )
    assert report["fit"]["label"] == "predicted"
    assert report["decode_launches"] == 1 * 2 * 2


def test_the_report_labels_depth2_unmeasured(tmp_path, monkeypatch):
    # Bug caught: a depth2 run reported like the measured per-block path.
    report = fake_model(tmp_path, monkeypatch, eval_policy="depth2").report()
    assert (report["eval_policy"], report["eval_policy_status"]) == ("depth2", "unmeasured")


# --- the constructor, end to end past the resolver ------------------------------------------------------------


def test_the_constructor_builds_from_local_dirs_with_the_variant_check_and_the_language_model_bytes(
    tmp_path, monkeypatch
):
    # Bug caught: __init__ wiring past the resolver untested: the config-less checkpoint not opened through
    # the layout table, the build without the variant check (the model name not passed), the base components read
    # from another directory, or the sizes taken from the encoder file (vision tower and last layer included) instead
    # of the evaluated language model.
    from tests._krea2_tiny import TINY, stub_components

    from mlx_dfloat import _layouts, _metal_decode
    from mlx_dfloat.mflux.krea2 import model as model_module

    _m, _c, layout = write_tiny_checkpoint(tmp_path / "df11", np.random.default_rng(TINY_SEED))
    monkeypatch.setattr(_layouts, "KNOWN_LAYOUTS", (layout,))
    base = tmp_path / "base"
    (base / "text_encoder").mkdir(parents=True)
    # The language model: embed_tokens (10, 4) 80 B, two layers' up_proj (4, 4) 32 B each, final norm (4,) 8 B; the
    # last layer never runs. Never loaded: a vision tensor (8, 4) 64 B. 80 + 32 + 8 = 120.
    mx.save_safetensors(
        str(base / "text_encoder" / "model.safetensors"),
        {
            "language_model.embed_tokens.weight": mx.zeros((10, 4), dtype=mx.bfloat16),
            "language_model.layers.0.mlp.up_proj.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "language_model.layers.1.mlp.up_proj.weight": mx.zeros((4, 4), dtype=mx.bfloat16),
            "language_model.norm.weight": mx.zeros((4,), dtype=mx.bfloat16),
            "visual.blocks.0.attn.qkv.weight": mx.zeros((8, 4), dtype=mx.bfloat16),
        },
    )
    (base / "vae").mkdir()
    (base / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 100)

    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    loaded = []
    monkeypatch.setattr(
        kinit, "load_base", lambda root: (loaded.append(root), stub_components())[1]
    )
    built = []
    real_build = model_module.build_transformer

    def tiny_build(ckpt, **kwargs):
        built.append(kwargs)
        return real_build(ckpt, transformer_kwargs=TINY, **kwargs)

    from mlx_dfloat.mflux.krea2.transformer import seam_transformer_class

    monkeypatch.setattr(model_module, "build_transformer", tiny_build)
    model = model_module.DFloatKrea2(
        "krea-2-raw", df11_path=str(tmp_path / "df11"), base_path=str(base)
    )
    # The per-call class: the shipped build.
    assert built == [
        {"model": "krea-2-raw", "transformer_class": seam_transformer_class(per_call=True)}
    ]
    assert loaded == [base]
    report = model.report()
    assert report["sizes"]["encoders"] == 120
    assert report["sizes"]["vae"] == 100
    # Per-call decode: the seven non-block groups are not kept decoded with the set.
    assert report["sizes"]["nonblock"] == 0
    assert report["nonblock"] == "per-call"
    assert report["layout"].startswith("header and spot checks match layout tiny-krea")


# --- per-call caps ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raised", [None, RuntimeError("in the loop")], ids=["returns", "raises"])
def test_a_python_api_call_runs_under_the_commands_caps_and_restores_mlxs_defaults(
    tmp_path, monkeypatch, raised
):
    # Bug caught: generate_image not run under the per-call caps (a Python-API process sits at MLX's default wired
    # limit 0, while every VAE term is measured under the command's caps), or the caps left installed after a call
    # that returns or raises.
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


def test_encode_runs_under_the_commands_caps_and_restores_mlxs_defaults(tmp_path, monkeypatch):
    # Bug caught: encode() from Python run at MLX's default wired limit 0 (a prompt encode is one of the measured
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


# --- the tiny generation against stock mflux ----------------------------------------------------------------


@pytest.mark.parametrize("nonblock", ["resident", "per-call"])
@pytest.mark.parametrize(
    ("guidance", "negative"), [(1.0, None), (3.5, "n")], ids=["guidance-1", "cfg"]
)
def test_the_tiny_generation_matches_stock_krea2_latents(
    tmp_path, monkeypatch, guidance, negative, nonblock
):
    # Bug caught: anything between the checkpoint and mflux's loop changing the result: (embeds, neg_embeds) swapped by
    # the _encode_prompts override (the negative is 12 tokens, the prompt 1, so a swap changes the latents), a decoded
    # matrix on the wrong parameter, a non-block group or rename lost, an extra mis-routed, the CFG formula or the
    # er_sde seeded noise stream not mflux's. The reference is stock mflux over the same BF16 weights; mflux's predict
    # factory returns the plain closure when the chip check says M1/M2, so it is patched True for both sides (ours
    # bypasses compile on any chip). Equality is bit for bit on the float32 latents; the extras are random because
    # constant ones blind the RMSNorm-fed forward. The per-call build (the shipped one) releases the non-block weights
    # and clears MLX's cache before block 0 of every call: the latents must not move.
    from mflux.utils.apple_silicon import AppleSiliconUtil
    from tests._krea2_tiny import TINY, StubVAE, stock_krea2, stub_components

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.krea2.transformer import build_transformer, seam_transformer_class

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    rng = np.random.default_rng(11)
    matrices, extras, layout = write_tiny_checkpoint(tmp_path / "random", rng, random_extras=rng)
    ckpt = open_checkpoint(tmp_path / "random", layouts=(layout,))
    tokenizer = StubTokenizer({"n": 12})
    ours_vae, stock_vae = StubVAE(), StubVAE()
    components = stub_components(tokenizer)
    components.vae, components.text_encoder = ours_vae, StubTextEncoder()
    model = fake_model(
        tmp_path,
        monkeypatch,
        model="krea-2-raw",
        nonblock=nonblock,
        ckpt=ckpt,
        build=build_transformer(
            ckpt,
            transformer_kwargs=TINY,
            transformer_class=seam_transformer_class(per_call=nonblock == "per-call"),
        ),
        components=components,
    )
    stock = stock_krea2(tokenizer, StubTextEncoder(), stock_vae, matrices, extras)
    kwargs = {
        "seed": 3,
        "prompt": "p",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": guidance,
        "negative_prompt": negative,
        "scheduler": "er_sde",
    }
    model.generate_image(**kwargs)
    stock.generate_image(**kwargs)
    (ours,), (theirs,) = ours_vae.seen, stock_vae.seen
    assert model.report()["cfg_calls_per_step"] == (2 if negative else 1)
    assert model.report()["predict"] == "uncompiled"
    assert bool(mx.all(mx.isfinite(theirs)))
    assert ours.dtype == theirs.dtype == mx.float32
    assert ours.shape == theirs.shape == (1, 16, 8, 8)
    assert np.array_equal(np.array(ours.view(mx.uint32)), np.array(theirs.view(mx.uint32)))
