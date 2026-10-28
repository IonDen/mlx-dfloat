"""Our DF11 Qwen-Image 2.1 transformer against mflux 0.23's own default generation path, bit for bit. `mflux` lane.

mflux 0.23's default for a padding-free prompt inside ``inference_projections()``: packed Q/K/V and gate/up, the
fused QK-norm/RoPE kernel, the text-prefix K/V cache (``text_forward`` once per prompt) and the image-token step
(``image_forward``). Our side decodes each block's DF11 group just in time, packs it per block and runs both passes
uncompiled. The reference is stock mflux under ``mx.disable_compile()``: mflux compiles both step functions, and the
compiled bits differ from the same graph run plain (up to about 80 % of the elements by an ulp or more), so the
identity bar is the plain graph, as for every other family. Both sides of a comparison run under
``mflux_uncompiled()``: MLX's own activations are compiled too (``nn.silu`` is ``mx.compile(shapeless=True)``), and
for some weights the compiled bits differ from the plain body's. That a seamed transformer keeps mflux's step
functions plain is checked on its own below. Same weights on both sides; the comparison is ``view(uint16)``.
"""

import types

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from tests._qwen21_tiny import WIDE, mflux_uncompiled, tiny_config, tiny_groups, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.providers import DF11Provider

pytestmark = pytest.mark.mflux

COMPILED = type(mx.compile(lambda x: x))  # mlx.gc_func
STEP_METHODS = {"_step_fn": "_forward", "_image_step_fn": "_image_forward"}


def _inputs(*, text=6, padded=False, t=0, seed=0):
    """``Qwen21Transformer.__call__`` inputs at WIDE: 16 image tokens of 64 channels, ``text`` prompt tokens."""
    keys = mx.random.split(mx.random.key(seed), 2)
    mask = mx.ones((1, text), dtype=mx.int32)
    if padded:
        mask = mx.concatenate([mask[:, :-1], mx.zeros((1, 1), dtype=mx.int32)], axis=1)
    return {
        "t": t,
        "config": tiny_config(steps=2),
        "hidden_states": mx.random.normal((1, 16, 64), key=keys[0]).astype(mx.bfloat16),
        "encoder_hidden_states": mx.random.normal((1, text, 32), key=keys[1]).astype(mx.bfloat16),
        "encoder_hidden_states_mask": mask,
    }


def _pair(decoded=None):
    """(ours, native): our seamed WIDE transformer on a DF11 provider, and stock mflux holding the same weights.

    The non-block parameters are cast to bfloat16, as a checkpoint's extras load (the computed RoPE tables and
    timestep frequencies are not parameters in mflux 0.23 and keep mflux's own values on both sides); the block
    matrices are the DF11 groups' source bits.

    The non-block parameters come from MLX's global generator, seeded here so a test does not depend on the tests run
    before it. Seed 57 (mlx 0.32.2) gives weights where MLX's compiled ``nn.silu`` and its plain body differ in the
    last bit, so a comparison that runs one side compiled and the other plain fails on it.
    """
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    mx.random.seed(57)
    ours, shapes = tiny_seamed(n_layers=1, dims=WIDE)
    groups, names, sources = tiny_groups(shapes, np.random.default_rng(7))
    native = Qwen21Transformer(**WIDE)
    matrices = {m for per in names.values() for m in per}
    shared = [
        (k, v.astype(mx.bfloat16))
        for k, v in tree_flatten(native.parameters())
        if k not in matrices
    ]
    native.update(tree_unflatten(shared))
    ours.update(tree_unflatten(shared))
    native.update(
        tree_unflatten(
            [
                (m, mx.array(bits).view(mx.bfloat16))
                for per in sources.values()
                for m, bits in per.items()
            ]
        )
    )

    def decode(group):
        if decoded is not None:
            decoded.append(group.name)
        return decode_group(group, backend="reference")

    ours.attach(
        DF11Provider(groups, names, qwen21_name_map(), decode=decode), shapes, verify_in_call=True
    )
    return ours, native


def _bits(x):
    mx.eval(x)
    return x.view(mx.uint16).tolist()


def test_two_steps_match_mflux_s_default_packed_cached_path_uncompiled_bit_for_bit():
    # Bugs caught: the seam leaving text_forward or image_forward unseamed (a placeholder matmul raises), running the
    # separate projections (test_packing_changes_the_bits_at_this_shape shows they differ here), packing outside
    # the window or keeping a packed matrix across blocks, or a step reusing the wrong text cache. The precondition
    # lines catch a WIDE that no longer reaches the fused kernel.
    ours, native = _pair()
    assert native.transformer_blocks[0].attn.use_fused_prologue
    assert ours.transformer_blocks[0].attn.use_fused_prologue
    # One prompt array for both steps: mflux keys its text cache on the array's identity, as its loop passes it.
    steps = [_inputs(t=t) for t in (0, 1)]
    for step in steps[1:]:
        step["encoder_hidden_states"] = steps[0]["encoder_hidden_states"]
    with mflux_uncompiled(), native.inference_projections():
        want = [_bits(native(**step)) for step in steps]
    with mflux_uncompiled(), ours.inference_projections():
        got = [_bits(ours(**step)) for step in steps]
    assert got == want


def test_packing_changes_the_bits_at_this_shape():
    # Bug caught (the control): a WIDE where mflux's packed and separate projections agree, so the test above could
    # not tell a seam that skips the packing. Stock mflux only, uncompiled as the reference above, with
    # use_packed_projections off on one side.
    _ours, native = _pair()
    inputs = _inputs()
    with mflux_uncompiled():
        with native.inference_projections():
            packed = _bits(native(**inputs))
        native.use_packed_projections = False
        with native.inference_projections():
            separate = _bits(native(**inputs))
    assert packed != separate


def test_mflux_compiles_both_step_functions_and_ours_runs_them_plain():
    # Bug caught: the parity reference above made uncompiled while mflux itself also stopped compiling (the reference
    # would then be the only mflux path), or the seamed transformer letting mflux compile either pass. mflux 0.23
    # compiles on every call path (qwen21_transformer.py:96-98 joint, 111-113 image step), with no chip check.
    for padded, attr in ((True, "_step_fn"), (False, "_image_step_fn")):
        ours, native = (
            _pair()
        )  # one pair per path: mflux caches the geometry (and its mask) per shape
        inputs = _inputs(padded=padded)
        with native.inference_projections():
            mx.eval(native(**inputs))
            assert type(getattr(native, attr)) is COMPILED, attr
        with ours.inference_projections():
            mx.eval(ours(**inputs))
            fn = getattr(ours, attr)
            assert type(fn) is types.MethodType, attr
            assert fn.__func__ is getattr(type(native), STEP_METHODS[attr]), attr


def test_a_padded_prompt_takes_the_joint_pass_and_matches_mflux():
    # Bug caught: the block's own __call__ (the joint pass mflux runs for a padded prompt, qwen21_transformer.py:92-99)
    # unseamed or unpacked. _step_fn left compiled is test_mflux_compiles_both_step_functions_and_ours_runs_them_plain's.
    ours, native = _pair()
    inputs = _inputs(padded=True)
    with mflux_uncompiled(), native.inference_projections():
        want = _bits(native(**inputs))
    with mflux_uncompiled(), ours.inference_projections():
        got = _bits(ours(**inputs))
    assert got == want


def test_the_text_prefix_decodes_each_block_once_per_prompt():
    # Bug caught: the text pass rebuilt every step (a second decode per block per step) or never built (mflux's
    # cache keyed off another array). Two steps on one prompt: text pass + 2 image passes = 3 decodes of the block.
    decoded = []
    ours, _native = _pair(decoded)
    inputs = _inputs()
    with ours.inference_projections():
        for t in (0, 1):
            mx.eval(ours(**{**inputs, "t": t}))
    assert decoded == ["transformer_blocks.0"] * 3


@pytest.mark.parametrize("attr", ["_step_fn", "_image_step_fn"])
def test_a_compiled_step_function_is_refused_before_any_decode(attr):
    # Bug caught: the guard checking only _step_fn (0.20's one compiled function): a compiled _image_step_fn runs the
    # seam inside mx.compile, where the per-block eval raises.
    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    decoded = []
    ours, _native = _pair(decoded)
    method = "_forward" if attr == "_step_fn" else "_image_forward"
    setattr(ours, attr, mx.compile(getattr(ours, method)))
    assert not is_eager(ours)
    with pytest.raises(DFloatIntegrationError, match=rf"{attr} is .*compiled the forward pass"):
        ours(**_inputs())
    assert decoded == []


@pytest.mark.parametrize("attr", ["_step_fn", "_image_step_fn"])
def test_a_compiled_step_function_set_before_the_window_is_refused_on_entry(attr):
    # Bug caught: inference_projections() clearing a compiled step function before the guard looks (mflux 0.23's
    # _clear_inference_projections sets both to None), so a compile planted outside the window goes unreported by
    # the call while report() says "compiled"; the refusal must also come before any decode.
    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    decoded = []
    ours, _native = _pair(decoded)
    compiled = mx.compile(getattr(ours, STEP_METHODS[attr]))
    setattr(ours, attr, compiled)
    with (
        pytest.raises(DFloatIntegrationError, match=rf"{attr} is .*compiled the forward pass"),
        ours.inference_projections(),
    ):
        ours(**_inputs())
    assert decoded == []
    assert getattr(ours, attr) is compiled
    assert not is_eager(ours)
    assert ours.pack_switch.on is False


def test_the_seamed_transformer_runs_both_passes_plain():
    # Bug caught: mflux's compile reaching through the seamed class on the default (text-cache) path.
    from mlx_dfloat.mflux.qwen21.transformer import is_eager

    ours, _native = _pair()
    with ours.inference_projections():
        mx.eval(ours(**_inputs()))
        assert type(ours._image_step_fn) is types.MethodType
        assert is_eager(ours)


def test_the_generate_call_refuses_step_reuse(tmp_path, monkeypatch):
    # Bug caught: step_cache_ratio / teacache_ratio passed through (#779): mflux would skip transformer calls, an
    # image no identity check here has covered, or the refusal coming after the prompts were encoded.
    from tests._qwen21_tiny import fake_model

    from mlx_dfloat.errors import DFloatUnsupportedError

    model = fake_model(tmp_path, monkeypatch)
    for kwargs in ({"step_cache_ratio": 0.25}, {"teacache_ratio": 0.25}):
        with pytest.raises(DFloatUnsupportedError, match="step_cache_ratio/teacache_ratio"):
            model.generate_image(
                seed=1, prompt="a", num_inference_steps=2, height=64, width=64, **kwargs
            )
    assert model.prompt_cache == {}


def test_the_constructor_refuses_a_compute_precision():
    # Bug caught: compute_precision (#820) accepted: mflux casts the attention and FFN modules, another image.
    from mlx_dfloat.errors import DFloatUnsupportedError
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    with pytest.raises(DFloatUnsupportedError, match="compute_precision"):
        DFloatQwenImage21(compute_precision=mx.float16)
