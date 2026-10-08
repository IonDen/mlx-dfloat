"""The Krea 2 transformer: the class-swap seam, the compile bypass and the float32 residual, on a real tiny mflux one."""

import importlib.util
import types
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tests._krea2_tiny import TINY, tiny_groups, tiny_inputs, tiny_predict, tiny_seamed

from mlx_dfloat.decode import decode_group
from mlx_dfloat.errors import DFloatDependencyError
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.placeholders import get_attr_path
from mlx_dfloat.integrate.providers import DF11Provider, ResidentProvider
from mlx_dfloat.mflux._compile import uncompiled
from mlx_dfloat.mflux.krea2.transformer import block_lists, seam_transformer_class

COMPILED = type(
    mx.compile(lambda x: x)
)  # mlx.gc_func, a FunctionType subclass: only `type(f) is` tells them apart
BLOCKS = ["blocks.0", "blocks.1"]


@pytest.fixture
def not_m1(monkeypatch):
    """mflux's chip check answers "not a base or Pro M1/M2": stock mflux compiles Krea 2's predict there.

    On this M1 Max the unpatched check is already False ("max" in the chip name, apple_silicon.py:15-16); CI's base
    M1/M2 runners answer True, where mflux hands back the plain function without our override and a bypass test
    would pass vacuously. The fixture keeps every chip on the compiling side.
    """
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    return AppleSiliconUtil


def _stock_args(inputs, guidance=3.5):
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    return (Krea2Transformer(**TINY), inputs["embeds"], inputs["neg_embeds"], guidance)


def _random_weights(shapes, seed=11):
    """Every block matrix its own random bfloat16 values (small, so the tiny forward stays finite)."""
    rng = np.random.default_rng(seed)
    return {
        block: {
            attr: mx.array((rng.standard_normal(shape) * 0.05).astype(np.float32)).astype(
                mx.bfloat16
            )
            for attr, shape in per.items()
        }
        for block, per in shapes.items()
    }


def _constant_weights(shapes):
    """Block i's matrices filled with the constant i + 1."""
    return {
        block: {
            attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()
        }
        for i, (block, per) in enumerate(shapes.items())
    }


def _eager_reference(tf, shapes, weights):
    """Stock mflux holding tf's non-block parameters and ``weights`` on its blocks."""
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    plain = Krea2Transformer(**TINY)
    matrices = {f"{b}.{a}.weight" for b, per in shapes.items() for a in per}
    plain.update(
        tree_unflatten([(k, v) for k, v in tree_flatten(tf.parameters()) if k not in matrices])
    )
    for block, per in weights.items():
        idx = int(block.partition(".")[2])
        for attr, w in per.items():
            get_attr_path(plain.blocks[idx], attr).weight = w
    return plain


def _call(inputs):
    """``Krea2Transformer.__call__`` positional inputs (latents, timestep, context), as mflux's predict passes them."""
    return (inputs["latents"], inputs["timestep"], inputs["embeds"])


def test_block_lists_is_the_one_list_the_forward_runs():
    # Bug caught: another attribute than the list __call__ iterates (transformer.py:89-92: `for block in
    # self.blocks`), so no block would run through the seam.
    tf = SimpleNamespace(blocks=[1, 2], txtfusion=SimpleNamespace(layerwise_blocks=[3]))
    assert block_lists(tf) == [("blocks", [1, 2])]


def test_the_seamed_class_needs_the_mflux_extra(monkeypatch):
    # Bug caught: the adapter importing mflux without the guard (a bare ImportError instead of the package's own
    # dependency error naming the extra), for both classes. The check runs on every call, before the cached class.
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "mflux" else real(name, *a, **k),
    )
    with pytest.raises(DFloatDependencyError, match=r"mlx-dfloat\[mflux\]"):
        seam_transformer_class()
    with pytest.raises(DFloatDependencyError, match=r"mlx-dfloat\[mflux\]"):
        seam_transformer_class(per_call=True)


@pytest.mark.mflux
def test_off_m1_m2_the_stock_factory_compiles_and_on_them_it_does_not(not_m1, monkeypatch):
    # Bug caught (the control): the reason for the bypass gone (an mflux change: krea2.py:202-204 no longer compiling
    # off base and Pro M1/M2), or a bypass test that passes vacuously on a base M1/M2 CI runner (where the unpatched
    # chip check is True). On this M1 Max the unpatched check is already False: stock mflux compiles here.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    args = _stock_args(tiny_inputs())
    assert type(Krea2._predict(*args)) is COMPILED
    monkeypatch.setattr(not_m1, "is_m1_or_m2", classmethod(lambda cls: True))
    assert type(Krea2._predict(*args)) is types.FunctionType


@pytest.mark.mflux
def test_uncompiled_returns_mfluxs_plain_predict_on_any_chip(not_m1):
    # Bug caught: the chip override not reaching Krea2._predict (the seam's per-block mx.eval would then run inside
    # mx.compile), or leaking past the factory call (every later mflux factory would run uncompiled).
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    fn = uncompiled(Krea2._predict, *_stock_args(tiny_inputs()))
    assert type(fn) is types.FunctionType
    assert not_m1.is_m1_or_m2() is False


@pytest.mark.mflux
def test_the_compiled_predict_cannot_run_the_seam():
    # Bug caught: none in our code; it pins why the bypass exists (if the compiled predict could run per-block evals,
    # the bypass and its cost on Max/Ultra and M3+ chips would be unjustified).
    tf, shapes = tiny_seamed()
    tf.attach(ResidentProvider(_constant_weights(shapes)), shapes)
    with pytest.raises(ValueError, match=r"\[eval\] Attempting to eval"):
        mx.eval(tiny_predict(tf, tiny_inputs(), guidance=3.5, compiled=True))


@pytest.mark.mflux
def test_the_swap_keeps_parameter_paths_and_nothing_is_computed():
    # Bug caught: the class swap breaking on SingleStreamBlock or moving parameters under another path; a parameter
    # mflux computes at construction added upstream (no checkpoint holds it: every build would be refused by the
    # extras coverage), or a stale exemption hiding a parameter no checkpoint fills.
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
    from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock
    from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping

    from mlx_dfloat.integrate.blockseam import seam_blocks
    from mlx_dfloat.mflux.krea2.transformer import COMPUTED_PARAMS

    tf = seam_transformer_class()(**TINY)
    before = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert seam_blocks(block_lists(tf), tf.seam_cell) == ("blocks.0", "blocks.1")
    after = sorted(k for k, _v in tree_flatten(tf.parameters()))
    assert after == before
    assert isinstance(tf.blocks[1], SingleStreamBlock)

    params = {k for k, _v in tree_flatten(Krea2Transformer(**TINY).parameters())}
    targets = set()
    for t in Krea2WeightMapping.get_transformer_mapping():
        if "{layer}" in t.to_pattern:
            targets |= {t.to_pattern.replace("{layer}", str(i)) for i in range(2)}
        else:
            targets.add(t.to_pattern)
    assert frozenset(params - targets) == COMPUTED_PARAMS
    assert len(COMPUTED_PARAMS) == 0  # mflux 0.20.0: every parameter is a mapping target


@pytest.mark.mflux
def test_the_residual_stream_runs_float32_from_float32_latents():
    # Bug caught (in our model of the world): activation terms calibrated or predicted on a bf16 residual (half the
    # true size). With every weight in bf16, as mflux loads the published checkpoint (weight_loader.py:448-449), the
    # float32 latents (create_noise, krea2_latent_creator.py:6-10) still make the stream every block sees float32:
    # `first` promotes the patches and the bf16 text joins them by concatenation. The control: bf16 latents give a
    # bf16 stream, so the latents' dtype is what decides.
    from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer

    def run(latents):
        tf = Krea2Transformer(**TINY)
        tf.update(tree_map(lambda p: p.astype(mx.bfloat16), tf.parameters()))
        seen = []
        block = tf.blocks[0]

        def recording(x, *args):
            seen.append(x.dtype)
            return block(x, *args)

        tf.blocks[0] = recording
        inputs = tiny_inputs()
        out = tf(latents, inputs["timestep"], inputs["embeds"])
        mx.eval(out)
        return seen, out.dtype

    latents = tiny_inputs()["latents"]
    assert latents.dtype == mx.float32
    assert run(latents) == ([mx.float32], mx.float32)
    assert run(latents.astype(mx.bfloat16)) == ([mx.bfloat16], mx.bfloat16)


@pytest.mark.mflux
def test_a_forward_runs_each_block_with_its_own_weights_and_ends_on_placeholders(monkeypatch):
    # Bug caught: a block run on another block's weights (blocks.1 skipped or fed blocks.0's matrices: the output
    # differs from stock mflux holding the same weights), a block output not evaluated at its boundary, or the
    # placeholders not restored afterwards.
    tf, shapes = tiny_seamed()
    weights = _constant_weights(shapes)
    evals = []
    real_eval = seam._eval

    def recording_eval(*args):
        evals.append(args[0])
        real_eval(*args)

    monkeypatch.setattr(seam, "_eval", recording_eval)
    tf.attach(ResidentProvider(weights), shapes)
    inputs = tiny_inputs()
    out = tf(*_call(inputs))
    mx.eval(out)
    tf.verify_step()
    want = _eager_reference(tf, shapes, weights)(*_call(inputs))
    mx.eval(want)

    assert len(evals) == 2
    # Each block returns the joint [6 text | 16 image] sequence at width 32, float32 like the latents.
    assert [(tuple(e.shape), e.dtype) for e in evals] == [((1, 22, 32), mx.float32)] * 2
    assert out.shape == (1, 16, 8, 8)
    assert bool(mx.isfinite(out).all().item())
    assert out.dtype == want.dtype == mx.float32
    assert mx.array_equal(out.view(mx.uint32), want.view(mx.uint32)).item()
    for block, per in shapes.items():
        idx = int(block.partition(".")[2])
        for attr in per:
            assert get_attr_path(tf.blocks[idx], attr).weight.size == 0, (block, attr)


@pytest.mark.mflux
def test_depth2_drains_the_last_block(monkeypatch):
    # Bug caught: end_step not draining the tail (the last block's output never evaluated inside the step), or the
    # look-ahead evaluating in the wrong order. Expected (seam._seam_eval + end_step): async r0, async r1, eval r0,
    # eval r1.
    tf, shapes = tiny_seamed()
    events = []
    real_eval, real_async = seam._eval, seam._async_eval

    def recording_eval(*args):
        events.append(("eval", args[0]))
        real_eval(*args)

    def recording_async(*args):
        events.append(("async", args[0]))
        real_async(*args)

    monkeypatch.setattr(seam, "_eval", recording_eval)
    monkeypatch.setattr(seam, "_async_eval", recording_async)
    tf.attach(ResidentProvider(_constant_weights(shapes)), shapes, eval_policy="depth2")
    mx.eval(tf(*_call(tiny_inputs())))
    tf.verify_step()

    outputs = []
    for _kind, obj in events:
        if not any(obj is seen for seen in outputs):
            outputs.append(obj)
    order = [(kind, next(i for i, o in enumerate(outputs) if o is obj)) for kind, obj in events]
    assert order == [("async", 0), ("async", 1), ("eval", 0), ("eval", 1)]


@pytest.mark.mflux
@pytest.mark.parametrize(
    ("guidance", "negative", "expected"),
    [(3.5, True, BLOCKS + BLOCKS), (1.0, False, BLOCKS)],
)
def test_cfg_runs_two_calls_and_decodes_each_block_once_per_call(guidance, negative, expected):
    # Bug caught: a decode cached across the two CFG calls without verification, or the second call skipped. mflux
    # runs a CFG step as two batch-1 transformer calls (krea2.py:195-200): 4 decodes per step at 2 blocks; at
    # guidance 1.0 with no negative, one call and 2 decodes.
    from mlx_dfloat.mflux.krea2.names import krea2_name_map

    tf, shapes = tiny_seamed()
    groups, names, _src = tiny_groups(shapes, np.random.default_rng(9))
    calls = []

    def count(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    tf.attach(
        DF11Provider(groups, names, krea2_name_map(), decode=count), shapes, verify_in_call=True
    )
    inputs = tiny_inputs()
    if not negative:
        inputs["neg_embeds"] = None
    out = tiny_predict(tf, inputs, guidance=guidance)
    mx.eval(out)
    assert calls == expected
    assert out.shape == (1, 16, 8, 8)
    assert bool(mx.isfinite(out).all().item())


@pytest.mark.mflux
def test_a_cfg_step_through_the_seam_gives_stock_mfluxs_prediction_bit_for_bit(monkeypatch):
    # Bug caught: the two CFG calls wrong through the seam where one call is right: the second call run on the first
    # call's placeholders-restored blocks, the two prompts' contexts (6 and 3 tokens) mixed, or the guidance
    # combination fed the wrong call. The reference is mflux's own predict over a stock transformer holding the same
    # weights, both uncompiled; equality is bit for bit.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2
    from mflux.utils.apple_silicon import AppleSiliconUtil

    tf, shapes = tiny_seamed()
    weights = _random_weights(shapes)
    tf.attach(ResidentProvider(weights), shapes, verify_in_call=True)
    inputs = tiny_inputs()
    ours = tiny_predict(tf, inputs, guidance=3.5)
    mx.eval(ours)
    plain = _eager_reference(tf, shapes, weights)
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    predict = Krea2._predict(plain, inputs["embeds"], inputs["neg_embeds"], 3.5)
    assert type(predict) is types.FunctionType  # stock, uncompiled like ours
    theirs = predict(latents=inputs["latents"], timestep=inputs["timestep"])
    mx.eval(theirs)
    assert ours.shape == theirs.shape == (1, 16, 8, 8)
    assert ours.dtype == theirs.dtype == mx.float32
    assert mx.array_equal(ours.view(mx.uint32), theirs.view(mx.uint32)).item()


@pytest.mark.mflux
def test_the_krea_schedulers_are_registered_by_the_adapters_own_import():
    # Bug caught: the adapter's import path no longer registering er_sde / euler (mflux 0.20.0 registers them in
    # mflux/models/krea2/__init__.py, which any mflux.models.krea2 import runs), so the user's first call fails with
    # NotImplementedError when Config resolves the name (krea2.py:78 reads config.scheduler.sigmas). A fresh interpreter: before any Krea import the scheduler is unknown (the
    # control), after the adapter builds its transformer class it constructs.
    import subprocess
    import sys

    code = """
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig

def make():  # Config.scheduler is a lazy property (config.py:145-159): reading it is what resolves the name
    return Config(model_config=ModelConfig.krea2_raw(), num_inference_steps=2, height=64, width=64,
                  guidance=3.5, scheduler="er_sde").scheduler

try:
    make()
    print("registered-before")
except NotImplementedError:
    print("unknown-before")
from mlx_dfloat.mflux.krea2.transformer import seam_transformer_class
seam_transformer_class()
print(type(make()).__name__)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["unknown-before", "Krea2FlowScheduler"]


def test_importing_the_transformer_module_does_not_import_mflux():
    # Bug caught: a module-level mflux import in the transformer module (a user without the extra gets a bare
    # ImportError on import).
    import subprocess
    import sys

    code = "import sys, mlx_dfloat.mflux.krea2.transformer; print('mflux' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- per-call non-block decode (the shipped mode) ------------------------------------------------------------------

TPROJ = "tproj.linear.weight"


def _per_call_build(tmp_path, *, per_call, n_layers=2, seed=4):
    """A tiny config-less checkpoint (random extras, so the forward sees every weight), built resident or per-call;
    returns (build, groups, names): the block and non-block DF11 groups loaded, nothing decoded or bound yet."""
    from tests._krea2_tiny import write_tiny_checkpoint

    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.integrate.coverage import load_resident_set
    from mlx_dfloat.mflux.krea2.names import krea2_name_map
    from mlx_dfloat.mflux.krea2.transformer import build_transformer

    rng = np.random.default_rng(seed)
    root = tmp_path / ("per-call" if per_call else "resident")
    _m, _c, layout = write_tiny_checkpoint(root, rng, n_layers=n_layers, random_extras=rng)
    ckpt = open_checkpoint(root, layouts=(layout,))
    tf_class = seam_transformer_class(per_call=True) if per_call else None
    build = build_transformer(
        ckpt, transformer_kwargs={**TINY, "layers": n_layers}, transformer_class=tf_class
    )
    return build, load_resident_set(ckpt), krea2_name_map(), ckpt


def _split(groups):
    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS

    nonblock = {g: v for g, v in groups.items() if g in NONBLOCK_GROUPS}
    blocks = {g: v for g, v in groups.items() if g not in NONBLOCK_GROUPS}
    return nonblock, blocks


def _reference():
    from functools import partial

    return partial(decode_group, backend="reference")


def _attach_blocks(build, blocks, ckpt, names, *, eval_policy="per-block", provider_class=None):
    provider = DF11Provider(
        blocks, {n: ckpt.groups[n].matrix_names for n in blocks}, names, decode=_reference()
    )
    if provider_class is not None:
        provider = provider_class(provider)
    build.transformer.attach(provider, build.shapes, eval_policy=eval_policy, verify_in_call=True)
    return provider


@pytest.mark.mflux
def test_per_call_decode_gives_the_resident_paths_output_bit_for_bit(tmp_path):
    # Bug caught: the non-block weights released before the forward starts (the pre-block graph, text fusion, tmlp,
    # tproj, txtmlp, would read a placeholder: a zero-size matmul error), a non-block weight installed on another
    # parameter, or the second CFG call run on cleared weights. The release point itself is the size-recording tests'
    # (a release later in the call changes no output: the graph already holds the arrays). One uncompiled predict at
    # guidance 3.5 (two calls) on each path.
    from mlx_dfloat.integrate.resident import decode_nonblock, install_nonblock
    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS

    outs = []
    for per_call in (False, True):
        build, groups, names, ckpt = _per_call_build(tmp_path, per_call=per_call)
        nonblock, blocks = _split(groups)
        if per_call:
            build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
        else:
            matrices = {g: NONBLOCK_GROUPS[g] for g in nonblock}
            weights = decode_nonblock(
                nonblock, matrices, build.nonblock, names, decode=_reference()
            )
            install_nonblock(build.transformer, weights)
        _attach_blocks(build, blocks, ckpt, names)
        out = tiny_predict(build.transformer, tiny_inputs(), guidance=3.5)
        mx.eval(out)
        outs.append(out)
    resident, per_call = outs
    assert per_call.dtype == resident.dtype == mx.float32
    assert bool(mx.isfinite(per_call).all().item())
    assert mx.array_equal(per_call.view(mx.uint32), resident.view(mx.uint32)).item()


@pytest.mark.mflux
@pytest.mark.parametrize(("guidance", "neg", "per_group"), [(3.5, True, 2), (1.0, False, 1)])
def test_per_call_decode_runs_once_per_transformer_call(tmp_path, guidance, neg, per_group):
    # Bug caught: the decode cached across the CFG pair (the second call reading cleared placeholders: a zero-size
    # matmul) or run twice in one call. Two calls per CFG step: each of the seven groups decoded twice; one at 1.0.
    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS

    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    seen = []
    reference = _reference()

    def counting(group):
        seen.append(group.name)
        return reference(group)

    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=counting)
    _attach_blocks(build, blocks, ckpt, names)
    inputs = tiny_inputs()
    if not neg:
        inputs["neg_embeds"] = None
    mx.eval(tiny_predict(build.transformer, inputs, guidance=guidance))
    assert sorted(seen) == sorted(list(NONBLOCK_GROUPS) * per_group)


# What MLX may still hold in its buffer cache right after a drain: a few tiny buffers (the drain's own one-element
# kick, a scalar) whose free lands after the final clear when the GPU's completion handler runs late, which a CI
# runner's paravirtual GPU showed as 4 to 16 B. The tests' released sets are far larger (the tiny model's activations
# and non-block weights, or a 4 MiB array), so a missed clear or drain still fails these bounds.
TINY_CACHE_BYTES = 1024


class _Recording:
    """A provider that records, at each block request, the size of the transformer's tproj weight and the MLX cache,
    then delegates."""

    def __init__(self, inner, transformer, *, raise_at=None, events=None):
        self.inner, self.transformer, self.raise_at = inner, transformer, raise_at
        self.sizes, self.cache = [], []
        self.events = [] if events is None else events
        self.launches, self.launching, self.policies = 0, inner.launching, inner.policies

    def weights_for(self, name, shapes):
        self.sizes.append((name, int(self.transformer.tproj.linear.weight.size)))
        self.cache.append(int(mx.get_cache_memory()))
        self.events.append(("weights", name))
        if name == self.raise_at:
            raise RuntimeError(f"stop at {name}")
        return self.inner.weights_for(name, shapes)

    def verify(self):
        self.inner.verify()

    def reset(self):
        self.inner.reset()


@pytest.mark.mflux
@pytest.mark.parametrize(("policy", "n_layers"), [("per-block", 2), ("depth2", 3)])
def test_the_nonblock_weights_are_released_before_block_0_requests_its_weights(
    tmp_path, policy, n_layers
):
    # Bug caught: the non-block weights released after block 0's decode (at blocks.1, the first design, which
    # measured 0.14 GiB saved: the denoise peak is block 0's eval), released never, or released with their buffers
    # left in MLX's cache (the footprint keeps them). At every block request tproj (192 x 32 at TINY size) is a
    # placeholder, and at block 0's request the cache is empty.
    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True, n_layers=n_layers)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    recorder = _attach_blocks(
        build,
        blocks,
        ckpt,
        names,
        eval_policy=policy,
        provider_class=lambda inner: _Recording(inner, build.transformer),
    )
    inputs = tiny_inputs()
    inputs["neg_embeds"] = None
    mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
    want = [(f"blocks.{i}", 0) for i in range(n_layers)]
    assert recorder.sizes == want
    assert recorder.cache[0] < TINY_CACHE_BYTES
    assert build.transformer.tproj.linear.weight.size == 0  # and after the call
    # The attached provider is back after the call: a wrapper left in place would be wrapped again by the next call,
    # one level deeper per call (about 100 deep over a 50-step guided run).
    assert build.transformer.seam_cell.state.provider is recorder
    mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
    assert recorder.sizes == want + want  # the second call releases at the same point


@pytest.mark.mflux
def test_each_call_drains_the_cache_then_block_0s_inputs_are_evaluated_released_and_drained(
    tmp_path, monkeypatch
):
    # Bug caught: the call-start drain missing or placed after the non-block decode (call 2 then decodes and
    # builds its pre-block graph on top of call 1's cached activations: the two-call peak sat at the memory limit), the
    # release before block 0's inputs are evaluated (the lazy pre-block graph still holds the decoded arrays, so
    # nothing is freed), the drain before the release (the released buffers land in the cache and stay in the
    # footprint), or the inputs of a later block evaluated instead. Block 0 is called with (combined, tvec, freqs,
    # attention_mask): three arrays and None at guidance 1.0. The decode marks where the call's non-block decode runs.
    from mlx_dfloat.mflux.krea2 import transformer as ktf

    events = []
    real_eval, real_clear, real_drain = ktf._eval, ktf.clear_nonblock, ktf._release_and_drain

    def eval_(*arrays):
        events.append(("eval", len(arrays)))
        real_eval(*arrays)

    def clear(module, shapes):
        events.append(("clear", int(module.tproj.linear.weight.size)))
        real_clear(module, shapes)

    def drain():
        events.append(("drain",))
        return real_drain()

    reference = _reference()

    def decode(group):
        if not events or events[-1] != ("decode",):
            events.append(("decode",))
        return reference(group)

    monkeypatch.setattr(ktf, "_eval", eval_)
    monkeypatch.setattr(ktf, "clear_nonblock", clear)
    monkeypatch.setattr(ktf, "_release_and_drain", drain)
    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=decode)
    _attach_blocks(
        build,
        blocks,
        ckpt,
        names,
        provider_class=lambda inner: _Recording(inner, build.transformer, events=events),
    )
    inputs = tiny_inputs()
    inputs["neg_embeds"] = None
    events.clear()
    out = tiny_predict(build.transformer, inputs, guidance=1.0)
    assert events == [
        ("drain",),
        ("decode",),
        ("eval", 3),
        ("clear", 192 * 32),
        ("drain",),
        ("weights", "blocks.0"),
        ("weights", "blocks.1"),
        # The call ends with its output evaluated and the placeholders put back again (already placeholders).
        ("eval", 1),
        ("clear", 0),
    ]
    assert bool(mx.isfinite(out).all().item())


@pytest.mark.mflux
def test_a_guided_steps_second_call_starts_with_mlxs_cache_empty(tmp_path):
    # Bug caught: call 2 of a guided step starting with call 1's freed activations still in MLX's cache, under
    # a cache limit large enough to keep them (the planner's 4.5e9 B at 1024²; 1 GiB here). Recorded at each call's
    # first non-block decode: call 1 leaves its activations cached (checked), and both calls start from 0 B.
    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    reference = _reference()
    at_decode, seen = [], set()

    def decode(group):
        if group.name in seen:
            seen.clear()
        if not seen:
            at_decode.append(int(mx.get_cache_memory()))
        seen.add(group.name)
        return reference(group)

    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=decode)
    _attach_blocks(build, blocks, ckpt, names)
    previous = mx.set_cache_limit(1024**3)
    try:
        mx.eval(tiny_predict(build.transformer, tiny_inputs(), guidance=3.5))
        left = int(mx.get_cache_memory())
    finally:
        mx.set_cache_limit(previous)
    assert all(cached < TINY_CACHE_BYTES for cached in at_decode), at_decode
    assert (
        left > 0
    )  # the step's last call did leave buffers in the cache: the drain had something to clear


class _Drain:
    """The drain's inputs, scripted: MLX's cache bytes, a footprint sequence and a clock sequence; records calls."""

    def __init__(self, monkeypatch, *, cached, footprints, clock):
        from mlx_dfloat.mflux.krea2 import transformer as ktf

        self.events = []
        self._fp, self._clock = list(footprints), list(clock)
        monkeypatch.setattr(ktf, "_synchronize", lambda: self.events.append("sync"))
        monkeypatch.setattr(ktf, "_cache_memory", lambda: cached)
        monkeypatch.setattr(ktf, "_clear_cache", lambda: self.events.append("clear"))
        monkeypatch.setattr(ktf, "_kick", lambda: self.events.append("kick"))
        monkeypatch.setattr(ktf, "_footprint", self._footprint)
        monkeypatch.setattr(ktf, "_clock", lambda: self._clock.pop(0))
        monkeypatch.setattr(ktf, "_sleep", lambda s: self.events.append(("sleep", s)))
        self.drain = ktf._release_and_drain

    def _footprint(self):
        self.events.append("footprint")
        return self._fp.pop(0)


def test_the_drain_waits_until_the_footprint_has_dropped_by_90_percent_of_the_released_bytes(
    monkeypatch,
):
    # Bug caught: the drain returning at the clear (the released pages stay in the footprint for ~20 ms to 0.7 s, and
    # the next call's 1.33 GB decode stacks on them), waiting for the whole release, or reading the footprint after
    # the clear (the drop is measured from before it). Released 1e9 B, so 9e8 B of drop ends the wait: 5e8 and 8.5e8
    # wait, 9e8 stops.
    d = _Drain(
        monkeypatch,
        cached=1_000_000_000,
        footprints=[20_000_000_000, 19_500_000_000, 19_150_000_000, 19_100_000_000],
        clock=[0.0, 0.01, 0.02],
    )
    assert d.drain() is True
    assert d.events == [
        "sync",
        "footprint",
        "clear",
        "kick",
        "footprint",
        ("sleep", 0.0005),
        "footprint",
        ("sleep", 0.0005),
        "footprint",
        "sync",
        "clear",
    ]


def test_the_drain_gives_up_at_a_quarter_second(monkeypatch):
    # Bug caught: a drain that never returns when the footprint does not fall (another allocation in flight), or a cap
    # other than 0.25 s: 0.24 s still waits, 0.26 s stops.
    d = _Drain(
        monkeypatch, cached=1_000_000_000, footprints=[20_000_000_000] * 4, clock=[0.0, 0.24, 0.26]
    )
    assert d.drain() is False
    assert d.events == [
        "sync",
        "footprint",
        "clear",
        "kick",
        "footprint",
        ("sleep", 0.0005),
        "footprint",
        "sync",
        "clear",
    ]


def test_the_drain_empties_the_cache_of_buffers_whose_free_was_still_pending():
    # Bug caught: the drain clearing the cache before MLX's pending frees have landed in it (a buffer freed after its
    # evaluation reaches the cache from the GPU's completion handler, after the Python reference is gone), so the call
    # starts with them cached after all. Without the synchronisation first, a freed 4 MiB array was still in the cache
    # after the clear in 1980 of 2000 tries on an M1 Max (mlx 0.32.2); with it, in none. 50 tries here.
    from mlx_dfloat.mflux.krea2 import transformer as ktf

    left = []
    for _ in range(50):
        big = mx.ones((1024, 1024))
        mx.eval(big)
        del big
        ktf._release_and_drain()
        left.append(int(mx.get_cache_memory()))
    assert all(cached < TINY_CACHE_BYTES for cached in left), left


def test_a_release_under_64_mib_is_cleared_and_kicked_without_a_wait(monkeypatch):
    # Bug caught: every block boundary of a small model paying the poll (up to 0.25 s when a few MB cannot show in
    # the footprint), the clear and the driver kick skipped for a small cache, or the kick's own buffers left behind:
    # they reach the cache once its command completes (16 B on an M1 Max), so the drain waits for that and clears
    # again. 64 MiB - 1 B released.
    d = _Drain(monkeypatch, cached=64 * 1024**2 - 1, footprints=[], clock=[])
    assert d.drain() is True
    assert d.events == ["sync", "clear", "kick", "sync", "clear"]


@pytest.mark.mflux
def test_the_none_policy_keeps_the_nonblock_weights_until_the_call_ends(tmp_path):
    # Bug caught: a release under "none" (no per-block eval: evaluating block 0's inputs there would force the
    # pre-block graph mid-call, which that policy promises not to do). Two blocks, the most "none" allows with a
    # launching provider, and verify_step() after the step instead of verify_in_call.
    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    inner = DF11Provider(
        blocks, {n: ckpt.groups[n].matrix_names for n in blocks}, names, decode=_reference()
    )
    recorder = _Recording(inner, build.transformer)
    build.transformer.attach(recorder, build.shapes, eval_policy="none")
    inputs = tiny_inputs()
    inputs["neg_embeds"] = None
    mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
    build.transformer.verify_step()
    assert recorder.sizes == [("blocks.0", 192 * 32), ("blocks.1", 192 * 32)]
    assert build.transformer.tproj.linear.weight.size == 0


@pytest.mark.mflux
def test_an_exception_mid_call_leaves_placeholders(tmp_path, monkeypatch):
    # Bug caught: a call that raises leaving the decoded non-block weights installed (1.23 GiB on the real model kept
    # past the call, and a retry would see them). The evaluation of block 0's inputs raises, before the release, so
    # only the call's own cleanup can put the placeholders back.
    from mlx_dfloat.mflux.krea2 import transformer as ktf

    def boom(*_arrays):
        raise RuntimeError("stop before the release")

    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    _attach_blocks(build, blocks, ckpt, names)
    monkeypatch.setattr(ktf, "_eval", boom)
    inputs = tiny_inputs()
    inputs["neg_embeds"] = None
    with pytest.raises(RuntimeError, match="stop before the release"):
        mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
    for param in build.nonblock:
        assert get_attr_path(build.transformer, param.removesuffix(".weight")).weight.size == 0, (
            param
        )


@pytest.mark.mflux
def test_a_per_call_decode_evaluates_the_seven_groups_in_one_eval(tmp_path, monkeypatch):
    # Bug caught: the per-call decode paying one synchronisation per non-block group (seven per transformer call, 14
    # per guided step) where one eval of all seven status words and 37 matrices does. Two calls at guidance 3.5.
    from mlx_dfloat.integrate import resident

    seen = []
    real = resident._eval
    monkeypatch.setattr(resident, "_eval", lambda *a: (seen.append(len(a)), real(*a)))
    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    _attach_blocks(build, blocks, ckpt, names)
    mx.eval(tiny_predict(build.transformer, tiny_inputs(), guidance=3.5))
    assert seen == [7 + 37, 7 + 37]


@pytest.mark.mflux
def test_the_bound_groups_live_outside_the_module_tree_and_detach_forgets_them(tmp_path):
    # Bug caught: the binding stored as a dict attribute, which nn.Module.__setattr__ routes into the parameter tree
    # (the compressed groups would show up as parameters: the extras coverage and load_weights break); or detach
    # keeping the compressed non-block groups alive after the set drops (about 0.9 GB on the real model, through the
    # VAE phase). A call without a binding is refused naming the method to call.
    from mlx_dfloat.errors import DFloatIntegrationError

    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    before = sorted(k for k, _v in tree_flatten(build.transformer.parameters()))
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    assert sorted(k for k, _v in tree_flatten(build.transformer.parameters())) == before
    _attach_blocks(build, blocks, ckpt, names)
    build.transformer.detach()
    assert build.transformer.nonblock_bound() is False
    _attach_blocks(build, blocks, ckpt, names)
    with pytest.raises(DFloatIntegrationError, match="bind_nonblock"):
        tiny_predict(build.transformer, tiny_inputs(), guidance=1.0)


@pytest.mark.mflux
def test_the_release_wrapper_carries_the_providers_attributes(tmp_path):
    # Bug caught: the wrapper exposing the Protocol's methods but not its attributes (`launches`, `launching`,
    # `policies`): the seam and the model's launch count read them through the state's provider during a call.
    from mlx_dfloat.mflux.krea2.transformer import ReleaseAtFirstBlock

    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    _nonblock, blocks = _split(groups)
    inner = DF11Provider(
        blocks, {n: ckpt.groups[n].matrix_names for n in blocks}, names, decode=_reference()
    )
    wrapper = ReleaseAtFirstBlock(inner, build.transformer, build.nonblock, policy="per-block")
    assert (wrapper.launching, wrapper.policies) == (inner.launching, inner.policies)
    wrapper.weights_for("blocks.0", build.shapes["blocks.0"])
    wrapper.verify()
    assert wrapper.launches == inner.launches == 1
    # The Protocol's attribute is writable; the count lives in the wrapped provider.
    wrapper.launches = 0
    assert inner.launches == 0


@pytest.mark.mflux
def test_a_per_call_step_is_refused_while_the_last_steps_status_words_are_unchecked(tmp_path):
    # Bug caught: the release wrapper hiding the provider's `pending` list, so the seam's unchecked-status
    # refusal (begin_step reads state.provider.pending) never fires on the per-call class: a caller attached without
    # verify_in_call who forgets verify_step() would pile status words up unnoticed.
    from mlx_dfloat.errors import DFloatIntegrationError
    from mlx_dfloat.mflux.krea2.transformer import ReleaseAtFirstBlock

    build, groups, names, ckpt = _per_call_build(tmp_path, per_call=True)
    nonblock, blocks = _split(groups)
    build.transformer.bind_nonblock(nonblock, build.nonblock, names, decode=_reference())
    inner = DF11Provider(
        blocks, {n: ckpt.groups[n].matrix_names for n in blocks}, names, decode=_reference()
    )
    assert (
        ReleaseAtFirstBlock(inner, build.transformer, build.nonblock, policy="per-block").pending
        is inner.pending
    )
    build.transformer.attach(inner, build.shapes, verify_in_call=False)
    inputs = tiny_inputs()
    inputs["neg_embeds"] = None
    mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
    assert len(inner.pending) == 2  # one status word per block, unchecked
    with pytest.raises(
        DFloatIntegrationError, match="2 decode status words from an earlier step are unchecked"
    ):
        tiny_predict(build.transformer, inputs, guidance=1.0)
    build.transformer.verify_step()
    mx.eval(tiny_predict(build.transformer, inputs, guidance=1.0))
