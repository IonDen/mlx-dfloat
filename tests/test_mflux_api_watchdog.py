"""A model built with no ``watchdog`` argument stops a denoise overrun, and the next call generates.

One real tiny generation per family through mflux's own loop. The autouse ``_api_watchdog_fakes`` fixture
(``tests/conftest.py``) puts the fake allocator in front of the watchdog and fixes the host ceiling at
30_064_771_072 bytes (a 32 GiB Mac minus 4 GiB), so the expected numbers below do not depend on the runner. A wrapper
around the provider's per-block ``weights_for`` raises the fake above the stop line on the first block fetched in
the denoise phase, waits for the watchdog to trip, and counts the blocks fetched after that. Every test is
`@pytest.mark.mflux`.
"""

import pytest

from mlx_dfloat import _guard
from mlx_dfloat.errors import DFloatWatchdogError

pytestmark = pytest.mark.mflux

# 27 GiB: the 30_064_771_072-byte ceiling minus the default 1 GiB stop margin.
STOP_LINE = 28_991_029_248


# --- one tiny model per family, the arguments of one short generation, and the provider class it fetches blocks from


def _flux1(tmp_path, monkeypatch):
    # The fake-width transformer cannot run mflux's loop, so this is the seamed mflux transformer at 1 + 1 blocks
    # over a resident set of zeros (tests/test_flux1_model.py, the upstream-loop test). Two steps: 4 block runs.
    import mlx.core as mx
    from mflux.models.common.config.model_config import ModelConfig
    from tests.test_flux1_model import StubTokenizer, _fake_model, _ShapedEncoder, _StubVAE

    from mlx_dfloat.integrate.placeholders import install_placeholders
    from mlx_dfloat.integrate.providers import ResidentProvider
    from mlx_dfloat.mflux.flux1.init import BaseComponents
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
    model = _fake_model(
        tmp_path, monkeypatch, model="schnell", transformer=tf, components=components
    )
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
    run = {
        "seed": 1,
        "prompt": "p",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": 0.0,
    }
    return model, "schnell", run, ResidentProvider


def _zimage(tmp_path, monkeypatch):
    # 2 layers (4 blocks), 2 steps, one transformer call per step: 8 block runs. The VAE decode is a zero image.
    import mlx.core as mx
    from mflux.models.z_image.variants.z_image import ZImage
    from tests._zimage_tiny import fake_model

    from mlx_dfloat.integrate.providers import DF11Provider

    monkeypatch.setattr(ZImage, "_decode_latents", lambda self, **kw: mx.zeros((1, 3, 32, 32)))
    model = fake_model(tmp_path, monkeypatch, model="z-image-turbo")
    run = {
        "seed": 3,
        "prompt": "p",
        "num_inference_steps": 2,
        "height": 32,
        "width": 32,
        "guidance": 4.0,
    }
    return model, "z-image-turbo", run, DF11Provider


def _flux2(tmp_path, monkeypatch):
    # 1 double + 2 single blocks, 2 steps, no CFG at guidance 1.0: 6 block runs.
    from tests._flux2_tiny import fake_model

    from mlx_dfloat.integrate.providers import DF11Provider

    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    run = {
        "seed": 1,
        "prompt": "a",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": 1.0,
    }
    return model, "flux2-klein-base-4b", run, DF11Provider


def _qwen21(tmp_path, monkeypatch):
    # 2 blocks, 2 steps, no CFG at guidance 1.0.
    from tests._qwen21_tiny import fake_model

    from mlx_dfloat.integrate.providers import DF11Provider

    model = fake_model(tmp_path, monkeypatch, model="qwen-image-2.1")
    run = {
        "seed": 1,
        "prompt": "a",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": 1.0,
    }
    run["negative_prompt"] = ""
    return model, "qwen-image-2.1", run, DF11Provider


def _ernie(tmp_path, monkeypatch):
    # 2 blocks, 2 steps: 4 block runs.
    from tests._ernie_tiny import fake_model

    from mlx_dfloat.integrate.providers import DF11Provider

    model = fake_model(tmp_path, monkeypatch, model="ernie-image-turbo")
    run = {
        "seed": 1,
        "prompt": "a",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": 1.0,
    }
    return model, "ernie-image-turbo", run, DF11Provider


def _krea2(tmp_path, monkeypatch):
    # 2 blocks, 2 steps, no CFG at guidance 1.0: 4 block runs. Counting blocks at weights_for, not decodes at the
    # decode hook, keeps the per-call non-block decodes of this build out of the count.
    from tests._krea2_tiny import fake_model

    from mlx_dfloat.integrate.providers import DF11Provider

    model = fake_model(tmp_path, monkeypatch, model="krea-2")
    run = {
        "seed": 1,
        "prompt": "a",
        "num_inference_steps": 2,
        "height": 64,
        "width": 64,
        "guidance": 1.0,
    }
    return model, "krea-2", run, DF11Provider


FAMILIES = [
    pytest.param(_flux1, id="flux1"),
    pytest.param(_zimage, id="zimage"),
    pytest.param(_flux2, id="flux2"),
    pytest.param(_qwen21, id="qwen21"),
    pytest.param(_ernie, id="ernie"),
    pytest.param(_krea2, id="krea2"),
]


class DenoiseOverrun:
    """Raises the fake allocator over the stop line on the first block fetched in the denoise phase.

    One-shot, and a pass-through when no watchdog is active, so a call that is not armed simply returns (the red
    run's "DID NOT RAISE") and a retry is never tripped again. It records and never asserts: the test asserts after
    the call. ``blocks_after_trip`` counts the block whose fetch tripped (it still runs) and every block fetched later.
    """

    def __init__(self, model, counters):
        self.model = model
        self.counters = counters
        self.armed = True
        self.tripped_in_time = None
        self.blocks_after_trip = 0

    def install(self, monkeypatch, provider_cls):
        real = provider_cls.weights_for

        def weights_for(provider, block_name, shapes):
            if self.tripped_in_time is not None:
                self.blocks_after_trip += 1
            elif self.armed and self.model.open_phase == "denoise":
                watchdog = _guard.active()
                if watchdog is not None:
                    self.armed = False
                    self.counters.raise_to(STOP_LINE + 1)
                    self.tripped_in_time = watchdog.tripped.wait(5.0)
                    self.blocks_after_trip = 1
            return real(provider, block_name, shapes)

        monkeypatch.setattr(provider_cls, "weights_for", weights_for)


def _overrun(build, tmp_path, monkeypatch, request):
    from mlx_dfloat.mflux import _pipeline

    fakes = request.getfixturevalue("_api_watchdog_fakes")
    aborts = tmp_path / "aborts"
    monkeypatch.setattr(_pipeline, "_artifact_root", lambda: aborts)
    model, model_name, run, provider_cls = build(tmp_path, monkeypatch)
    overrun = DenoiseOverrun(model, fakes.counters)
    overrun.install(monkeypatch, provider_cls)
    return fakes, aborts, model, model_name, run, overrun


@pytest.mark.parametrize("build", FAMILIES)
def test_default_watchdog_stops_a_denoise_overrun(build, tmp_path, monkeypatch, request):
    # Bug caught: a model built with no watchdog argument running unwatched (the default flipped to None, or the
    # class's generate_image not armed), or the block seam's checkpoints missing, so a tripped call runs every
    # remaining block and stops only at call_end (site "call_end", blocks after the trip > 1).
    _fakes, aborts, model, model_name, run, overrun = _overrun(
        build, tmp_path, monkeypatch, request
    )

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**run)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "denoise"
    assert err.reason == "memory"
    assert err.site in {"block_start", "block_end"}
    assert overrun.blocks_after_trip <= 1
    assert err.record["verdict_counter"] == "footprint"
    # The injected 32 GiB host, not this runner's RAM.
    assert err.record["ceiling"] == 30_064_771_072
    assert err.record["context"]["model"] == model_name
    assert err.artifact is not None
    assert err.artifact.is_file()
    assert err.artifact.parent == aborts


@pytest.mark.parametrize("build", FAMILIES)
def test_retry_after_abort_generates(build, tmp_path, monkeypatch, request):
    # Bug caught: an abort that leaves the model unusable (the set marked resident while its provider is gone, block
    # placeholders not restored, the guard's scope or a stopped watchdog leaking into the next call), so the second
    # call fails or reuses a half-dropped set instead of loading it again (set_loads stays 1).
    fakes, _aborts, model, _model_name, run, _overrun_hook = _overrun(
        build, tmp_path, monkeypatch, request
    )
    with pytest.raises(DFloatWatchdogError):
        model.generate_image(**run)

    fakes.counters.raise_to(0)
    image = model.generate_image(**run)

    assert image is not None
    assert model._lifecycle.counters.set_loads == 2


# --- the watchdog argument at construction

MODEL_CLASSES = [
    pytest.param("mlx_dfloat.mflux.flux1.model", "DFloatFlux1", id="flux1"),
    pytest.param("mlx_dfloat.mflux.zimage.model", "DFloatZImage", id="zimage"),
    pytest.param("mlx_dfloat.mflux.flux2.model", "DFloatFlux2Klein", id="flux2"),
    pytest.param("mlx_dfloat.mflux.qwen21.model", "DFloatQwenImage21", id="qwen21"),
    pytest.param("mlx_dfloat.mflux.ernie.model", "DFloatErnieImage", id="ernie"),
    pytest.param("mlx_dfloat.mflux.krea2.model", "DFloatKrea2", id="krea2"),
]


class _BuildStartedError(Exception):
    """Raised by the patched GPU decoder self-check: the constructor got past its argument checks."""


def _construct(module, name, monkeypatch, **kwargs):
    # The decoder self-check is the constructor's first step after its argument checks; stopping there keeps this
    # test from resolving a checkpoint or loading a base, whatever the local Hub cache holds.
    import importlib

    from mlx_dfloat import _metal_decode

    def ensure_canary(**_kwargs):
        raise _BuildStartedError

    monkeypatch.setattr(_metal_decode, "ensure_canary", ensure_canary)
    getattr(importlib.import_module(module), name)(**kwargs)


@pytest.mark.parametrize(("module", "name"), MODEL_CLASSES)
def test_watchdog_false_is_refused_at_construction(module, name, monkeypatch):
    # Bug caught: a constructor that does not check its watchdog argument, so watchdog=False builds the model and
    # fails later, at the first call, with an AttributeError on a bool.
    from mlx_dfloat.errors import DFloatUnsupportedError

    with pytest.raises(DFloatUnsupportedError, match="watchdog=None"):
        _construct(module, name, monkeypatch, watchdog=False)


@pytest.mark.parametrize(("module", "name"), MODEL_CLASSES)
def test_watchdog_none_is_accepted_at_construction(module, name, monkeypatch):
    # Bug caught: a check that tests truthiness (`if not watchdog`) and so refuses None, the documented way to run
    # the calls unwatched.
    with pytest.raises(_BuildStartedError):
        _construct(module, name, monkeypatch, watchdog=None)
