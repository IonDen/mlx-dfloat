"""Where an armed model call stops, per phase, and what it leaves behind: FLUX.2 Klein and FLUX.1 tiny models.

The autouse ``_api_watchdog_fakes`` fixture (``tests/conftest.py``) puts the fake allocator in front of the watchdog,
fixes the host ceiling at 30_064_771_072 bytes (a 32 GiB Mac minus 4 GiB) and records any process exit. Each test
wraps one phase's component: the wrapper raises the fake above the stop line and waits on the watchdog's ``tripped``
event (5 s timeout), which is a trip during an evaluation with no checkpoint. The tests then assert where the call
stopped (``site``) and how much work ran after the trip, not only ``phase``: ``phase`` is read at trip time, so a
later checkpoint would still report it.

Where FLUX.1's VAE evaluates (mflux 0.24.0): ``Flux1.generate_image`` calls ``VAEUtil.decode`` (lazy) and then
``ImageUtil.to_image``, which evaluates the decode once at ``np.array`` in ``ImageUtil._to_numpy``
(``mflux/utils/image_util.py:134``), the same shape as Klein. No checkpoint runs inside it; a VAE trip on FLUX.1 is
taken at ``call_end`` (``test_flux1_abort_in_vae_raises_at_call_end``). Every test is `@pytest.mark.mflux`.
"""

import threading
import weakref

import pytest

from mlx_dfloat import _guard
from mlx_dfloat.errors import DFloatWatchdogError

pytestmark = pytest.mark.mflux

# 27 GiB: the 30_064_771_072-byte ceiling minus the default 1 GiB stop margin.
STOP_LINE = 28_991_029_248
THREAD = "mlx-dfloat-watchdog"

# One short Klein generation: 1 double + 2 single blocks, 2 steps, no CFG at guidance 1.0 (6 block runs).
KLEIN_RUN = {
    "seed": 1,
    "prompt": "a",
    "num_inference_steps": 2,
    "height": 64,
    "width": 64,
    "guidance": 1.0,
}


@pytest.fixture
def fakes(request):
    """The autouse ``_api_watchdog_fakes`` value: the fake allocator and the exit recorder of this test."""
    return request.getfixturevalue("_api_watchdog_fakes")


@pytest.fixture(autouse=True)
def aborts(tmp_path, monkeypatch):
    """Every abort record of this file lands under the test's own folder, never in the real temp dir.

    On the way out the model's reference cycle (model and lifecycle) is collected and MLX's cache emptied, so the
    tiny model's arrays (about 1 GiB for the 1 + 1 block FLUX.1 set of zeros) are not still active in a later test
    that reads ``mx.get_active_memory()`` as its baseline.
    """
    import gc

    import mlx.core as mx

    from mlx_dfloat.mflux import _pipeline

    root = tmp_path / "aborts"
    monkeypatch.setattr(_pipeline, "_artifact_root", lambda: root)
    yield root
    monkeypatch.undo()  # the patched wrappers' closures reach the model; release them before the collect
    gc.collect()
    mx.clear_cache()


class Overrun:
    """One-shot: raises the fake over the stop line the first time ``fire`` runs in ``phase`` under an armed call.

    A pass-through otherwise, so an unarmed call simply returns and a retry is never tripped again. It records and
    never asserts. ``calls`` counts every ``fire``; ``after_trip`` counts the firing call and every later one.
    """

    def __init__(self, model, counters, phase):
        self.model, self.counters, self.phase = model, counters, phase
        self.armed = True
        self.tripped_in_time = None
        self.calls = 0
        self.after_trip = 0

    def fire(self):
        self.calls += 1
        if self.tripped_in_time is not None:
            self.after_trip += 1
            return
        if not self.armed or self.model.open_phase != self.phase:
            return
        watchdog = _guard.active()
        if watchdog is None:
            return
        self.armed = False
        self.counters.raise_to(STOP_LINE + 1)
        self.tripped_in_time = watchdog.tripped.wait(5.0)
        self.after_trip = 1


def _klein(tmp_path, monkeypatch, **overrides):
    from tests._flux2_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b", **overrides)


def _wrap_text_encoder(monkeypatch, before_call):
    from tests._flux2_tiny import StubTextEncoder

    real = StubTextEncoder.get_prompt_embeds

    def get_prompt_embeds(self, *args, **kwargs):
        before_call()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(StubTextEncoder, "get_prompt_embeds", get_prompt_embeds)


def _wrap_group_reads(monkeypatch, before_read):
    from mlx_dfloat.integrate import coverage

    real = coverage.load_group_mx

    def load_group_mx(group):
        out = real(group)
        before_read()
        return out

    monkeypatch.setattr(coverage, "load_group_mx", load_group_mx)


def test_klein_abort_in_encode_stops_in_encode(tmp_path, monkeypatch, fakes):
    # Bug caught: the prompt_encoded checkpoint missing from Lifecycle.ensure_embeddings (alone, or with
    # encoder_loaded), so a trip during the prompt's encoder pass caches the pair, runs on into the set load and stops
    # there (site "set_load_start").
    model = _klein(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "encode")
    _wrap_text_encoder(monkeypatch, overrun.fire)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "encode"
    assert err.site == "prompt_encoded"
    assert model._lifecycle.counters.set_loads == 0
    assert model._lifecycle.encoders_loaded is False


def test_klein_abort_in_set_load_stops_before_the_next_group(tmp_path, monkeypatch, fakes):
    # Bug caught: the set_group checkpoint missing from load_resident_set, so a tripped load reads every remaining
    # group (wrapper calls 8, not 1) and stops only at the first block (site "block_start"). And: the abort returning
    # without emptying MLX's cache when the set is not yet resident, so the groups already read stay in the buffer
    # pool (with the set load's raised cache limit) after the error leaves.
    import mlx.core as mx

    model = _klein(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "set_load")
    _wrap_group_reads(monkeypatch, overrun.fire)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "set_load"
    assert err.site == "set_group"
    assert overrun.calls == 1
    assert model._lifecycle.set_resident is False
    assert mx.get_cache_memory() == 0


def test_klein_abort_in_denoise_stops_at_the_next_block(tmp_path, monkeypatch, fakes):
    # Bug caught: the block_end checkpoint missing from seam.run_block, so a trip in a block's decode runs that block
    # and stops at the next block's block_start (or, with block_start gone too, runs every remaining block and the VAE
    # and stops at call_end); or an abort that leaves a decoded weight installed in a block or MLX's limits changed.
    from functools import partial

    from mlx_dfloat.bench.capped import current_limits
    from mlx_dfloat.decode import decode_group
    from mlx_dfloat.integrate.placeholders import PLACEHOLDER, get_attr_path

    real_decode = partial(decode_group, backend="reference")
    holder = {}

    def decode(group):
        holder["overrun"].fire()
        return real_decode(group)

    model = _klein(tmp_path, monkeypatch, decode=decode)
    overrun = holder["overrun"] = Overrun(model, fakes.counters, "denoise")
    limits_before = current_limits()

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "denoise"
    assert err.site == "block_end"
    assert overrun.after_trip <= 1
    not_placeholder = [
        f"{block}.{attr}"
        for block, per in model._shapes.items()
        for attr in per
        if get_attr_path(model.transformer, f"{block}.{attr}").weight is not PLACEHOLDER
    ]
    assert not_placeholder == []
    assert current_limits() == limits_before


def test_klein_abort_in_vae_raises_at_call_end(tmp_path, monkeypatch, fakes):
    # Bug caught: the decorator's call_end checkpoint missing, so a trip during the VAE decode (which has no
    # checkpoint) returns the image as if nothing happened, and the set stays resident past the abort.
    from tests._flux2_tiny import StubVAE

    model = _klein(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "vae")
    real = StubVAE.decode_packed_latents

    def decode_packed_latents(self, packed, tiling_config=None):
        overrun.fire()
        return real(self, packed, tiling_config=tiling_config)

    monkeypatch.setattr(StubVAE, "decode_packed_latents", decode_packed_latents)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "vae"
    assert err.site == "call_end"
    assert model._lifecycle.set_resident is False


def test_klein_abort_in_an_encoder_reload_stops_at_encoder_loaded(tmp_path, monkeypatch, fakes):
    # Bug caught: the encoder_loaded checkpoint alone missing from Lifecycle.ensure_embeddings, so a trip while a
    # new prompt reloads the text encoders runs the encoder pass anyway and stops at prompt_encoded.
    from mlx_dfloat.mflux.flux2 import init as finit

    model = _klein(tmp_path, monkeypatch)
    # Encodes "a", then drops the encoders: the next prompt reloads them.
    model.generate_image(**KLEIN_RUN)
    overrun = Overrun(model, fakes.counters, "encode")
    real_load = finit.load_text_encoder

    def load_text_encoder(root, model_config):
        encoder = real_load(root, model_config)
        overrun.fire()
        return encoder

    monkeypatch.setattr(finit, "load_text_encoder", load_text_encoder)
    passes = []
    _wrap_text_encoder(monkeypatch, lambda: passes.append(model.open_phase))

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**{**KLEIN_RUN, "prompt": "b"})

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "encode"
    assert err.site == "encoder_loaded"
    assert passes == []


def test_klein_abort_before_the_set_load_stops_at_set_load_start(tmp_path, monkeypatch, fakes):
    # Bug caught: the set_load_start checkpoint alone missing from Lifecycle.ensure_set, so a trip during the
    # pre-load memory check starts the load, reads no group but stops at the first set_group instead.
    model = _klein(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "set_load")
    real_bound = model._lifecycle.retained_bound

    def retained_bound():
        overrun.fire()  # a no-op outside the set-load phase (the encoders' reclaim also calls it)
        return real_bound()

    monkeypatch.setattr(model._lifecycle, "retained_bound", retained_bound)
    reads = []
    _wrap_group_reads(monkeypatch, lambda: reads.append(1))

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "set_load"
    assert err.site == "set_load_start"
    assert reads == []


def test_klein_abort_at_a_step_start_stops_at_block_start(tmp_path, monkeypatch, fakes):
    # Bug caught: the block_start checkpoint alone missing from seam.run_block, so a trip at the start of a denoise
    # step (before any block) decodes and runs the first block and stops at its block_end.
    from mlx_dfloat.integrate import seam

    model = _klein(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "denoise")
    real_begin = seam.begin_step

    def begin_step(state):
        overrun.fire()
        return real_begin(state)

    monkeypatch.setattr(seam, "begin_step", begin_step)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**KLEIN_RUN)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "denoise"
    assert err.site == "block_start"


def test_klein_nested_encode_arms_once(tmp_path, monkeypatch):
    # Bug caught: the decorator's active-guard check missing, so generate_image's nested self.encode() arms a second
    # watchdog (two sampler threads, two stop lines) instead of running under the outer one.
    model = _klein(tmp_path, monkeypatch)
    counts = []

    def count_threads():
        if model.open_phase == "encode":
            counts.append(
                sum(1 for t in threading.enumerate() if t.name == THREAD and t.is_alive())
            )

    _wrap_text_encoder(monkeypatch, count_threads)

    image = model.generate_image(**KLEIN_RUN)

    assert image is not None
    assert counts == [1]


def _flux1_fake(tmp_path, monkeypatch):
    """FLUX.1 over the fake-width seamed transformer and a real 2 + 2 block DF11 set, its real ``_load_set``."""
    from tests.test_flux1_model import _fake_model

    return _fake_model(tmp_path, monkeypatch)


def test_flux1_abort_in_set_load_leaves_no_set(tmp_path, monkeypatch, fakes):
    # Bug caught: a FLUX.1 set-load abort (its _load_set has no unload guard) that marks the set resident or leaves a
    # half-attached set, so the retry skips the load or is refused; or the load running on past the trip (the set
    # resident and the stop at call_end, not set_group).
    from tests.test_flux1_model import _patch_upstream_generate

    model = _flux1_fake(tmp_path, monkeypatch)
    _patch_upstream_generate(monkeypatch, model)
    overrun = Overrun(model, fakes.counters, "set_load")
    _wrap_group_reads(monkeypatch, overrun.fire)
    run = {"seed": 1, "prompt": "p", "height": 256, "width": 256}

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**run)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "set_load"
    assert err.site == "set_group"
    assert model._lifecycle.set_resident is False

    fakes.counters.raise_to(0)
    assert model.generate_image(**run) == "image"
    assert model._lifecycle.counters.set_loads == 1


def test_flux1_abort_traceback_does_not_pin_the_provider(tmp_path, monkeypatch, fakes):
    # Bug caught: the decorator not clearing the abort's frames (clear_call_frames). FLUX.1 clears frames only for
    # DFloatFormatError, so the seam frames in the traceback would keep the compressed set's provider alive for as
    # long as the caller holds the error.
    from mflux.models.flux.variants.txt2img.flux import Flux1
    from tests._flux_fakes import inputs

    from mlx_dfloat.integrate.providers import DF11Provider

    model = _flux1_fake(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "denoise")
    refs = []
    real_weights_for = DF11Provider.weights_for

    def weights_for(provider, block_name, shapes):
        if not refs:
            refs.append(weakref.ref(provider))
        overrun.fire()
        return real_weights_for(provider, block_name, shapes)

    monkeypatch.setattr(DF11Provider, "weights_for", weights_for)

    def upstream(self, **kwargs):
        # One step of the seamed fake-width transformer, as mflux's loop would run it.
        self.transformer(*inputs())
        return "image"

    monkeypatch.setattr(Flux1, "generate_image", upstream)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(seed=1, prompt="p", height=256, width=256)

    assert overrun.tripped_in_time is True
    assert excinfo.value.phase == "denoise"
    assert model._provider is None
    assert len(refs) == 1
    assert refs[0]() is None  # dead while excinfo.value is still held


def test_flux1_abort_in_vae_raises_at_call_end(tmp_path, monkeypatch, fakes):
    # Bug caught: the decorator's call_end checkpoint missing, so a trip during FLUX.1's VAE decode (evaluated at
    # np.array inside ImageUtil.to_image, with no checkpoint) returns the image. Runs mflux's real Flux1 loop on the
    # seamed 1 + 1 block transformer over a resident set of zeros (tests/test_mflux_api_watchdog.py).
    from tests.test_flux1_model import _StubVAE
    from tests.test_mflux_api_watchdog import _flux1

    model, _name, run, _provider_cls = _flux1(tmp_path, monkeypatch)
    overrun = Overrun(model, fakes.counters, "vae")
    real = _StubVAE.decode

    def decode(self, latents):
        overrun.fire()
        return real(self, latents)

    monkeypatch.setattr(_StubVAE, "decode", decode)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(**run)

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.phase == "vae"
    assert err.site == "call_end"
    assert model._lifecycle.set_resident is False


# --- a direct encode() after an abort, and the phase it reports


class DirectEncodeOverrun:
    """One-shot: raises the fake over the stop line at the first encoder pass under an armed call, in any phase.

    Wraps the lifecycle's encode callback, so it fires in a direct ``model.encode()`` (which opens no phase) as well
    as in a call. ``phase_seen`` is ``model.open_phase`` at the firing pass. It records and never asserts.
    """

    def __init__(self, model, counters):
        self.model, self.counters = model, counters
        self.armed = True
        self.tripped_in_time = None
        self.phase_seen = "not fired"

    def install(self, monkeypatch):
        real = self.model._lifecycle._encode

        def encode(prompt):
            watchdog = _guard.active()
            if self.armed and watchdog is not None:
                self.armed = False
                self.phase_seen = self.model.open_phase
                self.counters.raise_to(STOP_LINE + 1)
                self.tripped_in_time = watchdog.tripped.wait(5.0)
            return real(prompt)

        monkeypatch.setattr(self.model._lifecycle, "_encode", encode)


def test_klein_direct_encode_overrun_stops_at_prompt_encoded(tmp_path, monkeypatch, fakes):
    # Bug caught: encode() not armed on its own (the decorator missing from the public encode), so a direct
    # model.encode() runs the encoder pass unwatched, caches the pair and returns.
    model = _klein(tmp_path, monkeypatch)
    overrun = DirectEncodeOverrun(model, fakes.counters)
    overrun.install(monkeypatch)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.encode("p")

    err = excinfo.value
    assert overrun.tripped_in_time is True
    assert err.site == "prompt_encoded"
    assert err.phase is None
    assert "p" not in model.prompt_cache


def _zimage_fake(tmp_path, monkeypatch):
    from tests._zimage_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, model="z-image-turbo")


def _trip_in_encode(model, counters, monkeypatch):
    overrun = Overrun(model, counters, "encode")
    real = model._lifecycle._encode

    def encode(prompt):
        overrun.fire()
        return real(prompt)

    monkeypatch.setattr(model._lifecycle, "_encode", encode)
    return overrun


def _trip_in_set_load(model, counters, monkeypatch):
    overrun = Overrun(model, counters, "set_load")
    _wrap_group_reads(monkeypatch, overrun.fire)
    return overrun


@pytest.mark.parametrize("build", [_flux1_fake, _zimage_fake], ids=["flux1", "zimage"])
@pytest.mark.parametrize(
    ("trip", "phase", "site"),
    [
        pytest.param(_trip_in_encode, "encode", "prompt_encoded", id="encode"),
        pytest.param(_trip_in_set_load, "set_load", "set_group", id="set_load"),
    ],
)
def test_an_abort_closes_its_phase_for_a_later_direct_encode(
    build, trip, phase, site, tmp_path, monkeypatch, fakes
):
    # Bug caught: FLUX.1 or Z-Image closing the encode or set-load phase only on a normal return (no `finally`), so
    # after an abort there the phase stays open and a later direct encode() that trips names that stale phase in
    # its error and its record. A direct encode() opens no phase, as on FLUX.2 Klein
    # (test_klein_direct_encode_overrun_stops_at_prompt_encoded).
    model = build(tmp_path, monkeypatch)
    first = trip(model, fakes.counters, monkeypatch)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.generate_image(seed=1, prompt="p", height=256, width=256)

    assert first.tripped_in_time is True
    assert (excinfo.value.phase, excinfo.value.site) == (phase, site)
    assert model.open_phase is None

    fakes.counters.raise_to(0)
    later = DirectEncodeOverrun(model, fakes.counters)
    later.install(monkeypatch)

    with pytest.raises(DFloatWatchdogError) as excinfo:
        model.encode("q")

    err = excinfo.value
    assert later.tripped_in_time is True
    assert later.phase_seen is None
    assert err.site == "prompt_encoded"
    assert err.phase is None
    assert err.record["context"]["phase"] is None
