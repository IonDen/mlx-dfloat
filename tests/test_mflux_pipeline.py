"""The family-free pieces of an mflux model class: refusals, the phase tracker, the VAE pool guard, the call plan.

Offline: no mflux, no checkpoint. Expected values are hand-worked literals.
"""

import logging

import mlx.core as mx
import pytest

from mlx_dfloat.errors import DFloatResourceError, DFloatUnsupportedError
from mlx_dfloat.integrate.memory import CallPlan, FitEstimate
from mlx_dfloat.mflux import _pipeline
from mlx_dfloat.mflux._phases import FamilySizes, PhaseConstants

GIB = 1024**3
# Klein 4B decoded group sizes (S3 plan arithmetic table): double block 490_733_568 B, single block 245_366_784 B.
LARGEST = {"transformer_blocks": 490_733_568, "single_transformer_blocks": 245_366_784}
LOG = logging.getLogger("tests.pipeline")


def _constants(**over):
    base = {
        "overhead_bytes": 0,
        "vae_transient_bytes": 0,
        "denoise_activation_at_reference": 0,
        "reference_tokens": 4096 + 512,
    }
    return PhaseConstants(**{**base, **over})


def _sizes(**over):
    base = {"compressed": 0, "extras": 0, "nonblock": 0, "encoders": 0, "vae": 0}
    return FamilySizes(**{**base, **over})


def _plan(**over):
    args = {
        "constants": _constants(),
        "sizes": _sizes(),
        "largest": LARGEST,
        "policy": "per-block",
        "cache_limit_override": None,
        "fit_check": True,
        "budget": 40 * GIB,
        "height": 1024,
        "width": 1024,
        "text_tokens": 512,
        "log": LOG,
    }
    return _pipeline.plan_call_for(**{**args, **over})


# --- refusals -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"quantize": 8}, "quantize"),
        ({"lora_paths": ["a.safetensors"]}, "lora_paths"),
        ({"lora_scales": [1.0]}, "lora_scales"),
        ({"bake_lora": False}, "bake_lora"),
    ],
)
def test_each_construction_argument_this_path_cannot_honour_is_refused_by_name(kwargs, name):
    # Bug caught: a quantize / LoRA argument silently accepted (the output would no longer be the checkpoint's).
    args = {"quantize": None, "lora_paths": None, "lora_scales": None, "bake_lora": True, **kwargs}
    with pytest.raises(DFloatUnsupportedError, match=rf"^{name}: .*not on the DFloat11 path"):
        _pipeline.refuse_construction_args(**args)


def test_the_default_construction_arguments_pass():
    # Bug caught: the defaults themselves refused (empty LoRA lists are falsy and must pass too).
    _pipeline.refuse_construction_args(quantize=None, lora_paths=[], lora_scales=[], bake_lora=True)


def test_an_eval_policy_outside_the_familys_is_refused():
    # Bug caught: policy "none" (every decoded block kept resident) accepted.
    _pipeline.check_eval_policy("depth2", ("per-block", "depth2"))
    with pytest.raises(DFloatUnsupportedError, match="'none'"):
        _pipeline.check_eval_policy("none", ("per-block", "depth2"))


# --- the call plan ------------------------------------------------------------------------------------------------


def test_plan_call_rounds_to_16_before_the_estimate():
    # Bug caught: planning 1030x1030 while mflux runs 1024x1024. At 1024^2 + 512 text tokens the allowance is
    # 1.5e9 * 4608 / 4352 = 1_588_235_294; + 490_733_568 + 245_366_784 = 2_324_335_646. At an unrounded 1030^2 the
    # token count is 4144 + 512 = 4656: allowance 1_604_779_411, limit 2_340_879_763.
    plan = _plan(height=1030, width=1030)
    assert plan.cache_limit == 2_324_335_646
    assert _plan(cache_limit_override=7).cache_limit == 7


def test_above_1024_squared_is_refused_with_fit_check_and_warns_without(caplog):
    # Bug caught: 1040x1024 (above the measured size once rounded) accepted without a measurement, or refused even
    # when the caller asked to run on an extrapolation.
    with pytest.raises(DFloatResourceError, match="1024x1024"):
        _plan(height=1040, width=1024)
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        plan = _plan(height=1040, width=1024, fit_check=False)
    assert isinstance(plan, CallPlan)
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "extrapolation" in caplog.text


def test_the_set_is_dropped_before_the_vae_only_when_the_vae_phase_exceeds_the_budget():
    # Bug caught: `>=` for `>` (dropped at exactly the budget), or the decision taken on the peak phase instead of the
    # VAE phase. VAE phase with the set = compressed 1e9 + transient 1e10 = 11_000_000_000 (denoise 3_815_069_214).
    c = _constants(vae_transient_bytes=10_000_000_000)
    s = _sizes(compressed=1_000_000_000)
    at = _plan(constants=c, sizes=s, budget=11_000_000_000)
    assert at.drop_set_before_vae is False
    assert at.estimate.phases["vae"] == 11_000_000_000
    over = _plan(constants=c, sizes=s, budget=10_999_999_999)
    assert over.drop_set_before_vae is True
    assert over.estimate.phases["vae"] == 10_000_000_000  # the set left the phase
    assert over.estimate.fits
    # Denoise (1e9 + 490_733_568 + 2_324_335_646 = 3_815_069_214) is the peak and over a 2e9 budget, the VAE phase
    # (1e9) is not: no drop.
    peak_not_vae = _plan(sizes=s, budget=2_000_000_000, fit_check=False)
    assert peak_not_vae.estimate.peak_phase == "denoise"
    assert peak_not_vae.drop_set_before_vae is False


def test_a_refused_estimate_names_the_peak_phase_and_every_phase(caplog):
    # Bug caught: a refusal that names neither the phase over budget nor the others (the user cannot tell whether a
    # smaller image or a bigger Mac helps), or fit_check=False still refusing.
    s = _sizes(compressed=20 * GIB)
    with pytest.raises(DFloatResourceError) as info:
        _plan(sizes=s, budget=10 * GIB)
    message = str(info.value)
    assert "in the denoise phase" in message
    assert "10.0 GiB" in message
    for phase in ("encode", "denoise", "vae"):
        assert f"{phase} " in message
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        plan = _plan(sizes=s, budget=10 * GIB, fit_check=False)
    assert not plan.estimate.fits
    assert "running anyway" in caplog.text


def test_a_cache_limit_below_the_two_largest_groups_warns(caplog):
    # Bug caught: an override below the two largest decoded groups accepted silently (every block's decode output
    # allocated fresh), or the minimum taken from one group (one byte under the two-group sum passing quietly).
    two = 490_733_568 + 245_366_784
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(cache_limit_override=two)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(cache_limit_override=two - 1)
    assert "below the derived minimum" in caplog.text


# --- the phase tracker --------------------------------------------------------------------------------------------


def test_the_phase_tracker_resets_the_peak_and_ignores_an_end_for_another_phase():
    # Bug caught: the MLX peak not reset at a phase start (phase b reports phase a's 64 MiB spike), or an end for a
    # phase that is not open closing the open one.
    tracker = _pipeline.PhaseTracker()
    assert tracker.open_phase is None
    tracker.begin("a")
    spike = mx.zeros(16 * 1024 * 1024, dtype=mx.float32)  # 64 MiB
    mx.eval(spike)
    del spike
    mx.synchronize()  # the buffer is released when its command buffer completes (mlx 0.32.2)
    mx.clear_cache()
    tracker.end("b")  # not the open phase: ignored
    assert tracker.open_phase == "a"
    assert tracker.peaks["a"]["footprint_end"] == 0
    tracker.end("a")
    assert tracker.open_phase is None
    assert tracker.peaks["a"]["footprint_end"] > 0
    tracker.begin("b")
    small = mx.zeros(1024, dtype=mx.float32)
    mx.eval(small)
    tracker.end("b")
    assert tracker.peaks["b"]["mlx_peak"] < tracker.peaks["a"]["mlx_peak"] - 32 * 1024 * 1024
    assert set(tracker.peaks["a"]) == {
        "footprint_start",
        "mlx_peak",
        "footprint_end",
        "active_at_start",
    }


# --- the VAE pool guard -------------------------------------------------------------------------------------------


def _estimate():
    return FitEstimate(phases={"vae": 1}, peak_phase="vae", peak_bytes=1, budget_bytes=2)


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (None, ["end denoise", "clear", "limit 0", "begin vae"]),
        (
            CallPlan(cache_limit=5, estimate=_estimate(), drop_set_before_vae=False),
            ["end denoise", "clear", "limit 0", "begin vae"],
        ),
        (
            CallPlan(cache_limit=5, estimate=_estimate(), drop_set_before_vae=True),
            ["end denoise", "drop set", "clear", "limit 0", "begin vae"],
        ),
    ],
)
def test_the_vae_guard_ends_denoise_drops_when_planned_and_caps_the_pool(
    monkeypatch, plan, expected
):
    # Bug caught: the set dropped on every call (a reload per call on a roomy Mac) or kept when the plan says the
    # decode cannot hold it; the pool not emptied and capped before the decode; the phases closed out of order.
    events = []
    monkeypatch.setattr(mx, "clear_cache", lambda: events.append("clear"))
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: events.append(f"limit {n}") or 0)
    guard = _pipeline.VaePoolGuard(
        plan=lambda: plan,
        end_denoise=lambda: events.append("end denoise"),
        drop_set=lambda: events.append("drop set"),
        begin_vae=lambda: events.append("begin vae"),
    )
    guard.call_after_loop(seed=1, prompt="p", latents=None, config=None)
    assert events == expected
