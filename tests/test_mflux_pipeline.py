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
# Klein 4B decoded group sizes (from the header's matrix shapes): double block 490_733_568 B, single block 245_366_784 B.
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
    assert "(the two largest decoded groups)" in caplog.text


def test_a_one_kind_familys_derived_minimum_includes_the_activation_allowance(caplog):
    # Bug caught: a one-kind family's minimum taken as its one decoded group (any override of a block or more passes
    # silently, though the activations then evict the decoded buffer and every block is allocated fresh). One kind of
    # 436_207_616 B; allowance at 1024^2 + 512 tokens 1_588_235_294: minimum 2_024_442_910.
    one = {"transformer_blocks": 436_207_616}
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(largest=one, cache_limit_override=2_024_442_910)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(largest=one, cache_limit_override=2_024_442_909)
    assert (
        "below the derived minimum 2024442910 (the largest decoded group and the activation allowance)"
        in caplog.text
    )


def test_a_one_kind_familys_depth2_minimum_holds_the_look_ahead_group_too(caplog):
    # Bug caught: depth2's look-ahead buffer left out of a one-kind family's minimum (an override with room for one
    # decoded group passes silently though two are in flight). One kind of 436_207_616 B and the allowance at
    # 1024^2 + 512 tokens 1_588_235_294: per-block minimum 2_024_442_910, depth2 2_460_650_526.
    one = {"transformer_blocks": 436_207_616}
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(largest=one, policy="depth2", cache_limit_override=2_460_650_526)
    assert "below the derived minimum" not in caplog.text
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(largest=one, policy="depth2", cache_limit_override=2_460_650_525)
    assert (
        "below the derived minimum 2460650526 (the largest decoded group twice, for depth2's look-ahead, and the "
        "activation allowance)" in caplog.text
    )


def test_a_one_kind_warning_states_the_fresh_allocation_as_a_prediction(caplog):
    # Bug caught: the warning asserting as fact that every block is allocated fresh: Qwen-Image 2.1's A/B of real
    # steps at the derived limit and below it measured no difference in step time, so the warning may only predict it.
    with caplog.at_level("WARNING", logger="tests.pipeline"):
        _plan(largest={"transformer_blocks": 436_207_616}, cache_limit_override=1)
    assert "may allocate each block's decode output fresh" in caplog.text
    assert "will be allocated fresh" not in caplog.text


# --- the per-call memory caps -------------------------------------------------------------------------------------


class FakeLimits:
    """MLX's limit setters as mlx 0.32.2 has them (each returns the previous value) and its device report."""

    def __init__(self, *, wired, memory, recommended=26_800_603_136):
        self.wired, self.memory, self.recommended = wired, memory, recommended

    def set_wired_limit(self, n):
        previous, self.wired = self.wired, n
        return previous

    def set_memory_limit(self, n):
        previous, self.memory = self.memory, n
        return previous

    def device_info(self):
        return {"max_recommended_working_set_size": self.recommended}


# MLX's default memory limit on this Mac: 1.5 x the recommended working set.
DEFAULT_MEMORY = 40_200_904_704


def test_a_call_without_caps_runs_under_the_commands_caps_and_restores_mlxs_defaults():
    # Bug caught: the Python API running a call at MLX's default wired limit 0 (a Klein VAE decode's MLX peak went
    # 6.89 -> 9.49 GiB there, and every VAE term was measured under the caps), or the caps left installed after the
    # call. This Mac's 26_800_603_136 B working set floors to 24 GiB: wired 20 GiB, memory 22 GiB (the command's caps).
    limits = FakeLimits(wired=0, memory=DEFAULT_MEMORY)
    with _pipeline.call_caps(limits) as installed:
        assert (limits.wired, limits.memory) == (21_474_836_480, 23_622_320_128)
        assert installed == (21_474_836_480, 23_622_320_128)
    assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)


def test_a_call_that_raises_still_restores_mlxs_defaults():
    # Bug caught: the restore skipped on an exception (an interrupted generation leaves the process capped).
    limits = FakeLimits(wired=0, memory=DEFAULT_MEMORY)
    with pytest.raises(KeyboardInterrupt), _pipeline.call_caps(limits):
        raise KeyboardInterrupt
    assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)


def test_caps_already_in_force_are_left_as_they_are():
    # Bug caught: a call replacing the caps the command (or a CAPPED tier: wired 8 GiB, memory 10 GiB at 16 GB)
    # installed with the host's, or restoring them to MLX's defaults afterwards.
    limits = FakeLimits(wired=8 * GIB, memory=10 * GIB)
    with _pipeline.call_caps(limits) as installed:
        assert (limits.wired, limits.memory) == (8 * GIB, 10 * GIB)
        assert installed == (0, 0)
    assert (limits.wired, limits.memory) == (8 * GIB, 10 * GIB)


def test_a_device_that_reports_no_working_set_gets_no_caps():
    # Bug caught: caps of 0 bytes installed (a memory limit of 0) on a device that reports nothing, as
    # install_memory_caps itself installs nothing there.
    limits = FakeLimits(wired=0, memory=DEFAULT_MEMORY, recommended=0)
    with _pipeline.call_caps(limits) as installed:
        assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)
        assert installed == (0, 0)
    assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)


def test_the_command_path_keeps_its_caps_through_a_call():
    # Bug caught: the command's own caps (installed before the model is built, as tests/conftest.py installs them
    # here) changed by a call: real MLX, the limits read before, inside and after.
    from tests._mlx_limits import current_limits

    before = current_limits()
    assert before["wired"] > 0
    with _pipeline.call_caps():
        inside = current_limits()
    assert inside == before
    assert current_limits() == before


class RecordingLimits(FakeLimits):
    """FakeLimits that records every wired-limit call; with ``refuse_wired`` it raises as MLX does for a wired limit
    above what the system allows (``[metal::set_wired_limit] ... is not allowed``, a ValueError)."""

    def __init__(self, *, refuse_wired=False, **kwargs):
        super().__init__(**kwargs)
        self.wired_calls = []
        self.refuse_wired = refuse_wired

    def set_wired_limit(self, n):
        self.wired_calls.append(n)
        if self.refuse_wired and n > 0:
            raise ValueError(
                "[metal::set_wired_limit] Setting a wired limit larger than the maximum is not allowed."
            )
        return super().set_wired_limit(n)


def test_a_wired_cap_mlx_refuses_still_installs_the_memory_cap_and_warns_once(monkeypatch):
    # Bug caught (P1): a Python generate_image raising MLX's raw ValueError when the system's wired limit was lowered
    # (sysctl iogpu.wired_limit_mb), where install_memory_caps keeps the memory cap and goes on; or a warning on every
    # call. The memory cap (22 GiB here) is installed for the block and MLX's default restored after it.
    monkeypatch.setattr(_pipeline, "_wired_refusal_warned", False)
    limits = RecordingLimits(refuse_wired=True, wired=0, memory=DEFAULT_MEMORY)
    import warnings

    inside = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _pipeline.call_caps(limits) as installed:
            inside.append((installed, limits.wired, limits.memory))
    assert [str(w.message) for w in caught if "wired" in str(w.message)] != []
    assert inside == [((0, 23_622_320_128), 0, 23_622_320_128)]
    assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with _pipeline.call_caps(limits) as again:
            assert again == (0, 23_622_320_128)


def test_a_second_call_does_not_probe_the_wired_limit():
    # Bug caught (P2): the probe (set the wired limit to 0, read the previous value, set it back) run at every call,
    # with a compressed set resident, when this process already knows the limit it left in force. First call: probe
    # (0, then the 0 it read), install 20 GiB, restore 0; the second: install and restore only.
    limits = RecordingLimits(wired=0, memory=DEFAULT_MEMORY)
    with _pipeline.call_caps(limits):
        pass
    with _pipeline.call_caps(limits):
        pass
    assert limits.wired_calls == [0, 0, 21_474_836_480, 0, 21_474_836_480, 0]


def test_the_command_path_never_probes(monkeypatch):
    # Bug caught (P2): a call probing the wired limit after the command installed its caps (install_memory_caps
    # records what it installed): the only wired-limit call is the command's own 20 GiB.
    from mlx_dfloat import _memory_caps

    limits = RecordingLimits(wired=0, memory=DEFAULT_MEMORY)
    monkeypatch.setattr(_memory_caps, "mx", limits)
    assert _memory_caps.install_memory_caps() == (20, 22)
    with _pipeline.call_caps(limits) as installed:
        assert installed == (0, 0)
    assert limits.wired_calls == [21_474_836_480]


def test_a_wired_cap_the_command_could_not_install_is_recorded_so_calls_never_probe(monkeypatch):
    # Bug caught (R2): install_memory_caps leaving the wired limit unknown when MLX refuses its cap, so the first call
    # probes it after all (set to 0 and back). Recorded as 0: the call tries the cap once (refused again, one warning)
    # and never writes 0. The only wired-limit calls are the two refused 20 GiB attempts.
    import warnings

    from mlx_dfloat import _memory_caps

    monkeypatch.setattr(_pipeline, "_wired_refusal_warned", False)
    limits = RecordingLimits(refuse_wired=True, wired=0, memory=DEFAULT_MEMORY)
    monkeypatch.setattr(_memory_caps, "mx", limits)
    assert _memory_caps.install_memory_caps() == (0, 22)
    assert _memory_caps.known_wired_limit(limits) == 0
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with _pipeline.call_caps(limits) as installed:
            assert installed == (0, 23_622_320_128)
    assert limits.wired_calls == [21_474_836_480, 21_474_836_480]


def test_a_nested_call_leaves_the_outer_calls_caps_alone():
    # Bug caught: an encode() run inside generate_image (both capped) restoring MLX's defaults when it returns, so
    # the rest of the generation runs uncapped.
    limits = RecordingLimits(wired=0, memory=DEFAULT_MEMORY)
    with _pipeline.call_caps(limits):
        with _pipeline.call_caps(limits) as inner:
            assert inner == (0, 0)
        assert (limits.wired, limits.memory) == (21_474_836_480, 23_622_320_128)
    assert (limits.wired, limits.memory) == (0, DEFAULT_MEMORY)


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
