import pytest

from mlx_dfloat.integrate.memory import fit_estimate
from mlx_dfloat.mflux.flux1.memory import activation_allowance, cache_limit_for, flux_phases

GIB = 1024**3
LARGEST = {"transformer_blocks": 679_000_000, "single_transformer_blocks": 283_000_000}


def test_allowance_scales_with_image_and_text_tokens_and_has_a_floor():
    # Bug caught: using a fixed 1.5e9 allowance regardless of token count (dropping the linear scale).
    assert activation_allowance(height=1024, width=1024, text_tokens=256) == pytest.approx(1.5e9)
    assert activation_allowance(height=1024, width=1024, text_tokens=512) == pytest.approx(
        1.5e9 * 4608 / 4352
    )
    assert activation_allowance(height=256, width=256, text_tokens=256) == 500_000_000  # the floor


def test_cache_limit_at_schnell_1024_is_the_measured_working_point():
    # Bug caught: omitting the 2.5 GB measured-minimum floor at 1024x1024, returning the raw ~2.46 GB sum.
    # 0.68 + 0.28 + 1.5 = 2.46 GB derived, floored to the measured 2.5 GB.
    assert (
        cache_limit_for(LARGEST, policy="per-block", height=1024, width=1024, text_tokens=256)
        == 2_500_000_000
    )


def test_cache_limit_adds_a_group_under_depth2_and_honours_an_override():
    # Bug caught: the depth2 policy not adding an extra decoded group's worth of headroom, or an
    # explicit override not short-circuiting the whole computation.
    per_block = cache_limit_for(
        LARGEST, policy="per-block", height=1024, width=1024, text_tokens=512
    )
    depth2 = cache_limit_for(LARGEST, policy="depth2", height=1024, width=1024, text_tokens=512)
    assert depth2 - per_block == 679_000_000
    assert (
        cache_limit_for(
            LARGEST,
            policy="depth2",
            height=1024,
            width=1024,
            text_tokens=512,
            override=3_000_000_000,
        )
        == 3_000_000_000
    )


def test_the_denoise_phase_estimate_lands_in_the_measured_band_for_schnell_1024():
    # The estimate must never refuse the configuration it was built on: the 0.1 GiB above the band
    # is its whole margin of conservatism. Measured 19.2-19.6 GiB with the 16.3 GB set resident,
    # per-block, 2.5 GB cache; the estimate is 19.66 GiB, just above that band.
    # Bug caught: a term dropped from (or double-counted in) the denoise sum, the per-block in-flight
    # buffer taken as the smaller kind, or the VAE phase no longer the peak at this size.
    phases = flux_phases(
        compressed_bytes=16_330_000_000,
        extras_bytes=100_000_000,
        largest=LARGEST,
        policy="per-block",
        cache_limit=2_500_000_000,
        allowance=1_500_000_000,
        encoders_bytes=9_900_000_000,
        vae_bytes=160_000_000,
        vae_transient_bytes=int(3.5 * GIB),
        overhead_bytes=int(1.4 * GIB),
    )
    est = fit_estimate(phases, budget_bytes=int(23.0 * GIB))
    denoise = 16_330_000_000 + 100_000_000 + 679_000_000 + 2_500_000_000 + int(1.4 * GIB)
    assert est.phases["denoise"] == denoise
    assert 19.2 * GIB <= est.phases["denoise"] <= (19.6 + 0.1) * GIB
    assert est.peak_phase == "vae"
    assert est.fits


def test_fit_estimate_refuses_over_budget_with_the_peak_phase_named():
    # Bug caught: picking the first phase instead of the largest-total phase as the peak, or comparing
    # peak_bytes to budget_bytes with the wrong direction (fits True when over budget).
    est = fit_estimate({"a": {"x": 10}, "b": {"y": 30, "z": 5}}, budget_bytes=20)
    assert est.peak_phase == "b"
    assert est.peak_bytes == 35
    assert est.fits is False


def test_flux_phases_depth2_doubles_the_in_flight_decoded_buffer():
    # Bug caught: an inverted or missing multiplier under-estimating the depth-2 in-flight decoded
    # buffers (or over-estimating the per-block ones).
    common = {
        "compressed_bytes": 16_330_000_000,
        "extras_bytes": 100_000_000,
        "largest": LARGEST,
        "cache_limit": 2_500_000_000,
        "allowance": 1_500_000_000,
        "encoders_bytes": 9_900_000_000,
        "vae_bytes": 160_000_000,
        "vae_transient_bytes": int(3.5 * GIB),
        "overhead_bytes": int(1.4 * GIB),
    }
    per_block = flux_phases(policy="per-block", **common)
    depth2 = flux_phases(policy="depth2", **common)
    assert per_block["denoise"]["decoded"] == max(LARGEST.values())
    assert depth2["denoise"]["decoded"] == 2 * max(LARGEST.values())


def test_cache_limit_below_1024_squared_is_not_floored():
    # Bug caught: an inverted resolution condition force-capping every non-1024x1024 run at 2.5 GB.
    expected = (
        LARGEST["transformer_blocks"]
        + LARGEST["single_transformer_blocks"]
        + activation_allowance(height=512, width=512, text_tokens=256)
    )
    assert expected < 2_500_000_000
    assert (
        cache_limit_for(LARGEST, policy="per-block", height=512, width=512, text_tokens=256)
        == expected
    )
