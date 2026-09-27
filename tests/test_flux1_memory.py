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
    # Review focus 5: the estimate must never refuse the configuration it was built on.
    # 0021 measured 19.2-19.6 GiB with the 16.3 GB set resident, per-block, 2.5 GB cache.
    # Bug caught: dropping a term from the denoise phase sum (e.g. forgetting the cache limit or the
    # overhead), which would push the estimate below the measured band.
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
    assert 19.0 * GIB <= est.phases["denoise"] <= 20.0 * GIB
    assert est.fits
    assert est.peak_phase in {"denoise", "vae"}


def test_fit_estimate_refuses_over_budget_with_the_peak_phase_named():
    # Bug caught: picking the first phase instead of the largest-total phase as the peak, or comparing
    # peak_bytes to budget_bytes with the wrong direction (fits True when over budget).
    est = fit_estimate({"a": {"x": 10}, "b": {"y": 30, "z": 5}}, budget_bytes=20)
    assert est.peak_phase == "b"
    assert est.peak_bytes == 35
    assert est.fits is False
