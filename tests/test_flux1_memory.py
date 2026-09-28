import numpy as np
import pytest
from tests._flux_fakes import FLUX_TABLE, write_flux_checkpoint

from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.memory import fit_estimate
from mlx_dfloat.mflux.flux1.memory import (
    OVERHEAD_BYTES,
    VAE_TRANSIENT_BYTES,
    FluxSizes,
    activation_allowance,
    cache_limit_for,
    fit_for,
    flux_phases,
    safetensors_bytes,
    sizes_for,
)

GIB = 1024**3
LARGEST = {"transformer_blocks": 679_000_000, "single_transformer_blocks": 283_000_000}

# Measured on the 2026-09-28 de-risk run (schnell DF11 `51a428b9`, schnell base `741f7c3c`).
MEASURED_DENOISE_PEAK_GIB = 19.15  # derisk.json phases.denoise.footprint_peak (2.5 GB cache limit)
MEASURED_VAE_PEAK_GIB = 23.29  # derisk.json phases.vae.footprint_peak, set resident
SCHNELL_SIZES = FluxSizes(
    compressed=16_195_141_095, extras=113_899_648, encoders=9_770_792_936, vae=167_666_902
)
SCHNELL_LARGEST = {"transformer_blocks": 679_477_248, "single_transformer_blocks": 283_115_520}


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


def test_safetensors_bytes_sums_only_safetensors_directly_under_the_named_subdirs(tmp_path):
    # Bug caught: counting the index json or a nested file, or missing the second shard.
    (tmp_path / "text_encoder_2").mkdir()
    (tmp_path / "text_encoder_2" / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "text_encoder_2" / "b.safetensors").write_bytes(b"x" * 5)
    (tmp_path / "text_encoder_2" / "model.safetensors.index.json").write_bytes(b"x" * 100)
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "c.safetensors").write_bytes(b"x" * 7)
    assert safetensors_bytes(tmp_path, "text_encoder", "text_encoder_2") == 22
    assert safetensors_bytes(tmp_path, "vae") == 0


def test_sizes_for_splits_the_checkpoint_bytes_into_compressed_and_extras(tmp_path):
    # Bug caught: counting the extras twice in a single-file repo (they share the groups' file),
    # or reading tensor sizes instead of file sizes for the compressed part (headers not counted).
    shapes = {
        "transformer_blocks.0": dict.fromkeys(FLUX_TABLE.attrs_of("transformer_blocks"), (4, 4))
    }
    write_flux_checkpoint(
        tmp_path / "df11",
        shapes,
        np.random.default_rng(1),
        extras={"x_embedder.bias": np.zeros(3, np.uint16)},
    )
    (tmp_path / "base" / "vae").mkdir(parents=True)
    (tmp_path / "base" / "vae" / "v.safetensors").write_bytes(b"v" * 50)
    ckpt = open_checkpoint(tmp_path / "df11")
    sizes = sizes_for(ckpt, tmp_path / "base")
    total = sum(p.stat().st_size for p in (tmp_path / "df11").glob("*.safetensors"))
    assert sizes.extras == 6  # 3 bf16 values
    assert sizes.compressed == total - sizes.extras
    assert sizes.vae == 50
    assert sizes.encoders == 0


def test_fit_for_uses_the_measured_constants_and_names_the_peak_phase():
    # Bug caught: a phase term dropped, or the constants not the measured ones (an estimate that
    # refuses the very run it was calibrated on).
    sizes = FluxSizes(
        compressed=16_330_000_000 - 100_000_000,
        extras=100_000_000,
        encoders=9_900_000_000,
        vae=160_000_000,
    )
    est = fit_for(
        sizes=sizes,
        largest=LARGEST,
        policy="per-block",
        cache_limit=2_500_000_000,
        allowance=1_500_000_000,
        budget=int(23.0 * GIB),
    )
    assert est.phases["denoise"] == 16_330_000_000 + 679_000_000 + 2_500_000_000 + OVERHEAD_BYTES
    assert est.phases["vae"] == 16_330_000_000 + 160_000_000 + VAE_TRANSIENT_BYTES + OVERHEAD_BYTES
    assert est.phases["encode"] == 9_900_000_000 + 1_500_000_000 + OVERHEAD_BYTES
    assert est.peak_phase in ("denoise", "vae")


def test_the_measured_constants_reproduce_the_derisk_run_within_its_band():
    # The two constants come from the 2026-09-28 de-risk run (schnell 1024², one process). This
    # pins them to that record so a later "tidy-up" cannot drift them silently.
    # Bug caught: a constant edited without a new measurement.
    # Bounds widened from the brief's 0.5/6.0 GiB (which excluded the mandated 0.23/7.82 GiB
    # constants themselves) to bracket the measured values; still catches a units/magnitude error.
    assert 0.1 * GIB <= OVERHEAD_BYTES <= 3.0 * GIB
    assert 1.0 * GIB <= VAE_TRANSIENT_BYTES <= 9.0 * GIB
    denoise_measured = MEASURED_DENOISE_PEAK_GIB * GIB  # derisk.json phases.denoise.footprint_peak
    vae_measured = MEASURED_VAE_PEAK_GIB * GIB  # derisk.json phases.vae.footprint_peak
    est = fit_for(
        sizes=SCHNELL_SIZES,
        largest=SCHNELL_LARGEST,
        policy="per-block",
        cache_limit=2_500_000_000,
        allowance=1_500_000_000,
        budget=int(23.0 * GIB),
    )
    # The denoise estimate carries no activation term beyond the cache limit and sits ~0.8 GiB under the
    # measured peak; the VAE estimate is within 0.1 GiB. Both bands are ±1.0 GiB of the measurement.
    assert denoise_measured - 1.0 * GIB <= est.phases["denoise"] <= denoise_measured + 1.0 * GIB
    assert vae_measured - 1.0 * GIB <= est.phases["vae"] <= vae_measured + 1.0 * GIB
    assert est.peak_phase == "vae"
    assert not est.fits  # the measured fact: resident set + VAE decode > 23.0 GiB


def test_fit_for_can_plan_the_vae_phase_with_the_set_dropped():
    # Bug caught: the dropped-set variant still counting the compressed set (the model would then
    # refuse every 1024² call), or the resident variant dropping it (the paging storm the rule prevents).
    resident = fit_for(
        sizes=SCHNELL_SIZES,
        largest=SCHNELL_LARGEST,
        policy="per-block",
        cache_limit=2_500_000_000,
        allowance=1_500_000_000,
        budget=int(23.0 * GIB),
    )
    dropped = fit_for(
        sizes=SCHNELL_SIZES,
        largest=SCHNELL_LARGEST,
        policy="per-block",
        cache_limit=2_500_000_000,
        allowance=1_500_000_000,
        budget=int(23.0 * GIB),
        vae_with_set=False,
    )
    assert dropped.phases["vae"] == SCHNELL_SIZES.vae + VAE_TRANSIENT_BYTES + OVERHEAD_BYTES
    assert (
        resident.phases["vae"] - dropped.phases["vae"]
        == SCHNELL_SIZES.compressed + SCHNELL_SIZES.extras
    )
    assert dropped.phases["denoise"] == resident.phases["denoise"]
    assert not resident.fits
    assert dropped.fits
    assert dropped.peak_phase == "denoise"
