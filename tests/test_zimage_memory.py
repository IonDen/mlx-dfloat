"""Z-Image's memory model. Expected values are hand arithmetic from the committed constants' source files.

Constants come from ``derisk-{turbo,base}/derisk.json`` (2026-10-08, git 3da18bb, mlx 0.32.2, mflux 0.20.0):
overhead max(510_030_936, 470_479_960); denoise activation max(12_781_477_784 - 11_583_577_874, 13_061_987_888 -
11_538_787_646) = 1_523_200_242. The VAE transient comes from the measured 1024² generate run of the base model
(bench/results/tiers/z-image-1024.json, mflux 0.23): footprint peak 15_462_030_800 minus (8_369_046_600 + 6_088_832 +
19_660_800 + 167_666_902 + 510_030_936) = 6_389_536_730, the highest of every committed 1024² sample.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests._family_fakes import ZIMAGE_FULL_TABLE, FakeZImageFull, write_fake_zimage_checkpoint

from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.mflux.zimage.memory import (
    DENOISE_ACTIVATION_AT_REFERENCE,
    OVERHEAD_BYTES,
    VAE_TRANSIENT_BYTES,
    ZImageSizes,
    activation_allowance,
    cache_limit_for,
    denoise_activation_bytes,
    fit_for,
    sizes_for,
    text_tokens,
    vae_transient_bytes,
    zimage_phases,
)
from mlx_dfloat.mflux.zimage.transformer import block_lists

LARGEST = {"noise_refiner": 362_000_000, "context_refiner": 330_000_000, "layers": 362_000_000}


def test_the_constants_are_the_derisk_measurements():
    # Bug caught: a constant taken from one unit instead of the larger of the two, or a FLUX.1 value left in.
    assert OVERHEAD_BYTES == 510_030_936
    assert VAE_TRANSIENT_BYTES == 6_389_536_730
    assert DENOISE_ACTIVATION_AT_REFERENCE == 1_523_200_242


def test_cache_limit_is_the_two_largest_decoded_groups_plus_the_allowance():
    # Bug caught: only one decoded buffer budgeted (the next block's decode allocates fresh every time), or the
    # depth-2 extra buffer missing. Allowance at 1024² + 512 tokens = 1.5e9 * 4608 / 4352 = 1_588_235_294.
    assert (
        cache_limit_for(LARGEST, policy="per-block", height=1024, width=1024, text_tokens=512)
        == 2_312_235_294
    )
    assert (
        cache_limit_for(LARGEST, policy="depth2", height=1024, width=1024, text_tokens=512)
        == 2_674_235_294
    )
    assert (
        cache_limit_for(
            LARGEST, policy="per-block", height=1024, width=1024, text_tokens=512, override=7
        )
        == 7
    )


def test_cache_limit_has_no_1024_floor_and_follows_the_smaller_allowance():
    # Bug caught: FLUX.1's 2.5 GB floor copied over (Z-Image's measured working point is 2.31 GB at 1024²).
    # 512²: tokens 1024 + 512 = 1536; 1.5e9 * 1536 / 4352 = 529_411_764; plus 724_000_000.
    assert (
        cache_limit_for(LARGEST, policy="per-block", height=512, width=512, text_tokens=512)
        == 1_253_411_764
    )


def test_the_allowance_scales_with_tokens_and_has_a_floor():
    # Bug caught: a fixed allowance regardless of size, or no floor on tiny images.
    assert activation_allowance(height=1024, width=1024, text_tokens=256) == 1_500_000_000
    # 256² with 256 text tokens: 1.5e9 * 1280 / 4352 = 441_176_470 -> the 500 MB floor.
    assert activation_allowance(height=256, width=256, text_tokens=256) == 500_000_000


def test_the_vae_transient_is_floored_at_the_measurement_and_linear_above_1024_squared():
    # Bug caught: a smaller image predicting less than the measured 1024² value, or no growth above it.
    assert vae_transient_bytes(height=512, width=512) == 6_389_536_730
    assert vae_transient_bytes(height=1024, width=1024) == 6_389_536_730
    assert vae_transient_bytes(height=2048, width=1024) == 12_779_073_460  # 2 x 6_389_536_730


def test_the_denoise_activation_is_linear_in_the_token_count():
    # Bug caught: the reference token count taken as FLUX.1's 4352 instead of Z-Image's 4608.
    assert denoise_activation_bytes(height=1024, width=1024, text_tokens=512) == 1_523_200_242
    # 512² + 512 text tokens = 1536 tokens = one third of 4608: 1_523_200_242 / 3.
    assert denoise_activation_bytes(height=512, width=512, text_tokens=512) == 507_733_414


def test_the_estimate_uses_the_text_length_limit_not_the_prompt():
    # Bug caught: a short prompt under-predicting denoise memory (the tokenizer pads/truncates to 512; the
    # estimate must not shrink with the prompt). Z-Image's limit is read from the model config.
    assert text_tokens(SimpleNamespace(max_sequence_length=512)) == 512
    assert text_tokens(SimpleNamespace(max_sequence_length=77)) == 77


@pytest.mark.mflux
def test_text_tokens_of_the_real_turbo_config_is_512():
    # Bug caught: reading the wrong config field (mflux model_config.py:674-685 sets 512 for both Z-Image models).
    from mflux.models.common.config.model_config import ModelConfig

    assert text_tokens(ModelConfig.z_image_turbo()) == 512


_PHASE_KW = {
    "compressed_bytes": 8_000_000_000,
    "extras_bytes": 6_000_000,
    "nonblock_bytes": 20_000_000,
    "largest": {"noise_refiner": 300_000_000, "layers": 360_000_000},
    "cache_limit": 2_300_000_000,
    "allowance": 1_500_000_000,
    "encoders_bytes": 8_000_000_000,
    "vae_bytes": 160_000_000,
    "vae_transient_bytes": 3_900_000_000,
    "overhead_bytes": 500_000_000,
    "denoise_activation_bytes": 1_500_000_000,
}


def test_phases_count_each_term_once():
    # Bug caught: the non-block decoded bytes left out of denoise (or the vae phase), or counted twice.
    phases = zimage_phases(policy="per-block", **_PHASE_KW)
    assert phases["encode"] == {
        "encoders": 8_000_000_000,
        "activations": 1_500_000_000,
        "overhead": 500_000_000,
    }
    assert phases["denoise"] == {
        "compressed": 8_000_000_000,
        "extras": 6_000_000,
        "nonblock": 20_000_000,
        "decoded": 360_000_000,
        "cache": 2_300_000_000,
        "activations": 1_500_000_000,
        "overhead": 500_000_000,
    }
    assert phases["vae"] == {
        "compressed": 8_000_000_000,
        "extras": 6_000_000,
        "nonblock": 20_000_000,
        "vae": 160_000_000,
        "transient": 3_900_000_000,
        "overhead": 500_000_000,
    }


def test_depth2_doubles_the_in_flight_decoded_buffer():
    # Bug caught: depth-2 evaluation budgeted like per-block (one decoded group in flight).
    phases = zimage_phases(policy="depth2", **_PHASE_KW)
    assert phases["denoise"]["decoded"] == 720_000_000


def test_fit_for_sums_the_phases_and_drops_the_set_from_the_vae_phase_when_asked():
    # Bug caught: vae_with_set=False keeping the compressed set / non-block bytes in the VAE phase or dropping the
    # extras (they stay on the transformer when the set goes), or the constants not the measured ones. Hand sums (denoise at 1024², 512 tokens):
    #   8_000_000_000 + 6_000_000 + 20_000_000 + 360_000_000 + 2_300_000_000 + 1_523_200_242 + 510_030_936
    #   = 12_719_231_178 ; vae with set: 8_026_000_000 + 160_000_000 + 6_389_536_730 + 510_030_936 = 15_085_567_666.
    sizes = ZImageSizes(
        compressed=8_000_000_000,
        extras=6_000_000,
        nonblock=20_000_000,
        encoders=8_000_000_000,
        vae=160_000_000,
    )
    common = {
        "sizes": sizes,
        "largest": {"noise_refiner": 300_000_000, "layers": 360_000_000},
        "policy": "per-block",
        "cache_limit": 2_300_000_000,
        "allowance": 1_500_000_000,
        "budget": 24_000_000_000,
        "height": 1024,
        "width": 1024,
        "text_tokens": 512,
    }
    kept = fit_for(**common)
    assert kept.phases["denoise"] == 12_719_231_178
    assert kept.phases["vae"] == 15_085_567_666
    assert kept.phases["encode"] == 8_000_000_000 + 1_500_000_000 + 510_030_936
    assert kept.peak_phase == "vae"
    dropped = fit_for(vae_with_set=False, **common)
    # 6_000_000 + 160_000_000 + 6_389_536_730 + 510_030_936
    assert dropped.phases["vae"] == 7_065_567_666
    assert dropped.peak_phase == "denoise"
    assert dropped.phases["denoise"] == kept.phases["denoise"]


REPO = Path(__file__).resolve().parents[1]
_BASE_SIZES = ZImageSizes(
    compressed=8_369_046_600,
    extras=6_088_832,
    nonblock=19_660_800,
    encoders=8_044_982_000,
    vae=167_666_902,
)


def _record_sample(path):
    """A committed 1024² record's sizes and its watched peak (verify_image's df11 side: the report and its footprint)."""
    record = json.loads((REPO / path).read_text())
    if "report" in record:
        return ZImageSizes(**record["report"]["sizes"]), record["footprint_peak_bytes"]
    return ZImageSizes(**record["sizes"]), record["watched_peak_bytes"]


# Every committed 1024² sample with the set resident through the VAE decode, each peak reached in the VAE phase:
# the records in the tree (mflux 0.23: Z-Image's MEASURED run and identity df11 side; mflux 0.20: Z-Image-Turbo's
# MEASURED run and its five repeats) and the two mflux 0.20 Z-Image records they replaced (git ed2b0ca, the 0.2.0 tag:
# bench/results/tiers/z-image-1024.json watched_peak_bytes and identity/z-image-1024/df11-result.json
# footprint_peak_bytes, the same sizes as the 0.23 records).
_SAMPLES_1024 = {
    "z-image-measured-mflux023": _record_sample("bench/results/tiers/z-image-1024.json"),
    "z-image-identity-mflux023": _record_sample(
        "bench/results/identity/z-image-1024/df11-result.json"
    ),
    "z-image-turbo-measured-mflux020": _record_sample(
        "bench/results/tiers/z-image-turbo-1024.json"
    ),
    **{
        f"z-image-turbo-repeat-{n}-mflux020": _record_sample(
            f"bench/results/repeats/z-image-turbo-1024/turbo-run-{n}.json"
        )
        for n in range(1, 6)
    },
    "z-image-measured-mflux020": (_BASE_SIZES, 14_115_020_240),
    "z-image-identity-mflux020": (_BASE_SIZES, 14_066_196_160),
}


def _fit_1024(sizes):
    # The 1024² working point of the measured runs: their cache limit and the in-flight decoded group the reports'
    # denoise estimate implies, 13_101_538_864 - (8_369_046_600 + 6_088_832 + 19_660_800 + 2_311_752_734
    # + 1_523_200_242 + 510_030_936) = 361_759_720 (the turbo report gives the same).
    return fit_for(
        sizes=sizes,
        largest={"layers": 361_759_720},
        policy="per-block",
        cache_limit=2_311_752_734,
        allowance=1_588_235_294,
        budget=24_653_119_488,
        height=1024,
        width=1024,
        text_tokens=512,
    )


@pytest.mark.parametrize("sample", sorted(_SAMPLES_1024))
def test_the_vae_phase_never_under_predicts_a_committed_1024_sample(sample):
    # Bug caught: a VAE transient calibrated on a subset of the samples, so a tier whose budget falls between the
    # estimate and a measured peak is told the set can stay through the decode. On mflux 0.23 the 0.20 constant
    # puts Z-Image's VAE phase 1_347_010_560 B under its MEASURED run and 970_653_696 B under its identity run, and
    # it was already 1_188_073_004 B under the 0.20 Turbo repeat 4.
    sizes, measured = _SAMPLES_1024[sample]
    est = _fit_1024(sizes)
    assert measured <= est.phases["vae"]
    assert est.peak_phase == "vae"


def test_the_vae_phase_equals_the_binding_sample():
    # Bug caught: a transient padded beyond the overall-peak rule (more set drops than the samples justify), or taken
    # from another sample than the highest: the mflux 0.23 MEASURED Z-Image run binds, exactly.
    sizes, measured = _SAMPLES_1024["z-image-measured-mflux023"]
    assert measured == 15_462_030_800
    assert _fit_1024(sizes).phases["vae"] == 15_462_030_800


def _checkpoint(tmp_path):
    probe = FakeZImageFull(None, n_refiner_layers=1, n_layers=2)
    shapes = install_placeholders(block_lists(probe), ZIMAGE_FULL_TABLE)
    write_fake_zimage_checkpoint(tmp_path / "df11", shapes, np.random.default_rng(5))
    return open_checkpoint(tmp_path / "df11")


def test_sizes_for_reads_the_single_file_and_the_base_directories(tmp_path):
    # Bug caught: compressed counting the extras, encoders reading FLUX's text_encoder_2 (absent here, 0 bytes),
    # or nonblock counting the compressed bytes instead of the decoded ones (cap_embedder is 4x4 BF16 = 32 bytes).
    ckpt = _checkpoint(tmp_path)
    base = tmp_path / "base"
    (base / "text_encoder").mkdir(parents=True)
    (base / "text_encoder" / "a.safetensors").write_bytes(b"x" * 10)
    (base / "text_encoder" / "b.safetensors").write_bytes(b"x" * 5)
    (base / "text_encoder" / "model.safetensors.index.json").write_bytes(b"x" * 100)
    (base / "text_encoder_2").mkdir()
    (base / "text_encoder_2" / "c.safetensors").write_bytes(b"x" * 1000)
    (base / "vae").mkdir()
    (base / "vae" / "v.safetensors").write_bytes(b"v" * 50)
    sizes = sizes_for(ckpt, base)
    total = sum(p.stat().st_size for p in (tmp_path / "df11").glob("*.safetensors"))
    assert sizes.extras > 0
    assert sizes.compressed + sizes.extras == total
    assert sizes.nonblock == 32
    assert sizes.encoders == 15
    assert sizes.vae == 50
