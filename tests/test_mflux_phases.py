"""The shared phase arithmetic of the mflux families: hand-worked literals, and a differential pin to Z-Image's rules."""

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.mflux._phases import (
    FamilySizes,
    PhaseConstants,
    activation_allowance,
    cache_limit_for,
    denoise_activation_bytes,
    family_phases,
    fit_for,
    sizes_for,
    vae_transient_bytes,
)

# Klein 4B decoded group sizes (from the header's matrix shapes: double block 490_733_568 B, single block 245_366_784 B).
KLEIN_4B_LARGEST = {"transformer_blocks": 490_733_568, "single_transformer_blocks": 245_366_784}


def _constants(**over):
    base = {
        "overhead_bytes": 0,
        "vae_transient_bytes": 0,
        "denoise_activation_at_reference": 0,
        "reference_tokens": 4096 + 512,
    }
    return PhaseConstants(**{**base, **over})


def test_cache_limit_for_klein_4b_at_1024_is_the_two_kinds_largest_plus_the_allowance():
    # Bug caught: the per-kind largest groups not summed (one decoded buffer budgeted), or depth2's extra buffer
    # missing. Allowance at 1024^2 + 512 text tokens: 1.5e9 * 4608 / 4352 = 1_588_235_294.
    # 490_733_568 + 245_366_784 + 1_588_235_294 = 2_324_335_646; depth2 adds 490_733_568.
    c = _constants()
    common = {"height": 1024, "width": 1024, "text_tokens": 512}
    assert cache_limit_for(c, KLEIN_4B_LARGEST, policy="per-block", **common) == 2_324_335_646
    assert cache_limit_for(c, KLEIN_4B_LARGEST, policy="depth2", **common) == 2_815_069_214
    assert cache_limit_for(c, KLEIN_4B_LARGEST, policy="per-block", override=7, **common) == 7


def test_the_allowance_is_linear_in_tokens_with_a_floor_and_reads_its_rate_from_the_constants():
    # Bug caught: the floor missing (a tiny image budgets almost no cache), the token count leaving out the text, or
    # the module's own rate used instead of the record's.
    c = _constants()
    # 256^2 / 256 + 0 = 256 tokens: 1.5e9 * 256 / 4352 = 88_235_294 < floor 500_000_000.
    assert activation_allowance(c, height=256, width=256, text_tokens=0) == 500_000_000
    # 1024^2 / 256 + 256 = 4352 tokens: exactly the reference, 1_500_000_000.
    assert activation_allowance(c, height=1024, width=1024, text_tokens=256) == 1_500_000_000
    other = _constants(allowance_at_reference=3_000_000_000, allowance_floor=1)
    assert activation_allowance(other, height=1024, width=1024, text_tokens=256) == 3_000_000_000


def test_the_vae_transient_is_floored_at_the_measured_value_and_linear_above():
    # Bug caught: a smaller image predicted to need less than measured (the floor), or no growth above 1024^2.
    c = _constants(vae_transient_bytes=4_000_000_000)
    assert vae_transient_bytes(c, height=512, width=512) == 4_000_000_000
    assert vae_transient_bytes(c, height=1024, width=2048) == 8_000_000_000


def test_the_denoise_activation_scales_with_tokens_from_the_reference():
    # Bug caught: the activation not scaled (a 512^2 call costed like 1024^2), or scaled against the allowance's
    # reference (4352) instead of the activation's own (4608). 512^2: 1024 + 512 = 1536 tokens;
    # 1_000_000 * 1536 / 4608 = 333_333.
    c = _constants(denoise_activation_at_reference=1_000_000)
    assert denoise_activation_bytes(c, height=1024, width=1024, text_tokens=512) == 1_000_000
    assert denoise_activation_bytes(c, height=512, width=512, text_tokens=512) == 333_333


def test_a_denoise_activation_floor_holds_below_the_reference_and_is_off_by_default():
    # Bug caught: a family whose step holds a size-independent buffer (ERNIE-Image's float32 copies of a block's
    # weights) predicted below it at a small image, or the floor applied to every family (the default must leave the
    # scaled term as it was). 512^2 + 512: 333_333 scaled, floored at 400_000; 1024^2 + 512: 1_000_000 above the floor.
    floored = _constants(
        denoise_activation_at_reference=1_000_000, denoise_activation_floor_bytes=400_000
    )
    assert denoise_activation_bytes(floored, height=512, width=512, text_tokens=512) == 400_000
    assert denoise_activation_bytes(floored, height=1024, width=1024, text_tokens=512) == 1_000_000
    assert _constants().denoise_activation_floor_bytes is None
    plain = _constants(denoise_activation_at_reference=1_000_000)
    assert denoise_activation_bytes(plain, height=512, width=512, text_tokens=512) == 333_333


SIZES = FamilySizes(compressed=5_000, extras=10, nonblock=300, encoders=8_000, vae=160)
SMALL_LARGEST = {"a": 400, "b": 200}


def _fit(*, policy="per-block", vae_with_set=True):
    c = _constants(overhead_bytes=50, vae_transient_bytes=1_000)
    return fit_for(
        c,
        sizes=SIZES,
        largest=SMALL_LARGEST,
        policy=policy,
        cache_limit=900,
        allowance=700,
        budget=10_000,
        height=1024,
        width=1024,
        text_tokens=512,
        vae_with_set=vae_with_set,
    )


def test_phases_count_each_term_once_and_drop_the_set_from_the_vae_phase_when_asked():
    # Bug caught: the decoded non-block bytes left out of denoise or counted twice; vae_with_set=False keeping the
    # compressed set or the non-block bytes, or dropping the extras (they stay on the transformer when the set goes).
    # encode = 8_000 + 700 + 50 = 8_750; denoise = 5_000 + 10 + 300 + 400 + 900 + 0 + 50 = 6_660;
    # vae = 5_000 + 10 + 300 + 160 + 1_000 + 50 = 6_520; without the set: 10 + 160 + 1_000 + 50 = 1_220.
    fit = _fit()
    assert fit.phases == {"encode": 8_750, "denoise": 6_660, "vae": 6_520}
    assert (fit.peak_phase, fit.peak_bytes, fit.budget_bytes) == ("encode", 8_750, 10_000)
    assert _fit(vae_with_set=False).phases["vae"] == 1_220


@pytest.mark.parametrize(("term", "encode"), [(30, 8_080), (0, 8_050)])
def test_a_measured_encode_term_replaces_the_allowance_in_the_encode_phase_only(term, encode):
    # Bug caught: the family's measured encode term ignored (Qwen's encode sized by the denoise allowance, +1.08 GiB),
    # added on top of the allowance, a 0 term read as "unset" (`or` for `is None`), or the term leaking into the
    # denoise phase. encode = 8_000 + term + 50; denoise 6_660 and vae 6_520 as without the term.
    c = _constants(overhead_bytes=50, vae_transient_bytes=1_000, encode_activation_bytes=term)
    fit = fit_for(
        c,
        sizes=SIZES,
        largest=SMALL_LARGEST,
        policy="per-block",
        cache_limit=900,
        allowance=700,
        budget=10_000,
        height=1024,
        width=1024,
        text_tokens=512,
    )
    assert fit.phases == {"encode": encode, "denoise": 6_660, "vae": 6_520}


def test_a_one_kind_familys_cache_limit_is_one_decoded_group_plus_the_allowance():
    # Bug caught: a one-kind family budgeted two decoded buffers (per-block evaluation frees one block-sized buffer
    # at a time), or depth2's look-ahead buffer missing. Allowance at 1024^2 + 512 tokens 1_588_235_294;
    # 436_207_616 + 1_588_235_294 = 2_024_442_910; depth2 adds 436_207_616.
    c = _constants()
    one = {"transformer_blocks": 436_207_616}
    common = {"height": 1024, "width": 1024, "text_tokens": 512}
    assert cache_limit_for(c, one, policy="per-block", **common) == 2_024_442_910
    assert cache_limit_for(c, one, policy="depth2", **common) == 2_460_650_526


def test_depth2_holds_two_of_the_largest_decoded_groups_in_flight():
    # Bug caught: depth2's look-ahead buffer missing from the denoise phase. 6_660 + 400 = 7_060.
    assert _fit(policy="depth2").phases["denoise"] == 7_060


def test_family_phases_name_every_term():
    # Bug caught: a term renamed or merged (the report and the plan read terms by name).
    phases = family_phases(
        compressed_bytes=1,
        extras_bytes=2,
        nonblock_bytes=3,
        largest={"a": 4},
        policy="per-block",
        cache_limit=5,
        allowance=6,
        encoders_bytes=7,
        vae_bytes=8,
        vae_transient_bytes=9,
        overhead_bytes=10,
        denoise_activation_bytes=11,
    )
    assert phases == {
        "encode": {"encoders": 7, "activations": 6, "overhead": 10},
        "denoise": {
            "compressed": 1,
            "extras": 2,
            "nonblock": 3,
            "decoded": 4,
            "cache": 5,
            "activations": 11,
            "overhead": 10,
        },
        "vae": {
            "compressed": 1,
            "extras": 2,
            "nonblock": 3,
            "vae": 8,
            "transient": 9,
            "overhead": 10,
        },
    }
    with_term = family_phases(
        compressed_bytes=1,
        extras_bytes=2,
        nonblock_bytes=3,
        largest={"a": 4},
        policy="per-block",
        cache_limit=5,
        allowance=6,
        encoders_bytes=7,
        vae_bytes=8,
        vae_transient_bytes=9,
        overhead_bytes=10,
        denoise_activation_bytes=11,
        encode_activation_bytes=12,
    )
    assert with_term["encode"] == {"encoders": 7, "activations": 12, "overhead": 10}
    assert with_term["denoise"] == phases["denoise"]


@pytest.mark.parametrize(
    ("height", "width", "policy", "vae_with_set"),
    [
        (1024, 1024, "per-block", True),
        (512, 768, "depth2", True),
        (1024, 2048, "per-block", False),
        (256, 256, "per-block", True),
    ],
)
def test_with_zimage_constants_the_shared_arithmetic_equals_zimage_memory(
    height, width, policy, vae_with_set
):
    # Bug caught: the shared module drifting from the Z-Image rules it will replace (the later migration would change
    # Z-Image's estimates). Differential: same inputs, both implementations, every result equal. Z-Image's constants
    # are inputs to both sides here, not expected values.
    from mlx_dfloat.mflux.zimage import memory as z

    c = PhaseConstants(
        overhead_bytes=z.OVERHEAD_BYTES,
        vae_transient_bytes=z.VAE_TRANSIENT_BYTES,
        denoise_activation_at_reference=z.DENOISE_ACTIVATION_AT_REFERENCE,
        reference_tokens=z.REFERENCE_TOKENS,
        allowance_at_reference=z.ALLOWANCE_AT_REFERENCE,
        allowance_reference_tokens=z.ALLOWANCE_REFERENCE_TOKENS,
        allowance_floor=z.ALLOWANCE_FLOOR,
        max_measured_pixels=z.MAX_MEASURED_PIXELS,
    )
    sizes = {
        "compressed": 7_000_000_000,
        "extras": 6_000_000,
        "nonblock": 20_000_000,
        "encoders": 8_000_000_000,
        "vae": 160_000_000,
    }
    largest = {"noise_refiner": 362_000_000, "context_refiner": 330_000_000, "layers": 362_000_000}
    common = {"height": height, "width": width, "text_tokens": 512}
    lim_z = z.cache_limit_for(largest, policy=policy, **common)
    assert cache_limit_for(c, largest, policy=policy, **common) == lim_z
    allow = z.activation_allowance(**common)
    assert activation_allowance(c, **common) == allow
    assert vae_transient_bytes(c, height=height, width=width) == z.vae_transient_bytes(
        height=height, width=width
    )
    assert denoise_activation_bytes(c, **common) == z.denoise_activation_bytes(**common)
    args = dict(
        largest=largest,
        policy=policy,
        cache_limit=lim_z,
        allowance=allow,
        budget=23 * 1024**3,
        vae_with_set=vae_with_set,
        **common,
    )
    ours = fit_for(c, sizes=FamilySizes(**sizes), **args)
    theirs = z.fit_for(sizes=z.ZImageSizes(**sizes), **args)
    assert (ours.phases, ours.peak_phase, ours.peak_bytes) == (
        theirs.phases,
        theirs.peak_phase,
        theirs.peak_bytes,
    )


def test_sizes_for_counts_the_nonblock_groups_named_and_the_base_directories(tmp_path):
    # Bug caught: compressed counting the extras, nonblock counting a group not named (or its compressed bytes
    # instead of the decoded ones: ctx is 4 x 6 BF16 = 48 bytes), or the encoders reading text_encoder_2 (absent
    # in FLUX.2 and Z-Image bases; a stray one here must not count).
    rng = np.random.default_rng(9)
    write_checkpoint(
        tmp_path / "df11",
        groups={"blk.0": [random_bf16(rng, (8, 4))], "ctx": [random_bf16(rng, (4, 6))]},
        patterns={r"blk\.\d+": ("m",), "ctx": ()},
        extras={"x.weight": np.zeros((3, 5), dtype=np.uint16)},
        single_file=True,
    )
    ckpt = open_checkpoint(tmp_path / "df11")
    base = tmp_path / "base"
    (base / "text_encoder").mkdir(parents=True)
    (base / "text_encoder" / "a.safetensors").write_bytes(b"x" * 10)
    (base / "text_encoder" / "b.safetensors").write_bytes(b"x" * 5)
    (base / "text_encoder" / "model.safetensors.index.json").write_bytes(b"x" * 100)
    (base / "text_encoder_2").mkdir()
    (base / "text_encoder_2" / "c.safetensors").write_bytes(b"x" * 1000)
    (base / "vae").mkdir()
    (base / "vae" / "v.safetensors").write_bytes(b"v" * 50)

    sizes = sizes_for(ckpt, base, nonblock_groups=("ctx", "absent_group"))
    total = (tmp_path / "df11" / "model.safetensors").stat().st_size
    assert sizes.extras == 30  # 3 x 5 BF16
    assert sizes.compressed == total - 30
    assert sizes.nonblock == 48
    assert sizes.encoders == 15
    assert sizes.vae == 50
    assert sizes_for(ckpt, base, nonblock_groups=()).nonblock == 0
