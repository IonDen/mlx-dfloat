import numpy as np
import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st
from tests._df11_fixtures import codec_for, compress_group, random_bf16, write_checkpoint
from tests._upstream.dfloat11_encoder import encode_weights

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import GroupArrays, open_checkpoint
from mlx_dfloat.reference import (
    decode_group,
    decode_matrices,
    max_code_length,
    max_elements_per_block,
    thread_gaps,
)


def _as_arrays(stored):
    return GroupArrays(
        encoded_exponent=stored["encoded_exponent"],
        sign_mantissa=stored["sign_mantissa"],
        luts=stored["luts"],
        gaps=stored["gaps"],
        output_positions=stored["output_positions"].view("<u4"),
        split_positions=stored["split_positions"],
    )


def _encode(bits, counter):
    codec, _, luts = codec_for(counter)
    encoded, other, positions, gaps, split = encode_weights([bits], codec, 8, 512)
    return GroupArrays(
        encoded_exponent=encoded,
        sign_mantissa=other,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=split,
    )


def _roundtrip(matrices):
    stored = compress_group(matrices)
    arrays = _as_arrays(stored)
    np.testing.assert_array_equal(
        decode_group(arrays), np.concatenate([m.reshape(-1) for m in matrices])
    )
    return arrays


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(1, 70_000),
    low=st.integers(0, 200),
    width=st.integers(1, 39),
)
@example(seed=1, n=70_000, low=100, width=30)  # always exercise a many-block group
def test_uniform_exponent_windows_roundtrip(seed, n, low, width):
    rng = np.random.default_rng(seed)
    arrays = _roundtrip(
        [random_bf16(rng, (n,), exponent_low=low, exponent_high=min(low + width, 240))]
    )
    event(f"blocks={arrays.n_blocks}")


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(2_000, 60_000),
    ratio=st.sampled_from([0.5, 0.35, 0.2]),
)
def test_geometric_exponent_spread_reaches_long_codes(seed, n, ratio):
    # Geometric frequencies (p_k ~ ratio^k) give long codes, deep LUT chains and large gaps.
    rng = np.random.default_rng(seed)
    exps = (127 - np.minimum(rng.geometric(1 - ratio, size=n) - 1, 100)).astype(np.uint16)
    bits = (exps << 7) | (rng.integers(0, 65536, size=n, dtype=np.uint16) & 0x807F)
    arrays = _roundtrip([bits])
    event(f"max_code={max_code_length(arrays.luts) // 4 * 4}+")
    event(f"max_gap>=24={bool(thread_gaps(arrays.gaps, arrays.n_threads).max() >= 24)}")


def test_special_values_roundtrip():
    values = np.array(
        [0x0000, 0x8000, 0x0001, 0x807F, 239 << 7, (239 << 7) | 0x807F, 0x3F80], np.uint16
    )
    _roundtrip([np.tile(values, 50)])


def test_single_symbol_codebook_roundtrips():
    # EOF gets '0' here; the seeded vendored encoder maps its range to a zero-length code, which
    # only appears after the data and must be tolerated in the last thread.
    _roundtrip([np.full(5000, 0x3F80, np.uint16)])


def test_32_bit_codes_and_four_level_chains_roundtrip():
    counter = {}
    a, b = 1, 1
    for exp in range(100, 140):
        counter[exp] = a
        a, b = b, a + b
    _, table, luts = codec_for(counter)
    lengths = {k: v[0] for k, v in table.items() if isinstance(k, int)}
    assert max(lengths.values()) == 32
    assert luts.shape[0] >= 5  # rows for '', 8-, 16- and 24-bit prefixes, plus lengths: 4 levels
    rare = sorted(lengths, key=lengths.get, reverse=True)[:6]
    rng = np.random.default_rng(3)
    exps = rng.choice([*rare, max(counter, key=counter.get)], size=6000).astype(np.uint16)
    bits = (exps << 7) | (rng.integers(0, 65536, size=6000, dtype=np.uint16) & 0x807F)
    arrays = _encode(bits, counter)
    np.testing.assert_array_equal(decode_group(arrays), bits)


def test_skewed_one_bit_code_fills_a_block_with_32k_elements():
    rng = np.random.default_rng(4)
    bits = np.full(70_000, 0x3F80, np.uint16)
    bits[rng.choice(70_000, 50, replace=False)] = 0x4000
    arrays = _roundtrip([bits])
    assert max_elements_per_block(arrays.output_positions) > 30_000


def test_byte_aligned_end_without_eof_roundtrips():
    # Review focus 5a. Pick a 1-bit symbol, then 8 of it = 8 bits exactly: no EOF emitted.
    counter = {127: 1000, 126: 2, 125: 1}
    _, table, _ = codec_for(counter)
    one_bit = next(k for k, v in table.items() if isinstance(k, int) and v[0] == 1)
    bits = np.full(8, one_bit << 7, np.uint16)
    arrays = _encode(bits, counter)
    assert arrays.n_bytes == 1
    assert arrays.output_positions.tolist() == [0, 8]
    np.testing.assert_array_equal(decode_group(arrays), bits)


def test_last_code_straddling_into_a_code_free_tail_block_roundtrips():
    # Review focus 5b: 32767 one-bit codes + one 9-bit code at bit 32767 -> 32776 bits = 4097 bytes,
    # byte-aligned, no EOF; block 1 has no code start (2 positions for 2 blocks).
    counter = {127 - i: 2 ** (20 - i) for i in range(12)}
    _, table, _ = codec_for(counter)
    one_bit = next(k for k, v in table.items() if isinstance(k, int) and v[0] == 1)
    nine_bit = next(k for k, v in table.items() if isinstance(k, int) and v[0] == 9)
    bits = np.array([one_bit << 7] * 32767 + [nine_bit << 7], np.uint16)
    arrays = _encode(bits, counter)
    assert arrays.n_bytes == 4097
    assert arrays.output_positions.tolist() == [0, 32768]
    np.testing.assert_array_equal(decode_group(arrays), bits)


def test_dropped_entry_for_a_block_that_starts_codes_is_refused():
    # Review focus 5c: remove the last block's start from a real 4-block group.
    rng = np.random.default_rng(6)
    arrays = _roundtrip([random_bf16(rng, (20_000,))])
    assert arrays.n_blocks >= 3
    positions = arrays.output_positions
    dropped = GroupArrays(
        encoded_exponent=arrays.encoded_exponent,
        sign_mantissa=arrays.sign_mantissa,
        luts=arrays.luts,
        gaps=arrays.gaps,
        output_positions=np.concatenate([positions[:-2], positions[-1:]]),
        split_positions=arrays.split_positions,
    )
    with pytest.raises(DFloatFormatError, match="no entry"):
        decode_group(dropped)


@pytest.mark.parametrize(
    ("group", "pattern", "subs", "names"),
    [
        ("lm_head", r"lm_head", (), ["lm_head.weight"]),
        (
            "blocks.0",
            r"blocks\.\d+",
            ("a", "b", "c"),
            ["blocks.0.a.weight", "blocks.0.b.weight", "blocks.0.c.weight"],
        ),
    ],
)
def test_matrices_decode_by_name_through_a_checkpoint(tmp_path, group, pattern, subs, names):
    rng = np.random.default_rng(5)
    mats = [random_bf16(rng, (7 + i, 13)) for i in range(len(names))]
    root = write_checkpoint(tmp_path / "c", groups={group: mats}, pattern=pattern, sub_paths=subs)
    decoded = decode_matrices(open_checkpoint(root).groups[group])
    assert list(decoded) == names  # hard-coded, not read back from the same source
    for name, mat in zip(names, mats, strict=True):
        np.testing.assert_array_equal(decoded[name], mat.reshape(-1))
