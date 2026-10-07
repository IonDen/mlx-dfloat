"""Builders for MxGroup / decode-backend tests: compress bits with the vendored upstream encoder."""

from pathlib import Path

import numpy as np
from tests._df11_fixtures import codec_for
from tests._upstream.dfloat11_encoder import encode_weights, exponent_counter

from mlx_dfloat import reference
from mlx_dfloat.format import GroupArrays

_SLICE_FIXTURE = (
    Path(__file__).parents[1]
    / "src"
    / "mlx_dfloat"
    / "_canary_data"
    / "qwen3_4b_layer0_4blocks.npz"
)


def encoder_group(bits, *splits):
    """Compress uint16 BF16 bits (optionally split into matrices) with the vendored upstream encoder."""
    mats = np.split(bits, list(splits)) if splits else [bits]
    codec, _, luts = codec_for(exponent_counter(bits))
    encoded, other, positions, gaps, split = encode_weights(list(mats), codec, 8, 512)
    return GroupArrays(
        encoded_exponent=encoded,
        sign_mantissa=other,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=split,
    )


def short_form_group():
    """The short output_positions form: 32,767 one-bit codes then one longer code straddling into a code-free
    tail block. Copy the exact construction of tests/test_reference_roundtrip.py::
    test_last_code_straddling_into_a_code_free_tail_block_roundtrips (counter {127-i: 2**(20-i) for i in range(12)}).
    Verified through the encoder there: 4,097 bytes, n_blocks 2, positions [0, 32768]."""
    counter = {127 - i: 2 ** (20 - i) for i in range(12)}
    codec, table, luts = codec_for(counter)
    one_bit = next(k for k, v in table.items() if isinstance(k, int) and v[0] == 1)
    nine_bit = next(k for k, v in table.items() if isinstance(k, int) and v[0] == 9)
    bits = np.array([one_bit << 7] * 32767 + [nine_bit << 7], np.uint16)
    encoded, other, positions, gaps, split = encode_weights([bits], codec, 8, 512)
    arrays = GroupArrays(
        encoded_exponent=encoded,
        sign_mantissa=other,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=split,
    )
    assert (arrays.n_bytes, arrays.n_blocks, arrays.output_positions.tolist()) == (
        4097,
        2,
        [0, 32768],
    ), "fixture drifted"
    return arrays, bits


def slice_group():
    """The committed Qwen3-4B layer-0 slice: 16,392 bytes truncated mid-block (5 blocks, 4 launched
    thread-groups, real 27-bit codes and 5 LUT rows). Loads
    src/mlx_dfloat/_canary_data/qwen3_4b_layer0_4blocks.npz exactly as tests/test_upstream_slice.py does."""
    data = np.load(_SLICE_FIXTURE)
    arrays = GroupArrays(
        encoded_exponent=data["encoded_exponent"],
        sign_mantissa=data["sign_mantissa"],
        luts=data["luts"],
        gaps=data["gaps"],
        output_positions=data["output_positions"].astype(np.uint32),
        split_positions=data["split_positions"],
    )
    return arrays, data["expected_bf16"]


def fibonacci_group():
    """32-bit codes across a four-level LUT chain: copies the construction of
    tests/test_reference_roundtrip.py::test_32_bit_codes_and_four_level_chains_roundtrip."""
    counter = {}
    a, b = 1, 1
    for exp in range(100, 140):
        counter[exp] = a
        a, b = b, a + b
    codec, table, luts = codec_for(counter)
    lengths = {k: v[0] for k, v in table.items() if isinstance(k, int)}
    rare = sorted(lengths, key=lengths.get, reverse=True)[:6]
    rng = np.random.default_rng(3)
    exps = rng.choice([*rare, max(counter, key=counter.get)], size=6000).astype(np.uint16)
    bits = (exps << 7) | (rng.integers(0, 65536, size=6000, dtype=np.uint16) & 0x807F)
    encoded, other, positions, gaps, split = encode_weights([bits], codec, 8, 512)
    arrays = GroupArrays(
        encoded_exponent=encoded,
        sign_mantissa=other,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=split,
    )
    return arrays, bits


def long_code_canary_group():
    """The packaged long-code canary: 32-bit codes across a four-level LUT chain, spread over more than seven
    launched blocks, then one block too dense for the staged path (over 16,192 elements), then a short tail.

    Same Fibonacci codebook idea as `fibonacci_group`: 8,000 draws from the six longest codes (about 30 bits
    each, so over seven 4,096-byte blocks), a 50,000-element run of the most frequent exponent (a 1-bit code, so
    one block holds 32,768 of them), and 500 more rare draws that leave the last block partial. Seeded RNG;
    `src/mlx_dfloat/_canary_data/long_codes.npz` is this group's arrays plus its input bits as `expected_bf16`.
    """
    counter = {}
    a, b = 1, 1
    for exp in range(100, 140):
        counter[exp] = a
        a, b = b, a + b
    codec, table, luts = codec_for(counter)
    lengths = {k: v[0] for k, v in table.items() if isinstance(k, int)}
    rare = sorted(lengths, key=lengths.get, reverse=True)[:6]
    frequent = max(counter, key=counter.get)
    rng = np.random.default_rng(11)
    exps = np.concatenate(
        [
            rng.choice(rare, size=8000),
            np.full(50000, frequent),
            rng.choice(rare, size=500),
        ]
    ).astype(np.uint16)
    bits = (exps << 7) | (rng.integers(0, 65536, size=exps.size, dtype=np.uint16) & 0x807F)
    encoded, other, positions, gaps, split = encode_weights([bits], codec, 8, 512)
    arrays = GroupArrays(
        encoded_exponent=encoded,
        sign_mantissa=other,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=split,
    )
    return arrays, bits


# --- Hand-derived fixtures (moved verbatim from tests/test_reference_hand.py) ---------------------------------


def _arrays(encoded, sign_mantissa, luts, gaps, positions, split=()):
    return GroupArrays(
        encoded_exponent=np.array(encoded, np.uint8),
        sign_mantissa=np.array(sign_mantissa, np.uint8),
        luts=luts,
        gaps=np.asarray(gaps, np.uint8),
        output_positions=np.array(positions, np.uint32),
        split_positions=np.array(split, np.int64),
    )


def _lens(pairs):
    lens = np.zeros(256, np.uint8)
    for sym, bits in pairs:
        lens[sym] = bits
    return lens


def _h1_luts():
    # H1: 127 '0', 126 '10', 128 '110', EOF '111' (EOF's range inherits 128, as upstream).
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:256] = 127, 126, 128
    return np.stack([row0, _lens([(126, 2), (127, 1), (128, 3)])])


def _h2_rows(pointer_row_value):
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:224], row0[224:240] = 127, 126, 125, 124
    row0[240:248], row0[248:252], row0[252:254], row0[254], row0[255] = (
        123,
        122,
        121,
        120,
        pointer_row_value,
    )
    tail = np.zeros(256, np.uint8)
    tail[0:128], tail[128:256] = 119, 118
    lens = _lens(zip(range(127, 117, -1), range(1, 11), strict=True))
    return row0, tail, lens


def _h3():
    # 23 x '110' (exponent 128) + EOF: DB 6D B6 DB 6D B6 DB 6D B7. Symbol 22 starts at bit 66 ->
    # thread 1's gap is 2: gaps bytes 0x00, 0x80.
    gaps = np.zeros(320, np.uint8)
    gaps[1] = 0x80
    return [0xDB, 0x6D, 0xB6, 0xDB, 0x6D, 0xB6, 0xDB, 0x6D, 0xB7], gaps


def _h4():
    # Unary codebook: exponent 128-L has code '1'*(L-1)+'0' for L = 1..32; EOF = '1'*32.
    # Rows: '' / '1'*8 / '1'*16 / '1'*24 -> pointers 255, 254, 253; the 32-bit code needs all 4 levels.
    rows = np.zeros((4, 256), np.uint8)
    for level in range(4):
        for k in range(8):  # code ends in this byte after k ones: byte starts with k ones then 0
            lo, hi = 256 - (1 << (8 - k)), 256 - (1 << (7 - k))
            rows[level, lo:hi] = 128 - (8 * level + k + 1)
        rows[level, 255] = 255 - level if level < 3 else 96  # last row: EOF inherits 96
    lens = _lens((128 - length, length) for length in range(1, 33))
    luts = np.vstack([rows, lens])
    # Data: 63 x exponent 127 (1 bit, bits 0..62), then exponent 96 (32 bits) at 63..94, then 96
    # again at 95..126, then EOF's first bit at 127. Thread 1's first code starts at 95 -> gap 31.
    encoded = [0x00] * 7 + [0x01, 0xFF, 0xFF, 0xFF, 0xFD, 0xFF, 0xFF, 0xFF, 0xFD]
    gaps = np.zeros(320, np.uint8)
    gaps[0], gaps[1] = 0x07, 0xC0  # 5-bit fields 00000, 11111
    sm = [i % 128 for i in range(63)] + [0x85, 0x7F]
    expected = [0x3F80 | (i % 128) for i in range(63)] + [
        0x8000 | (96 << 7) | 0x05,
        (96 << 7) | 0x7F,
    ]
    return _arrays(encoded, sm, luts, gaps, [0, 65]), expected


# --- (GroupArrays, expected) builders for the parity tests: each copies its test_h* case exactly -------------


def h1():
    # test_h1_single_level_lut: [126, 127, 128] -> 0x9B; sm [0x00, 0x80, 0x7F].
    arrays = _arrays([0x9B], [0x00, 0x80, 0x7F], _h1_luts(), np.zeros(320), [0, 3])
    return arrays, [0x3F00, 0xBF80, 0x407F]


def h2():
    # test_h2_two_level_lut_with_9_and_10_bit_codes: [118, 119, 127] -> FF BF CF; sm [0x01, 0x81, 0x00].
    row0, row1, lens = _h2_rows(255)
    arrays = _arrays(
        [0xFF, 0xBF, 0xCF], [0x01, 0x81, 0x00], np.stack([row0, row1, lens]), np.zeros(320), [0, 3]
    )
    return arrays, [0x3B01, 0xBB81, 0x3F80]


def h3():
    # test_h3_code_straddling_a_64_bit_chunk_and_nonzero_gap: sm[i] = i -> out[i] = 0x4000 + i.
    encoded, gaps = _h3()
    arrays = _arrays(encoded, list(range(23)), _h1_luts(), gaps, [0, 23])
    return arrays, [0x4000 + i for i in range(23)]


def h4():
    # test_h4_32_bit_code_four_level_chain_and_gap_31.
    return _h4()


def h5():
    # test_h5_same_length_prefixes_route_to_the_right_rows: [117, 119, 121] -> FF FF 3F 3F;
    # row0[254] -> row 1, row0[255] -> row 2. sm [0, 0x80, 0x01] -> 0x3A80, 0xBB80, 0x3C81.
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:224], row0[224:240] = 127, 126, 125, 124
    row0[240:248], row0[248:252], row0[252], row0[253], row0[254], row0[255] = (
        123,
        122,
        121,
        120,
        255,
        254,
    )
    row1 = np.zeros(256, np.uint8)
    row1[0:128], row1[128:256] = 119, 118
    row2 = np.zeros(256, np.uint8)
    row2[0:128], row2[128:256] = 118, 117
    lens = _lens(
        [
            (127, 1),
            (126, 2),
            (125, 3),
            (124, 4),
            (123, 5),
            (122, 6),
            (121, 8),
            (120, 8),
            (119, 9),
            (118, 9),
            (117, 9),
        ]
    )
    arrays = _arrays(
        [0xFF, 0xFF, 0x3F, 0x3F],
        [0x00, 0x80, 0x01],
        np.stack([row0, row1, row2, lens]),
        np.zeros(320),
        [0, 3],
    )
    return arrays, [0x3A80, 0xBB80, 0x3C81]


def h4_gap_zero():
    """A 32-bit code at bit 0 (gap 0), 33 one-bit codes, EOF: verified against the strict reference.
    Stream (9 bytes): FF FF FF FE | 00 00 00 00 | 7F  -> exponent 96 (31 ones + 0), 33 zeros = 33 x exponent 127,
    then EOF's leading bits pad the last byte (upstream emits one byte after the data)."""
    luts = h4()[0].luts
    encoded = [0xFF, 0xFF, 0xFF, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x7F]
    # Thread 0 gap 0; thread 1's first code starts at bit 64 -> gap 0.
    gaps = np.zeros(320, np.uint8)
    sm = [0x05] + [i % 128 for i in range(33)]
    expected = [(96 << 7) | 0x05] + [0x3F80 | (i % 128) for i in range(33)]
    return _arrays(encoded, sm, luts, gaps, [0, 34]), expected


def hand_fixtures():
    cases = [
        ("h1", *h1()),
        ("h2", *h2()),
        ("h3", *h3()),
        ("h4", *h4()),
        ("h5", *h5()),
        ("h4_gap_zero", *h4_gap_zero()),
    ]
    # Every hand fixture is pinned by the strict reference first.
    for name, arrays, expected in cases:
        assert reference.decode_group(arrays).tolist() == expected, name
    return cases
