import numpy as np
import pytest
from tests._decode_fixtures import _arrays, _h1_luts, _h2_rows, _h3, _h4, _lens

from mlx_dfloat.errors import DFloatFormatError, DFloatResourceError
from mlx_dfloat.reference import (
    decode_group,
    estimate_decode_bytes,
    max_code_length,
    max_elements_per_block,
    split_matrices,
    thread_gaps,
)


def test_h1_single_level_lut():
    # [126, 127, 128] -> '10' '0' '110' + EOF '111' -> upstream emits 0b10011011 = 0x9B.
    # sm [0x00, 0x80, 0x7F] -> 0x3F00, 0xBF80, 0x407F.
    arrays = _arrays([0x9B], [0x00, 0x80, 0x7F], _h1_luts(), np.zeros(320), [0, 3])
    assert decode_group(arrays).tolist() == [0x3F00, 0xBF80, 0x407F]


def test_h2_two_level_lut_with_9_and_10_bit_codes():
    # [118, 119, 127] -> FF BF CF; sm [0x01, 0x81, 0x00] -> 0x3B01, 0xBB81, 0x3F80.
    row0, row1, lens = _h2_rows(255)
    arrays = _arrays(
        [0xFF, 0xBF, 0xCF], [0x01, 0x81, 0x00], np.stack([row0, row1, lens]), np.zeros(320), [0, 3]
    )
    assert decode_group(arrays).tolist() == [0x3B01, 0xBB81, 0x3F80]


def test_pointer_240_reaches_row_16_of_an_18_row_table():
    # Same stream as H2, but the second-level row sits at index 16 (pointer 240): row 0, 15
    # filler rows, row 16, lengths row = 18 rows, the largest table upstream can emit.
    row0, row16, lens = _h2_rows(240)
    luts = np.vstack([row0, np.full((15, 256), 127, np.uint8), row16, lens])
    assert luts.shape == (18, 256)
    arrays = _arrays([0xFF, 0xBF, 0xCF], [0x01, 0x81, 0x00], luts, np.zeros(320), [0, 3])
    assert decode_group(arrays).tolist() == [0x3B01, 0xBB81, 0x3F80]


def test_h3_code_straddling_a_64_bit_chunk_and_nonzero_gap():
    # sm[i] = i makes every output distinct: out[i] = 128<<7 | i = 0x4000 + i.
    encoded, gaps = _h3()
    assert decode_group(_arrays(encoded, list(range(23)), _h1_luts(), gaps, [0, 23])).tolist() == [
        0x4000 + i for i in range(23)
    ]


def test_h4_32_bit_code_four_level_chain_and_gap_31():
    arrays, expected = _h4()
    assert max_code_length(arrays.luts) == 32
    assert thread_gaps(arrays.gaps, 512)[:2].tolist() == [0, 31]
    assert decode_group(arrays).tolist() == expected


def test_h5_same_length_prefixes_route_to_the_right_rows():
    # H5 (see test_upstream_encoder): [117, 119, 121] -> FF FF 3F 3F; row0[254] -> row 1,
    # row0[255] -> row 2. sm [0, 0x80, 0x01] -> 0x3A80, 0xBB80, 0x3C81.
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
    assert decode_group(arrays).tolist() == [0x3A80, 0xBB80, 0x3C81]


def test_thread_gaps_unpacks_msb_first_5_bit_fields():
    _, gaps = _h3()
    assert thread_gaps(gaps, 512)[:3].tolist() == [0, 2, 0]


def test_corrupt_gap_breaks_continuity():
    # Thread 1's gap corrupted from 2 to 0: thread 0 ends at bit 66 but thread 1 would start at 64.
    encoded, gaps = _h3()
    gaps[1] = 0x00
    with pytest.raises(DFloatFormatError, match="corrupt gaps"):
        decode_group(_arrays(encoded, list(range(23)), _h1_luts(), gaps, [0, 23]))


def test_check_stream_end_false_still_runs_the_gap_continuity_check():
    # Bug caught: check_stream_end=False used as a general "lenient" switch that also skips
    # continuity; the committed upstream slice then stops proving the gaps are consistent.
    encoded, gaps = _h3()
    gaps[1] = 0x00
    arrays = _arrays(encoded, list(range(23)), _h1_luts(), gaps, [0, 23])
    with pytest.raises(DFloatFormatError, match="corrupt gaps"):
        decode_group(arrays, check_stream_end=False)


def test_check_stream_end_false_skips_only_the_eof_byte_count():
    # H1 plus one trailing byte: refused by default (see the trailing-bytes test), decoded to the
    # H1 values when only the stream-end check is off.
    arrays = _arrays([0x9B, 0x00], [0x00, 0x80, 0x7F], _h1_luts(), np.zeros(320), [0, 3])
    assert decode_group(arrays, check_stream_end=False).tolist() == [0x3F00, 0xBF80, 0x407F]


def test_invalid_code_in_a_non_final_thread_is_a_format_error():
    # H3 stream, but '110' (exponent 128) has length 0: thread 0 (not the last real thread) hits
    # an invalid code in its first code. Only the pass-1 per-thread check catches this.
    encoded, gaps = _h3()
    luts = _h1_luts()
    luts[1, 128] = 0
    with pytest.raises(DFloatFormatError, match="thread 0"):
        decode_group(_arrays(encoded, list(range(23)), luts, gaps, [0, 23]))


def test_invalid_code_in_the_only_thread_is_reported():
    luts = _h1_luts()
    luts[1, 126] = 0
    with pytest.raises(DFloatFormatError, match="invalid code"):
        decode_group(_arrays([0x9B], [0, 0, 0], luts, np.zeros(320), [0, 3]))


def test_five_level_pointer_chain_is_invalid():
    # Rows 0..3 each point onward on byte 0xFF; row 3's pointer (252 -> row 4) would be a 5th level.
    luts = np.zeros((6, 256), np.uint8)
    luts[:5, :255] = 127
    luts[0, 255], luts[1, 255], luts[2, 255], luts[3, 255] = 255, 254, 253, 252
    luts[5, 127] = 1
    with pytest.raises(DFloatFormatError, match="invalid code"):
        decode_group(_arrays([0xFF, 0xFF, 0xFF, 0xFF, 0xFF], [0], luts, np.zeros(320), [0, 1]))


@pytest.mark.parametrize(("encoded", "message"), [([0x9B, 0x00], "should be 1 bytes")])
def test_trailing_bytes_after_eof_are_a_format_error(encoded, message):
    # H1 data ends at bit 6 -> exactly ceil(6/8) = 1 byte; a second byte is a malformed stream.
    with pytest.raises(DFloatFormatError, match=message):
        decode_group(_arrays(encoded, [0x00, 0x80, 0x7F], _h1_luts(), np.zeros(320), [0, 3]))


def test_missing_final_byte_is_a_format_error():
    # H3 without its last byte: element 22's code runs past the stream.
    encoded, gaps = _h3()
    with pytest.raises(DFloatFormatError, match=r"codes, expected|should be"):
        decode_group(_arrays(encoded[:-1], list(range(23)), _h1_luts(), gaps, [0, 23]))


def test_output_positions_disagreeing_with_counts_is_a_format_error():
    # Block 0 = 4096 zero bytes = 32768 one-bit codes; claim block 1 starts at 32767.
    n = 32768 + 8
    arrays = _arrays(np.zeros(4097), np.zeros(n), _h1_luts(), np.zeros(640), [0, 32767, n])
    with pytest.raises(DFloatFormatError, match="block 1"):
        decode_group(arrays)


def test_decode_refuses_to_exceed_its_memory_budget():
    arrays = _arrays([0x9B], [0x00, 0x80, 0x7F], _h1_luts(), np.zeros(320), [0, 3])
    assert estimate_decode_bytes(arrays) > 1000
    with pytest.raises(DFloatResourceError, match="budget"):
        decode_group(arrays, max_memory_bytes=1000)


def test_split_matrices_and_block_stats():
    parts = split_matrices(np.arange(10, dtype=np.uint16), np.array([3, 7], np.int64))
    assert [p.tolist() for p in parts] == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
    assert max_elements_per_block(np.array([0, 5, 25, 30], np.uint32)) == 20
