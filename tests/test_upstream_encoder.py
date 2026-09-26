import numpy as np
import pytest
from tests._upstream.dfloat11_encoder import encode, get_luts

# Codebook "H2": 127:'0', 126:'10', 125:'110', 124:'1110', 123:'11110', 122:'111110',
# 121:'1111110', 120:'11111110', 119:'111111110' (9 bits), 118:'1111111110' (10 bits), EOF '1111111111'.
H2_TABLE = {
    127: (1, 0b0),
    126: (2, 0b10),
    125: (3, 0b110),
    124: (4, 0b1110),
    123: (5, 0b11110),
    122: (6, 0b111110),
    121: (7, 0b1111110),
    120: (8, 0b11111110),
    119: (9, 0b111111110),
    118: (10, 0b1111111110),
    "EOF": (10, 0b1111111111),
}

# Codebook "H5": two 8-bit prefixes of the same length ('11111110' -> row 1, '11111111' -> row 2,
# in table order) and EOF as the leftmost code under row 2, so row 2 indices 0..127 are never
# assigned and inherit row 1's last value (118): upstream's stale-accumulator carry.
H5_TABLE = {
    127: (1, 0b0),
    126: (2, 0b10),
    125: (3, 0b110),
    124: (4, 0b1110),
    123: (5, 0b11110),
    122: (6, 0b111110),
    121: (8, 0b11111100),
    120: (8, 0b11111101),
    119: (9, 0b111111100),
    118: (9, 0b111111101),
    "EOF": (9, 0b111111110),
    117: (9, 0b111111111),
}


def _lens(table):
    lens = np.zeros(256, np.uint8)
    for sym, (bits, _) in table.items():
        if isinstance(sym, int):
            lens[sym] = bits
    return lens


def h2_expected_luts():
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:224], row0[224:240] = 127, 126, 125, 124
    row0[240:248], row0[248:252], row0[252:254], row0[254], row0[255] = 123, 122, 121, 120, 255
    row1 = np.zeros(256, np.uint8)
    row1[0:128], row1[128:256] = 119, 118
    return np.stack([row0, row1, _lens(H2_TABLE)])


def h5_expected_luts():
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:224], row0[224:240] = 127, 126, 125, 124
    row0[240:248], row0[248:252], row0[252], row0[253] = 123, 122, 121, 120
    row0[254], row0[255] = 255, 254  # '11111110' -> row 1, '11111111' -> row 2
    row1 = np.zeros(256, np.uint8)
    row1[0:128], row1[128:256] = 119, 118
    row2 = np.zeros(256, np.uint8)
    row2[0:128], row2[128:256] = 118, 117  # 0..127: stale carry of row 1's last value
    return np.stack([row0, row1, row2, _lens(H5_TABLE)])


def test_get_luts_matches_hand_derived_h2_tables():
    # Bug caught: a transcription error in the vendored get_luts (row order, pointer value, fill).
    np.testing.assert_array_equal(get_luts(H2_TABLE), h2_expected_luts())


def test_get_luts_orders_same_length_prefixes_and_carries_stale_values():
    # Bug caught: sorting same-length prefixes differently (pointer numbers change) or "fixing"
    # upstream's cross-row carry (row 2's unassigned entries would become 0 instead of 118).
    np.testing.assert_array_equal(get_luts(H5_TABLE), h5_expected_luts())


def _upstream_get_luts_verbatim(table):
    # Upstream dfloat11_utils.get_luts @457733886c, verbatim except torch -> NumPy and no print:
    # curr_val is NOT seeded, so it raises UnboundLocalError when EOF owns row-0 index 0.
    prefixes = [""]
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)
    prefixes.sort(key=len)
    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)
    for pi, p in enumerate(prefixes):
        bytes_dict = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, "0")
                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8) :].ljust(8, "0"), 2)
                        dict_value = key
                    else:
                        dict_key = int(bin_val[(pl * 8) : (pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[: (pl * 8 + 8)])
                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                    bytes_dict[dict_key] = dict_value
        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val  # noqa: F821
    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits
    return np.concatenate((luts, lens), axis=0)


def test_seeded_get_luts_equals_upstream_wherever_upstream_works():
    # Bug caught: the seed (or any other edit) changing a table upstream could actually ship.
    from dahuffman import HuffmanCodec

    rng = np.random.default_rng(2026)
    compared = crashed = 0
    for _ in range(300):
        n_symbols = int(rng.integers(1, 40))
        exps = rng.choice(np.arange(60, 200), size=n_symbols, replace=False)
        counter = {
            int(e): int(f)
            for e, f in zip(exps, rng.integers(1, 10_000, size=n_symbols), strict=True)
        }
        table = HuffmanCodec.from_frequencies(counter).get_code_table()
        try:
            expected = _upstream_get_luts_verbatim(table)
        except UnboundLocalError:
            crashed += 1
            continue
        np.testing.assert_array_equal(get_luts(table), expected)
        compared += 1
    assert compared > 200  # get_luts branch exercised
    assert crashed > 0  # UnboundLocalError branch exercised


def test_upstream_cannot_build_a_single_symbol_table():
    table = {"EOF": (1, 0b0), 127: (1, 0b1)}
    with pytest.raises(UnboundLocalError):
        _upstream_get_luts_verbatim(table)


def test_get_luts_seeded_accumulator_handles_eof_as_all_zeros():
    # EOF '0', one symbol '1': upstream raises UnboundLocalError here; the seed maps 0..127 to 0.
    luts = get_luts({"EOF": (1, 0b0), 127: (1, 0b1)})
    assert luts[0, :128].tolist() == [0] * 128
    assert luts[0, 128:].tolist() == [127] * 128
    assert luts[1, 0] == 0
    assert luts[1, 127] == 1  # exponent 0 has no code (length 0)


class _HandCodec:
    """Minimal stand-in exposing the two attributes upstream's encode() reads."""

    _eof = "EOF"

    def __init__(self, table):
        self._table = table


def test_encode_matches_hand_derived_h2_stream():
    # 118 + 119 + 127 + EOF -> FF BF CF (upstream emits only the byte holding EOF's start).
    encoded, gaps, positions = encode([118, 119, 127], _HandCodec(H2_TABLE), 8, 512)
    assert encoded.tolist() == [0xFF, 0xBF, 0xCF]
    assert positions.tolist() == [0, 3]
    assert gaps.size == 320
    assert not gaps.any()


def test_encode_records_gap_of_first_code_starting_in_next_chunk():
    # 23 x '110' = 69 bits; symbol 22 starts at bit 66 -> chunk 1's gap is 2.
    h1 = {127: (1, 0b0), 126: (2, 0b10), 128: (3, 0b110), "EOF": (3, 0b111)}
    encoded, gaps, positions = encode([128] * 23, _HandCodec(h1), 8, 512)
    assert encoded.tolist() == [0xDB, 0x6D, 0xB6, 0xDB, 0x6D, 0xB6, 0xDB, 0x6D, 0xB7]
    assert gaps[:2].tolist() == [0x00, 0x80]
    assert positions.tolist() == [0, 23]
