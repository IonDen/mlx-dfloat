"""Builders for MxGroup / decode-backend tests: compress bits with the vendored upstream encoder."""

import numpy as np
from tests._df11_fixtures import codec_for
from tests._upstream.dfloat11_encoder import encode_weights, exponent_counter

from mlx_dfloat.format import GroupArrays


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
