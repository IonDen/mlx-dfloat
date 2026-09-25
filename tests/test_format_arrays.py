import numpy as np
import pytest

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import GroupArrays, n_blocks_for, validate_group_arrays


def _valid(**overrides):
    luts = np.zeros((2, 256), np.uint8)
    luts[0, :] = 127
    luts[1, 127] = 1
    base = {
        "encoded_exponent": np.zeros(1, np.uint8),
        "sign_mantissa": np.zeros(8, np.uint8),
        "luts": luts,
        "gaps": np.zeros(320, np.uint8),
        "output_positions": np.array([0, 8], np.uint32),
        "split_positions": np.array([], np.int64),
    }
    base.update(overrides)
    return GroupArrays(**base)


def _two_blocks(**overrides):
    return _valid(
        encoded_exponent=np.zeros(4097, np.uint8), gaps=np.zeros(640, np.uint8), **overrides
    )


def test_valid_arrays_pass_and_expose_sizes():
    arrays = _valid()
    validate_group_arrays(arrays, name="g")
    assert (arrays.n_elements, arrays.n_bytes, arrays.n_blocks, arrays.n_threads) == (8, 1, 1, 512)


@pytest.mark.parametrize(("n_bytes", "blocks"), [(1, 1), (4096, 1), (4097, 2), (8192, 2)])
def test_n_blocks_for(n_bytes, blocks):
    assert n_blocks_for(n_bytes) == blocks


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        (_valid(output_positions=np.array([0, 7], np.uint32)), "last output position"),
        (_valid(output_positions=np.array([1, 8], np.uint32)), "first output position"),
        (_two_blocks(output_positions=np.array([0, 9, 8], np.uint32)), "not monotonic"),
        (_valid(output_positions=np.array([0, 4, 6, 8], np.uint32)), "entries"),
        (_valid(gaps=np.zeros(319, np.uint8)), "gaps"),
        (_valid(luts=np.zeros((2, 255), np.uint8)), "luts"),
        (_valid(luts=np.zeros((1, 256), np.uint8)), "luts"),
        (_valid(luts=np.zeros((19, 256), np.uint8)), "luts"),
        (_valid(split_positions=np.array([0], np.int64)), "split_positions"),
        (_valid(split_positions=np.array([8], np.int64)), "split_positions"),
        (_valid(split_positions=np.array([5, 3], np.int64)), "split_positions"),
        (_valid(encoded_exponent=np.zeros(0, np.uint8)), "empty"),
        (
            _valid(
                sign_mantissa=np.zeros(0, np.uint8), output_positions=np.array([0, 0], np.uint32)
            ),
            "empty",
        ),
    ],
)
def test_structural_violations_are_format_errors(arrays, message):
    with pytest.raises(DFloatFormatError, match=message):
        validate_group_arrays(arrays, name="g")


def test_eighteen_row_table_is_accepted():
    luts = np.zeros((18, 256), np.uint8)
    luts[0, 255] = 240  # pointer 240 -> row 16, the largest legal target
    validate_group_arrays(_valid(luts=luts), name="g")


@pytest.mark.parametrize(("rows", "pointer"), [(2, 255), (2, 250), (18, 239 + 0)])
def test_pointer_bounds(rows, pointer):
    luts = np.zeros((rows, 256), np.uint8)
    luts[0, 200] = pointer
    if pointer < 240:  # 239 is a symbol, not a pointer: always accepted
        validate_group_arrays(_valid(luts=luts), name="g")
        return
    # 255 on a 2-row table targets row 1, the lengths row: must be refused (off-by-one guard).
    with pytest.raises(DFloatFormatError, match="pointer"):
        validate_group_arrays(_valid(luts=luts), name="g")


def test_tail_block_without_a_code_start_is_accepted():
    # Review focus 5: n_bytes spans 2 blocks, no code starts in the second, 2 entries.
    validate_group_arrays(_two_blocks(output_positions=np.array([0, 8], np.uint32)), name="g")
