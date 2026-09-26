import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from mlx_dfloat._safetensors import MAX_HEADER_BYTES, read_array, read_header
from mlx_dfloat.errors import DFloatFormatError


def _write(path, tensors):
    # mx.save_safetensors appends ".safetensors" when missing, so always pass the full name.
    assert path.name.endswith(".safetensors")
    mx.save_safetensors(str(path), {k: mx.array(v) for k, v in tensors.items()})


def test_reads_back_what_mlx_wrote(tmp_path):
    # Independent writer (MLX, which records "__metadata__": null) vs our reader.
    path = tmp_path / "a.safetensors"
    u8 = np.arange(10, dtype=np.uint8)
    i64 = np.array([3, 7], dtype=np.int64)
    _write(path, {"x.u8": u8, "x.i64": i64})
    header = read_header(path)
    assert header["x.u8"].dtype == "U8"
    assert header["x.u8"].shape == (10,)
    np.testing.assert_array_equal(read_array(path, header["x.u8"]), u8)
    np.testing.assert_array_equal(read_array(path, header["x.i64"]), i64)


def test_bf16_is_returned_as_uint16_bits(tmp_path):
    path = tmp_path / "b.safetensors"
    bits = np.array([0x3F80, 0xBF80, 0x0000, 0x8000], dtype=np.uint16)
    mx.save_safetensors(str(path), {"w": mx.array(bits).view(mx.bfloat16)})
    info = read_header(path)["w"]
    assert info.dtype == "BF16"
    np.testing.assert_array_equal(read_array(path, info), bits)  # -0 stays 0x8000


def test_reads_through_a_symlink(tmp_path):
    # Review focus 1: HF snapshots are symlinks into blobs/.
    blob = tmp_path / "blob.safetensors"
    _write(blob, {"t": np.arange(4, dtype=np.uint8)})
    link = tmp_path / "model.safetensors"
    link.symlink_to(blob)
    np.testing.assert_array_equal(
        read_array(link, read_header(link)["t"]), np.arange(4, dtype=np.uint8)
    )


def test_dangling_symlink_is_a_format_error(tmp_path):
    link = tmp_path / "model.safetensors"
    link.symlink_to(tmp_path / "missing-blob")
    with pytest.raises(DFloatFormatError, match="cannot read"):
        read_header(link)


def test_truncated_file_is_a_format_error(tmp_path):
    # Review focus 2.
    path = tmp_path / "c.safetensors"
    _write(path, {"t": np.zeros(1000, dtype=np.uint8)})
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(DFloatFormatError, match="extends past end of file"):
        read_header(path)


def _raw_file(path, header_obj, payload=b"", *, raw=None):
    body = raw if raw is not None else json.dumps(header_obj).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"t": {"dtype": "Q9", "shape": [1], "data_offsets": [0, 1]}}, "unsupported dtype"),
        ({"t": {"dtype": ["U8"], "shape": [1], "data_offsets": [0, 1]}}, "unsupported dtype"),
        ({"t": {"dtype": "U8", "shape": [2], "data_offsets": [0, 1]}}, "size does not match"),
        ({"t": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}}, "shape"),
        ({"t": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}}, "shape"),
        ({"t": {"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}}, "offsets"),
        ({"t": "nonsense"}, "entry"),
    ],
)
def test_malformed_entries_are_format_errors(tmp_path, header, message):
    path = tmp_path / "m.safetensors"
    _raw_file(path, header, payload=b"\x00" * 4)
    with pytest.raises(DFloatFormatError, match=message):
        read_header(path)


def test_duplicate_keys_are_a_format_error(tmp_path):
    path = tmp_path / "d.safetensors"
    entry = '{"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}'
    _raw_file(path, None, payload=b"\x00", raw=f'{{"t": {entry}, "t": {entry}}}'.encode())
    with pytest.raises(DFloatFormatError, match="duplicate"):
        read_header(path)


def test_header_length_beyond_file_is_a_format_error(tmp_path):
    path = tmp_path / "h.safetensors"
    path.write_bytes(struct.pack("<Q", 10_000) + b"{}")
    with pytest.raises(DFloatFormatError, match="header"):
        read_header(path)


def test_oversized_header_is_refused_before_reading_it(tmp_path):
    path = tmp_path / "big.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", MAX_HEADER_BYTES + 1))
        handle.truncate(MAX_HEADER_BYTES + 16)  # sparse file; nothing real is read
    with pytest.raises(DFloatFormatError, match="header"):
        read_header(path)


@pytest.mark.parametrize("raw", [b"{x}", b"[" * 100_000 + b"]" * 100_000])
def test_invalid_or_deeply_nested_json_is_a_format_error(tmp_path, raw):
    path = tmp_path / "j.safetensors"
    _raw_file(path, None, raw=raw)
    with pytest.raises(DFloatFormatError, match="header"):
        read_header(path)


def test_empty_tensor_reads_as_empty_array(tmp_path):
    path = tmp_path / "e.safetensors"
    _raw_file(path, {"t": {"dtype": "I64", "shape": [0], "data_offsets": [0, 0]}})
    arr = read_array(path, read_header(path)["t"])
    assert arr.shape == (0,)
    assert arr.dtype == np.int64


def test_a_huge_zero_product_shape_is_a_format_error(tmp_path):
    # Bug caught: shape [0, 2**63] passes the product check (0 bytes) and np.empty later raises a
    # bare ValueError instead of DFloatFormatError.
    path = tmp_path / "z.safetensors"
    _raw_file(path, {"t": {"dtype": "U8", "shape": [0, 2**63], "data_offsets": [0, 0]}})
    with pytest.raises(DFloatFormatError, match="shape"):
        read_header(path)


@pytest.mark.parametrize(
    "entry",
    [
        {"dtype": "Q" * 10_000, "shape": [1], "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [-1] * 10_000, "data_offsets": [0, 1]},
        {"dtype": "U8", "shape": [1], "data_offsets": [0] * 10_000},
    ],
)
def test_untrusted_header_values_are_truncated_in_error_text(tmp_path, entry):
    # Bug caught: echoing a header value uncapped copies up to the 100 MB header into stderr and
    # into the parity summary's "error" field.
    path = tmp_path / "t.safetensors"
    _raw_file(path, {"n" * 10_000: entry}, payload=b"\x00")
    with pytest.raises(DFloatFormatError) as info:
        read_header(path)
    assert len(str(info.value)) < 400
