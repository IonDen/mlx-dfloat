"""Minimal safetensors reader: header parsing plus lazy, read-only memmap access.

Written against the safetensors format (8-byte little-endian header length, JSON header, raw
little-endian data). Files come from the network, so every field is validated and the header size
is capped. BF16 tensors are returned as their uint16 bit patterns.
"""

import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mlx_dfloat.errors import DFloatFormatError

MAX_HEADER_BYTES = 100_000_000

DTYPES: Mapping[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "F32": np.dtype("<f4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F64": np.dtype("<f8"),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorInfo:
    """Location and type of one tensor inside a safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate key in safetensors header")
    return dict(pairs)


def parse_header(
    raw: bytes, *, data_start: int, file_size: int, source: str
) -> dict[str, TensorInfo]:
    """Parse and validate a safetensors JSON header.

    Args:
        raw: The JSON header bytes.
        data_start: Absolute file offset where tensor data begins (8 + header length).
        file_size: Total file size in bytes, used to detect truncated downloads.
        source: File name used in error messages.

    Returns:
        Tensor name to TensorInfo, without the ``__metadata__`` entry.

    Raises:
        DFloatFormatError: The header is not valid JSON, has duplicate keys, or an entry is
            malformed or out of bounds.
    """
    try:
        header = json.loads(raw, object_pairs_hook=_no_duplicates)
    except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError, duplicate keys
        message = "duplicate key" if "duplicate" in str(exc) else "not valid JSON"
        raise DFloatFormatError(f"{source}: safetensors header is {message}") from exc
    except RecursionError as exc:
        raise DFloatFormatError(f"{source}: safetensors header is nested too deeply") from exc
    if not isinstance(header, dict):
        raise DFloatFormatError(f"{source}: safetensors header is not a JSON object")
    infos: dict[str, TensorInfo] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        infos[name] = _parse_entry(
            name, meta, data_start=data_start, file_size=file_size, source=source
        )
    return infos


def _parse_entry(
    name: str, meta: object, *, data_start: int, file_size: int, source: str
) -> TensorInfo:
    if not isinstance(meta, dict) or {"dtype", "shape", "data_offsets"} - meta.keys():
        raise DFloatFormatError(f"{source}: tensor entry {name!r} is malformed")
    dtype = meta["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise DFloatFormatError(f"{source}: tensor {name!r} has unsupported dtype {dtype!r}")
    shape = meta["shape"]
    if not isinstance(shape, list) or not all(type(d) is int and d >= 0 for d in shape):
        raise DFloatFormatError(f"{source}: tensor {name!r} has an invalid shape {shape!r}")
    offsets = meta["data_offsets"]
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(o) is int for o in offsets)
        or not 0 <= offsets[0] <= offsets[1]
    ):
        raise DFloatFormatError(f"{source}: tensor {name!r} has invalid data offsets {offsets!r}")
    nbytes = offsets[1] - offsets[0]
    if nbytes != math.prod(shape) * DTYPES[dtype].itemsize:
        raise DFloatFormatError(
            f"{source}: tensor {name!r} byte size does not match its shape and dtype"
        )
    if data_start + offsets[1] > file_size:
        raise DFloatFormatError(
            f"{source}: tensor {name!r} extends past end of file ({file_size} bytes); partial download?"
        )
    return TensorInfo(
        name=name, dtype=dtype, shape=tuple(shape), offset=data_start + offsets[0], nbytes=nbytes
    )


def read_header(path: Path) -> dict[str, TensorInfo]:
    """Read and validate the header of a safetensors file without reading tensor data.

    Raises:
        DFloatFormatError: The file cannot be read, or its header is invalid or oversized.
    """
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise DFloatFormatError(f"{path.name}: too short for a safetensors header")
            (length,) = struct.unpack("<Q", prefix)
            if length == 0 or length > MAX_HEADER_BYTES or 8 + length > file_size:
                raise DFloatFormatError(
                    f"{path.name}: safetensors header length {length} is invalid"
                )
            raw = handle.read(length)
    except OSError as exc:
        raise DFloatFormatError(f"{path}: cannot read file ({exc.strerror or exc})") from exc
    return parse_header(raw, data_start=8 + length, file_size=file_size, source=path.name)


def read_array(path: Path, info: TensorInfo) -> npt.NDArray[Any]:
    """Return a read-only, lazily paged view of one tensor (BF16 as uint16 bits)."""
    dtype = DTYPES[info.dtype]
    if info.nbytes == 0:
        return np.empty(info.shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=info.offset, shape=info.shape)


__all__ = ["DTYPES", "MAX_HEADER_BYTES", "TensorInfo", "parse_header", "read_array", "read_header"]
