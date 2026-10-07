"""The packaged canary groups: known inputs with known BF16 bits, decoded before any real decode."""

import functools
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

import numpy as np
import numpy.typing as npt

from mlx_dfloat.errors import DFloatBackendError
from mlx_dfloat.format import GroupArrays

SLICE_NAME = "canary-qwen3-4b-slice"
LONG_NAME = "canary-long-codes"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryGroup:
    """One canary: the group's host arrays and the BF16 bits it must decode to."""

    name: str
    arrays: GroupArrays
    expected: npt.NDArray[np.uint16]


def data_file(name: str) -> Traversable:
    """A file in the packaged canary data directory."""
    return resources.files("mlx_dfloat") / "_canary_data" / name


_FIELDS = (
    "encoded_exponent",
    "sign_mantissa",
    "luts",
    "gaps",
    "output_positions",
    "split_positions",
    "expected_bf16",
)


def _require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def _read(file: str) -> dict[str, npt.NDArray[Any]]:
    """The packaged file's arrays, each checked for the dtype and rank the decoder expects."""
    with data_file(file).open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        fields = {key: data[key] for key in _FIELDS}
    for key in ("encoded_exponent", "sign_mantissa", "gaps"):
        _require(fields[key].dtype == np.uint8 and fields[key].ndim == 1, f"{key} is not 1-D uint8")
    _require(fields["luts"].dtype == np.uint8 and fields["luts"].ndim == 2, "luts is not 2-D uint8")
    _require(fields["output_positions"].ndim == 1, "output_positions is not 1-D")
    _require(
        fields["split_positions"].dtype == np.int64 and fields["split_positions"].ndim == 1,
        "split_positions is not 1-D int64",
    )
    expected = fields["expected_bf16"]
    _require(expected.dtype == np.uint16 and expected.ndim == 1, "expected_bf16 is not 1-D uint16")
    _require(
        expected.size == fields["sign_mantissa"].size,
        "expected_bf16 and sign_mantissa differ in length",
    )
    # canary_failure prefills its output with a word the group never decodes to; one must exist
    _require(np.unique(expected).size < 1 << 16, "expected_bf16 holds every 16-bit value")
    return fields


def _load(file: str, name: str) -> CanaryGroup:
    try:
        fields = _read(file)
        arrays = GroupArrays(
            encoded_exponent=fields["encoded_exponent"],
            sign_mantissa=fields["sign_mantissa"],
            luts=fields["luts"],
            gaps=fields["gaps"],
            output_positions=fields["output_positions"].astype(np.uint32),
            split_positions=fields["split_positions"],
        )
        expected = fields["expected_bf16"]
    except Exception as exc:  # missing file, bad archive, missing key, wrong dtype or shape
        raise DFloatBackendError(f"canary data {name} is unreadable or malformed: {exc}") from exc
    return CanaryGroup(name=name, arrays=arrays, expected=expected)


@functools.cache
def canary_groups() -> tuple[CanaryGroup, ...]:
    """The canary groups: a slice cut from a real checkpoint, then one with the longest codes the format allows."""
    return (
        _load("qwen3_4b_layer0_4blocks.npz", SLICE_NAME),
        _load("long_codes.npz", LONG_NAME),
    )
