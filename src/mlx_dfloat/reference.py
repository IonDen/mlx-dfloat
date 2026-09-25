"""Pure-NumPy, bit-exact DFloat11 decoder: the oracle every GPU kernel is tested against.

It emulates upstream's CUDA algorithm in lockstep: each of 512 threads per 4096-byte block owns one
64-bit chunk of the exponent stream and decodes the codes that start in it (from its 5-bit gap); an
exclusive scan turns per-thread counts into output indices. Structural checks run as it decodes.
"""

import os

import numpy as np
import numpy.typing as npt

from mlx_dfloat.errors import DFloatFormatError, DFloatResourceError
from mlx_dfloat.format import (
    BYTES_PER_THREAD,
    LUT_POINTER_MIN,
    THREADS_PER_BLOCK,
    DF11Group,
    GroupArrays,
    validate_group_arrays,
)

_BITS_PER_THREAD = 8 * BYTES_PER_THREAD

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
U16Array = npt.NDArray[np.uint16]


def thread_gaps(gaps: npt.NDArray[np.uint8], n_threads: int) -> IntArray:
    """Unpack the MSB-first 5-bit start offsets, one per thread."""
    need = -(-5 * n_threads // 8)
    bits = np.unpackbits(np.asarray(gaps[:need], dtype=np.uint8))[: 5 * n_threads].reshape(
        n_threads, 5
    )
    packed = (
        (bits[:, 0] << 4) | (bits[:, 1] << 3) | (bits[:, 2] << 2) | (bits[:, 3] << 1) | bits[:, 4]
    )
    return packed.astype(np.int64)


def estimate_decode_bytes(arrays: GroupArrays) -> int:
    """Upper estimate of the decoder's working memory for one group."""
    return 12 * arrays.n_threads * BYTES_PER_THREAD + 3 * arrays.n_elements


def default_memory_budget() -> int:
    """Half of physical RAM."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2


def max_code_length(luts: npt.NDArray[np.uint8]) -> int:
    """Longest code in a group's codebook (the maximum of the lengths row)."""
    return int(np.asarray(luts)[-1].max())


def _peek32(stream: npt.NDArray[np.uint8], pos: IntArray) -> npt.NDArray[np.uint32]:
    byte = pos >> 3
    shift = (pos & 7).astype(np.uint64)
    word = np.zeros(pos.size, dtype=np.uint64)
    for k in range(5):
        word = (word << np.uint64(8)) | stream[byte + k].astype(np.uint64)
    return ((word >> (np.uint64(8) - shift)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _lookup(
    luts: npt.NDArray[np.uint8], words: npt.NDArray[np.uint32]
) -> tuple[IntArray, IntArray, BoolArray]:
    n_luts = luts.shape[0]
    sym = luts[0, words >> 24].astype(np.int64)
    ok = np.ones(words.size, dtype=np.bool_)
    for level in (1, 2, 3):
        ptr = sym >= LUT_POINTER_MIN
        if not ptr.any():
            break
        row = 256 - sym[ptr]
        in_range = (row >= 1) & (row <= n_luts - 2)
        byte = (words[ptr] >> np.uint32(24 - 8 * level)) & np.uint32(0xFF)
        sym[ptr] = np.where(in_range, luts[np.where(in_range, row, 0), byte], LUT_POINTER_MIN)
        ok[ptr] &= in_range
    ok &= sym < LUT_POINTER_MIN
    length = np.where(ok, luts[n_luts - 1, np.minimum(sym, 255)], 0).astype(np.int64)
    ok &= length > 0
    return sym, length, ok


def _count(
    stream: npt.NDArray[np.uint8], luts: npt.NDArray[np.uint8], starts: IntArray, ends: IntArray
) -> tuple[IntArray, BoolArray, IntArray]:
    counts = np.zeros(starts.size, dtype=np.int64)
    invalid = np.zeros(starts.size, dtype=np.bool_)
    pos = starts.copy()
    active = np.flatnonzero(pos < ends)
    while active.size:
        _, length, ok = _lookup(luts, _peek32(stream, pos[active]))
        invalid[active[~ok]] = True
        good = active[ok]
        counts[good] += 1
        pos[good] += length[ok]
        active = good[pos[good] < ends[good]]
    return counts, invalid, pos


def _write(
    stream: npt.NDArray[np.uint8],
    luts: npt.NDArray[np.uint8],
    sign_mantissa: npt.NDArray[np.uint8],
    starts: IntArray,
    first: IntArray,
    stop: IntArray,
) -> tuple[U16Array, int]:
    n = sign_mantissa.size
    out = np.zeros(n, dtype=np.uint16)
    pos = starts.copy()
    idx = first.copy()
    active = np.flatnonzero(idx < stop)
    while active.size:
        sym, length, _ = _lookup(luts, _peek32(stream, pos[active]))  # validated in pass 1
        k = idx[active]
        sm = sign_mantissa[k].astype(np.uint16)
        out[k] = ((sm & 0x80) << 8) | (sym.astype(np.uint16) << 7) | (sm & 0x7F)
        idx[active] += 1
        pos[active] += length
        active = active[idx[active] < stop[active]]
    last_thread = int(np.searchsorted(first, n - 1, side="right") - 1)
    return out, int(pos[last_thread])


def decode_group(
    arrays: GroupArrays,
    *,
    name: str = "<group>",
    max_memory_bytes: int | None = None,
    check_stream_end: bool = True,
) -> U16Array:
    """Decode one compressed group to its BF16 bit patterns (flat, concatenation order).

    ``check_stream_end=False`` skips only the EOF byte-count check; it exists for fixtures that are
    deliberately truncated mid-stream (the committed upstream slice), never for real checkpoints.

    Raises:
        DFloatFormatError: The arrays are structurally invalid or internally inconsistent.
        DFloatResourceError: The estimated working memory exceeds ``max_memory_bytes`` (default:
            half of physical RAM).
    """
    validate_group_arrays(arrays, name=name)
    budget = default_memory_budget() if max_memory_bytes is None else max_memory_bytes
    need = estimate_decode_bytes(arrays)
    if need > budget:
        raise DFloatResourceError(
            f"{name}: decoding needs about {need} bytes, over the {budget}-byte budget"
        )
    n, n_bytes, n_threads = arrays.n_elements, arrays.n_bytes, arrays.n_threads
    n_bits = 8 * n_bytes
    last_real = (n_bits - 1) // _BITS_PER_THREAD
    g = np.arange(last_real + 1, dtype=np.int64)
    starts = g * _BITS_PER_THREAD + thread_gaps(arrays.gaps, n_threads)[: last_real + 1]
    ends = (g + 1) * _BITS_PER_THREAD
    stream = np.zeros(n_threads * BYTES_PER_THREAD + 8, dtype=np.uint8)
    stream[:n_bytes] = arrays.encoded_exponent
    luts = np.ascontiguousarray(arrays.luts)

    counts, invalid, ended = _count(stream, luts, starts, ends)
    bad = np.flatnonzero(invalid[:-1])
    if bad.size:
        t = int(bad[0])
        raise DFloatFormatError(
            f"{name}: invalid code in thread {t} (block {t // THREADS_PER_BLOCK})"
        )
    inside = ended[:-1] < n_bits
    broken = np.flatnonzero(inside & (ended[:-1] != starts[1:]))
    if broken.size:
        t = int(broken[0])
        raise DFloatFormatError(
            f"{name}: thread {t} ends at bit {int(ended[t])} but thread {t + 1} starts at bit "
            f"{int(starts[t + 1])} (corrupt gaps?)"
        )
    first = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=first[1:])
    positions = arrays.output_positions.astype(np.int64)
    block_starts = np.arange(positions.size - 1, dtype=np.int64) * THREADS_PER_BLOCK
    mismatch = np.flatnonzero(first[block_starts] != positions[:-1])
    if mismatch.size:
        b = int(mismatch[0])
        raise DFloatFormatError(
            f"{name}: block {b} starts at element {int(first[block_starts[b]])}, output_positions says "
            f"{int(positions[b])}"
        )
    if positions.size - 1 == arrays.n_blocks - 1:
        # The last block holds at least one byte, so its first thread is always a real one
        # (tail <= last_real) and first[tail] is in range.
        tail = (arrays.n_blocks - 1) * THREADS_PER_BLOCK
        if first[tail] < n:
            raise DFloatFormatError(
                f"{name}: block {arrays.n_blocks - 1} starts codes but output_positions has no entry for it"
            )
    total = int(first[-1] + counts[-1])
    if total < n:
        hint = " (invalid code in the last thread)" if invalid[-1] else ""
        raise DFloatFormatError(f"{name}: stream holds {total} codes, expected {n}{hint}")
    stop = np.minimum(first + counts, n)
    out, end_bit = _write(stream, luts, np.asarray(arrays.sign_mantissa), starts, first, stop)
    if check_stream_end and -(-end_bit // 8) != n_bytes:
        raise DFloatFormatError(
            f"{name}: data ends at bit {end_bit}, so the stream should be {-(-end_bit // 8)} bytes, not {n_bytes}"
        )
    return out


def split_matrices(flat: U16Array, split_positions: npt.NDArray[np.int64]) -> list[U16Array]:
    """Split a decoded group into its matrices (flat), at ``split_positions``."""
    return list(np.split(flat, split_positions.astype(np.int64)))


def decode_matrices(
    group: DF11Group, *, max_memory_bytes: int | None = None
) -> dict[str, U16Array]:
    """Decode a group and return its matrices keyed by weight name (flat BF16 bit patterns)."""
    arrays = group.load()
    parts = split_matrices(
        decode_group(arrays, name=group.name, max_memory_bytes=max_memory_bytes),
        arrays.split_positions,
    )
    return dict(zip(group.matrix_names, parts, strict=True))


def max_elements_per_block(output_positions: npt.NDArray[np.uint32]) -> int:
    """Largest number of elements whose codes start in one 4096-byte block (threadgroup sizing)."""
    return int(np.diff(output_positions.astype(np.int64)).max())


__all__ = [
    "decode_group",
    "decode_matrices",
    "default_memory_budget",
    "estimate_decode_bytes",
    "max_code_length",
    "max_elements_per_block",
    "split_matrices",
    "thread_gaps",
]
