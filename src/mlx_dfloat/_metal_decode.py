"""The Metal DF11 decode kernel: a per-thread, bounded, status-reporting mirror of the reference algorithm."""

from typing import Any

import mlx.core as mx
import numpy as np

from mlx_dfloat.decode import DecodeResult
from mlx_dfloat.errors import DFloatBackendError
from mlx_dfloat.format import GroupArrays, MxGroup

THREADS = 512
CAP = 16192
_SCRATCH_BYTES = 4 * (16 + 16 + 1 + 16)


def _metal_static_bytes(declared: int) -> int:
    """The static threadgroup memory Metal reserves for ``declared`` bytes of arrays.

    Metal reports ``staticThreadgroupMemoryLength`` at 16-byte granularity (measured with
    standalone compiles: 196 -> 208, 192 -> 192; mlx 0.32.2, macOS 27.0, M1 Max).
    """
    return -(-declared // 16) * 16


THREADGROUP_BYTES_STAGED = _metal_static_bytes(
    2 * CAP + _SCRATCH_BYTES
)  # 32,580 -> 32,592 < 32,768
THREADGROUP_BYTES_DIRECT = _metal_static_bytes(2 + _SCRATCH_BYTES)  # 198 -> 208

_KERNEL: Any | None = None
# (force_direct, poison_buf, unguarded_gap_read) -> warmed
_PIPELINES: dict[tuple[bool, bool, bool], bool] = {}

_SOURCE = r"""
    const uint t = thread_position_in_threadgroup.x;
    const uint b = threadgroup_position_in_grid.x;
    const uint g = b * 512u + t;
    const uint n_bytes = (uint)encoded_shape[0];
    const uint n = (uint)sm_shape[0];
    const uint n_luts = (uint)luts_shape[0];
    const uint n_launch = (uint)positions_shape[0] - 1u;
    const uint blk_start = positions[b];
    const uint blk_end = positions[b + 1u];
    const uint interval = blk_end - blk_start;
    const bool staged = (!FORCE_DIRECT) && (interval <= (uint)CAP);
    const uint last_real = (n_bytes - 1u) / 8u;
    const bool real = (8u * g) < n_bytes;

    threadgroup ushort buf[FORCE_DIRECT ? 1 : CAP];
    threadgroup uint simd_totals[16];
    threadgroup uint simd_pre[16];
    threadgroup uint block_total[1];
    threadgroup uint simd_bad[16];
    if (POISON_BUF && staged) { for (uint i = t; i < (uint)CAP; i += 512u) buf[i] = (ushort)0xDEAD; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint gbit = 5u * g;
    const uint gbyte = gbit >> 3;
    const uint gsh = gbit & 7u;
    const uint g0 = (uint)gaps[gbyte];
    const uint g1 = (UNGUARDED_GAP_READ || gsh > 3u) ? (uint)gaps[gbyte + 1u] : 0u;
    const uint gap = (((g0 << 8) | g1) >> (11u - gsh)) & 31u;

    ulong w = 0;
    for (uint k = 0; k < 8u; ++k) {
        const uint i = 8u * g + k;
        w = (w << 8) | (ulong)((i < n_bytes) ? encoded[i] : (uchar)0);
    }
    uint la = 0;
    for (uint k = 8u; k < 12u; ++k) {
        const uint i = 8u * g + k;
        la = (la << 8) | (uint)((i < n_bytes) ? encoded[i] : (uchar)0);
    }

    uint count = 0;
    uint p = gap;
    bool bad = false;
    if (real) {
        for (uint it = 0; it < 64u && p < 64u; ++it) {
            const uint word = (p <= 32u) ? (uint)(w >> (32u - p)) : ((uint)(w << (p - 32u)) | (la >> (64u - p)));
            uint sym = (uint)luts[word >> 24];
            for (uint level = 1u; level <= 3u && sym >= 240u; ++level) {
                const uint row = 256u - sym;
                if (row < 1u || row > n_luts - 2u) { sym = 240u; break; }
                sym = (uint)luts[row * 256u + ((word >> (24u - 8u * level)) & 0xFFu)];
            }
            const uint len = (sym < 240u) ? (uint)luts[(n_luts - 1u) * 256u + sym] : 0u;
            if (len == 0u) { bad = true; break; }
            count += 1u;
            p += len;
        }
    }
    const uint sg = t / 32u;
    const uint lane = t % 32u;
    const uint local = simd_prefix_exclusive_sum(count);

    // Chain check (reference `broken`): a clean thread before the last real one must end exactly where the
    // next thread's gap says its first code starts, unless its last code runs past the stream end.
    bool broken = false;
    if (real && !bad && g < last_real) {
        const uint nbit = 5u * (g + 1u);
        const uint nb0 = (uint)gaps[nbit >> 3];
        const uint nsh = nbit & 7u;
        const uint nb1 = (nsh > 3u) ? (uint)gaps[(nbit >> 3) + 1u] : 0u;
        const uint next_gap = (((nb0 << 8) | nb1) >> (11u - nsh)) & 31u;
        const ulong end_bit = 64ul * (ulong)g + (ulong)p;
        broken = (end_bit < 8ul * (ulong)n_bytes) && (p != 64u + next_gap);
    }
    const bool relevant_bad = bad && (g < last_real);
    const uint vote = (simd_any(relevant_bad) ? 1u : 0u) | (simd_any(broken) ? 4u : 0u);   // uniform flow

    if (lane == 31u) { simd_totals[sg] = local + count; simd_bad[sg] = vote; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint v = (lane < 16u) ? simd_totals[lane] : 0u;
        const uint pre = simd_prefix_exclusive_sum(v);
        if (lane < 16u) simd_pre[lane] = pre;
        if (lane == 15u) block_total[0] = pre + v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint first = blk_start + simd_pre[sg] + local;
    const uint total = block_total[0];

    if (real && count > 0u) {
        uint idx = first;
        const uint stop = min(min(first + count, n), blk_end);
        uint q = gap;
        for (uint it = 0; it < 64u && idx < stop; ++it) {
            const uint word = (q <= 32u) ? (uint)(w >> (32u - q)) : ((uint)(w << (q - 32u)) | (la >> (64u - q)));
            uint sym = (uint)luts[word >> 24];
            for (uint level = 1u; level <= 3u && sym >= 240u; ++level) {
                const uint row = 256u - sym;
                if (row < 1u || row > n_luts - 2u) { sym = 240u; break; }
                sym = (uint)luts[row * 256u + ((word >> (24u - 8u * level)) & 0xFFu)];
            }
            const uint len = (sym < 240u) ? (uint)luts[(n_luts - 1u) * 256u + sym] : 0u;
            if (len == 0u) break;
            const uint smv = (uint)sm[idx];
            const ushort v = (ushort)(((smv & 0x80u) << 8) | (sym << 7) | (smv & 0x7Fu));
            if (staged) buf[idx - blk_start] = v; else out[idx] = v;
            idx += 1u;
            q += len;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (staged) {
        const uint limit = min(interval, n - blk_start);
        for (uint i = t; i < limit; i += 512u) out[blk_start + i] = buf[i];
    }
    if (t == 511u) {
        uint flags = 0u;
        for (uint s = 0; s < 16u; ++s) flags |= simd_bad[s];
        uint word = flags;                                          // 1 invalid code, 4 broken chain
        const bool last_block = (b + 1u == n_launch);
        if ((!last_block && total != interval) || (last_block && total < interval)) word |= 2u;
        if (!staged) word |= 8u;                                    // informational: direct path taken
        status[b] = word;
    }
"""


def _build_kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="df11_decode",
            input_names=["encoded", "sm", "luts", "gaps", "positions"],
            output_names=["out", "status"],
            source=_SOURCE,
        )
    return _KERNEL


def _dispatch(
    group: MxGroup,
    *,
    force_direct: bool,
    poison_buf: bool,
    init_value: int | None,
    unguarded_gap_read: bool = False,
) -> tuple[mx.array, mx.array]:
    out, status = _build_kernel()(
        inputs=[
            group.encoded_exponent,
            group.sign_mantissa,
            group.luts,
            group.gaps,
            group.positions,
        ],
        template=[
            ("CAP", CAP),
            ("FORCE_DIRECT", force_direct),
            ("POISON_BUF", poison_buf),
            ("UNGUARDED_GAP_READ", unguarded_gap_read),
        ],
        grid=(THREADS * group.n_launch, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(group.n_elements,), (group.n_launch,)],
        output_dtypes=[mx.uint16, mx.uint32],
        init_value=init_value,
    )
    return out, status


def _warmup_group() -> MxGroup:
    # H1: three elements, one LUT level: enough to compile and dispatch every code path once.
    row0 = np.zeros(256, np.uint8)
    row0[0:128], row0[128:192], row0[192:256] = 127, 126, 128
    lens = np.zeros(256, np.uint8)
    lens[126], lens[127], lens[128] = 2, 1, 3
    return GroupArrays(
        encoded_exponent=np.array([0x9B], np.uint8),
        sign_mantissa=np.array([0, 0x80, 0x7F], np.uint8),
        luts=np.stack([row0, lens]),
        gaps=np.zeros(320, np.uint8),
        output_positions=np.array([0, 3], np.uint32),
        split_positions=np.zeros(0, np.int64),
    ).to_mx(name="warmup")


def ensure_pipeline(
    *, force_direct: bool, poison_buf: bool = False, unguarded_gap_read: bool = False
) -> None:
    """Compile and dispatch one tiny group for this instantiation, alone.

    Each distinct template tuple is its own JIT compile, so each is warmed once and the result
    is cached in `_PIPELINES`. The warm-up output is prefilled with 0, which none of the expected
    words equals, so an element the kernel fails to write cannot pass on recycled memory. The
    3-element warm-up group must run staged unless `force_direct` is set, so the status word's
    path bit is checked too. `unguarded_gap_read` is a test-only mutant (see `decode`).

    Raises:
        DFloatBackendError: Metal is unavailable, the kernel fails to compile or dispatch here
            (a pipeline ceiling below 512 threads, a driver refusal), the warm-up group
            decodes to the wrong bits, or it runs on the wrong write path.
    """
    key = (force_direct, poison_buf, unguarded_gap_read)
    if _PIPELINES.get(key):
        return
    if not mx.metal.is_available():
        raise DFloatBackendError("Metal is not available on this machine")
    try:
        out, status = _dispatch(
            _warmup_group(),
            force_direct=force_direct,
            poison_buf=poison_buf,
            init_value=0,
            unguarded_gap_read=unguarded_gap_read,
        )
        mx.eval(out, status)
    except Exception as exc:  # compile error, pipeline ceiling below 512, driver refusal
        raise DFloatBackendError(f"the Metal decode kernel cannot run here: {exc}") from exc
    word = int(np.array(status)[0])
    if np.array(out).tolist() != [0x3F00, 0xBF80, 0x407F] or word & 7:
        raise DFloatBackendError("the Metal decode kernel produced wrong bits on the warm-up group")
    if (word & 8) != (8 if force_direct else 0):
        want = "direct" if force_direct else "staged"
        raise DFloatBackendError(
            f"the Metal decode kernel took the wrong write path on the warm-up group (want {want})"
        )
    _PIPELINES[key] = True


def metal_ready() -> bool:
    """Whether both kernel instantiations (direct and staged) compile and pass their warm-up here."""
    try:
        ensure_pipeline(force_direct=True)
        ensure_pipeline(force_direct=False)
    except DFloatBackendError:
        return False
    return True


def decode(
    group: MxGroup,
    *,
    force_direct: bool = False,
    _init_value: int | None = None,
    _poison_buf: bool = False,
    _unguarded_gap_read: bool = False,
) -> DecodeResult:
    """Lazy Metal decode of one group.

    `_init_value` and `_poison_buf` are test-only poisons for the write-once tests: they prefill
    the output and the staging buffer so an unwritten element cannot pass as a correct one.
    `_unguarded_gap_read` is a test-only mutant for the shader-validation test: it drops the
    guard on the second `gaps` byte, so the last thread of a full group reads one byte past
    `gaps` (the value is shifted out, so the bits stay correct).

    Raises:
        DFloatBackendError: The kernel instantiation cannot be warmed up here (see
            `ensure_pipeline`).
    """
    ensure_pipeline(
        force_direct=force_direct, poison_buf=_poison_buf, unguarded_gap_read=_unguarded_gap_read
    )
    out, status = _dispatch(
        group,
        force_direct=force_direct,
        poison_buf=_poison_buf,
        init_value=_init_value,
        unguarded_gap_read=_unguarded_gap_read,
    )
    direct = group.n_launch if force_direct else int(np.count_nonzero(group.intervals > CAP))
    return DecodeResult(
        bits=out,
        status=status,
        backend="metal",
        direct_blocks=direct,
        threadgroup_bytes=THREADGROUP_BYTES_DIRECT if force_direct else THREADGROUP_BYTES_STAGED,
    )
