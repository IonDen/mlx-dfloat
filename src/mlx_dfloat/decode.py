"""Decode DF11 groups to BF16 bit patterns through an explicitly chosen backend."""

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from mlx_dfloat import reference
from mlx_dfloat.errors import DFloatBackendError, DFloatError, DFloatFormatError
from mlx_dfloat.format import GroupArrays, MxGroup

Backend = Literal["reference", "metal"]

STATUS_INVALID_CODE = 1
STATUS_COUNT_MISMATCH = 2
STATUS_BROKEN_CHAIN = 4
STATUS_PATH_DIRECT = 8  # informational: the block wrote straight to device memory
STATUS_ERROR_MASK = STATUS_INVALID_CODE | STATUS_COUNT_MISMATCH | STATUS_BROKEN_CHAIN
_STATUS_TEXT = {
    STATUS_INVALID_CODE: "invalid code",
    STATUS_COUNT_MISMATCH: "code count does not match output_positions",
    STATUS_BROKEN_CHAIN: "thread chain broken (corrupt gaps or stream)",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeResult:
    """One decoded group: its BF16 bits, a per-block status word and how it was decoded.

    ``threadgroup_bytes`` is the static threadgroup memory the kernel instantiation reserves (0 for
    the reference), not a per-block figure: every launched block reserves it, staged or direct.
    The bits of a block whose status word has an error bit are undefined; call ``check`` first.
    """

    bits: mx.array
    status: mx.array
    backend: Backend
    direct_blocks: int
    threadgroup_bytes: int


def _to_arrays(group: MxGroup) -> GroupArrays:
    return GroupArrays(
        encoded_exponent=np.array(group.encoded_exponent),
        sign_mantissa=np.array(group.sign_mantissa),
        luts=np.array(group.luts),
        gaps=np.array(group.gaps),
        output_positions=np.array(group.positions).astype(np.uint32),
        split_positions=np.array(group.split_positions, dtype=np.int64),
    )


def available_backends() -> tuple[Backend, ...]:
    """Backends that can run here; "metal" appears only after its kernel warm-up succeeds.

    The warm-up compiles and checks a pipeline for every input-binding signature a group can
    present (many blocks, one block, all-tiny), so once "metal" is listed every group's decode
    reuses a pipeline that already decoded bit-exactly here.
    """
    from mlx_dfloat import _metal_decode  # lazy: importing the package must not touch the GPU

    return ("reference", "metal") if _metal_decode.metal_ready() else ("reference",)


def decode_group(group: MxGroup, *, backend: Backend) -> DecodeResult:
    """Decode a group. The reference runs eagerly on the CPU; "metal" returns a lazy result.

    The reference backend skips only the stream-level EOF byte-count check (`check_stream_end=False`): the kernel
    cannot see it either, and the committed test slice is deliberately truncated. The reference module's own tests
    keep the strict path.

    Raises:
        DFloatBackendError: `backend` is not a known backend, or `backend="metal"` is requested and
            the Metal backend cannot run here.
        DFloatFormatError: The reference decoder finds the group structurally invalid.
    """
    if backend == "reference":
        out = reference.decode_group(_to_arrays(group), name=group.name, check_stream_end=False)
        return DecodeResult(
            bits=mx.array(out),
            status=mx.zeros((group.n_launch,), dtype=mx.uint32),
            backend="reference",
            direct_blocks=0,
            threadgroup_bytes=0,
        )
    if backend == "metal":
        from mlx_dfloat import _metal_decode

        return _metal_decode.decode(group)
    raise DFloatBackendError(f"unknown backend {backend!r}")


def check(result: DecodeResult, *, name: str = "<group>") -> None:
    """Evaluate the status words and refuse the group if any block reports an error.

    Raises:
        DFloatFormatError: A block's status word has an error bit set (informational bits, such
            as `STATUS_PATH_DIRECT`, are ignored).
    """
    check_status(result.status, name=name)


def check_status(status_words: mx.array, *, name: str = "<group>") -> None:
    """``check`` on a bare status array: reads it on the host, so call it after the step's eval.

    Raises:
        DFloatFormatError: A block's status word has an error bit set.
    """
    status = np.array(status_words) & STATUS_ERROR_MASK
    bad = np.flatnonzero(status)
    if bad.size:
        b = int(bad[0])
        reasons = ", ".join(t for flag, t in _STATUS_TEXT.items() if int(status[b]) & flag)
        raise DFloatFormatError(f"{name}: block {b}: {reasons}")


def split_matrices(flat: mx.array, split_positions: tuple[int, ...]) -> list[mx.array]:
    """Cut a decoded group into its matrices (views, no copies)."""
    return list(mx.split(flat, list(split_positions))) if split_positions else [flat]


SelftestPath = Literal["staged", "direct", "reference"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SelftestCheck:
    """One canary group decoded one way and compared with its known BF16 bits."""

    group: str
    path: SelftestPath
    elements: int
    ok: bool
    detail: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class SelftestReport:
    """The outcome of `selftest`: every check, and why there are none when the GPU decoder cannot run."""

    ok: bool
    reason: str | None
    device: str
    mlx_version: str
    package_version: str
    checks: tuple[SelftestCheck, ...]
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """The report as plain JSON-serialisable data."""
        return asdict(self)


def _device_summary() -> str:
    try:
        info = mx.device_info()
    except Exception:  # no device to describe
        return "unknown device"
    parts = []
    if info.get("device_name"):
        parts.append(str(info["device_name"]))
    if info.get("architecture"):
        parts.append(str(info["architecture"]))
    memory = info.get("memory_size")
    if isinstance(memory, int):
        parts.append(f"{memory / 2**30:.0f} GiB")
    return ", ".join(parts) or "unknown device"


def selftest() -> SelftestReport:
    """Decode the packaged canary groups on both Metal write paths and with the CPU reference.

    Each group is compared with the BF16 bits known in advance, three ways: the staged and direct
    write paths of the GPU kernel, and the CPU reference. One failing path never hides another. A
    GPU fault is reported in the result rather than raised: `ok` is False and the failing checks
    say what differed.
    """
    from mlx_dfloat import _canary, _metal_decode
    from mlx_dfloat._version import __version__

    start = time.perf_counter()

    def report(
        ok: bool, reason: str | None, checks: tuple[SelftestCheck, ...] = ()
    ) -> SelftestReport:
        return SelftestReport(
            ok=ok,
            reason=reason,
            device=_device_summary(),
            mlx_version=str(getattr(mx, "__version__", "unknown")),
            package_version=__version__,
            checks=checks,
            seconds=time.perf_counter() - start,
        )

    if not mx.metal.is_available():
        return report(False, "Metal is not available on this machine")
    try:
        _metal_decode.ensure_pipeline(force_direct=True)
        _metal_decode.ensure_pipeline(force_direct=False)
        canaries = _canary.canary_groups()
    except DFloatBackendError as exc:  # no pipeline, or the packaged canary data is unreadable
        return report(False, str(exc))
    checks: list[SelftestCheck] = []
    for canary in canaries:
        for path in ("staged", "direct"):
            failure = _metal_decode.canary_failure(canary, force_direct=path == "direct")
            checks.append(
                SelftestCheck(
                    group=canary.name,
                    path=path,
                    elements=int(canary.expected.size),
                    ok=failure is None,
                    detail=failure,
                )
            )
        try:
            got = reference.decode_group(canary.arrays, name=canary.name, check_stream_end=False)
        except DFloatError as exc:
            reference_failure: str | None = f"the reference decode raised: {exc}"
        else:
            wrong = np.flatnonzero(got != canary.expected)
            reference_failure = (
                None if wrong.size == 0 else f"wrong bits at element {int(wrong[0])}"
            )
        checks.append(
            SelftestCheck(
                group=canary.name,
                path="reference",
                elements=int(canary.expected.size),
                ok=reference_failure is None,
                detail=reference_failure,
            )
        )
    return report(all(c.ok for c in checks), None, tuple(checks))
