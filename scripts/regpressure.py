"""Compile-time pipeline limits of the DF11 decode kernel, read back through the Metal framework.

For each kernel instantiation the bench runs (direct and staged, no test poisons), this probe
prints two numbers from the compiled ``MTLComputePipelineState``:

- ``maxTotalThreadsPerThreadgroup``: the device lowers it as the per-thread register footprint
  grows. The kernel launches 512 threads per threadgroup, so a ceiling below 512 means it cannot
  run; the probe exits 2.
- ``staticThreadgroupMemoryLength``: the threadgroup memory the compiler actually reserved. It
  must equal the byte count ``_metal_decode`` declares for that path
  (``THREADGROUP_BYTES_DIRECT`` / ``THREADGROUP_BYTES_STAGED``); otherwise the probe prints both
  and exits 2.

The ceiling is a register-pressure telltale, not a performance verdict: two variants at the same
ceiling can differ in throughput, and only a measured rate decides between them.

How it works: the kernel is launched once with ``verbose=True``, which makes MLX print the exact
MSL it compiled. That dump comes from MLX's C++ layer straight to file descriptor 1, so it is
captured at the descriptor level. The banner line and code fences are stripped, the prelude MLX's
own compile step supplies is prepended, and the text is recompiled standalone with PyObjC's Metal
bindings. The probe group is zero-filled and seven blocks long, so every input has at least eight
entries: MLX binds smaller read-only inputs in the ``constant`` address space, which would compile
a different signature from the one real groups get.

Adapted from mlx-train-perf's register-pressure probe. Needs the ``bench`` dependency group:
    uv run --group bench python scripts/regpressure.py
Exit codes: 0 both variants within limits, 2 a limit violated or the probe could not run.
"""

import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from typing import IO, Any

import mlx.core as mx
import numpy as np

from mlx_dfloat import _metal_decode as md
from mlx_dfloat.format import GroupArrays, MxGroup

MIN_THREADS = md.THREADS
_BLOCKS = 7  # 8 output positions: every input clears MLX's small-array `constant` binding
_ELEMENTS_PER_BLOCK = md.THREADS * 64  # an all-zero stream is 64 one-bit codes per thread
_BANNER_RE = re.compile(r"Generated source code for `[^`]+`:\s*\n")
_PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"
# (label, FORCE_DIRECT, the threadgroup bytes _metal_decode declares for that path)
VARIANTS = (
    ("direct", True, md.THREADGROUP_BYTES_DIRECT),
    ("staged", False, md.THREADGROUP_BYTES_STAGED),
)


def _strip_banner_and_fences(raw: str) -> str:
    """Drop the ``verbose=True`` banner line and the markdown code fences around the dump."""
    text = _BANNER_RE.sub("", raw, count=1)
    return "\n".join(line for line in text.splitlines() if line.strip() != "```")


def _prepare_msl(raw: str) -> str:
    """Strip the capture noise and prepend the prelude a standalone compile needs."""
    return _PRELUDE + _strip_banner_and_fences(raw)


@contextlib.contextmanager
def _capture_fd_stdout() -> Iterator[IO[str]]:
    """Capture file descriptor 1, where MLX's C++ layer prints the ``verbose=True`` dump."""
    sys.stdout.flush()
    with tempfile.TemporaryFile(mode="w+") as buf:
        saved = os.dup(1)
        os.dup2(buf.fileno(), 1)
        try:
            yield buf
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def probe_group() -> MxGroup:
    """A structurally valid, zero-filled seven-block group using the warm-up group's LUTs."""
    luts = np.array(md._warmup_group().luts)
    return GroupArrays(
        encoded_exponent=np.zeros(_BLOCKS * 4096, np.uint8),
        sign_mantissa=np.zeros(_BLOCKS * _ELEMENTS_PER_BLOCK, np.uint8),
        luts=luts,
        gaps=np.zeros(320 * _BLOCKS, np.uint8),
        output_positions=(np.arange(_BLOCKS + 1) * _ELEMENTS_PER_BLOCK).astype(np.uint32),
        split_positions=np.zeros(0, np.int64),
    ).to_mx(name="regpressure-probe")


def capture_msl(group: MxGroup, *, force_direct: bool) -> str:
    """Launch one instantiation with ``verbose=True`` and return the cleaned MSL it compiled.

    Raises:
        RuntimeError: No kernel source appeared in the captured output.
    """
    with _capture_fd_stdout() as buf:
        out = md._build_kernel()(
            inputs=[
                group.encoded_exponent,
                group.sign_mantissa,
                group.luts,
                group.gaps,
                group.positions,
            ],
            template=[
                ("CAP", md.CAP),
                ("FORCE_DIRECT", force_direct),
                ("POISON_BUF", False),
                ("UNGUARDED_GAP_READ", False),
            ],
            grid=(md.THREADS * group.n_launch, 1, 1),
            threadgroup=(md.THREADS, 1, 1),
            output_shapes=[(group.n_elements,), (group.n_launch,)],
            output_dtypes=[mx.uint16, mx.uint32],
            init_value=0,
            verbose=True,
        )
        mx.eval(out)
        os.fsync(1)
        buf.seek(0)
        raw = buf.read()
    if "[[kernel]]" not in raw:
        raise RuntimeError(f"no MSL captured for the probe launch; got:\n{raw[:500]}")
    return _prepare_msl(raw)


def compile_limits(metal: Any, device: Any, msl: str) -> tuple[int, int]:
    """Compile ``msl`` standalone; return (maxTotalThreadsPerThreadgroup, staticThreadgroupMemoryLength).

    Raises:
        RuntimeError: The source does not compile to exactly one kernel, or no pipeline results.
    """
    library, error = device.newLibraryWithSource_options_error_(
        msl, metal.MTLCompileOptions.new(), None
    )
    if library is None:
        raise RuntimeError(f"standalone MSL recompile failed: {error}")
    names = list(library.functionNames())
    if len(names) != 1:
        raise RuntimeError(f"expected exactly one kernel function, got {names}")
    function = library.newFunctionWithName_(names[0])
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pipeline is None:
        raise RuntimeError(f"cannot create a compute pipeline state: {error}")
    return int(pipeline.maxTotalThreadsPerThreadgroup()), int(
        pipeline.staticThreadgroupMemoryLength()
    )


def main() -> int:
    """Probe both instantiations; 0 when both are within limits, else 2."""
    try:
        import Metal  # PyObjC, from the bench dependency group
    except ImportError:
        print("error: PyObjC Metal is missing; run with `uv run --group bench`", file=sys.stderr)
        return 2
    device = Metal.MTLCreateSystemDefaultDevice()
    print(f"mlx {mx.__version__}, device {device.name()}")
    group = probe_group()
    code = 0
    for label, force_direct, declared in VARIANTS:
        try:
            ceiling, tg_bytes = compile_limits(
                Metal, device, capture_msl(group, force_direct=force_direct)
            )
        except RuntimeError as exc:
            print(f"error: {label}: {exc}", file=sys.stderr)
            return 2
        print(
            f"{label}: maxTotalThreadsPerThreadgroup={ceiling} "
            f"staticThreadgroupMemoryLength={tg_bytes}"
        )
        if tg_bytes != declared:
            print(
                f"error: {label}: the compiler reserved {tg_bytes} threadgroup bytes, "
                f"_metal_decode declares {declared}",
                file=sys.stderr,
            )
            code = 2
        if ceiling < MIN_THREADS:
            print(
                f"error: {label}: ceiling {ceiling} is below the {MIN_THREADS} threads per "
                "threadgroup the kernel launches",
                file=sys.stderr,
            )
            code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
