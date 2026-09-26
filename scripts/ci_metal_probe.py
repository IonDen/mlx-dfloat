"""Report whether the Metal decode kernel can run on this machine (a CI runner, typically).

Prints ``mx.device_info()``, ``platform.mac_ver()``, then ``metal_ready=<bool>`` and, when the
kernel cannot run, the ``DFloatBackendError`` text that says why. It is a diagnostic: it always
exits 0, so a runner without a usable GPU shows its reason in the log instead of failing the job.

Usage:
    uv run python scripts/ci_metal_probe.py
"""

import platform
import traceback


def main() -> int:
    """Print the device, the OS version and the kernel's readiness; always return 0."""
    try:
        import mlx.core as mx  # an import failure is itself the diagnostic

        from mlx_dfloat import _metal_decode
        from mlx_dfloat.errors import DFloatBackendError
    except Exception:
        print(f"mac_ver={platform.mac_ver()}")
        print("metal_ready=False")
        traceback.print_exc()
        return 0
    try:
        print(f"device_info={dict(mx.device_info())}")
    except Exception as exc:
        print(f"device_info unavailable: {exc!r}")
    print(f"mac_ver={platform.mac_ver()}")
    try:
        ready = _metal_decode.metal_ready()
        print(f"metal_ready={ready}")
        if not ready:
            # metal_ready() swallows the reason; warm both instantiations again to print it.
            try:
                _metal_decode.ensure_pipeline(force_direct=True)
                _metal_decode.ensure_pipeline(force_direct=False)
            except DFloatBackendError as exc:
                print(f"DFloatBackendError: {exc}")
    except Exception:
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
