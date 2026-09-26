"""Metal decode-kernel backend; Task 3 replaces this stub with the real `mx.fast.metal_kernel`."""

from typing import TYPE_CHECKING

import mlx.core as mx

from mlx_dfloat.errors import DFloatBackendError
from mlx_dfloat.format import MxGroup

if TYPE_CHECKING:
    from mlx_dfloat.decode import DecodeResult

_PIPELINES: dict[tuple[bool, bool], bool] = {}


def metal_ready() -> bool:
    """Whether the Metal decode kernel is built and warmed up here. Always False in the stub."""
    return False


def decode(group: MxGroup, **kwargs: object) -> "DecodeResult":
    """Decode `group` on the Metal backend.

    Raises:
        DFloatBackendError: Metal is not available on this machine, or (when Metal is available)
            the decode kernel is not built yet.
    """
    if not mx.metal.is_available():
        raise DFloatBackendError("Metal is not available on this machine")
    raise DFloatBackendError("the Metal decode kernel is not built yet")
