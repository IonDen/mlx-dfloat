"""mflux's compiled ``predict``, turned off for the duration of one factory call.

mflux 0.20's ``_predict`` factories (Z-Image, FLUX.2 Klein, ERNIE-Image, Krea 2) return ``mx.compile(predict)``
unless ``AppleSiliconUtil.is_m1_or_m2()``, which they read at call time. The block seam evaluates at block
boundaries, which cannot happen inside ``mx.compile``, so the adapters take the plain function on every chip. The
override is process-wide but lasts only while the factory runs (it builds a closure; nothing else executes). A
factory that still returns a compiled function is refused, so a future mflux that reads the chip another way fails
with an error that names the factory.
"""

import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.mflux import require_mflux

T = TypeVar("T")


@contextmanager
def eager_factories() -> Iterator[None]:
    """Within the block, mflux's chip check answers "M1 or M2", the chips mflux runs uncompiled."""
    require_mflux()
    from mflux.utils.apple_silicon import AppleSiliconUtil

    original = AppleSiliconUtil.__dict__["is_m1_or_m2"]
    setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))  # noqa: B010
    try:
        yield
    finally:
        setattr(AppleSiliconUtil, "is_m1_or_m2", original)  # noqa: B010


def is_plain_function(fn: object) -> bool:
    """Whether ``fn`` is a plain Python function.

    ``mx.compile`` returns ``mlx.gc_func``, a ``FunctionType`` subclass, so only an exact type check tells them apart.
    """
    return type(fn) is types.FunctionType


def predict_mode(factory: Callable[..., object], *args: Any) -> str:
    """``"uncompiled"`` or ``"compiled"``: what the factory returns under the override (it builds a closure only)."""
    with eager_factories():
        fn = factory(*args)
    return "uncompiled" if is_plain_function(fn) else "compiled"


def uncompiled(factory: Callable[..., T], *args: Any) -> T:
    """Call one of mflux's ``_predict`` factories so it returns its plain, uncompiled function.

    Raises:
        DFloatIntegrationError: The factory still returned a compiled function (the override did not reach it).
    """
    with eager_factories():
        fn = factory(*args)
    if not is_plain_function(fn):
        name = getattr(factory, "__qualname__", repr(factory))
        raise DFloatIntegrationError(
            f"{name} returned a compiled function ({type(fn).__module__}.{type(fn).__qualname__}) despite the "
            "chip-check override: this mflux release decides compilation another way, and the block seam cannot "
            "evaluate inside mx.compile"
        )
    return fn
