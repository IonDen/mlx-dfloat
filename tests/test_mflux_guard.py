import builtins
import sys

import pytest

from mlx_dfloat import DFloatDependencyError, DFloatError, DFloatUnsupportedError


def test_the_new_errors_are_package_rooted_and_typed_like_their_causes():
    # Bug caught: either new error losing its DFloatError root (escapes `except DFloatError`) or
    # its builtin parent (DFloatUnsupportedError not a ValueError, DFloatDependencyError not an
    # ImportError), which would surprise a caller that only knows the builtin taxonomy.
    assert issubclass(DFloatUnsupportedError, DFloatError)
    assert issubclass(DFloatUnsupportedError, ValueError)
    assert issubclass(DFloatDependencyError, DFloatError)
    assert issubclass(DFloatDependencyError, ImportError)


def test_importing_the_mflux_package_needs_no_mflux(monkeypatch):
    # Review focus 4: the guard fires at first use, never at import.
    #
    # This test has two independent halves, each catching a different regression:
    # - The `no_mflux` __import__ hook catches `mlx_dfloat.mflux` eagerly importing mflux at
    #   module-import time (e.g. a top-level `import mflux` instead of the lazy
    #   `importlib.util.find_spec` check): `import mlx_dfloat.mflux as adapters` would raise
    #   ImportError instead of succeeding, because `find_spec`/module import for a name mflux
    #   itself resolves through `__import__` machinery only when mflux is actually imported, not
    #   when merely probed with `find_spec`.
    # - The `pytest.raises(DFloatDependencyError, ...)` block catches `require_mflux()` failing to
    #   raise the package-rooted error (e.g. returning None silently, or raising a bare
    #   ImportError/ModuleNotFoundError with no "install mlx-dfloat[mflux]" guidance) when mflux
    #   is unavailable.
    monkeypatch.delitem(sys.modules, "mlx_dfloat.mflux", raising=False)
    real_import = builtins.__import__

    def no_mflux(name, *args, **kwargs):
        if name == "mflux" or name.startswith("mflux."):
            raise ImportError("no mflux here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mflux)
    import mlx_dfloat.mflux as adapters  # must not raise

    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        adapters.require_mflux()
