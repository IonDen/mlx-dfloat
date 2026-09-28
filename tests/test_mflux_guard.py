import builtins
import importlib.machinery
import importlib.util
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


def _is_mflux(name):
    return name == "mflux" or name.startswith("mflux.")


def _hide_mflux(monkeypatch):
    """No mflux, whatever this venv holds: both the import and the ``find_spec`` probe miss it."""
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def no_mflux(name, *args, **kwargs):
        if _is_mflux(name):
            raise ImportError("no mflux here")
        return real_import(name, *args, **kwargs)

    def find_spec(name, *args, **kwargs):
        return None if _is_mflux(name) else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mflux)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_importing_the_mflux_package_needs_no_mflux(monkeypatch):
    # Bug caught: `mlx_dfloat.mflux` importing mflux at module import time (a top-level `import
    # mflux`), which the __import__ hook turns into an ImportError here.
    monkeypatch.delitem(sys.modules, "mlx_dfloat.mflux", raising=False)
    _hide_mflux(monkeypatch)
    import mlx_dfloat.mflux as adapters  # must not raise

    assert callable(adapters.require_mflux)


def test_require_mflux_raises_the_package_error_only_when_mflux_is_missing(monkeypatch):
    # Bug caught: require_mflux returning silently (or raising a bare ImportError without the
    # install hint) when mflux is absent, or refusing when mflux is present. find_spec is patched
    # both ways, so the verdict does not depend on what this venv happens to hold.
    import mlx_dfloat.mflux as adapters

    _hide_mflux(monkeypatch)
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        adapters.require_mflux()
    spec = importlib.machinery.ModuleSpec("mflux", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: spec)
    adapters.require_mflux()  # present: no raise


def test_the_first_mflux_entry_points_raise_the_package_error_without_mflux(monkeypatch):
    # Bug caught: flux_name_map(), seam_transformer_class() or the benches' build_transformer
    # reaching their lazy `from mflux...` import unguarded, so a user without the extra gets a bare
    # ModuleNotFoundError instead of the DFloatDependencyError the changelog promises.
    from scripts import _flux_rig as rig

    from mlx_dfloat.mflux.flux1 import names, transformer

    _hide_mflux(monkeypatch)
    transformer.seam_transformer_class.cache_clear()
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        names.flux_name_map()
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        transformer.seam_transformer_class()
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        rig.build_transformer("schnell", None)
