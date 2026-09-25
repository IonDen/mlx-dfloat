import pytest

import mlx_dfloat
from mlx_dfloat import DFloatError, DFloatFormatError


def test_version_is_exported():
    assert isinstance(mlx_dfloat.__version__, str)
    assert mlx_dfloat.__version__
    assert "__version__" in mlx_dfloat.__all__


def test_errors_are_exported():
    assert "DFloatError" in mlx_dfloat.__all__
    assert "DFloatFormatError" in mlx_dfloat.__all__
    assert "DFloatResourceError" in mlx_dfloat.__all__


def test_format_error_is_caught_by_package_root():
    # Bug this catches: a format error that does not derive from DFloatError escapes a
    # caller's `except DFloatError`.
    with pytest.raises(DFloatError):
        raise DFloatFormatError("bad checkpoint")


def test_format_error_is_a_value_error():
    # Callers that only know the builtin taxonomy still catch a malformed checkpoint.
    with pytest.raises(ValueError, match="bad checkpoint"):
        raise DFloatFormatError("bad checkpoint")
