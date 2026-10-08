import pytest

import mlx_dfloat
from mlx_dfloat import DFloatBackendError, DFloatError, DFloatFormatError


def test_version_is_exported():
    assert isinstance(mlx_dfloat.__version__, str)
    assert mlx_dfloat.__version__
    assert "__version__" in mlx_dfloat.__all__


def test_errors_are_exported():
    assert "DFloatError" in mlx_dfloat.__all__
    assert "DFloatFormatError" in mlx_dfloat.__all__
    assert "DFloatResourceError" in mlx_dfloat.__all__
    assert "DFloatBackendError" in mlx_dfloat.__all__
    assert "DFloatUnsupportedError" in mlx_dfloat.__all__
    assert "DFloatDependencyError" in mlx_dfloat.__all__
    assert "DFloatIntegrationError" in mlx_dfloat.__all__


def test_format_error_is_caught_by_package_root():
    # Bug this catches: a format error that does not derive from DFloatError escapes a
    # caller's `except DFloatError`.
    with pytest.raises(DFloatError):
        raise DFloatFormatError("bad checkpoint")


def test_format_error_is_a_value_error():
    # Callers that only know the builtin taxonomy still catch a malformed checkpoint.
    with pytest.raises(ValueError, match="bad checkpoint"):
        raise DFloatFormatError("bad checkpoint")


def test_backend_error_is_caught_by_package_root():
    # Bug this catches: a backend error that does not derive from DFloatError escapes a
    # caller's `except DFloatError`.
    with pytest.raises(DFloatError):
        raise DFloatBackendError("no Metal device")


def test_access_error_is_package_rooted_and_a_permission_error():
    # Bug caught: a gated-repo failure escaping `except DFloatError`, or not reading as a
    # PermissionError to a caller that only knows the builtin taxonomy.
    from mlx_dfloat import DFloatAccessError

    assert issubclass(DFloatAccessError, DFloatError)
    assert issubclass(DFloatAccessError, PermissionError)
    assert "DFloatAccessError" in mlx_dfloat.__all__


def test_dfloat_model_is_exported_from_the_mflux_package():
    # Bug caught: the name-based constructor missing from the adapters' public surface.
    import mlx_dfloat.mflux as adapters

    assert "DFloatModel" in adapters.__all__
    assert callable(adapters.DFloatModel)


@pytest.mark.mflux
def test_dfloat_zimage_is_reachable_lazily():
    # Bug caught: DFloatZImage missing from the lazy attribute hook (AttributeError on first use).
    import mlx_dfloat.mflux as adapters
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    assert adapters.DFloatZImage is DFloatZImage


def test_the_model_classes_stay_out_of_the_star_import():
    # Bug caught: a model class added to __all__ (`from mlx_dfloat.mflux import *` would then need mflux).
    import mlx_dfloat.mflux as adapters

    assert sorted(adapters.__all__) == ["DFloatModel", "require_mflux"]


@pytest.mark.mflux
def test_dfloat_flux2_klein_is_reachable_lazily():
    # Bug caught: DFloatFlux2Klein missing from the lazy attribute hook (AttributeError on first use).
    import mlx_dfloat.mflux as adapters
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    assert adapters.DFloatFlux2Klein is DFloatFlux2Klein


@pytest.mark.mflux
def test_dfloat_qwen_image_21_is_reachable_lazily():
    # Bug caught: DFloatQwenImage21 missing from the lazy attribute hook (AttributeError on first use), or added to
    # __all__ (a star-import would then need mflux; the star-import test pins __all__).
    import mlx_dfloat.mflux as adapters
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    assert adapters.DFloatQwenImage21 is DFloatQwenImage21


@pytest.mark.mflux
def test_dfloat_ernie_image_is_reachable_lazily():
    # Bug caught: DFloatErnieImage missing from the lazy attribute hook (AttributeError on first use), or added to
    # __all__ (a star-import would then need mflux; the star-import test pins __all__).
    import mlx_dfloat.mflux as adapters
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    assert adapters.DFloatErnieImage is DFloatErnieImage
