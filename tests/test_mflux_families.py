import subprocess
import sys
from dataclasses import replace

import pytest

from mlx_dfloat import DFloatUnsupportedError
from mlx_dfloat.mflux import families


def test_importing_the_registry_does_not_import_mflux():
    # Bug caught: a family module imported at registry import time (the CLI's --help would need mflux).
    code = "import sys, mlx_dfloat.mflux.families as f; assert 'mflux' not in sys.modules; print(len(f.MODELS))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "5"


def test_resolving_an_adapter_module_and_importing_the_package_loads_no_mlx():
    # Bug caught: an eager `from mlx_dfloat.mflux.families import DFloatModel` in mlx_dfloat/mflux/__init__.py.
    # pytest-cov's --cov=mlx_dfloat.mflux.flux1.model resolves that spec when coverage starts; the import chain then
    # loaded mlx.core there, and mlx's native types registered twice (the CI lane aborted with exit 134).
    code = (
        "import importlib.util, sys; importlib.util.find_spec('mlx_dfloat.mflux.flux1.model'); "
        "import mlx_dfloat.mflux; print('mlx.core' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_dfloat_model_is_still_importable_from_the_package():
    # Bug caught: the lazy attribute dropping DFloatModel (the README's `from mlx_dfloat.mflux import DFloatModel`).
    from mlx_dfloat.mflux import DFloatModel

    assert DFloatModel is families.DFloatModel


def test_dfloat_model_builds_the_family_class_with_the_name_and_options(monkeypatch):
    # Bug caught: dispatch on the family name instead of the model name (z-image-turbo built as z-image).
    seen = []
    monkeypatch.setitem(
        families.FAMILIES,
        "zimage",
        replace(
            families.FAMILIES["zimage"],
            load_model_class=lambda: lambda name, **kw: seen.append((name, kw)) or "built",
        ),
    )
    assert families.DFloatModel("z-image-turbo", eval_policy="depth2") == "built"
    assert seen == [("z-image-turbo", {"eval_policy": "depth2"})]


def test_an_unknown_name_is_refused_with_the_known_ones():
    # Bug caught: a KeyError instead of a package error a user can act on.
    with pytest.raises(DFloatUnsupportedError, match="z-image-turbo"):
        families.entry("zimage-turbo")
    with pytest.raises(DFloatUnsupportedError, match="z-image-turbo"):
        families.family_of("zimage-turbo")


@pytest.mark.parametrize(
    ("name", "family", "steps", "guidance", "scheduler", "two_calls", "negative"),
    [
        ("schnell", "flux1", 4, 3.5, "linear", False, False),
        ("dev", "flux1", 25, 3.5, "linear", False, False),
        ("krea-dev", "flux1", 25, 3.5, "linear", False, False),
        ("z-image", "zimage", 50, None, None, True, True),
        ("z-image-turbo", "zimage", 9, None, None, False, False),
    ],
)
def test_the_entries_carry_the_mflux_defaults(
    name, family, steps, guidance, scheduler, two_calls, negative
):
    # Bug caught: a default copied from the wrong model (Z-Image on FLUX's 25 steps, FLUX losing 3.5).
    e = families.entry(name)
    assert (e.family, e.default_steps, e.default_guidance, e.default_scheduler) == (
        family,
        steps,
        guidance,
        scheduler,
    )
    assert (e.cfg_two_calls, e.uses_negative_prompt) == (two_calls, negative)


def test_labels_and_repos_are_the_published_names():
    # Bug caught: a table naming "z-image-turbo" instead of the model's name, or a swapped repository.
    assert families.entry("schnell").label == "FLUX.1-schnell"
    assert families.entry("z-image").label == "Z-Image"
    turbo = families.entry("z-image-turbo")
    assert turbo.label == "Z-Image-Turbo"
    assert (turbo.df11_repo, turbo.base_repo) == (
        "mingyi456/Z-Image-Turbo-DF11",
        "Tongyi-MAI/Z-Image-Turbo",
    )


def test_the_zimage_default_checkpoints_are_pinned_and_the_flux_ones_are_not():
    # Bug caught: the Z-Image defaults (one person's Hub account) following whatever lands on main, or a pin
    # copied onto the wrong model.
    assert families.entry("z-image").df11_revision == "36d582560630bfe2f5b78183c4164c66d6919e5b"
    assert (
        families.entry("z-image-turbo").df11_revision == "422db601eac214d0f1e2cc99e2cf34906e8a7267"
    )
    assert [families.entry(n).df11_revision for n in ("schnell", "dev", "krea-dev")] == [None] * 3


def test_family_of_returns_the_spec_of_the_models_family():
    # Bug caught: family_of keyed on the model name in FAMILIES (KeyError for "schnell").
    assert families.family_of("schnell") is families.FAMILIES["flux1"]
    assert families.family_of("z-image") is families.FAMILIES["zimage"]


@pytest.mark.mflux
def test_the_flux_entries_match_the_flux_model_table():
    # Bug caught: the registry and DFloatFlux1's own MODELS drifting (the CLI resolving one repo, the class another).
    from mlx_dfloat.mflux.flux1.model import MODELS as FLUX

    assert {
        n: (e.df11_repo, e.base_repo) for n, e in families.MODELS.items() if e.family == "flux1"
    } == FLUX


@pytest.mark.mflux
def test_the_zimage_model_table_is_derived_from_the_registry():
    # Bug caught: Z-Image's own MODELS literal drifting from the registry the CLI resolves through.
    from mlx_dfloat.mflux.zimage.model import MODELS as ZIMAGE

    assert ZIMAGE == {
        "z-image": ("mingyi456/Z-Image-DF11", "Tongyi-MAI/Z-Image"),
        "z-image-turbo": ("mingyi456/Z-Image-Turbo-DF11", "Tongyi-MAI/Z-Image-Turbo"),
    }


@pytest.mark.mflux
def test_the_family_loaders_return_the_real_classes_and_maps():
    # Bug caught: a loader lambda pointing at the other family's class or map.
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    assert families.FAMILIES["flux1"].load_model_class() is DFloatFlux1
    assert families.FAMILIES["zimage"].load_model_class() is DFloatZImage
    assert families.FAMILIES["zimage"].load_name_map().kinds == (
        "noise_refiner",
        "context_refiner",
        "layers",
    )
