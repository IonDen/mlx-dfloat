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
    assert out.stdout.strip() == "9"


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
        # mflux 0.20.0 cli/defaults/defaults.py:41-45 (steps); flux2_generate.py:63-64 (guidance 1.0) and
        # flux2_klein.py:60 (scheduler); one call per step at guidance 1.0, two above it on the base models.
        ("flux2-klein-base-4b", "flux2", 50, 1.0, "flow_match_euler_discrete", True, False),
        ("flux2-klein-4b", "flux2", 4, 1.0, "flow_match_euler_discrete", False, False),
        ("flux2-klein-base-9b", "flux2", 50, 1.0, "flow_match_euler_discrete", True, False),
        ("flux2-klein-9b", "flux2", 4, 1.0, "flow_match_euler_discrete", False, False),
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


@pytest.mark.parametrize(
    ("name", "label", "df11", "revision", "base"),
    [
        (
            "flux2-klein-base-4b",
            "FLUX.2-klein-base-4B",
            "mingyi456/FLUX.2-klein-base-4B-DF11",
            "b887c73c5cbc3f4d50887a04c41509fb25a0a0f0",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        (
            "flux2-klein-4b",
            "FLUX.2-klein-4B",
            "mingyi456/FLUX.2-klein-4B-DF11",
            "d29a2c249ff0afeb678da101e707bd134596c96f",
            "black-forest-labs/FLUX.2-klein-4B",
        ),
        (
            "flux2-klein-base-9b",
            "FLUX.2-klein-base-9B",
            "mingyi456/FLUX.2-klein-base-9B-DF11",
            "50dc8e7cba7a41eeaf9e9c4fcec557d1a6888ded",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        (
            "flux2-klein-9b",
            "FLUX.2-klein-9B",
            "mingyi456/FLUX.2-klein-9B-DF11",
            "45a202a0bd19ec01ce8db2d5236890586cbac6a6",
            "black-forest-labs/FLUX.2-klein-9B",
        ),
    ],
)
def test_the_klein_entries_name_their_published_repos_and_pinned_checkpoints(
    name, label, df11, revision, base
):
    # Bug caught: a distilled model resolving the base DF11 (or the 4B one the 9B), a pin copied onto the wrong
    # model (the checkpoint verified here swapped for whatever lands on main), or a label that is not BFL's name.
    e = families.entry(name)
    assert (e.label, e.df11_repo, e.df11_revision, e.base_repo) == (label, df11, revision, base)


def test_only_the_distilled_klein_models_fix_their_guidance():
    # Bug caught: the fixed guidance on a base model (its --guidance 4 refused) or missing on a distilled one, or set
    # on a FLUX.1 / Z-Image entry (their --guidance refused). mflux flux2_generate.py:74: distilled = no "base".
    assert {n: e.fixed_guidance for n, e in families.MODELS.items()} == {
        "schnell": None,
        "dev": None,
        "krea-dev": None,
        "z-image": None,
        "z-image-turbo": None,
        "flux2-klein-base-4b": None,
        "flux2-klein-4b": 1.0,
        "flux2-klein-base-9b": None,
        "flux2-klein-9b": 1.0,
    }


def test_family_of_returns_the_spec_of_the_models_family():
    # Bug caught: family_of keyed on the model name in FAMILIES (KeyError for "schnell").
    assert families.family_of("schnell") is families.FAMILIES["flux1"]
    assert families.family_of("z-image") is families.FAMILIES["zimage"]
    assert families.family_of("flux2-klein-9b") is families.FAMILIES["flux2"]


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
def test_the_flux2_model_table_is_derived_from_the_registry():
    # Bug caught: the Klein class's own MODELS drifting from the registry the CLI resolves through.
    from mlx_dfloat.mflux.flux2.model import MODELS as KLEIN

    assert KLEIN == {
        "flux2-klein-base-4b": (
            "mingyi456/FLUX.2-klein-base-4B-DF11",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        "flux2-klein-4b": ("mingyi456/FLUX.2-klein-4B-DF11", "black-forest-labs/FLUX.2-klein-4B"),
        "flux2-klein-base-9b": (
            "mingyi456/FLUX.2-klein-base-9B-DF11",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        "flux2-klein-9b": ("mingyi456/FLUX.2-klein-9B-DF11", "black-forest-labs/FLUX.2-klein-9B"),
    }


@pytest.mark.mflux
def test_the_family_loaders_return_the_real_classes_and_maps():
    # Bug caught: a loader lambda pointing at the other family's class or map.
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    assert families.FAMILIES["flux1"].load_model_class() is DFloatFlux1
    assert families.FAMILIES["zimage"].load_model_class() is DFloatZImage
    assert families.FAMILIES["zimage"].load_name_map().kinds == (
        "noise_refiner",
        "context_refiner",
        "layers",
    )
    assert families.FAMILIES["flux2"].load_model_class() is DFloatFlux2Klein
    # mflux 0.20.0 Flux2Transformer.__call__ (transformer.py:132-163): double blocks, then single blocks.
    assert families.FAMILIES["flux2"].load_name_map().kinds == (
        "transformer_blocks",
        "single_transformer_blocks",
    )
