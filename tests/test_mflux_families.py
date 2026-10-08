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
    assert out.stdout.strip() == "14"


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


def test_resolving_the_ernie_coverage_specs_loads_no_mlx():
    # Bug caught: an eager import in mlx_dfloat/mflux/ernie/__init__.py. The mflux lane's
    # --cov=mlx_dfloat.mflux.ernie.model / .init / .transformer resolve those specs (importing the ernie package) when
    # coverage starts; MLX loaded there registers its native types twice and the lane aborts (exit 134).
    code = (
        "import importlib.util, sys; "
        "[importlib.util.find_spec(f'mlx_dfloat.mflux.ernie.{m}') for m in ('model', 'init', 'transformer')]; "
        "import mlx_dfloat.mflux; print('mlx.core' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_resolving_the_qwen21_coverage_specs_loads_no_mlx():
    # Bug caught: an eager import in mlx_dfloat/mflux/qwen21/__init__.py. The mflux lane's
    # --cov=mlx_dfloat.mflux.qwen21.model / .init resolve those specs (importing the qwen21 package) when coverage
    # starts; MLX loaded there registers its native types twice and the lane aborts (exit 134), as it did for FLUX.1.
    code = (
        "import importlib.util, sys; importlib.util.find_spec('mlx_dfloat.mflux.qwen21.model'); "
        "importlib.util.find_spec('mlx_dfloat.mflux.qwen21.init'); print('mlx.core' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_resolving_the_krea2_coverage_specs_loads_no_mlx():
    # Bug caught: an eager import in mlx_dfloat/mflux/krea2/__init__.py. The mflux lane's
    # --cov=mlx_dfloat.mflux.krea2.model / .init / .transformer resolve those specs (importing the krea2 package) when
    # coverage starts; MLX loaded there registers its native types twice and the lane aborts (exit 134).
    code = (
        "import importlib.util, sys; "
        "[importlib.util.find_spec(f'mlx_dfloat.mflux.krea2.{m}') for m in ('model', 'init', 'transformer')]; "
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
        # mflux 0.20.0 cli/defaults/defaults.py:52 (40 steps); qwen21_generate.py:60 (guidance 1.0);
        # cli/parser/parsers.py:180 (scheduler "linear"); qwen_image_21.py:89 (a second call per step above 1.0 with
        # a negative prompt, which mflux takes from --negative-prompt, parsers.py:177).
        ("qwen-image-2.1", "qwen21", 40, 1.0, "linear", True, True),
        # mflux 0.20.0 cli/defaults/defaults.py:35-36 (50 and 8 steps); ernie_image_generate.py:22 (guidance 4.0) and
        # ernie_image_turbo_generate.py:42-45 (1.0 only); "linear" (ernie_image_generate.py:33-34,
        # ernie_image.py:64-65). CFG is one batch-2 call (ernie_image.py:239-250), so never a second call; Turbo's
        # command ignores --negative-prompt (ernie_image_turbo_generate.py:10-12).
        ("ernie-image", "ernie", 50, 4.0, "linear", False, True),
        ("ernie-image-turbo", "ernie", 8, 1.0, "linear", False, False),
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
        "qwen-image-2.1": None,
        # mflux 0.20.0 ernie_image_turbo_generate.py:42-45: any guidance but 1.0 is an error for Turbo.
        "ernie-image-turbo": 1.0,
        "ernie-image": None,
        # mflux 0.20.0: Krea 2 Turbo and Raw take any guidance (krea2.py:58; CFG for any value other than 1.0).
        "krea-2": None,
        "krea-2-raw": None,
    }


def test_only_qwen_image_needs_a_negative_prompt_for_cfg():
    # Bug caught: the "no CFG without --negative-prompt" warning shown for Z-Image (whose base model does run CFG with
    # an empty negative) or missing for Qwen-Image 2.1 (mflux runs CFG only with both, qwen_image_21.py:89).
    assert {n for n, e in families.MODELS.items() if e.cfg_needs_negative} == {"qwen-image-2.1"}


def test_the_qwen_entry_names_its_published_repos_and_pinned_checkpoint():
    # Bug caught: the default checkpoint (one person's Hub account, a ComfyUI single file) following whatever lands on
    # main instead of the revision whose header and bytes were verified, or a swapped repository.
    e = families.entry("qwen-image-2.1")
    assert (e.label, e.df11_repo, e.df11_revision, e.base_repo) == (
        "Qwen-Image-2.1",
        "mingyi456/Qwen-Image-2.1-DF11-ComfyUI",
        "1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc",
        "Qwen/Qwen-Image-2.1",
    )


def test_pinned_bases_are_the_snapshots_the_recorded_runs_used_and_no_other_base_is():
    # Bug caught: a base (text encoder, VAE, tokenizer) following whatever lands on main instead of the snapshot its
    # recorded runs used (Qwen-Image 2.1: d26bb61, bench/results/tiers/qwen-image-2.1-1024.json; ERNIE-Image and
    # ERNIE-Image-Turbo: the snapshots their parity and de-risk runs read, 2026-10-08), or a pin set on a family whose
    # base no recorded run names.
    assert {
        n: e.base_revision for n, e in families.MODELS.items() if e.base_revision is not None
    } == {
        "qwen-image-2.1": "d26bb61231c349cf6b7896fa83353113880e1ba3",
        "ernie-image": "5346b31d68c9c23758ba56ef8be5e9dc174c7f99",
        "ernie-image-turbo": "bc68c81e2a1730a394d5fc9fae70713dee940140",
        # Krea 2: the gated base snapshots the range-read parity reads and every Krea run uses (2026-10-08).
        "krea-2": "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
        "krea-2-raw": "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603",
    }


def test_family_of_returns_the_spec_of_the_models_family():
    # Bug caught: family_of keyed on the model name in FAMILIES (KeyError for "schnell").
    assert families.family_of("schnell") is families.FAMILIES["flux1"]
    assert families.family_of("z-image") is families.FAMILIES["zimage"]
    assert families.family_of("flux2-klein-9b") is families.FAMILIES["flux2"]
    assert families.family_of("qwen-image-2.1") is families.FAMILIES["qwen21"]


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
def test_the_qwen21_model_table_is_derived_from_the_registry():
    # Bug caught: the Qwen class's own MODELS drifting from the registry the CLI resolves through.
    from mlx_dfloat.mflux.qwen21.model import MODELS as QWEN

    assert QWEN == {
        "qwen-image-2.1": ("mingyi456/Qwen-Image-2.1-DF11-ComfyUI", "Qwen/Qwen-Image-2.1")
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
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    assert families.FAMILIES["qwen21"].load_model_class() is DFloatQwenImage21
    # mflux 0.20.0 Qwen21Transformer._forward (qwen21_transformer.py:96-100): one block list.
    assert families.FAMILIES["qwen21"].load_name_map().kinds == ("transformer_blocks",)


def test_the_ernie_entries_name_their_published_repos_and_pinned_checkpoints():
    # Bug caught: a swapped repository or a pin copied onto the wrong model (the checkpoint verified here swapped for
    # whatever lands on main).
    got = {
        n: (e.label, e.df11_repo, e.df11_revision, e.base_repo)
        for n, e in families.MODELS.items()
        if e.family == "ernie"
    }
    assert got == {
        "ernie-image": (
            "ERNIE-Image",
            "mingyi456/ERNIE-Image-DF11",
            "c2dd30ad7dd5a928df2309581b282337f7cdb41f",
            "baidu/ERNIE-Image",
        ),
        "ernie-image-turbo": (
            "ERNIE-Image-Turbo",
            "mingyi456/ERNIE-Image-Turbo-DF11",
            "27f84b44a3b78fcfaaadbaeeeae7cc7f7d75153b",
            "baidu/ERNIE-Image-Turbo",
        ),
    }


@pytest.mark.mflux
def test_the_ernie_family_loads_its_class_and_one_kind_map():
    # Bug caught: the ernie loader pointing at another family's class or map, or the class's own MODELS drifting
    # from the registry.
    from mlx_dfloat.mflux.ernie.model import MODELS as ERNIE
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    assert families.FAMILIES["ernie"].load_model_class() is DFloatErnieImage
    assert families.FAMILIES["ernie"].load_name_map().kinds == ("layers",)
    assert families.family_of("ernie-image") is families.FAMILIES["ernie"]
    assert ERNIE == {
        "ernie-image": ("mingyi456/ERNIE-Image-DF11", "baidu/ERNIE-Image"),
        "ernie-image-turbo": ("mingyi456/ERNIE-Image-Turbo-DF11", "baidu/ERNIE-Image-Turbo"),
    }


# --- Krea 2 ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "label", "steps", "df11", "revision", "base", "base_revision"),
    [
        # mflux 0.20.0: 8 steps (cli/defaults/defaults.py:47), guidance 1.0 (krea2_generate.py:17; krea2.py:58).
        (
            "krea-2",
            "Krea 2 Turbo",
            8,
            "mingyi456/Krea-2-Turbo-DF11-ComfyUI",
            "978da5fb7647bd222d33125993abd8fdc2840cfc",
            "krea/Krea-2-Turbo",
            "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
        ),
        # mflux's Krea command refuses krea-2-raw (krea2_generate.py:46-49): the step table's fallback 25
        # (defaults.py:19,111-118) and generate_image's guidance 1.0 (krea2.py:58) are its only defaults.
        (
            "krea-2-raw",
            "Krea 2 Raw",
            25,
            "mingyi456/Krea-2-Raw-DF11-ComfyUI",
            "8320616b25ac9340a830a7fb21f1b0237e160e66",
            "krea/Krea-2-Raw",
            "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603",
        ),
    ],
)
def test_the_krea_entries_carry_mfluxs_defaults_and_their_pins(
    name, label, steps, df11, revision, base, base_revision
):
    # Bug caught: Raw carrying the model card's 52 / 3.5 (a CFG run nobody asked for, about 4x the time) or Turbo's 8
    # steps, a pin or repository swapped between the two models, or the CFG rule copied as `> 1.0`.
    e = families.entry(name)
    assert (e.family, e.label, e.df11_repo, e.df11_revision, e.base_repo, e.base_revision) == (
        "krea2",
        label,
        df11,
        revision,
        base,
        base_revision,
    )
    assert (e.default_steps, e.default_guidance, e.default_scheduler) == (steps, 1.0, "er_sde")
    assert (e.cfg_two_calls, e.uses_negative_prompt, e.cfg_needs_negative) == (True, True, False)
    assert (e.cfg_unless_guidance_one, e.schedulers) == (True, ("er_sde", "euler", "linear"))
    assert e.fixed_guidance is None


def test_new_fields_default_off_for_every_other_family():
    # Bug caught: another family's warnings or scheduler handling changed by the two Krea fields.
    others = [e for e in families.MODELS.values() if e.family != "krea2"]
    assert len(others) == 12
    assert all(e.cfg_unless_guidance_one is False for e in others)
    assert all(e.schedulers is None for e in others)


@pytest.mark.mflux
def test_the_krea2_family_loads_its_class_and_one_block_kind():
    # Bug caught: the family's loaders pointing at another family's class or map.
    from mlx_dfloat.mflux.krea2.model import MODELS as KREA
    from mlx_dfloat.mflux.krea2.model import DFloatKrea2

    assert families.FAMILIES["krea2"].load_model_class() is DFloatKrea2
    # mflux 0.20.0 Krea2Transformer.__call__ (transformer.py:89-92): one block list.
    assert families.FAMILIES["krea2"].load_name_map().kinds == ("blocks",)
    assert KREA == {
        "krea-2": ("mingyi456/Krea-2-Turbo-DF11-ComfyUI", "krea/Krea-2-Turbo"),
        "krea-2-raw": ("mingyi456/Krea-2-Raw-DF11-ComfyUI", "krea/Krea-2-Raw"),
    }
