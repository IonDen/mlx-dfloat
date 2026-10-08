"""The model registry: which model names the DFloat11 path runs, and what each family needs.

Importing this module never imports mflux; a family's adapter is imported only when its class or name map is
asked for.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from mlx_dfloat.errors import DFloatUnsupportedError
from mlx_dfloat.integrate.names import NameMap


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelEntry:
    """One model name the DFloat11 path runs."""

    name: str
    family: str
    label: str  # "FLUX.1-schnell", "Z-Image", "FLUX.2-klein-4B" (tables, README)
    df11_repo: str
    df11_revision: (
        str | None
    )  # the commit the default checkpoint is pinned to (None: the default branch)
    base_repo: str
    default_steps: int  # mflux 0.20 cli/defaults/defaults.py MODEL_INFERENCE_STEPS
    default_guidance: float | None  # None: the model's own rule
    default_scheduler: str | None  # None: the model's own rule
    cfg_two_calls: bool  # the step may call the transformer twice
    uses_negative_prompt: bool
    # A distilled model runs at this guidance only (the CLI refuses any other); None: any guidance.
    fixed_guidance: float | None = None
    # Classifier-free guidance runs only with a negative prompt as well as a guidance above 1.0 (mflux's rule for
    # Qwen-Image 2.1); the CLI warns when --guidance asks for it without --negative-prompt.
    cfg_needs_negative: bool = False
    # The commit the default base repository is pinned to (None: the default branch); a user's own base is taken as
    # given.
    base_revision: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FamilySpec:
    """What the generic layer needs to know about one mflux model family."""

    name: str
    refused_flags: Mapping[str, str]  # CLI flags refused for this family, beyond the common ones
    load_model_class: Callable[[], type]
    load_name_map: Callable[[], NameMap]


def _flux1_class() -> type:
    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    return DFloatFlux1


def _flux1_names() -> NameMap:
    from mlx_dfloat.mflux.flux1.names import flux_name_map

    return flux_name_map()


def _zimage_class() -> type:
    from mlx_dfloat.mflux.zimage.model import DFloatZImage

    return DFloatZImage


def _zimage_names() -> NameMap:
    from mlx_dfloat.mflux.zimage.names import zimage_name_map

    return zimage_name_map()


def _flux2_class() -> type:
    from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

    return DFloatFlux2Klein


def _flux2_names() -> NameMap:
    from mlx_dfloat.mflux.flux2.names import klein_name_map

    return klein_name_map()


def _qwen21_class() -> type:
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    return DFloatQwenImage21


def _qwen21_names() -> NameMap:
    from mlx_dfloat.mflux.qwen21.names import qwen21_name_map

    return qwen21_name_map()


def _ernie_class() -> type:
    from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

    return DFloatErnieImage


def _ernie_names() -> NameMap:
    from mlx_dfloat.mflux.ernie.names import ernie_name_map

    return ernie_name_map()


FAMILIES: dict[str, FamilySpec] = {
    "flux1": FamilySpec(
        name="flux1",
        refused_flags={},
        load_model_class=_flux1_class,
        load_name_map=_flux1_names,
    ),
    "zimage": FamilySpec(
        name="zimage",
        refused_flags={},
        load_model_class=_zimage_class,
        load_name_map=_zimage_names,
    ),
    "flux2": FamilySpec(
        name="flux2",
        # mflux 0.20.0's FLUX.2 Klein command has no scheduler flag (models/flux2/cli/flux2_generate.py).
        refused_flags={
            "--scheduler": "mflux's FLUX.2 Klein command runs flow_match_euler_discrete only"
        },
        load_model_class=_flux2_class,
        load_name_map=_flux2_names,
    ),
    "qwen21": FamilySpec(
        name="qwen21",
        refused_flags={},
        load_model_class=_qwen21_class,
        load_name_map=_qwen21_names,
    ),
    "ernie": FamilySpec(
        name="ernie",
        refused_flags={},
        load_model_class=_ernie_class,
        load_name_map=_ernie_names,
    ),
}


def _flux(name: str, label: str, steps: int, df11: str, base: str) -> ModelEntry:
    return ModelEntry(
        name=name,
        family="flux1",
        label=label,
        df11_repo=df11,
        df11_revision=None,
        base_repo=base,
        default_steps=steps,
        default_guidance=3.5,
        default_scheduler="linear",
        cfg_two_calls=False,
        uses_negative_prompt=False,
    )


def _klein(name: str, label: str, steps: int, revision: str, *, base: bool) -> ModelEntry:
    """A FLUX.2 Klein entry, with mflux 0.20.0's defaults for it.

    The default steps are 50 for a base model and 4 for a distilled one; guidance defaults to 1.0, and the scheduler
    is mflux's fixed one. A base model calls the transformer twice per step above guidance 1.0 (the negative is
    mflux's own blank prompt, so there is no negative prompt to pass); a distilled model runs at 1.0 only, as mflux's
    command enforces.
    """
    # Sources in mflux 0.20.0: the steps in cli/defaults/defaults.py; guidance 1.0 and the scheduler in
    # models/flux2/cli/flux2_generate.py and Flux2Klein.generate_image.
    return ModelEntry(
        name=name,
        family="flux2",
        label=label,
        df11_repo=f"mingyi456/{label}-DF11",
        df11_revision=revision,
        base_repo=f"black-forest-labs/{label}",
        default_steps=steps,
        default_guidance=1.0,
        default_scheduler="flow_match_euler_discrete",
        cfg_two_calls=base,
        uses_negative_prompt=False,
        fixed_guidance=None if base else 1.0,
    )


MODELS: dict[str, ModelEntry] = {
    e.name: e
    for e in (
        _flux(
            "schnell",
            "FLUX.1-schnell",
            4,
            "DFloat11/FLUX.1-schnell-DF11",
            "black-forest-labs/FLUX.1-schnell",
        ),
        _flux("dev", "FLUX.1-dev", 25, "DFloat11/FLUX.1-dev-DF11", "black-forest-labs/FLUX.1-dev"),
        _flux(
            "krea-dev",
            "FLUX.1-Krea-dev",
            25,
            "DFloat11/FLUX.1-Krea-dev-DF11",
            "black-forest-labs/FLUX.1-Krea-dev",
        ),
        ModelEntry(
            name="z-image",
            family="zimage",
            label="Z-Image",
            df11_repo="mingyi456/Z-Image-DF11",
            df11_revision="36d582560630bfe2f5b78183c4164c66d6919e5b",
            base_repo="Tongyi-MAI/Z-Image",
            default_steps=50,
            default_guidance=None,
            default_scheduler=None,
            cfg_two_calls=True,
            uses_negative_prompt=True,
        ),
        ModelEntry(
            name="z-image-turbo",
            family="zimage",
            label="Z-Image-Turbo",
            df11_repo="mingyi456/Z-Image-Turbo-DF11",
            df11_revision="422db601eac214d0f1e2cc99e2cf34906e8a7267",
            base_repo="Tongyi-MAI/Z-Image-Turbo",
            default_steps=9,
            default_guidance=None,
            default_scheduler=None,
            cfg_two_calls=False,
            uses_negative_prompt=False,
        ),
        _klein(
            "flux2-klein-base-4b",
            "FLUX.2-klein-base-4B",
            50,
            "b887c73c5cbc3f4d50887a04c41509fb25a0a0f0",
            base=True,
        ),
        _klein(
            "flux2-klein-4b",
            "FLUX.2-klein-4B",
            4,
            "d29a2c249ff0afeb678da101e707bd134596c96f",
            base=False,
        ),
        _klein(
            "flux2-klein-base-9b",
            "FLUX.2-klein-base-9B",
            50,
            "50dc8e7cba7a41eeaf9e9c4fcec557d1a6888ded",
            base=True,
        ),
        _klein(
            "flux2-klein-9b",
            "FLUX.2-klein-9B",
            4,
            "45a202a0bd19ec01ce8db2d5236890586cbac6a6",
            base=False,
        ),
        # mflux 0.20.0: 40 steps (cli/defaults/defaults.py:52), guidance 1.0 (models/qwen21/cli/qwen21_generate.py:60),
        # scheduler "linear" (cli/parser/parsers.py:180). A second transformer call per step runs only above guidance
        # 1.0 with a non-empty negative prompt (models/qwen21/variants/txt2img/qwen_image_21.py:89).
        ModelEntry(
            name="qwen-image-2.1",
            family="qwen21",
            label="Qwen-Image-2.1",
            df11_repo="mingyi456/Qwen-Image-2.1-DF11-ComfyUI",
            df11_revision="1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc",
            base_repo="Qwen/Qwen-Image-2.1",
            default_steps=40,
            default_guidance=1.0,
            default_scheduler="linear",
            cfg_two_calls=True,
            uses_negative_prompt=True,
            cfg_needs_negative=True,
            # The snapshot the identity check and the MEASURED row ran on.
            base_revision="d26bb61231c349cf6b7896fa83353113880e1ba3",
        ),
        # mflux 0.20.0: 8 steps (cli/defaults/defaults.py:36); guidance 1.0 only, any other is an error
        # (models/ernie_image/cli/ernie_image_turbo_generate.py:42-45), and --negative-prompt is ignored (:10-12, 40);
        # scheduler "linear" (ernie_image.py:64-65).
        ModelEntry(
            name="ernie-image-turbo",
            family="ernie",
            label="ERNIE-Image-Turbo",
            df11_repo="mingyi456/ERNIE-Image-Turbo-DF11",
            df11_revision="27f84b44a3b78fcfaaadbaeeeae7cc7f7d75153b",
            base_repo="baidu/ERNIE-Image-Turbo",
            default_steps=8,
            default_guidance=1.0,
            default_scheduler="linear",
            cfg_two_calls=False,
            uses_negative_prompt=False,
            fixed_guidance=1.0,
            # The snapshot the parity, de-risk and MEASURED runs used (text encoder, VAE, tokenizer).
            base_revision="bc68c81e2a1730a394d5fc9fae70713dee940140",
        ),
        # mflux 0.20.0: 50 steps (cli/defaults/defaults.py:35); guidance 4.0 (models/ernie_image/cli/
        # ernie_image_generate.py:22, 30-31); scheduler "linear" (:33-34). Classifier-free guidance runs above 1.0
        # with or without a negative prompt, as one batch-2 transformer call (ernie_image.py:164-168, 239-250).
        ModelEntry(
            name="ernie-image",
            family="ernie",
            label="ERNIE-Image",
            df11_repo="mingyi456/ERNIE-Image-DF11",
            df11_revision="c2dd30ad7dd5a928df2309581b282337f7cdb41f",
            base_repo="baidu/ERNIE-Image",
            default_steps=50,
            default_guidance=4.0,
            default_scheduler="linear",
            cfg_two_calls=False,  # CFG runs as one batch-2 call (ernie_image.py:239-250)
            uses_negative_prompt=True,
            # The snapshot the parity, identity, de-risk and MEASURED runs used.
            base_revision="5346b31d68c9c23758ba56ef8be5e9dc174c7f99",
        ),
    )
}


def entry(name: str) -> ModelEntry:
    """The registry entry of a model name.

    Raises:
        DFloatUnsupportedError: The name is not registered (the message lists the known names).
    """
    try:
        return MODELS[name]
    except KeyError:
        raise DFloatUnsupportedError(
            f"unknown model {name!r}; choose from {', '.join(MODELS)}"
        ) from None


def family_of(name: str) -> FamilySpec:
    """The family spec of a model name (same error as ``entry``)."""
    return FAMILIES[entry(name).family]


def DFloatModel(name: str, /, **kwargs: Any) -> Any:  # noqa: N802
    """A ready DFloat11 model for a registered name: the family's model class built with ``name`` and ``kwargs``.

    Imports mflux. ``DFloatModel("z-image-turbo")`` returns a ``DFloatZImage``; ``DFloatModel("schnell")`` a
    ``DFloatFlux1``; ``DFloatModel("flux2-klein-4b")`` a ``DFloatFlux2Klein``; ``DFloatModel("qwen-image-2.1")`` a
    ``DFloatQwenImage21``; ``DFloatModel("ernie-image")`` a ``DFloatErnieImage``.

    Args:
        name: A registered model name, such as ``"schnell"`` or ``"z-image-turbo"``.
        **kwargs: Passed to the family's model class.

    Raises:
        DFloatUnsupportedError: The name is not registered.
        DFloatDependencyError: The optional ``mflux`` extra is not installed.
    """
    return family_of(name).load_model_class()(name, **kwargs)
