"""The ``watchdog`` keyword on the six model classes: stored as ``model.watchdog`` by ``_assemble``.

Every test is `@pytest.mark.mflux`; the builders import mflux only when called, so the module collects without it.
"""

import pytest

from mlx_dfloat.mflux import WatchdogConfig

pytestmark = pytest.mark.mflux


def _flux1(tmp_path, monkeypatch, **overrides):
    from tests.test_flux1_model import _fake_model

    return _fake_model(tmp_path, monkeypatch, **overrides)


def _zimage(tmp_path, monkeypatch, **overrides):
    from tests._zimage_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, **overrides)


def _flux2(tmp_path, monkeypatch, **overrides):
    from tests._flux2_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, **overrides)


def _qwen21(tmp_path, monkeypatch, **overrides):
    from tests._qwen21_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, **overrides)


def _ernie(tmp_path, monkeypatch, **overrides):
    from tests._ernie_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, **overrides)


def _krea2(tmp_path, monkeypatch, **overrides):
    from tests._krea2_tiny import fake_model

    return fake_model(tmp_path, monkeypatch, **overrides)


BUILDERS = [
    pytest.param(_flux1, id="flux1"),
    pytest.param(_zimage, id="zimage"),
    pytest.param(_flux2, id="flux2"),
    pytest.param(_qwen21, id="qwen21"),
    pytest.param(_ernie, id="ernie"),
    pytest.param(_krea2, id="krea2"),
]


@pytest.mark.parametrize("build", BUILDERS)
def test_the_watchdog_keyword_defaults_to_the_default_config(build, tmp_path, monkeypatch):
    # Bug caught: an _assemble that never sets model.watchdog, or a default of None that leaves a Python caller
    # unwatched (the spec's default is WatchdogConfig()).
    model = build(tmp_path, monkeypatch)
    assert model.watchdog == WatchdogConfig()


@pytest.mark.parametrize("build", BUILDERS)
def test_the_watchdog_keyword_none_is_stored_as_none(build, tmp_path, monkeypatch):
    # Bug caught: an _assemble that drops the keyword (TypeError) or replaces None with the default config, so the
    # command's model would arm a second watchdog under the command's own.
    model = build(tmp_path, monkeypatch, watchdog=None)
    assert model.watchdog is None


@pytest.mark.parametrize("build", BUILDERS)
def test_the_watchdog_keyword_keeps_the_callers_config(build, tmp_path, monkeypatch):
    # Bug caught: an _assemble that copies or rebuilds the config (losing a tier's explicit ceiling) instead of keeping
    # the caller's object. stop_margin=1: the default 1 GiB margin is refused below a 123-byte ceiling.
    config = WatchdogConfig(ceiling=123, stop_margin=1)
    model = build(tmp_path, monkeypatch, watchdog=config)
    assert model.watchdog is config


# --- the constructor forwards the keyword to _assemble -------------------------------------------------------------
#
# The builders above enter through ``_from_parts``, so they cannot see a ``__init__`` that takes ``watchdog`` and then
# drops it: ``_assemble`` would quietly fall back to its own default. Each case below calls the real ``__init__`` with
# every step between the decoder self-check and ``_assemble`` replaced (``monkeypatch.setattr`` refuses a name that
# does not exist, so a renamed step fails here instead of resolving or loading anything), and records what reaches
# ``_assemble``.

# What every replaced step returns; nothing reads it, because _assemble is replaced too.
_STUB = object()


def _stub(*args, **kwargs):
    return _STUB


# family: (model module, class name, model name, init-module alias, memory-module alias or None, model-module steps)
INIT_STEPS = {
    "flux1": (
        "mlx_dfloat.mflux.flux1.model",
        "DFloatFlux1",
        "schnell",
        "base_init",
        None,
        {
            "open_checkpoint": _stub,
            "check_flux_groups": _stub,
            "build_transformer": lambda *a, **k: (_STUB, _STUB),  # (transformer, shapes)
            "sizes_for": _stub,
        },
    ),
    "zimage": (
        "mlx_dfloat.mflux.zimage.model",
        "DFloatZImage",
        "z-image-turbo",
        "zinit",
        None,
        {
            "open_checkpoint": _stub,
            "check_zimage_groups": _stub,
            "build_transformer": _stub,
            "sizes_for": _stub,
        },
    ),
    "flux2": (
        "mlx_dfloat.mflux.flux2.model",
        "DFloatFlux2Klein",
        "flux2-klein-4b",
        "finit",
        "kmem",
        {"open_checkpoint": _stub, "build_transformer": _stub},
    ),
    "qwen21": (
        "mlx_dfloat.mflux.qwen21.model",
        "DFloatQwenImage21",
        "qwen-image-2.1",
        "qinit",
        "qmem",
        {"open_checkpoint": _stub, "build_transformer": _stub},
    ),
    "ernie": (
        "mlx_dfloat.mflux.ernie.model",
        "DFloatErnieImage",
        "ernie-image-turbo",
        "einit",
        "emem",
        {"open_checkpoint": _stub, "build_transformer": _stub},
    ),
    "krea2": (
        "mlx_dfloat.mflux.krea2.model",
        "DFloatKrea2",
        "krea-2",
        "kinit",
        "kmem",
        {"open_checkpoint": _stub, "check_variant": _stub, "build_transformer": _stub},
    ),
}


@pytest.mark.parametrize("family", list(INIT_STEPS))
def test_init_forwards_the_watchdog_keyword(family, tmp_path, monkeypatch):
    # Bug caught: a class whose __init__ accepts `watchdog=` but leaves `watchdog=watchdog` out of its `_assemble`
    # call, so a caller's config (a tier's lower ceiling, or None under the command's own watchdog) is silently
    # replaced by the default config.
    import importlib

    from mlx_dfloat import _metal_decode
    from mlx_dfloat.mflux._hub import ResolvedRepo

    module_name, class_name, model_name, init_alias, memory_alias, steps = INIT_STEPS[family]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    init = getattr(module, init_alias)
    monkeypatch.setattr(_metal_decode, "ensure_canary", lambda **kwargs: None)
    monkeypatch.setattr(
        init,
        "resolve",
        lambda spec, *, patterns, revision=None: ResolvedRepo(
            root=tmp_path, repo_id=spec, revision=revision
        ),
    )
    monkeypatch.setattr(init, "load_base", _stub)
    if memory_alias is not None:
        monkeypatch.setattr(getattr(module, memory_alias), "sizes_for", _stub)
    for name, replacement in steps.items():
        monkeypatch.setattr(module, name, replacement)
    received = []
    monkeypatch.setattr(cls, "_assemble", lambda self, **parts: received.append(parts))

    config = WatchdogConfig(ceiling=123, stop_margin=1)
    cls(model_name, watchdog=config)

    assert len(received) == 1
    assert received[0].get("watchdog", "<not passed>") is config
