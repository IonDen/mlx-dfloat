"""The q8 builder's release order and refusals, and the pinned-snapshot resolver.

Every mflux piece (loader, applier, transformer factory, component) is injected, so no test here
imports mflux or loads a model; the real build is ``tests/test_bench_q8_slow.py``.
"""

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from scripts import _q8_rig as q8

from mlx_dfloat.errors import DFloatDependencyError, DFloatIntegrationError

BASE_REPO = "black-forest-labs/FLUX.1-schnell"
BASE_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
COMPONENT = SimpleNamespace(name="transformer")


class _Arrays:
    """Stands in for the BF16 arrays dict of one component (a plain dict cannot be weak-referenced)."""


class _Weights:
    """A ``LoadedWeights`` look-alike that sits in a reference cycle, so only ``gc.collect`` frees it."""

    def __init__(self, level):
        self.meta_data = SimpleNamespace(quantization_level=level)
        self.components = {COMPONENT.name: _Arrays()}
        self.cycle = self


def _rig(*, level=None, log=None):
    """Injected fakes that log the calls in order and keep only weak references to the weights."""
    log = [] if log is None else log
    refs: list[weakref.ref] = []

    def loader(component, root):
        log.append(f"load {component.name} {root}")
        weights = _Weights(level)
        refs.extend([weakref.ref(weights), weakref.ref(weights.components[component.name])])
        return weights

    def make_transformer(model, n_double, n_single):
        log.append(f"make {model} {n_double} {n_single}")
        return SimpleNamespace(applied=None)

    def applier(weights, transformer, component):
        log.append("apply")
        # Records a name, never a reference: what the real applier keeps is the leaf arrays.
        transformer.applied = type(weights.components[component.name]).__name__

    def evaluate(transformer):
        log.append(f"evaluate alive={[r() is not None for r in refs]}")

    kwargs = {
        "loader": loader,
        "applier": applier,
        "evaluate": evaluate,
        "make_transformer": make_transformer,
        "component": COMPONENT,
    }
    return log, kwargs


@pytest.fixture
def no_mflux(monkeypatch):
    """``require_mflux`` raises, as it does in a venv without the extra."""

    def refuse():
        raise DFloatDependencyError("no mflux")

    monkeypatch.setattr(q8, "require_mflux", refuse)


@pytest.fixture
def clear_log(monkeypatch):
    """Record ``mx.clear_cache`` calls instead of making them."""
    calls: list[str] = []
    monkeypatch.setattr(q8.mx, "clear_cache", lambda: calls.append("clear_cache"))
    return calls


def test_the_bf16_dict_is_released_before_the_first_eval(clear_log, no_mflux):
    # Bug caught: evaluating while the LoadedWeights dict is alive keeps every BF16 shard resident
    # next to the q8 copy, ~34 GiB on a 32 GB Mac. Also red when `gc.collect()` is dropped (the
    # fake weights sit in a cycle, as a real object graph may), when the eval runs before the
    # applier, or when the cache is cleared before the eval instead of after.
    log, kwargs = _rig(log=clear_log)  # one log: the fakes' calls and mx.clear_cache, in order
    transformer = q8.build_q8_transformer("schnell", Path("/snap"), **kwargs)
    assert log == [
        "load transformer /snap",
        "make schnell 19 38",
        "apply",
        "evaluate alive=[False, False]",
        "clear_cache",
    ]
    assert transformer.applied == "_Arrays"  # the applier saw the loaded component


def test_the_build_passes_the_reduced_depth_to_the_transformer(clear_log, no_mflux):
    # Bug caught: n_double / n_single dropped (a 1+1 validation build allocates all 57 blocks).
    log, kwargs = _rig()
    q8.build_q8_transformer("dev", Path("/snap"), n_double=1, n_single=2, **kwargs)
    assert "make dev 1 2" in log


def test_a_stored_quantization_is_refused_before_anything_is_built(clear_log, no_mflux):
    # Bug caught: a pre-quantized base accepted, so mflux keeps its stored bits (not 8) under the q8
    # name, or the refusal coming after the transformer was built and the weights applied.
    log, kwargs = _rig(level=4)
    with pytest.raises(DFloatIntegrationError, match="quantized"):
        q8.build_q8_transformer("schnell", Path("/snap"), **kwargs)
    assert log == ["load transformer /snap"]
    gc.collect()  # the refused weights' cycle, so it does not outlive the test


def test_an_unknown_model_is_refused_before_loading(clear_log, no_mflux):
    # Bug caught: a model name outside the step bench's two passed through to the loader (a
    # krea-dev base would be quantized under a schnell/dev config).
    log, kwargs = _rig()
    with pytest.raises(DFloatIntegrationError, match="krea-dev"):
        q8.build_q8_transformer("krea-dev", Path("/snap"), **kwargs)
    assert log == []


def test_with_every_piece_injected_mflux_is_never_required(clear_log, no_mflux):
    # Bug caught: `require_mflux()` (or an mflux import) at the top of the builder, so the fake-driven
    # tests above could not run in the dev venv and the defaults were not the only mflux surface.
    _log, kwargs = _rig()
    q8.build_q8_transformer("schnell", Path("/snap"), **kwargs)


def test_a_default_piece_requires_mflux(clear_log, no_mflux):
    # Bug caught: a default used without `require_mflux()` (an ImportError instead of the package's
    # install hint), here for the default loader.
    _log, kwargs = _rig()
    kwargs["loader"] = None
    with pytest.raises(DFloatDependencyError):
        q8.build_q8_transformer("schnell", Path("/snap"), **kwargs)


class _Part:
    """A block (or the whole model) whose ``parameters()`` is a token naming it."""

    def __init__(self, name):
        self.name = name

    def parameters(self):
        return f"params of {self.name}"


class _BlockedTransformer(_Part):
    def __init__(self):
        super().__init__("model")
        self.transformer_blocks = [_Part("double 0"), _Part("double 1")]
        self.single_transformer_blocks = [_Part("single 0")]


def test_the_default_evaluate_runs_block_by_block_then_the_whole_model(
    monkeypatch, clear_log, no_mflux
):
    # Bug caught: a default evaluate that is one whole-model `mx.eval(t.parameters())`, which
    # materialises every block's BF16 sources in one graph (the ~22 GiB BF16 transformer next to the
    # q8 copy), or one that skips a block list. Uses the default (evaluate=None) without mflux.
    evals: list[str] = []
    monkeypatch.setattr(q8.mx, "eval", lambda *args: evals.extend(args))
    _log, kwargs = _rig()
    kwargs["evaluate"] = None
    kwargs["make_transformer"] = lambda model, n_double, n_single: _BlockedTransformer()
    kwargs["applier"] = lambda weights, transformer, component: None
    q8.build_q8_transformer("schnell", Path("/snap"), **kwargs)
    assert evals == [
        "params of double 0",
        "params of double 1",
        "params of single 0",
        "params of model",
    ]


# --- pinned_snapshot -------------------------------------------------------------------------------


def _fake_download(monkeypatch, result, calls):
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(q8, "snapshot_download", snapshot_download)


def test_pinned_snapshot_returns_the_revision_directory_from_the_local_cache_only(
    monkeypatch, tmp_path
):
    # Bug caught: a call that may download (local_files_only missing), drops the revision (the newest
    # cached snapshot would be used), or widens the patterns.
    snap = tmp_path / BASE_REVISION
    calls: list[dict] = []
    _fake_download(monkeypatch, str(snap), calls)
    got = q8.pinned_snapshot(BASE_REPO, BASE_REVISION, allow_patterns=("transformer/*",))
    assert got == snap
    assert calls == [
        {
            "repo_id": BASE_REPO,
            "revision": BASE_REVISION,
            "allow_patterns": ["transformer/*"],
            "local_files_only": True,
        }
    ]


def test_pinned_snapshot_refuses_a_directory_named_otherwise(monkeypatch, tmp_path):
    # Bug caught: a branch name ("main") or another commit's snapshot accepted as the pinned base.
    _fake_download(monkeypatch, str(tmp_path / "main"), [])
    with pytest.raises(DFloatIntegrationError, match=BASE_REVISION):
        q8.pinned_snapshot(BASE_REPO, BASE_REVISION, allow_patterns=["transformer/*"])


def test_pinned_snapshot_turns_a_missing_local_entry_into_the_download_hint(monkeypatch):
    # Bug caught: huggingface_hub's error escaping without the command to fix it, or a hint that
    # joins two patterns into one --include (hf ignores that form).
    _fake_download(monkeypatch, LocalEntryNotFoundError("not cached"), [])
    with pytest.raises(DFloatIntegrationError) as exc:
        q8.pinned_snapshot(BASE_REPO, BASE_REVISION, allow_patterns=["transformer/*", "vae/*"])
    assert (
        f'hf download {BASE_REPO} --revision {BASE_REVISION} --include "transformer/*" '
        '--include "vae/*"'
    ) in str(exc.value)
