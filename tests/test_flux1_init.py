import builtins
import importlib.util

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from mlx_dfloat.errors import DFloatAccessError, DFloatDependencyError, DFloatFormatError
from mlx_dfloat.mflux.flux1 import init


def _hub_http_error(cls: type[Exception], message: str, status: int) -> Exception:
    """Build a real huggingface_hub HTTP error with the ``response`` its ``__init__`` requires.

    huggingface_hub 1.33's HTTP errors are httpx-based and take a mandatory keyword-only
    ``response``; a bare ``GatedRepoError("gated")`` (no response) raises ``TypeError`` before
    ``hub_error`` ever sees it, so every fixture here carries a minimal real ``httpx.Response``.
    """
    request = httpx.Request("GET", "https://huggingface.co/api/models/x/y")
    return cls(message, response=httpx.Response(status, request=request))


def _is_mflux(name: str) -> bool:
    return name == "mflux" or name.startswith("mflux.")


def _hide_mflux(monkeypatch: pytest.MonkeyPatch) -> None:
    """No mflux, whatever this venv holds: both the import and the ``find_spec`` probe miss it.

    Same pattern as ``tests/test_mflux_guard.py``'s ``_hide_mflux``, duplicated here so this
    module's dependency-guard test does not depend on import order across test modules.
    """
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def no_mflux(name: str, *args: object, **kwargs: object) -> object:
        if _is_mflux(name):
            raise ImportError("no mflux here")
        return real_import(name, *args, **kwargs)

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        return None if _is_mflux(name) else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mflux)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_base_patterns_fetch_encoders_vae_and_tokenizers_but_never_the_transformer():
    # Bug caught: a pattern list that pulls `transformer/` (23.8 GB of BF16 the DF11 model never
    # uses) or forgets a component (the loader then fails at the first missing subdir).
    assert not any(p.startswith("transformer") for p in init.BASE_PATTERNS)
    heads = {p.split("/")[0] for p in init.BASE_PATTERNS}
    assert heads == {"text_encoder", "text_encoder_2", "vae", "tokenizer", "tokenizer_2"}
    assert init.DF11_PATTERNS == ("*.safetensors", "config.json")


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("DFloat11/FLUX.1-schnell-DF11", True),
        ("black-forest-labs/FLUX.1-dev", True),
        ("./local/dir", False),
        ("~/models/x", False),
        ("/abs/path", False),
        ("nodash", False),
        ("a/b/c", False),
    ],
)
def test_is_hub_id_follows_mflux_rule_and_prefers_an_existing_path(spec, want, tmp_path):
    # Bug caught: treating `org/name` as a repo id when a directory of that relative name exists
    # (mflux resolves local first), or accepting `a/b/c` as a repo id.
    assert init.is_hub_id(spec) is want
    (tmp_path / "org").mkdir()
    (tmp_path / "org" / "name").mkdir()
    assert init.is_hub_id(str(tmp_path / "org" / "name")) is False


def test_hub_revision_reads_the_snapshot_sha_and_nothing_else(tmp_path):
    # Bug caught: recording a local directory's name as a "revision", or missing the SHA of a
    # cache snapshot (the report would then say the checkpoint has no revision).
    sha = "51a428b928197e0531cb93d6e438941e2d0b247e"
    snap = tmp_path / "models--x--y" / "snapshots" / sha
    snap.mkdir(parents=True)
    assert init.hub_revision(snap) == sha
    assert init.hub_revision(tmp_path) is None


def test_hub_errors_map_to_access_or_format_and_leave_the_rest_alone():
    # Bug caught: a gated-licence refusal surfacing as a bare HfHubHTTPError traceback (the CLI
    # would exit with an unmapped error), or a missing repo reading as an access problem.
    gated = init.hub_error(
        _hub_http_error(GatedRepoError, "gated", 403), "black-forest-labs/FLUX.1-dev"
    )
    assert isinstance(gated, DFloatAccessError)
    assert "accept the licence" in str(gated)
    assert "hf auth login" in str(gated)
    missing = init.hub_error(_hub_http_error(RepositoryNotFoundError, "404", 401), "no/such")
    assert isinstance(missing, DFloatFormatError)
    other = ValueError("unrelated")
    assert init.hub_error(other, "x/y") is other
    plain_http = _hub_http_error(HfHubHTTPError, "500", 500)
    assert init.hub_error(plain_http, "x/y") is plain_http


@pytest.mark.parametrize("status", [401, 403])
def test_a_bare_unauthorised_http_error_maps_to_access_not_format(status):
    # Bug caught: a plain HfHubHTTPError (no GatedRepoError/RepositoryNotFoundError subclass) with
    # a 401 or 403 response falling through to `return exc` unmapped, so a private-without-token
    # or an otherwise-unauthorised repo read surfaces as a bare HTTP error instead of the
    # DFloatAccessError with the login hint the CLI maps to its access exit code.
    unauthorised = init.hub_error(_hub_http_error(HfHubHTTPError, "no", status), "x/y")
    assert isinstance(unauthorised, DFloatAccessError)
    assert "hf auth login" in str(unauthorised)
    assert str(status) in str(unauthorised)


def test_resolve_takes_a_local_directory_as_is_and_refuses_a_missing_one(tmp_path):
    # Bug caught: a local path going through snapshot_download (a network call for a directory
    # on disk), or a typo'd path being mistaken for a Hub id.
    resolved = init.resolve(str(tmp_path), patterns=init.DF11_PATTERNS)
    assert resolved == init.ResolvedRepo(root=tmp_path, repo_id=None, revision=None)
    with pytest.raises(DFloatFormatError, match="not a directory or a Hub repository id"):
        init.resolve(str(tmp_path / "nope"), patterns=init.DF11_PATTERNS)


def test_resolve_downloads_a_hub_id_with_the_patterns_and_wraps_a_gated_refusal(
    monkeypatch, tmp_path
):
    # Bug caught: downloading the whole repo (no allow_patterns), or letting GatedRepoError out.
    calls = []
    sha = "d" * 40
    snap = tmp_path / "snapshots" / sha
    snap.mkdir(parents=True)

    def fake_download(*, repo_id, allow_patterns):
        calls.append((repo_id, tuple(allow_patterns)))
        if repo_id == "gated/repo":
            raise _hub_http_error(GatedRepoError, "no", 403)
        return str(snap)

    monkeypatch.setattr(init, "_snapshot_download", fake_download)
    resolved = init.resolve("DFloat11/FLUX.1-schnell-DF11", patterns=init.DF11_PATTERNS)
    assert resolved == init.ResolvedRepo(
        root=snap, repo_id="DFloat11/FLUX.1-schnell-DF11", revision=sha
    )
    assert calls == [("DFloat11/FLUX.1-schnell-DF11", init.DF11_PATTERNS)]
    with pytest.raises(DFloatAccessError):
        init.resolve("gated/repo", patterns=init.BASE_PATTERNS)


class _Meta:
    def __init__(self, level):
        self.quantization_level = level


class _Weights:
    def __init__(self, level):
        self.meta_data = _Meta(level)


def test_a_quantized_base_is_refused_and_a_bf16_one_passes(tmp_path):
    # Bug caught: WeightLoader applying a saved 8-bit quantisation to the encoders even with
    # quantize_arg=None (mflux honours the stored level), so the "bit-exact" run silently encodes
    # with q8 encoders.
    with pytest.raises(DFloatFormatError, match=r"8-bit"):
        init.refuse_quantized(_Weights(8), tmp_path)
    init.refuse_quantized(_Weights(None), tmp_path)


def test_load_encoders_and_load_vae_need_mflux_before_their_mflux_imports_run(
    monkeypatch, tmp_path
):
    # Bug caught: `from mflux....` module-level imports at the top of load_encoders/load_vae ran
    # before any require_mflux() guard, so a caller without the mflux extra (e.g. load_base) got a
    # bare ModuleNotFoundError instead of the package-rooted DFloatDependencyError the changelog
    # promises. Runs in every venv (with or without mflux installed): _hide_mflux forces the
    # "missing" condition regardless of what this venv actually holds.
    _hide_mflux(monkeypatch)
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        init.load_encoders(tmp_path)
    with pytest.raises(DFloatDependencyError, match=r"install mlx-dfloat\[mflux\]"):
        init.load_vae(tmp_path)


@pytest.mark.mflux
def test_the_definitions_hold_only_their_components_and_the_flux_download_patterns_minus_the_transformer():
    # Bug caught: a definition still listing the transformer (the loader would look for
    # transformer/ under the base and pull 23.8 GB), or patterns drifting from mflux's own list.
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    assert [c.name for c in init.encoders_definition().get_components()] == [
        "t5_encoder",
        "clip_encoder",
    ]
    assert [c.name for c in init.vae_definition().get_components()] == ["vae"]
    mflux_patterns = {
        p for p in FluxWeightDefinition.get_download_patterns() if not p.startswith("transformer/")
    }
    assert set(init.ENCODER_PATTERNS + init.VAE_PATTERNS) == mflux_patterns


@pytest.mark.mflux
def test_a_directory_without_the_encoders_is_a_format_error(tmp_path):
    # Bug caught: mflux's FileNotFoundError ("No safetensors files found in .../text_encoder_2")
    # escaping unwrapped, so the CLI cannot map "you pointed --base at the DF11 repo" to exit 2.
    (tmp_path / "text_encoder").mkdir()
    with pytest.raises(DFloatFormatError, match=r"not a FLUX\.1 base"):
        init.load_encoders(tmp_path)


@pytest.mark.mflux
def test_load_encoders_builds_fresh_modules_and_keeps_no_loaded_weights_object(
    monkeypatch, tmp_path
):
    # Bug caught: the returned encoders sharing one module object across reloads (a stale T5
    # after a drop), or the LoadedWeights dict kept alive on the module (pinning 9.5 GB).
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

    empty = LoadedWeights(
        components={"t5_encoder": {}, "clip_encoder": {}, "vae": {}},
        meta_data=MetaData(quantization_level=None, mflux_version=None),
    )
    monkeypatch.setattr(WeightLoader, "load", staticmethod(lambda **kw: empty))
    t5_a, clip_a = init.load_encoders(tmp_path)
    t5_b, _clip_b = init.load_encoders(tmp_path)
    assert isinstance(t5_a, T5Encoder)
    assert t5_a is not t5_b
    assert not any(isinstance(v, LoadedWeights) for v in vars(t5_a).values())
    assert clip_a.__class__.__name__ == "CLIPEncoder"


@pytest.mark.mflux
def test_a_quantized_base_is_refused_before_the_applier_runs(monkeypatch, tmp_path):
    # Bug caught: apply_and_quantize called on a q8 save (mflux honours the stored level and
    # the encoders come out quantized with no error).
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    q8 = LoadedWeights(
        components={"vae": {}}, meta_data=MetaData(quantization_level=8, mflux_version="0.20.0")
    )
    monkeypatch.setattr(WeightLoader, "load", staticmethod(lambda **kw: q8))
    monkeypatch.setattr(
        WeightApplier, "apply_and_quantize", staticmethod(lambda **kw: pytest.fail("applied"))
    )
    with pytest.raises(DFloatFormatError, match="8-bit"):
        init.load_vae(tmp_path)


@pytest.mark.mflux
@pytest.mark.network
@pytest.mark.parametrize(("model", "want"), [("schnell", 256), ("dev", 512), ("krea-dev", 512)])
def test_tokenizers_load_from_the_public_schnell_repo_with_the_models_t5_length(model, want):
    # Bug caught: the T5 max length left at the definition's 256 for dev (the encoder then sees
    # 256 tokens and the embeddings differ from mflux's own), or the tokenizer subdirs misnamed.
    from mflux.models.common.config.model_config import ModelConfig

    resolved = init.resolve("black-forest-labs/FLUX.1-schnell", patterns=init.TOKENIZER_PATTERNS)
    tokenizers = init.load_tokenizers(
        resolved.root, ModelConfig.from_name(model_name=model, base_model=None)
    )
    assert set(tokenizers) == {"clip", "t5"}
    assert tokenizers["t5"].max_length == want
    assert tokenizers["clip"].max_length == 77
