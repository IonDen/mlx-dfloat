"""Z-Image base components (text encoder, VAE, tokenizer) and the Hub helpers shared with FLUX.1."""

from fnmatch import fnmatch

import pytest

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.mflux import _hub
from mlx_dfloat.mflux.flux1 import init as flux_init
from mlx_dfloat.mflux.zimage.init import (
    BASE_PATTERNS,
    encoder_definition,
    load_base,
    load_text_encoder,
    load_tokenizers,
    load_vae,
    vae_definition,
)


def test_the_base_patterns_never_fetch_the_transformer():
    # Bug caught: a pattern that pulls the 12.3 GB BF16 transformer (or Turbo's 24.6 GB FP32 one) on first use.
    assert not any(p.startswith(("transformer", "*")) for p in BASE_PATTERNS)


def test_flux1_init_keeps_every_shared_helper_under_its_old_name():
    # Bug caught: a copy left behind in flux1/init.py (a fix to one Hub helper missing from the other), or an import
    # path FLUX.1's callers use dropped by the move.
    for name in (
        "ResolvedRepo",
        "is_hub_id",
        "hub_revision",
        "hub_error",
        "resolve",
        "refuse_quantized",
    ):
        assert getattr(flux_init, name) is getattr(_hub, name), name


def test_a_subset_definition_naming_an_unknown_component_is_refused():
    # Bug caught: a name typo giving an empty or short subset, so mflux's loader silently loads nothing.
    class Source:
        @staticmethod
        def get_components():
            return [type("C", (), {"name": "vae"})()]

        @staticmethod
        def get_tokenizers():
            return []

        @staticmethod
        def quantization_predicate(path, module):
            return False

    with pytest.raises(DFloatIntegrationError, match="does not define"):
        _hub.subset_definition(Source, ("vae", "nope"), ("vae/*",))
    sub = _hub.subset_definition(Source, ("vae",), ("vae/*",))
    assert [c.name for c in sub.get_components()] == ["vae"]
    assert sub.get_download_patterns() == ["vae/*"]


@pytest.mark.mflux
def test_the_definitions_name_only_their_component():
    # Bug caught: a subset that keeps the transformer component (mflux's loader would read transformer/).
    assert [c.name for c in encoder_definition().get_components()] == ["text_encoder"]
    assert [c.name for c in vae_definition().get_components()] == ["vae"]


@pytest.mark.mflux
def test_a_directory_without_the_text_encoder_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's loader with a bare FileNotFoundError.
    with pytest.raises(DFloatFormatError, match="Z-Image base"):
        load_text_encoder(tmp_path)


@pytest.mark.mflux
def test_a_directory_without_the_vae_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's VAE loader with a bare error instead of
    # naming what the directory lacks.
    with pytest.raises(DFloatFormatError, match=r"not a Z-Image base with \['vae'\]"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_a_directory_without_the_tokenizers_is_a_format_error(tmp_path):
    # Bug caught: mflux's tokenizer loader error (FileNotFoundError or RuntimeError) escaping unwrapped, so the user
    # sees an mflux traceback instead of the base that lacks the files.
    with pytest.raises(DFloatFormatError, match="no usable Z-Image tokenizers"):
        load_tokenizers(tmp_path)


@pytest.mark.mflux
def test_load_base_reads_the_vae_first_and_refuses_an_empty_directory(tmp_path):
    # Bug caught: load_base skipping a component (the refusal would come later, at the first call), or swallowing the
    # loader's format error.
    with pytest.raises(DFloatFormatError, match=r"\['vae'\]"):
        load_base(tmp_path)


@pytest.mark.network
@pytest.mark.parametrize("repo", ["Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"])
def test_the_base_patterns_match_text_encoder_vae_and_tokenizer_files_on_the_hub(repo):
    # Bug caught: a pattern that matches nothing in the real repository (an empty download that fails late).
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo)
    hit = [f for f in files if any(fnmatch(f, p) for p in BASE_PATTERNS)]
    assert {f.split("/", 1)[0] for f in hit} == {"text_encoder", "vae", "tokenizer"}
