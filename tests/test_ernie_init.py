"""ERNIE-Image base components (text encoder, VAE, tokenizer) without the transformer or the prompt enhancer."""

import os
import re
from fnmatch import fnmatch
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import _hub
from mlx_dfloat.mflux.ernie.init import BASE_PATTERNS, DF11_PATTERNS, TOKENIZER_PATTERNS

# baidu/ERNIE-Image @ 5346b31d68c9c23758ba56ef8be5e9dc174c7f99, every file on the Hub (listed 2026-10-08; the same
# paths as baidu/ERNIE-Image-Turbo @ bc68c81e).
BASE_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "pe/chat_template.jinja",
    "pe/config.json",
    "pe/generation_config.json",
    "pe/model.safetensors",
    "pe/tokenizer.json",
    "pe/tokenizer_config.json",
    "pe_tokenizer/chat_template.jinja",
    "pe_tokenizer/tokenizer.json",
    "pe_tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
# The calibration prompt and its token count (mflux 0.20.0 TokenizerLoader on both base snapshots, 2026-10-08).
PROMPT = (
    "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the "
    "water"
)
PROMPT_TOKENS = 26


def _hits(files):
    return sorted(f for f in files if any(fnmatch(f, p) for p in BASE_PATTERNS))


def test_base_patterns_fetch_the_encoder_vae_and_tokenizer_and_never_the_transformer_or_the_prompt_enhancer():
    # Bug caught: a pattern pulling the 16 GB BF16 transformer, the 7.7 GB prompt enhancer (pe/) or its tokenizer
    # (pe_tokenizer/, whose tokenizer.json a loose "*tokenizer*" pattern would match), or one that misses a file of
    # the encoder, the VAE or the tokenizer.
    assert _hits(BASE_FILES) == [
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]


def test_the_tokenizer_pattern_fetches_tokenizer_files_only():
    # Bug caught: the tokenizer pattern widened back to the directory ("tokenizer/*" or mflux's "tokenizer/**"),
    # fetching whatever a later revision drops there (binaries, nested copies). The listing is the snapshot's two
    # files plus files a later revision could add.
    listing = (
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/extra/tokenizer.json",
        "tokenizer/model.bin",
        "tokenizer/README.md",
        "tokenizer/special_tokens_map.json",
    )
    assert _hits(listing) == [
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    assert all(not p.endswith("*") for p in TOKENIZER_PATTERNS)


def test_df11_patterns_fetch_the_checkpoint_file_and_its_config():
    # Bug caught: the DF11 repository resolved without its config (the reader could not split the groups) or with
    # its README. mingyi456/ERNIE-Image-DF11 @ c2dd30ad holds these four files (Hub, 2026-10-08).
    files = (".gitattributes", "README.md", "config.json", "model.safetensors")
    assert [f for f in files if any(fnmatch(f, p) for p in DF11_PATTERNS)] == [
        "config.json",
        "model.safetensors",
    ]


# --- mflux lane -----------------------------------------------------------------------------------------------


@pytest.mark.mflux
def test_the_definitions_name_only_their_component_and_the_tokenizer_subdir():
    # Bug caught: a subset keeping the transformer component (mflux's loader would read transformer/), the encoder
    # losing its language-model prefix filter (the vision tower would load into nothing), or the tokenizer looked up
    # anywhere but tokenizer/ (mflux 0.20.0 ernie_weight_definition.py:26-47).
    from mlx_dfloat.mflux.ernie.init import encoder_definition, vae_definition

    (encoder,) = encoder_definition().get_components()
    assert (encoder.name, encoder.weight_prefix_filters) == (
        "text_encoder",
        ["language_model.model."],
    )
    assert [c.name for c in vae_definition().get_components()] == ["vae"]
    (tokenizer,) = encoder_definition().get_tokenizers()
    assert (tokenizer.name, tokenizer.hf_subdir, tokenizer.max_length) == (
        "ernie",
        "tokenizer",
        2048,
    )


# A two-layer Mistral text stack (tiny widths; mflux 0.20.0 ErnieMistralTextEncoder's constructor arguments).
TINY_ENCODER = {
    "vocab_size": 100,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "intermediate_size": 128,
}


@pytest.mark.mflux
def test_the_computed_exemptions_are_exactly_what_the_base_files_do_not_hold():
    # Bug caught: a computed buffer missing from COMPUTED (every real base refused), or a loaded weight listed there (a
    # missing tensor never noticed). mflux 0.20.0: the encoder's RoPE frequencies are the underscore attribute
    # _inv_freq (text_encoder.py:137-138), not a parameter, so every encoder parameter sits under the
    # language_model.model. prefix the base file holds; the VAE maps every parameter (flux2_weight_mapping.py).
    from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import (
        ErnieMistralTextEncoder,
    )
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping

    from mlx_dfloat.mflux.ernie.init import COMPUTED

    encoder = {n for n, _ in tree_flatten(ErnieMistralTextEncoder(**TINY_ENCODER).parameters())}
    assert {n for n in encoder if not n.startswith("language_model.model.")} == set()
    # The VAE targets are templated ("...resnets.{i}.conv1.weight"): an index token stands for any number.
    targets = [
        re.compile(re.sub(r"\\\{[a-z_]+\\\}", r"[0-9]+", re.escape(t.to_pattern)) + "$")
        for t in Flux2WeightMapping.get_vae_mapping()
    ]
    vae = {n for n, _ in tree_flatten(Flux2VAE().parameters())}
    assert {n for n in vae if not any(t.match(n) for t in targets)} == set()
    assert sorted(COMPUTED) == ["text_encoder", "vae"]
    assert [len(v) for v in COMPUTED.values()] == [0, 0]


def _write_tiny_encoder(root, *, drop=()):
    """The tiny encoder under the base file's names (mflux's paths: ``language_model.model.*``), plus a vision tower
    and a projector tensor as the real file carries (2026-10-08 header: 218 and 4 tensors), which mflux's prefix filter
    leaves out."""
    from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import (
        ErnieMistralTextEncoder,
    )

    params = dict(tree_flatten(ErnieMistralTextEncoder(**TINY_ENCODER).parameters()))
    flat = {n: mx.full(p.shape, 0.5, dtype=mx.bfloat16) for n, p in params.items() if n not in drop}
    flat["vision_tower.transformer.layers.0.attention.q_proj.weight"] = mx.zeros(
        (4, 4), dtype=mx.bfloat16
    )
    flat["multi_modal_projector.linear_1.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "text_encoder" / "model.safetensors"), flat)


@pytest.mark.mflux
def test_an_encoder_tensor_missing_from_the_base_is_refused_by_name(tmp_path):
    # Bug caught: mflux's applier updates with strict=False, so a base missing one tensor (a lost
    # shard, a renamed key) loads without error and that layer keeps its random float32 init: a wrong image, no error.
    from mlx_dfloat.mflux.ernie.init import load_text_encoder

    missing = "language_model.model.layers.1.self_attn.q_proj.weight"
    _write_tiny_encoder(tmp_path, drop={missing})
    with pytest.raises(DFloatFormatError, match=re.escape(missing) + ".*not in the base"):
        load_text_encoder(tmp_path, dims=TINY_ENCODER)


@pytest.mark.mflux
def test_a_complete_encoder_loads_from_the_language_model_tensors_only(tmp_path):
    # Bug caught: the vision tower or the projector in the same file breaking the load, or the file's weights not
    # landing (still the float32 init).
    from mlx_dfloat.mflux.ernie.init import load_text_encoder

    _write_tiny_encoder(tmp_path)
    encoder = load_text_encoder(tmp_path, dims=TINY_ENCODER)
    q = encoder.language_model.model.layers[1].self_attn.q_proj.weight
    assert q.dtype == mx.bfloat16
    assert float(q[0, 0]) == 0.5


@pytest.mark.mflux
def test_a_base_with_another_encoder_width_is_refused_before_any_encode(monkeypatch, tmp_path):
    # Bug caught: a wrong base (a Qwen3-4B-wide embedding) loading silently: mflux's applier assigns weights without a
    # shape check, so the failure would be a matmul error at the first encode, not a refusal.
    from mlx_dfloat.mflux.ernie.init import load_text_encoder

    def wrong_width(root, definition, models, *, what):
        model = models["text_encoder"].language_model.model
        model.embed_tokens.weight = mx.zeros((151936, 2560), dtype=mx.bfloat16)

    monkeypatch.setattr(_hub, "load_into", wrong_width)
    with pytest.raises(
        DFloatFormatError, match=r"embed_tokens\.weight has shape \(151936, 2560\).*\(100, 64\)"
    ):
        load_text_encoder(tmp_path, dims=TINY_ENCODER)


def _write_vae(root, *, drop=()):
    """A VAE file under the base's diffusers names, one-element tensors (the applier checks no shape).

    ERNIE's VAE is FLUX.2's (mflux 0.20.0 ernie_weight_mapping.py:111-113): mflux's parameter paths, except that the
    two mid-block attentions' output projection is ``to_out.0`` on disk (flux2_weight_mapping.py:319-325, 369-375),
    plus ``bn.num_batches_tracked``, which mflux does not map.
    """
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

    names = [n.replace(".to_out.", ".to_out.0.") for n, _ in tree_flatten(Flux2VAE().parameters())]
    names.append("bn.num_batches_tracked")
    flat = {n: mx.zeros((1,), dtype=mx.bfloat16) for n in names if n not in drop}
    (root / "vae").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "vae" / "diffusion_pytorch_model.safetensors"), flat)


@pytest.mark.mflux
@pytest.mark.parametrize("dropped", ["decoder.conv_out.weight", "bn.running_mean"])
def test_a_vae_tensor_missing_from_the_base_is_refused_by_name(tmp_path, dropped):
    # Bug caught: the VAE loaded through the same strict=False applier with no coverage check (a
    # lost decoder conv keeps its random init and the decode returns noise), or the batch-norm statistics exempted as
    # computed buffers (they come from the file: flux2_weight_mapping.py:202-209).
    from mlx_dfloat.mflux.ernie.init import load_vae

    _write_vae(tmp_path, drop={dropped})
    with pytest.raises(DFloatFormatError, match=re.escape(dropped) + ".*not in the base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_a_complete_vae_loads(tmp_path):
    # Bug caught: a complete VAE refused (a check that compares on-disk names such as to_out.0 instead of the
    # parameters the applier replaced).
    from mlx_dfloat.mflux.ernie.init import load_vae

    _write_vae(tmp_path)
    vae = load_vae(tmp_path)
    assert vae.bn.running_mean.dtype == mx.bfloat16  # the file's tensor (the init is float32)


@pytest.mark.mflux
def test_a_directory_without_the_text_encoder_or_the_vae_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's loader with a bare FileNotFoundError.
    from mlx_dfloat.mflux.ernie.init import load_text_encoder, load_vae

    with pytest.raises(DFloatFormatError, match="ERNIE-Image base"):
        load_text_encoder(tmp_path)
    with pytest.raises(DFloatFormatError, match="ERNIE-Image base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_missing_tokenizer_files_are_a_format_error(tmp_path):
    # Bug caught: an empty base directory surfacing mflux's own tokenizer error instead of the package's.
    from mlx_dfloat.mflux.ernie.init import load_tokenizers

    with pytest.raises(DFloatFormatError, match="no usable ERNIE-Image tokenizer"):
        load_tokenizers(tmp_path)


@pytest.mark.mflux
def test_a_partial_tokenizer_directory_is_a_format_error(monkeypatch, tmp_path):
    # Bug caught: transformers' plain OSError for a tokenizer/ missing one of its files escaping as a raw error
    # instead of the package's format error naming the base.
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader

    from mlx_dfloat.mflux.ernie.init import load_tokenizers

    def partial(**_kwargs):
        raise OSError("Can't load tokenizer for 'tokenizer'")

    monkeypatch.setattr(TokenizerLoader, "load_all", staticmethod(partial))
    with pytest.raises(
        DFloatFormatError, match=r"no usable ERNIE-Image tokenizer.*Can't load tokenizer"
    ):
        load_tokenizers(tmp_path)


def test_load_base_returns_the_three_components(monkeypatch, tmp_path):
    # Bug caught: load_base wiring the encoder into the vae slot (or the reverse), or the tokenizers dropped.
    from mlx_dfloat.mflux.ernie import init as einit

    monkeypatch.setattr(einit, "load_vae", lambda root: ("vae", root))
    monkeypatch.setattr(einit, "load_text_encoder", lambda root: ("encoder", root))
    monkeypatch.setattr(einit, "load_tokenizers", lambda root: {"ernie": root})
    parts = einit.load_base(tmp_path)
    assert (parts.vae, parts.text_encoder, parts.tokenizers) == (
        ("vae", tmp_path),
        ("encoder", tmp_path),
        {"ernie": tmp_path},
    )


@pytest.mark.mflux
@pytest.mark.slow
def test_the_real_tokenizer_loads_from_the_pattern_files_alone(tmp_path):
    # Bug caught: the tokenizer subdir not where mflux looks, or the narrowed list missing a file mflux's
    # TokenizerLoader (transformers' AutoTokenizer) needs, so a user's first fetch would load no tokenizer. Links only
    # the snapshot's tokenizer/ files a pattern selects. Reads a local base snapshot (no network): gated on --run-slow
    # and MLX_DFLOAT_ERNIE_BASE, like the real-blocks tests, so CI never runs it; run it by hand before a release.
    base = os.environ.get("MLX_DFLOAT_ERNIE_BASE")
    if not base:
        pytest.skip(
            "MLX_DFLOAT_ERNIE_BASE is not set (an ERNIE-Image base snapshot with tokenizer/)"
        )
    from mlx_dfloat.mflux.ernie.init import load_tokenizers

    (tmp_path / "tokenizer").mkdir()
    for path in sorted(Path(base, "tokenizer").iterdir()):
        name = f"tokenizer/{path.name}"
        if any(fnmatch(name, p) for p in BASE_PATTERNS):
            (tmp_path / name).symlink_to(path.resolve())
    linked = sorted(f"tokenizer/{p.name}" for p in (tmp_path / "tokenizer").iterdir())
    assert linked == ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    tokenizer = load_tokenizers(tmp_path)["ernie"]
    assert tokenizer.tokenize("a").input_ids.tolist() == [[1, 1097]]
    assert int(mx.sum(tokenizer.tokenize(PROMPT).attention_mask[0])) == PROMPT_TOKENS
