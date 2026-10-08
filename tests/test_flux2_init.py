"""FLUX.2 Klein base components (text encoder, VAE, tokenizer) without the transformer."""

import re
from fnmatch import fnmatch

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import _hub
from mlx_dfloat.mflux.flux2.init import BASE_PATTERNS, DF11_PATTERNS

# black-forest-labs/FLUX.2-klein-base-4B, every file on the Hub (listed 2026-10-08, revision a3b4f48).
BASE_4B_FILES = (
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "editing.jpg",
    "flux-2-klein-base-4b.safetensors",
    "model_index.json",
    "others.jpg",
    "realism.jpg",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


def _matching(patterns):
    return sorted(f for f in BASE_4B_FILES if any(fnmatch(f, p) for p in patterns))


def test_base_patterns_never_fetch_a_transformer_or_the_root_checkpoint():
    # Bug caught: a pattern like "*.safetensors" or "transformer/*" pulling the 7.75 GB BF16 transformer and the
    # 7.75 GB root BFL checkpoint on first use, or a pattern that misses a tokenizer file mflux needs.
    assert _matching(BASE_PATTERNS) == [
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]


def test_df11_patterns_fetch_the_one_checkpoint_file_and_its_config():
    # Bug caught: the DF11 repository resolved without config.json (no pattern_dict) or without its weights. The
    # mingyi456 Klein DF11 repositories hold model.safetensors + config.json (Hub, 2026-10-08).
    files = (".gitattributes", "README.md", "config.json", "model.safetensors")
    assert sorted(f for f in files if any(fnmatch(f, p) for p in DF11_PATTERNS)) == [
        "config.json",
        "model.safetensors",
    ]


# --- mflux lane -----------------------------------------------------------------------------------------------


@pytest.mark.mflux
def test_the_definitions_name_only_their_component():
    # Bug caught: a subset that keeps the transformer component (mflux's loader would read transformer/).
    from mlx_dfloat.mflux.flux2.init import encoder_definition, vae_definition

    assert [c.name for c in encoder_definition().get_components()] == ["text_encoder"]
    assert [c.name for c in vae_definition().get_components()] == ["vae"]
    assert [t.name for t in encoder_definition().get_tokenizers()] == ["qwen3"]


@pytest.mark.mflux
@pytest.mark.parametrize(
    ("name", "width"),
    # Qwen3 hidden size per Klein size: mflux 0.20.0 common/config/model_config.py:515 (base 4B: 2560) and :544
    # (base 9B: 4096); vocabulary 151936 (qwen3_text_encoder.py:12).
    [("flux2_klein_base_4b", 2560), ("flux2_klein_base_9b", 4096)],
)
def test_the_encoder_is_built_with_the_models_overrides(monkeypatch, tmp_path, name, width):
    # Bug caught: the 9B encoder built at mflux's 4B default width (the overrides not passed), so the 9B base's
    # weights would land on a 2560-wide module.
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.mflux.flux2.init import load_text_encoder

    def complete_base(root, definition, models, *, what):
        # A base holding every tensor at the built shapes (lazy zeros: nothing is allocated).
        module = models["text_encoder"]
        module.update(tree_map(lambda p: mx.zeros(p.shape, dtype=mx.bfloat16), module.parameters()))

    monkeypatch.setattr(_hub, "load_into", complete_base)
    encoder = load_text_encoder(tmp_path, getattr(ModelConfig, name)())
    assert encoder.embed_tokens.weight.shape == (151936, width)


@pytest.mark.mflux
def test_a_base_of_the_other_size_is_refused_before_any_encode(monkeypatch, tmp_path):
    # Bug caught (a 4B base passed for a 9B model): mflux's applier assigns the mismatched weights without a shape
    # check, so the failure would be a matmul error minutes later instead of a refusal naming the parameter.
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.mflux.flux2.init import load_text_encoder

    def wrong_width(root, definition, models, *, what):
        models["text_encoder"].embed_tokens.weight = mx.zeros((151936, 2560), dtype=mx.bfloat16)

    monkeypatch.setattr(_hub, "load_into", wrong_width)
    with pytest.raises(DFloatFormatError, match=r"embed_tokens\.weight.*\(151936, 2560\).*4096"):
        load_text_encoder(tmp_path, ModelConfig.flux2_klein_base_9b())


@pytest.mark.mflux
def test_a_directory_without_the_text_encoder_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's loader with a bare FileNotFoundError.
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.mflux.flux2.init import load_text_encoder

    with pytest.raises(DFloatFormatError, match=r"FLUX\.2 Klein base"):
        load_text_encoder(tmp_path, ModelConfig.flux2_klein_base_4b())


@pytest.mark.mflux
def test_a_directory_without_the_vae_is_a_format_error(tmp_path):
    # Bug caught: load_vae reaching mflux's loader without the package's translation (a bare FileNotFoundError).
    from mlx_dfloat.mflux.flux2.init import load_vae

    with pytest.raises(DFloatFormatError, match=r"FLUX\.2 Klein base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_missing_tokenizer_files_are_a_format_error(tmp_path):
    # Bug caught: an empty base directory surfacing mflux's own tokenizer error instead of the package's.
    from mlx_dfloat.mflux.flux2.init import load_tokenizers

    with pytest.raises(DFloatFormatError, match=r"no usable FLUX\.2 tokenizers"):
        load_tokenizers(tmp_path)


@pytest.mark.network
def test_the_base_patterns_match_text_encoder_vae_and_tokenizer_files_on_the_hub():
    # Bug caught: a pattern that matches nothing in the real repository (an empty download that fails late).
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files("black-forest-labs/FLUX.2-klein-base-4B")
    hit = [f for f in files if any(fnmatch(f, p) for p in BASE_PATTERNS)]
    assert {f.split("/", 1)[0] for f in hit} == {"text_encoder", "vae", "tokenizer"}


# --- base-component coverage: a tensor the base lacks is refused, not left at its random init -----------------------

# A two-layer Qwen3 text encoder (the shape of the real one, tiny widths).
TINY_ENCODER = {
    "vocab_size": 100,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
}


def _write_tiny_encoder(root, *, drop=()):
    """The tiny encoder's checkpoint under the names a Qwen3 checkpoint uses (``model.`` + mflux's path).

    ``rotary_emb.inv_freq`` is left out, as in the real files: it is computed in ``Qwen3TextRotaryEmbedding.__init__``
    (mflux 0.20.0 qwen3_text_rotary_embedding.py:18), and the base-4B encoder headers name only ``model.embed_tokens``,
    ``model.norm`` and ``model.layers.*`` tensors (read 2026-10-08, revision a3b4f48).
    """
    from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(Qwen3TextEncoder(**TINY_ENCODER).parameters()))
    flat = {
        f"model.{name}": mx.full(p.shape, 0.5, dtype=mx.bfloat16)
        for name, p in params.items()
        if name != "rotary_emb.inv_freq" and f"model.{name}" not in drop
    }
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "text_encoder" / "model.safetensors"), flat)


def _write_vae(root, *, drop=()):
    """A VAE file under the base's diffusers names, one-element tensors (the applier checks no shape).

    The names are the base-4B VAE file's (read 2026-10-08, revision a3b4f48): mflux's parameter paths, except that
    the two mid-block attentions' output projection is ``to_out.0`` on disk (mflux 0.20.0 flux2_weight_mapping.py:
    319-325, 369-375), plus ``bn.num_batches_tracked``, which mflux does not map.
    """
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mlx.utils import tree_flatten

    names = [n.replace(".to_out.", ".to_out.0.") for n, _ in tree_flatten(Flux2VAE().parameters())]
    names.append("bn.num_batches_tracked")
    flat = {n: mx.zeros((1,), dtype=mx.bfloat16) for n in names if n not in drop}
    (root / "vae").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "vae" / "diffusion_pytorch_model.safetensors"), flat)


@pytest.mark.mflux
def test_an_encoder_tensor_missing_from_the_base_is_refused_by_name(tmp_path):
    # Bug caught (review 2026-10-08): mflux's applier updates with strict=False, so a base missing one tensor (a lost
    # shard, a renamed key) loads without error and that layer keeps its random float32 init: a wrong image, no error.
    import types

    from mlx_dfloat.mflux.flux2.init import load_text_encoder

    _write_tiny_encoder(tmp_path, drop={"model.layers.1.self_attn.q_proj.weight"})
    config = types.SimpleNamespace(text_encoder_overrides=TINY_ENCODER)
    with pytest.raises(
        DFloatFormatError, match=r"layers\.1\.self_attn\.q_proj\.weight.*not in the base"
    ):
        load_text_encoder(tmp_path, config)


@pytest.mark.mflux
def test_a_complete_encoder_loads_and_its_computed_rotary_buffer_is_not_asked_of_the_base(tmp_path):
    # Bug caught: the coverage check demanding rotary_emb.inv_freq (computed at construction, in no checkpoint), so
    # every real base would be refused; or the check passing while the file's weights did not land.
    import types

    from mlx_dfloat.mflux.flux2.init import load_text_encoder

    _write_tiny_encoder(tmp_path)
    encoder = load_text_encoder(
        tmp_path, types.SimpleNamespace(text_encoder_overrides=TINY_ENCODER)
    )
    q = encoder.layers[1].self_attn.q_proj.weight
    assert q.dtype == mx.bfloat16
    assert float(q[0, 0]) == 0.5  # the file's value, not the float32 random init


@pytest.mark.mflux
@pytest.mark.parametrize("dropped", ["decoder.conv_out.weight", "bn.running_mean"])
def test_a_vae_tensor_missing_from_the_base_is_refused_by_name(tmp_path, dropped):
    # Bug caught: the VAE loaded through the same strict=False applier with no coverage check (a lost conv keeps its
    # random init and the decode returns noise), or the batch-norm statistics exempted as computed buffers (they come
    # from the file: mflux 0.20.0 flux2_weight_mapping.py:202-209; a missing one would leave zeros / ones).
    from mlx_dfloat.mflux.flux2.init import load_vae

    _write_vae(tmp_path, drop={dropped})
    with pytest.raises(DFloatFormatError, match=re.escape(dropped) + ".*not in the base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_a_complete_vae_loads(tmp_path):
    # Bug caught: a complete VAE refused (a check that compares on-disk names such as to_out.0 instead of the
    # parameters the applier replaced).
    from mlx_dfloat.mflux.flux2.init import load_vae

    _write_vae(tmp_path)
    vae = load_vae(tmp_path)
    # Replaced by the file's tensor (the init is float32 zeros).
    assert vae.bn.running_mean.dtype == mx.bfloat16
