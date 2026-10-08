"""Qwen-Image 2.1 base components (text encoder, VAE, tokenizer) without the transformer."""

import os
import re
from fnmatch import fnmatch
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from tests._df11_fixtures import write_bf16_original

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import _hub
from mlx_dfloat.mflux.qwen21.init import BASE_PATTERNS, DF11_PATTERNS

# Qwen/Qwen-Image-2.1 @ d26bb61231c349cf6b7896fa83353113880e1ba3, every file on the Hub (listed 2026-10-08).
BASE_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "assets/qr.png",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


# The tokenizer's files in processor/ (what transformers' AutoTokenizer reads for a Qwen2 tokenizer); the two
# preprocessor configs there belong to the image and video processors, which mflux never builds.
TOKENIZER_FILES = (
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/vocab.json",
)


def test_base_patterns_fetch_the_encoder_vae_and_tokenizer_and_never_the_transformer():
    # Bug caught: a pattern such as "*.safetensors" or "transformer/*" pulling the 14.2 GB BF16 transformer on first
    # use, or one that misses a tokenizer file mflux's tokenizer loader reads (its hf_subdir is "processor").
    hit = sorted(f for f in BASE_FILES if any(fnmatch(f, p) for p in BASE_PATTERNS))
    assert hit == sorted(
        [f for f in BASE_FILES if f.split("/", 1)[0] in ("text_encoder", "vae")]
        + list(TOKENIZER_FILES)
    )
    assert not [f for f in hit if f.startswith("transformer/")]


def test_the_processor_pattern_fetches_tokenizer_files_only():
    # Bug caught: "processor/*" fetching whatever the repository adds there on a future revision (code, pickles, large
    # binaries), when the tokenizer needs seven named files. The listing is the Hub's plus files a later revision
    # could add.
    listing = (
        *BASE_FILES,
        "processor/tokenization_qwen.py",
        "processor/processor.pkl",
        "processor/extra/tokenizer.json",
    )
    hit = sorted(
        f
        for f in listing
        if f.startswith("processor/") and any(fnmatch(f, p) for p in BASE_PATTERNS)
    )
    assert hit == sorted(TOKENIZER_FILES)


def test_df11_patterns_fetch_the_one_comfyui_file():
    # Bug caught: the config-less repository resolved without its only weight file (a pattern requiring a config).
    # mingyi456/Qwen-Image-2.1-DF11-ComfyUI @ 1b22a3a1 holds these three files (Hub, 2026-10-08).
    files = (".gitattributes", "README.md", "qwen_image_2.1_bf16-DF11.safetensors")
    assert [f for f in files if any(fnmatch(f, p) for p in DF11_PATTERNS)] == [
        "qwen_image_2.1_bf16-DF11.safetensors"
    ]


# --- mflux lane -----------------------------------------------------------------------------------------------


@pytest.mark.mflux
def test_the_definitions_name_only_their_component_and_the_processor_tokenizer():
    # Bug caught: a subset keeping the transformer component (mflux's loader would read transformer/), or the
    # tokenizer looked up anywhere but processor/ (mflux 0.20.0 qwen21_weight_definition.py:38-50).
    from mlx_dfloat.mflux.qwen21.init import encoder_definition, vae_definition

    assert [c.name for c in encoder_definition().get_components()] == ["text_encoder"]
    assert [c.name for c in vae_definition().get_components()] == ["vae"]
    (tokenizer,) = encoder_definition().get_tokenizers()
    assert (tokenizer.name, tokenizer.hf_subdir, tokenizer.max_length) == (
        "qwen21",
        "processor",
        2048,
    )


@pytest.mark.mflux
def test_the_computed_exemptions_are_exactly_what_mfluxs_mapping_does_not_load():
    # Bug caught: a computed buffer missing from COMPUTED (every real base refused), or a mapped weight listed there
    # (a missing tensor never noticed). mflux 0.20.0: the VAE maps every parameter (qwen21_weight_mapping.py:91 on),
    # the text encoder all but rotary_emb.inv_freq (qwen3_vl_rope.py:21). A two-layer encoder: the mapping names 36.
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
    from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
    from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping

    from mlx_dfloat.mflux.qwen21.init import COMPUTED

    def unmapped(module, mapping):
        targets = {t.to_pattern for t in mapping}
        return {n for n, _ in tree_flatten(module.parameters())} - targets

    encoder = Qwen21TextEncoder(num_hidden_layers=2)
    assert unmapped(encoder, Qwen21WeightMapping.get_text_encoder_mapping()) == {
        "rotary_emb.inv_freq"
    }
    assert unmapped(Qwen21VAE(), Qwen21WeightMapping.get_vae_mapping()) == set()
    assert sorted(COMPUTED["text_encoder"]) == ["rotary_emb.inv_freq"]
    assert sorted(COMPUTED["vae"]) == []


@pytest.mark.mflux
def test_a_base_with_another_encoder_width_is_refused_before_any_encode(monkeypatch, tmp_path):
    # Bug caught: Qwen-Image 1's base (a Qwen2.5-VL encoder 3584 wide) passed as --base: mflux's applier assigns its
    # weights without a shape check, so the failure would be a matmul error at the first encode, not a refusal.
    from mlx_dfloat.mflux.qwen21.init import load_text_encoder

    def wrong_width(root, definition, models, *, what):
        models["text_encoder"].embed_tokens.weight = mx.zeros((151936, 3584), dtype=mx.bfloat16)

    monkeypatch.setattr(_hub, "load_into", wrong_width)
    with pytest.raises(DFloatFormatError, match=r"embed_tokens\.weight.*\(151936, 3584\).*4096"):
        load_text_encoder(tmp_path)


# A two-layer Qwen3-VL text stack (tiny widths; mflux 0.20.0 Qwen21TextEncoder's constructor arguments).
TINY_ENCODER = {
    "vocab_size": 100,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "head_dim": 32,
}


def _write_tiny_encoder(root, *, drop=()):
    """The tiny encoder under the base's names: ``model.language_model.`` + mflux's path, plus a vision tower tensor
    and an untied ``lm_head`` as the real shards carry (mflux maps neither), in two shards with an index.

    ``rotary_emb.inv_freq`` is left out: no checkpoint holds it (computed in ``Qwen3VLRotaryEmbedding.__init__``).
    """
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder

    bits = np.uint16(0x3F00)  # bf16 0.5
    params = dict(tree_flatten(Qwen21TextEncoder(**TINY_ENCODER).parameters()))
    tensors = {
        f"model.language_model.{name}": np.full(p.shape, bits, np.uint16)
        for name, p in params.items()
        if name != "rotary_emb.inv_freq"
    }
    tensors["model.visual.blocks.0.attn.qkv.weight"] = np.zeros((4, 4), np.uint16)
    tensors["lm_head.weight"] = np.zeros((100, 64), np.uint16)
    write_bf16_original(root / "text_encoder", {n: t for n, t in tensors.items() if n not in drop})


@pytest.mark.mflux
def test_an_encoder_tensor_missing_from_the_base_is_refused_by_name(tmp_path):
    # Bug caught: mflux's applier updates with strict=False, so a base missing one tensor (a lost shard, a renamed key)
    # loads without error and that layer keeps its random float32 init: a wrong image, no error.
    from mlx_dfloat.mflux.qwen21.init import load_text_encoder

    _write_tiny_encoder(tmp_path, drop={"model.language_model.layers.1.self_attn.q_proj.weight"})
    with pytest.raises(
        DFloatFormatError, match=r"layers\.1\.self_attn\.q_proj\.weight.*not in the base"
    ):
        load_text_encoder(tmp_path, dims=TINY_ENCODER)


@pytest.mark.mflux
def test_a_complete_encoder_loads_from_the_language_model_tensors_only(tmp_path):
    # Bug caught: the coverage check demanding rotary_emb.inv_freq (every real base refused), the vision tower or the
    # lm_head in the same shards breaking the load, or the file's weights not landing (still the float32 init).
    from mlx_dfloat.mflux.qwen21.init import load_text_encoder

    _write_tiny_encoder(tmp_path)
    encoder = load_text_encoder(tmp_path, dims=TINY_ENCODER)
    q = encoder.layers[1].self_attn.q_proj.weight
    assert q.dtype == mx.bfloat16
    assert float(q[0, 0]) == 0.5


def _replace_all_but(component, missing):
    """A ``load_into`` that replaces every parameter of ``component`` but ``missing`` with lazy zeros (no allocation)."""

    def fake(root, definition, models, *, what):
        module = models[component]
        flat = dict(tree_flatten(module.parameters()))
        new = {n: mx.zeros(p.shape, dtype=mx.bfloat16) for n, p in flat.items() if n != missing}
        module.update(tree_unflatten(list(new.items())), strict=False)

    return fake


@pytest.mark.mflux
def test_a_vae_tensor_missing_from_the_base_is_refused_by_name(monkeypatch, tmp_path):
    # Bug caught: the VAE loaded through the same strict=False applier with no coverage check: a lost decoder conv
    # keeps its random init and the decode returns noise.
    from mlx_dfloat.mflux.qwen21.init import load_vae

    missing = "decoder.conv_out.conv.weight"
    monkeypatch.setattr(_hub, "load_into", _replace_all_but("vae", missing))
    with pytest.raises(DFloatFormatError, match=re.escape(missing) + ".*not in the base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_a_complete_vae_loads(monkeypatch, tmp_path):
    # Bug caught: a complete VAE refused (an exemption list naming something the module does not have, or a check
    # comparing on-disk names instead of the parameters the applier replaced).
    from mlx_dfloat.mflux.qwen21.init import load_vae

    monkeypatch.setattr(_hub, "load_into", _replace_all_but("vae", None))
    vae = load_vae(tmp_path)
    assert vae.decoder.conv_out.conv.weight.dtype == mx.bfloat16


@pytest.mark.mflux
def test_a_directory_without_the_text_encoder_or_the_vae_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's loader with a bare FileNotFoundError.
    from mlx_dfloat.mflux.qwen21.init import load_text_encoder, load_vae

    with pytest.raises(DFloatFormatError, match=r"Qwen-Image 2\.1 base"):
        load_text_encoder(tmp_path)
    with pytest.raises(DFloatFormatError, match=r"Qwen-Image 2\.1 base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_missing_tokenizer_files_are_a_format_error(tmp_path):
    # Bug caught: an empty base directory surfacing mflux's own tokenizer error instead of the package's.
    from mlx_dfloat.mflux.qwen21.init import load_tokenizers

    with pytest.raises(DFloatFormatError, match=r"no usable Qwen-Image 2\.1 tokenizer"):
        load_tokenizers(tmp_path)


@pytest.mark.mflux
def test_a_partial_processor_directory_is_a_format_error(monkeypatch, tmp_path):
    # Bug caught (fable M7): transformers' plain OSError for a processor/ missing one of its files escaping as a raw
    # error instead of the package's format error naming the base.
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader

    from mlx_dfloat.mflux.qwen21.init import load_tokenizers

    def partial(**_kwargs):
        raise OSError("Can't load tokenizer for 'processor'")

    monkeypatch.setattr(TokenizerLoader, "load_all", staticmethod(partial))
    with pytest.raises(
        DFloatFormatError, match=r"no usable Qwen-Image 2\.1 tokenizer.*Can't load tokenizer"
    ):
        load_tokenizers(tmp_path)


def test_load_base_returns_the_three_components(monkeypatch, tmp_path):
    # Bug caught: load_base wiring the encoder into the vae slot (or the reverse), or the tokenizers dropped.
    from mlx_dfloat.mflux.qwen21 import init as qinit

    monkeypatch.setattr(qinit, "load_vae", lambda root: ("vae", root))
    monkeypatch.setattr(qinit, "load_text_encoder", lambda root: ("encoder", root))
    monkeypatch.setattr(qinit, "load_tokenizers", lambda root: {"qwen21": root})
    parts = qinit.load_base(tmp_path)
    assert (parts.vae, parts.text_encoder, parts.tokenizers) == (
        ("vae", tmp_path),
        ("encoder", tmp_path),
        {"qwen21": tmp_path},
    )


@pytest.mark.mflux
@pytest.mark.network
def test_the_real_processor_tokenizer_loads():
    # Bug caught: the processor subdir not where mflux's loader looks, or the template not applied (the prefix mflux
    # drops would then be missing from the ids). Needs the base's processor/ files (a few MB).
    base = os.environ.get("MLX_DFLOAT_QWEN21_BASE")
    if not base:
        pytest.skip("MLX_DFLOAT_QWEN21_BASE is not set (a Qwen-Image 2.1 snapshot with processor/)")
    from mlx_dfloat.mflux.qwen21.init import load_tokenizers

    tokenizer = load_tokenizers(Path(base))["qwen21"]
    ids = tokenizer.tokenize("a").input_ids
    assert ids.shape[0] == 1
    assert ids.shape[1] > 1


@pytest.mark.mflux
@pytest.mark.network
def test_the_tokenizer_loads_from_the_files_the_base_patterns_fetch(tmp_path):
    # Bug caught: a narrowed processor pattern that leaves out a file mflux's TokenizerLoader (transformers'
    # AutoTokenizer) needs, so a fresh download loads no tokenizer. Copies only the matched processor/ files of a real
    # snapshot and loads from the copy. Needs the base's processor/ files (a few MB).
    import shutil

    base = os.environ.get("MLX_DFLOAT_QWEN21_BASE")
    if not base:
        pytest.skip("MLX_DFLOAT_QWEN21_BASE is not set (a Qwen-Image 2.1 snapshot with processor/)")
    from mlx_dfloat.mflux.qwen21.init import load_tokenizers
    from mlx_dfloat.mflux.qwen21.memory import prompt_tokens

    (tmp_path / "processor").mkdir()
    for path in sorted(Path(base, "processor").iterdir()):
        name = f"processor/{path.name}"
        if any(fnmatch(name, p) for p in BASE_PATTERNS):
            shutil.copyfile(path, tmp_path / name)
    copied = sorted(f"processor/{p.name}" for p in (tmp_path / "processor").iterdir())
    assert copied == sorted(TOKENIZER_FILES)
    tokenizer = load_tokenizers(tmp_path)["qwen21"]
    # The calibration's count: the negative prompt " " is 9 text tokens with the full processor/ (test_qwen21_memory).
    assert prompt_tokens(tokenizer, " ") == 9
