"""Krea 2 base components (text encoder, VAE, tokenizer) without the transformer."""

import os
import re
from fnmatch import fnmatch
from pathlib import Path

import httpx
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_dfloat.errors import DFloatAccessError, DFloatFormatError
from mlx_dfloat.mflux import _hub
from mlx_dfloat.mflux.krea2.init import BASE_PATTERNS, DF11_PATTERNS, TOKENIZER_PATTERNS

# krea/Krea-2-Raw @ 6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603, the Hub file listing (2026-10-08); its
# images/ directory (sample pictures, about 40 MB) is represented by its first and last file.
RAW_FILES = (
    ".gitattributes",
    "LICENSE.pdf",
    "README.md",
    "images/00.png",
    "images/header.jpg",
    "model_index.json",
    "raw.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
# krea/Krea-2-Turbo @ 98e0fe11: turbo.safetensors in place of raw.safetensors, no images/.
TURBO_FILES = tuple(
    "turbo.safetensors" if f == "raw.safetensors" else f
    for f in RAW_FILES
    if not f.startswith("images/")
)
WANTED = [
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
# The calibration prompt and its text tokens: 64 ids, the chat-template prefix ends at 34, so 30 reach the transformer
# (mflux 0.20.0 TokenizerLoader on both base snapshots, 2026-10-08).
PROMPT = (
    "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the "
    "water"
)


def _hits(files):
    return sorted(f for f in files if any(fnmatch(f, p) for p in BASE_PATTERNS))


@pytest.mark.parametrize("files", [RAW_FILES, TURBO_FILES])
def test_base_patterns_fetch_encoder_vae_and_tokenizer_and_never_a_transformer(files):
    # Bug caught: 26 GB of transformer pulled on a user's first call (mflux's own Raw patterns fetch
    # transformer/*.safetensors, krea2_weight_definition.py:120-124), the root raw.safetensors / turbo.safetensors
    # matched by a loose "*.safetensors", the sample images, or a file of the encoder, the VAE or the tokenizer missed.
    assert _hits(files) == WANTED


def test_the_tokenizer_pattern_fetches_tokenizer_files_only():
    # Bug caught: the tokenizer pattern widened back to the directory (mflux's "tokenizer/**"), fetching whatever a
    # later revision drops there. The listing is the snapshot's three files plus files a later revision could add.
    listing = (
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/extra/tokenizer.json",
        "tokenizer/model.bin",
        "tokenizer/README.md",
        "tokenizer/special_tokens_map.json",
    )
    assert _hits(listing) == [
        "tokenizer/chat_template.jinja",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    assert all(not p.endswith("*") for p in TOKENIZER_PATTERNS)


def test_df11_patterns_fetch_the_one_comfyui_file():
    # Bug caught: the config-less repository resolved without its only weight file, or with its README.
    # mingyi456/Krea-2-Raw-DF11-ComfyUI @ 8320616b holds these three files (Hub, 2026-10-08).
    files = (".gitattributes", "README.md", "krea2_raw_bf16-DF11.safetensors")
    assert [f for f in files if any(fnmatch(f, p) for p in DF11_PATTERNS)] == [
        "krea2_raw_bf16-DF11.safetensors"
    ]


def _gated(status=403):
    """huggingface_hub's GatedRepoError with the httpx response its __init__ requires."""
    from huggingface_hub.errors import GatedRepoError

    request = httpx.Request("GET", "https://huggingface.co/api/models/krea/Krea-2-Raw")
    return GatedRepoError("gated", response=httpx.Response(status, request=request))


def test_a_gated_base_without_access_names_the_licence_page(monkeypatch):
    # Bug caught: a user without the licence accepted sees a bare Hub traceback or a generic "on the Hub", not the
    # page to accept it on (both bases are gated, auto-approved).
    from mlx_dfloat.mflux.krea2 import init

    def refuse(**_kwargs):
        raise _gated()

    monkeypatch.setattr(_hub, "_snapshot_download", refuse)
    with pytest.raises(DFloatAccessError) as err:
        init.resolve("krea/Krea-2-Raw", patterns=BASE_PATTERNS)
    assert str(err.value) == (
        "krea/Krea-2-Raw: this repository is gated or this account may not read it; accept the licence at "
        "https://huggingface.co/krea/Krea-2-Raw and run `hf auth login`"
    )


def test_hub_errors_that_are_not_access_refusals_pass_through_unchanged(monkeypatch):
    # Bug caught: every Hub failure turned into a licence message (an offline cache miss or a typo'd repo id would
    # send the user to accept a licence). resolve is the one path that names the licence page.
    from mlx_dfloat.mflux.krea2 import init

    other = ValueError("offline")

    def offline(**_kwargs):
        raise other

    monkeypatch.setattr(_hub, "_snapshot_download", offline)
    with pytest.raises(ValueError, match="offline"):
        init.resolve("krea/Krea-2-Raw", patterns=BASE_PATTERNS)


# --- mflux lane -----------------------------------------------------------------------------------------------


@pytest.mark.mflux
def test_the_definitions_name_only_their_component_and_the_tokenizer_subdir():
    # Bug caught: a subset keeping the transformer component (mflux's loader would read raw.safetensors or
    # transformer/), the encoder losing its prefix transform (the vision tower and every language-model tensor would
    # miss), or the tokenizer looked up anywhere but tokenizer/ (krea2_weight_definition.py:96-108).
    from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

    from mlx_dfloat.mflux.krea2.init import encoder_definition, vae_definition

    (encoder,) = encoder_definition().get_components()
    assert encoder.name == "text_encoder"
    assert encoder.key_transform is Krea2WeightDefinition.strip_te_prefix
    assert [c.name for c in vae_definition().get_components()] == ["vae"]
    (tokenizer,) = encoder_definition().get_tokenizers()
    assert (tokenizer.name, tokenizer.hf_subdir, tokenizer.max_length) == (
        "qwen3vl",
        "tokenizer",
        1024,
    )


# A two-layer Qwen3-VL text stack (tiny widths; mflux 0.20.0 Krea2TextEncoder's constructor arguments).
TINY_ENCODER = {
    "vocab_size": 100,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "head_dim": 32,
}


@pytest.mark.mflux
def test_the_computed_exemptions_are_exactly_what_the_base_files_do_not_hold():
    # Bug caught: a computed buffer missing from COMPUTED (every real base refused), or a loaded weight listed there (a
    # missing tensor never noticed). mflux 0.20.0: the encoder's rotary_emb.inv_freq is computed in
    # Qwen3TextRotaryEmbedding.__init__ and the base file holds no such tensor (its 713 tensors are language_model.*
    # and visual.*, 2026-10-08 header); every VAE parameter is a target of mflux's VAE mapping.
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
    from mflux.models.krea2.weights.krea2_weight_mapping import Krea2WeightMapping
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

    from mlx_dfloat.mflux.krea2.init import COMPUTED

    encoder = {n for n, _ in tree_flatten(Krea2TextEncoder(**TINY_ENCODER).parameters())}
    held = {n for n in encoder if n.startswith(("embed_tokens.", "layers.", "norm."))}
    assert encoder - held == {"rotary_emb.inv_freq"}
    targets = [
        re.compile(re.sub(r"\\\{[a-z_]+\\\}", r"[0-9]+", re.escape(t.to_pattern)) + "$")
        for t in Krea2WeightMapping.get_vae_mapping()
    ]
    vae = {n for n, _ in tree_flatten(QwenVAE().parameters())}
    assert {n for n in vae if not any(t.match(n) for t in targets)} == set()
    assert sorted(COMPUTED["text_encoder"]) == ["rotary_emb.inv_freq"]
    assert sorted(COMPUTED["vae"]) == []


def _write_tiny_encoder(root, *, drop=()):
    """The tiny encoder under the base file's names (``language_model.`` + mflux's path), plus a vision tower tensor
    as the real file carries (``visual.*``, which mflux's key transform drops), in one file.

    ``rotary_emb.inv_freq`` is left out: no base holds it (computed at construction).
    """
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    params = dict(tree_flatten(Krea2TextEncoder(**TINY_ENCODER).parameters()))
    flat = {
        f"language_model.{n}": mx.full(p.shape, 0.5, dtype=mx.bfloat16)
        for n, p in params.items()
        if n != "rotary_emb.inv_freq" and f"language_model.{n}" not in drop
    }
    flat["visual.blocks.0.attn.qkv.weight"] = mx.zeros((4, 4), dtype=mx.bfloat16)
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(root / "text_encoder" / "model.safetensors"), flat)


@pytest.mark.mflux
def test_a_missing_encoder_tensor_is_refused(tmp_path):
    # Bug caught: mflux's applier updates with strict=False, so a base missing one tensor (a lost
    # shard, a renamed key) loads without error and that layer keeps its random float32 init: a wrong image, no error.
    from mlx_dfloat.mflux.krea2.init import load_text_encoder

    _write_tiny_encoder(tmp_path, drop={"language_model.layers.1.self_attn.q_proj.weight"})
    with pytest.raises(
        DFloatFormatError, match=r"layers\.1\.self_attn\.q_proj\.weight not in the base"
    ):
        load_text_encoder(tmp_path, dims=TINY_ENCODER)


@pytest.mark.mflux
def test_a_complete_encoder_loads_from_the_language_model_tensors_only(tmp_path):
    # Bug caught: the coverage check demanding rotary_emb.inv_freq (every real base refused), the vision tower in the
    # same file breaking the load, or the file's weights not landing (still the float32 init).
    from mlx_dfloat.mflux.krea2.init import load_text_encoder

    _write_tiny_encoder(tmp_path)
    encoder = load_text_encoder(tmp_path, dims=TINY_ENCODER)
    q = encoder.layers[1].self_attn.q_proj.weight
    assert q.dtype == mx.bfloat16
    assert float(q[0, 0]) == 0.5


@pytest.mark.mflux
def test_a_base_with_another_encoder_width_is_refused_before_any_encode(monkeypatch, tmp_path):
    # Bug caught: Qwen-Image 2.1's base (a Qwen3-VL encoder 4096 wide) passed as --base: mflux's applier assigns its
    # weights without a shape check, so the failure would be a matmul error at the first encode, not a refusal.
    from mlx_dfloat.mflux.krea2.init import load_text_encoder

    def wrong_width(root, definition, models, *, what):
        models["text_encoder"].embed_tokens.weight = mx.zeros((151936, 4096), dtype=mx.bfloat16)

    monkeypatch.setattr(_hub, "load_into", wrong_width)
    with pytest.raises(
        DFloatFormatError,
        match=r"embed_tokens\.weight has shape \(151936, 4096\), this model needs \(151936, 2560\)",
    ):
        load_text_encoder(tmp_path)


def _replace_all_but(component, missing):
    """A ``load_into`` that replaces every parameter of ``component`` but ``missing`` with lazy zeros (no allocation)."""

    def fake(root, definition, models, *, what):
        module = models[component]
        flat = dict(tree_flatten(module.parameters()))
        new = {n: mx.zeros(p.shape, dtype=mx.bfloat16) for n, p in flat.items() if n != missing}
        module.update(tree_unflatten(list(new.items())), strict=False)

    return fake


@pytest.mark.mflux
def test_a_missing_vae_tensor_is_refused(monkeypatch, tmp_path):
    # Bug caught: the VAE loaded through the same strict=False applier with no coverage check: a lost
    # decoder conv keeps its random init and the decode returns noise.
    from mlx_dfloat.mflux.krea2.init import load_vae

    missing = "decoder.conv_out.conv3d.weight"
    monkeypatch.setattr(_hub, "load_into", _replace_all_but("vae", missing))
    with pytest.raises(DFloatFormatError, match=re.escape(missing) + " not in the base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_a_complete_vae_loads_as_qwen_image_1s_vae(monkeypatch, tmp_path):
    # Bug caught: a complete VAE refused, or Krea 2 given Qwen-Image 2.1's VAE class (a different module: Krea 2 uses
    # Qwen-Image 1's QwenVAE, krea2.py:19 and krea2_weight_mapping.py get_vae_mapping).
    from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

    from mlx_dfloat.mflux.krea2.init import load_vae

    monkeypatch.setattr(_hub, "load_into", _replace_all_but("vae", None))
    vae = load_vae(tmp_path)
    assert type(vae) is QwenVAE
    assert vae.decoder.conv_out.conv3d.weight.dtype == mx.bfloat16


@pytest.mark.mflux
def test_a_directory_without_the_text_encoder_or_the_vae_is_a_format_error(tmp_path):
    # Bug caught: a DF11 checkpoint passed as --base crashing inside mflux's loader with a bare FileNotFoundError.
    from mlx_dfloat.mflux.krea2.init import load_text_encoder, load_vae

    with pytest.raises(DFloatFormatError, match="Krea 2 base"):
        load_text_encoder(tmp_path)
    with pytest.raises(DFloatFormatError, match="Krea 2 base"):
        load_vae(tmp_path)


@pytest.mark.mflux
def test_missing_tokenizer_files_are_a_format_error(tmp_path):
    # Bug caught: an empty base directory surfacing mflux's own tokenizer error instead of the package's.
    from mlx_dfloat.mflux.krea2.init import load_tokenizers

    with pytest.raises(DFloatFormatError, match="no usable Krea 2 tokenizer"):
        load_tokenizers(tmp_path)


@pytest.mark.mflux
def test_a_partial_tokenizer_directory_is_a_format_error(monkeypatch, tmp_path):
    # Bug caught: transformers' plain OSError for a tokenizer/ missing one of its files escaping as a raw error
    # instead of the package's format error naming the base.
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader

    from mlx_dfloat.mflux.krea2.init import load_tokenizers

    def partial(**_kwargs):
        raise OSError("Can't load tokenizer for 'tokenizer'")

    monkeypatch.setattr(TokenizerLoader, "load_all", staticmethod(partial))
    with pytest.raises(
        DFloatFormatError, match=r"no usable Krea 2 tokenizer.*Can't load tokenizer"
    ):
        load_tokenizers(tmp_path)


def test_load_base_returns_the_three_components(monkeypatch, tmp_path):
    # Bug caught: load_base wiring the encoder into the vae slot (or the reverse), or the tokenizers dropped.
    from mlx_dfloat.mflux.krea2 import init as kinit

    monkeypatch.setattr(kinit, "load_vae", lambda root: ("vae", root))
    monkeypatch.setattr(kinit, "load_text_encoder", lambda root: ("encoder", root))
    monkeypatch.setattr(kinit, "load_tokenizers", lambda root: {"qwen3vl": root})
    parts = kinit.load_base(tmp_path)
    assert (parts.vae, parts.text_encoder, parts.tokenizers) == (
        ("vae", tmp_path),
        ("encoder", tmp_path),
        {"qwen3vl": tmp_path},
    )


@pytest.mark.mflux
@pytest.mark.network
def test_the_real_tokenizer_loads_from_the_pattern_files_alone(tmp_path):
    # Bug caught: a narrowed tokenizer pattern that leaves out a file mflux's TokenizerLoader (transformers'
    # AutoTokenizer) needs, so a fresh download loads no tokenizer; or the template not applied (the 34-token prefix
    # mflux strips would be missing). Links only the snapshot's tokenizer/ files a pattern selects. Marked network as
    # the Qwen-Image 2.1 twin is, but it reads only a local base snapshot (MLX_DFLOAT_KREA_BASE): no download, no
    # Hub request (it passes with HF_HUB_OFFLINE=1).
    base = os.environ.get("MLX_DFLOAT_KREA_BASE")
    if not base:
        pytest.skip("MLX_DFLOAT_KREA_BASE is not set (a Krea 2 base snapshot with tokenizer/)")
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    from mlx_dfloat.mflux.krea2.init import load_tokenizers

    (tmp_path / "tokenizer").mkdir()
    for path in sorted(Path(base, "tokenizer").iterdir()):
        name = f"tokenizer/{path.name}"
        if any(fnmatch(name, p) for p in BASE_PATTERNS):
            (tmp_path / name).symlink_to(path.resolve())
    linked = sorted(f"tokenizer/{p.name}" for p in (tmp_path / "tokenizer").iterdir())
    assert linked == [
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    tokenizer = load_tokenizers(tmp_path)["qwen3vl"]
    assert tokenizer.tokenize("a").input_ids.shape == (1, 40)
    ids = tokenizer.tokenize(PROMPT).input_ids
    assert (int(ids.shape[1]), Krea2TextEncoder._template_end(ids)) == (64, 34)
