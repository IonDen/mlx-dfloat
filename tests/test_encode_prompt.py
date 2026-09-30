"""The pure parts of the prompt encoder: shapes per model, synthetic embeddings, the metadata."""

import mlx.core as mx
import pytest
from scripts.encode_prompt import (
    DEFAULT_PROMPT,
    build_metadata,
    embed_shapes,
    synthetic_embeds,
    t5_max_length,
)


@pytest.mark.parametrize(
    ("model", "want"),
    [("schnell", ((1, 256, 4096), (1, 768))), ("dev", ((1, 512, 4096), (1, 768)))],
)
def test_embed_shapes_follow_the_models_t5_length(model, want):
    # Bug caught: dev encoded at 256 tokens (mflux's dev max_sequence_length is 512).
    assert embed_shapes(model) == want


@pytest.mark.parametrize(("model", "want"), [("schnell", 256), ("dev", 512)])
def test_t5_max_length_per_model(model, want):
    assert t5_max_length(model) == want


def test_synthetic_embeds_have_the_models_shapes_and_are_seeded():
    # Bug caught: one key used for both arrays (identical rows), or an unseeded draw (two runs of
    # the encoder would give the bench two different inputs).
    prompt, pooled = synthetic_embeds("dev", seed=42)
    again, _pooled_again = synthetic_embeds("dev", seed=42)
    other, _ = synthetic_embeds("dev", seed=43)
    assert prompt.shape == (1, 512, 4096)
    assert pooled.shape == (1, 768)
    assert mx.array_equal(prompt, again)
    assert not mx.array_equal(prompt, other)
    assert not mx.array_equal(prompt[0, 0, :768], pooled[0])


def test_metadata_values_are_all_strings_and_record_the_run():
    # Bug caught: an int seed or bool synthetic in the metadata; mx.save_safetensors only accepts
    # str -> str and would raise at the very end of a multi-minute encoder run.
    md = build_metadata(
        model="schnell", seed=42, prompt=DEFAULT_PROMPT, root="/r", token_length=256, synthetic=True
    )
    assert all(isinstance(v, str) for v in md.values())
    assert md["seed"] == "42"
    assert md["synthetic"] == "true"
    assert md["prompt"] == DEFAULT_PROMPT
    assert md["token_length"] == "256"
    assert "mflux" in md


@pytest.mark.parametrize(
    ("root", "synthetic", "want"),
    [
        (
            "/hub/models--x/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9",
            False,
            "741f7c3ce8b383c54771c7003378a50191e9efe9",
        ),
        ("synthetic", True, "synthetic"),
    ],
)
def test_metadata_records_the_base_revision_from_the_root_dir_name(root, synthetic, want):
    # Bug caught: no base_revision (the scenario orchestrator could not tell embeddings from another
    # encoder snapshot apart and would reuse them), or the whole root path recorded in its place.
    md = build_metadata(
        model="schnell", seed=42, prompt="p", root=root, token_length=256, synthetic=synthetic
    )
    assert md["base_revision"] == want
