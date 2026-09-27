import mlx.core as mx
import mlx.nn as nn
import pytest
from tests._flux_fakes import (
    EXPECTED_PATHS,
    FLUX_TABLE,
    FakeDoubleBlock,
    FakeTransformer,
    Recorder,
    all_block_weights,
    block_lists,
)

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.names import Placement
from mlx_dfloat.integrate.placeholders import (
    PLACEHOLDER,
    get_attr_path,
    install_placeholders,
    set_attr_path,
)


@pytest.mark.parametrize(("matrix_name", "expected"), EXPECTED_PATHS)
def test_static_map_places_every_flux_matrix_name(matrix_name, expected):
    block, attr = expected
    assert FLUX_TABLE.place(matrix_name) == Placement(block=block, attr=attr)


@pytest.mark.parametrize(
    "name",
    [
        "transformer_blocks.4.attn.to_q.bias",
        "x_embedder.weight",
        "transformer_blocks.x.attn.to_q.weight",
        "single_transformer_blocks.1.ff.linear1.weight",
    ],
)
def test_static_map_refuses_names_outside_the_block_matrix_grammar(name):
    with pytest.raises(DFloatIntegrationError):
        FLUX_TABLE.place(name)


def test_static_map_kind_and_attrs():
    assert FLUX_TABLE.kinds == ("transformer_blocks", "single_transformer_blocks")
    assert FLUX_TABLE.kind_of("single_transformer_blocks.37") == "single_transformer_blocks"
    assert FLUX_TABLE.attrs_of("single_transformer_blocks") == (
        "norm.linear",
        "proj_mlp",
        "proj_out",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
    )
    with pytest.raises(DFloatIntegrationError, match="not a block"):
        FLUX_TABLE.kind_of("norm_out.linear")


@pytest.mark.parametrize(
    ("df11_name", "expected"),
    [
        ("transformer_blocks.3.ff.net.0.proj.bias", "transformer_blocks.3.ff.linear1.bias"),
        ("transformer_blocks.3.attn.norm_q.weight", "transformer_blocks.3.attn.norm_q.weight"),
        ("x_embedder.bias", "x_embedder.bias"),
    ],
)
def test_static_map_renames_only_the_mapped_block_extras(df11_name, expected):
    assert FLUX_TABLE.param_name(df11_name) == expected


def test_get_and_set_attr_path_walk_list_indices():
    block = FakeDoubleBlock(Recorder())
    assert get_attr_path(block, "attn.to_out.0") is block.attn.to_out[0]
    new = nn.Linear(4, 4)
    set_attr_path(block, "attn.to_out.0", new)
    assert block.attn.to_out[0] is new
    with pytest.raises(DFloatIntegrationError, match="no 'nope'"):
        get_attr_path(block, "attn.nope")


def test_install_placeholders_replaces_every_block_matrix_and_records_its_shape():
    # Bug caught: a matrix left resident (its weight stays a real array), or a shape recorded
    # transposed (the provider would reshape into (in, out)).
    tf = FakeTransformer(Recorder(), n_double=2, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    assert list(shapes) == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "single_transformer_blocks.0",
    ]
    assert shapes["transformer_blocks.0"]["norm1.linear"] == (24, 4)
    assert shapes["single_transformer_blocks.0"]["proj_out"] == (4, 12)
    assert all(w.size == 0 for w in all_block_weights(tf))
    assert (
        tf.transformer_blocks[0].attn.norm_q.weight.size == 4
    )  # a non-matrix parameter is untouched


def test_install_placeholders_fails_on_a_linear_the_map_does_not_cover():
    # Bug caught: an mflux layer added or renamed keeps a real weight resident per block.
    tf = FakeTransformer(Recorder(), n_double=0, n_single=1)
    tf.single_transformer_blocks[0].extra = nn.Linear(4, 4)
    with pytest.raises(
        DFloatIntegrationError, match=r"Linear layers the map does not cover: \['extra'\]"
    ):
        install_placeholders(block_lists(tf), FLUX_TABLE)


def test_install_placeholders_fails_when_a_mapped_matrix_is_missing():
    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    del tf.transformer_blocks[0].ff.linear2
    with pytest.raises(DFloatIntegrationError, match=r"missing from the block: \['ff.linear2'\]"):
        install_placeholders(block_lists(tf), FLUX_TABLE)


def test_placeholder_is_a_zero_size_bf16_array():
    assert PLACEHOLDER.size == 0
    assert PLACEHOLDER.dtype == mx.bfloat16
