import re

import mlx.core as mx
import mlx.nn as nn
import pytest
from tests._flux_fakes import (
    EXPECTED_PATHS,
    FF,
    FLUX_TABLE,
    D,
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
    # Bug caught: a missing or wrong rename (ff.net.0.proj must become ff.linear1, not ff.net.0.proj).
    block, attr = expected
    assert FLUX_TABLE.place(matrix_name) == Placement(block=block, attr=attr)


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("transformer_blocks.4.attn.to_z.weight", "not a transformer_blocks matrix the map covers"),
        (
            "single_transformer_blocks.1.ff.net.2.weight",  # a double-block sub-path on a single block
            "not a single_transformer_blocks matrix the map covers",
        ),
        (
            "single_transformer_blocks.1.ff.linear1.weight",  # the module's name, not the checkpoint's
            "not a single_transformer_blocks matrix the map covers",
        ),
        ("transformer_blocks.4.attn.to_q.bias", "is not a block matrix name"),
        ("x_embedder.weight", "is not a block matrix name"),
        ("transformer_blocks.x.attn.to_q.weight", "is not a block matrix name"),
        (
            "transformer_blocks.\u0663.attn.to_q.weight",
            "is not a block matrix name",
        ),  # Arabic-Indic 3
    ],
)
def test_static_map_refuses_names_outside_the_block_matrix_grammar(name, reason):
    # Bug caught: an unknown or non-matrix name (a bias, a non-block name, an unmapped sub-path, a
    # non-ASCII digit that int() still reads as 3) passing through as itself and landing on the wrong
    # attribute or block instead of being refused with the name in the message.
    with pytest.raises(DFloatIntegrationError, match=re.escape(repr(name)) + ".*" + reason):
        FLUX_TABLE.place(name)


def test_static_map_kind_and_attrs():
    # Bug caught: the table content drifting (a single-block matrix dropped, renamed or duplicated in
    # the table the fakes build), or `kind_of` accepting a non-block name (a norm_out weight, a
    # non-ASCII index) as if it named a block. The order below is the table's own; nothing consumes it.
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
    for name in ("norm_out.linear", "transformer_blocks.\u00b2", "transformer_blocks.\u0663"):
        with pytest.raises(DFloatIntegrationError, match="not a block"):
            FLUX_TABLE.kind_of(name)


@pytest.mark.parametrize(
    ("df11_name", "expected"),
    [
        ("transformer_blocks.3.ff.net.0.proj.bias", "transformer_blocks.3.ff.linear1.bias"),
        (
            "transformer_blocks.3.ff_context.net.2.bias",
            "transformer_blocks.3.ff_context.linear2.bias",
        ),
        ("transformer_blocks.3.attn.to_out.0.bias", "transformer_blocks.3.attn.to_out.0.bias"),
        ("transformer_blocks.3.attn.norm_q.weight", "transformer_blocks.3.attn.norm_q.weight"),
        (
            "transformer_blocks.3.attn.norm_added_q.weight",
            "transformer_blocks.3.attn.norm_added_q.weight",
        ),
        (
            "single_transformer_blocks.0.norm.linear.bias",
            "single_transformer_blocks.0.norm.linear.bias",
        ),
        ("norm_out.linear.weight", "norm_out.linear.weight"),
        (
            "time_text_embed.timestep_embedder.linear_1.bias",
            "time_text_embed.timestep_embedder.linear_1.bias",
        ),
        ("x_embedder.bias", "x_embedder.bias"),
        # A non-ASCII digit is not a block index: the name is not renamed onto block 3.
        (
            "transformer_blocks.\u0663.ff.net.0.proj.bias",
            "transformer_blocks.\u0663.ff.net.0.proj.bias",
        ),
    ],
)
def test_static_map_renames_only_the_mapped_block_extras(df11_name, expected):
    # Bug caught: a block bias that keeps its DF11 name would be dropped by load_weights(strict=False)
    # and leave the module's random init in place; a non-block name "renamed" by mistake would go
    # missing the same way; a Unicode-digit index read by int() would override block 3's extra.
    assert FLUX_TABLE.param_name(df11_name) == expected


def test_get_and_set_attr_path_walk_list_indices():
    # Bug caught: a digit component looked up with getattr instead of indexing, so `attn.to_out.0`
    # raises instead of indexing into the list.
    block = FakeDoubleBlock(Recorder())
    assert get_attr_path(block, "attn.to_out.0") is block.attn.to_out[0]
    new = nn.Linear(4, 4)
    set_attr_path(block, "attn.to_out.0", new)
    assert block.attn.to_out[0] is new
    # Bug caught: a non-digit leaf assigned onto the wrong parent (the head split one level too
    # shallow), leaving the block's own attribute holding its old value.
    plain = nn.Linear(4, 4)
    set_attr_path(block, "attn.to_q", plain)
    assert block.attn.to_q is plain
    with pytest.raises(DFloatIntegrationError, match="no 'nope'"):
        get_attr_path(block, "attn.nope")
    # Bug caught: an out-of-range list index escaping as a bare IndexError instead of the package
    # error naming the path.
    with pytest.raises(DFloatIntegrationError, match=r"attn\.to_out\.7"):
        get_attr_path(block, "attn.to_out.7")
    # Bug caught: a non-ASCII digit component reaching int() (a bare ValueError for "\u00b2") instead
    # of the package error.
    with pytest.raises(DFloatIntegrationError, match="no '\u00b2'"):
        get_attr_path(block, "attn.to_out.\u00b2")


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
    assert shapes["transformer_blocks.0"]["norm1.linear"] == (6 * D, D)
    assert shapes["transformer_blocks.1"]["ff.linear1"] == (FF, D)
    assert shapes["transformer_blocks.1"]["ff.linear2"] == (D, FF)
    assert shapes["single_transformer_blocks.0"]["proj_out"] == (D, D + FF)
    assert len(shapes["transformer_blocks.0"]) == 14
    assert len(shapes["single_transformer_blocks.0"]) == 6
    weights = list(all_block_weights(tf))
    assert len(weights) == 2 * 14 + 6
    assert all(w.size == 0 and w.dtype == mx.bfloat16 for w in weights)
    # Biases and norm scales are extras loaded from the checkpoint, never placeholders.
    assert tf.transformer_blocks[0].attn.to_q.bias.shape == (D,)
    assert tf.transformer_blocks[0].attn.norm_q.weight.shape == (D,)


def test_install_placeholders_fails_on_a_linear_the_map_does_not_cover():
    # Bug caught: an mflux layer added or renamed keeps a real weight resident per block.
    tf = FakeTransformer(Recorder(), n_double=0, n_single=1)
    tf.single_transformer_blocks[0].extra = nn.Linear(4, 4)
    with pytest.raises(
        DFloatIntegrationError, match=r"Linear layers the map does not cover: \['extra'\]"
    ):
        install_placeholders(block_lists(tf), FLUX_TABLE)


def test_install_placeholders_fails_when_a_mapped_matrix_is_missing():
    # Bug caught: a renamed or removed attribute (say ff.linear2 -> ff.out) going unnoticed until the
    # first step crashes inside a matmul instead of being refused up front.
    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    del tf.transformer_blocks[0].ff.linear2
    with pytest.raises(DFloatIntegrationError, match=r"missing from the block: \['ff.linear2'\]"):
        install_placeholders(block_lists(tf), FLUX_TABLE)


def test_placeholder_is_a_zero_size_bf16_array():
    # Bug caught: a placeholder of the wrong dtype promoting a block's matmul to float32, or a
    # non-zero-size placeholder keeping real memory resident before any weight is assigned.
    assert PLACEHOLDER.size == 0
    assert PLACEHOLDER.dtype == mx.bfloat16
