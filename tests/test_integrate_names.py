"""Non-block renames and extra transforms in the static name map."""

import mlx.core as mx
import pytest

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.names import StaticNameMap

TABLES = {
    "layers": {"attention.to_q": "attention.to_q", "adaLN_modulation.0": "adaLN_modulation.0"}
}


def test_non_block_names_follow_the_renames_and_block_extras_follow_the_tables():
    # Bug caught: a non-block rename never applied (Z-Image's t_embedder.mlp.0.weight would load nowhere and the
    # coverage would refuse it), or the renames shadowing a block extra's table placement.
    m = StaticNameMap(TABLES, renames={"t_embedder.mlp.0.weight": "t_embedder.linear1.weight"})
    assert m.param_name("t_embedder.mlp.0.weight") == "t_embedder.linear1.weight"
    assert m.param_name("layers.3.adaLN_modulation.0.bias") == "layers.3.adaLN_modulation.0.bias"
    assert m.param_name("layers.3.attention.norm_q.weight") == "layers.3.attention.norm_q.weight"
    assert m.param_name("x_pad_token") == "x_pad_token"


def test_a_rename_whose_source_is_a_block_name_is_refused():
    # Bug caught: a rename silently overriding the block table for one block only.
    with pytest.raises(DFloatIntegrationError, match="block"):
        StaticNameMap(TABLES, renames={"layers.0.attention.to_q.weight": "x.weight"})


def test_transform_of_is_keyed_by_the_parameter_name():
    # Bug caught: the transform looked up by the checkpoint name (a renamed extra would never be transformed).
    t = mx.transpose
    m = StaticNameMap(TABLES, renames={"a.weight": "b.weight"}, transforms={"b.weight": t})
    assert m.transform_of("b.weight") is t
    assert m.transform_of("a.weight") is None
