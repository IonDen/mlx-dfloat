"""Name maps derived from an mflux WeightMapping, on stand-in targets (no mflux)."""

from types import SimpleNamespace

import pytest

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.mflux._mapping import name_map_from_mapping


def T(to, *frm, transform=None):  # noqa: N802
    return SimpleNamespace(to_pattern=to, from_pattern=list(frm), transform=transform)


SUBS = {"layers": ("attention.to_q", "feed_forward.w1")}
BASE = [
    T("layers.{layer}.attention.to_q.weight", "layers.{layer}.attention.to_q.weight"),
    T("layers.{layer}.feed_forward.w1.weight", "layers.{layer}.feed_forward.w1.weight"),
    T("layers.{layer}.attention.norm_q.weight", "layers.{layer}.attention.norm_q.weight"),
]


def test_tables_hold_exactly_the_compressed_subs_and_renames_the_non_block_targets():
    # Bug caught: a norm scale put in the matrix table (install_placeholders would refuse the real block), or a
    # non-block rename (t_embedder.mlp.0 -> linear1) dropped.
    m = name_map_from_mapping(
        [*BASE, T("t_embedder.linear1.weight", "t_embedder.mlp.0.weight")], matrix_subs=SUBS
    )
    assert m.attrs_of("layers") == ("attention.to_q", "feed_forward.w1")
    assert m.param_name("t_embedder.mlp.0.weight") == "t_embedder.linear1.weight"


def test_a_compressed_sub_with_no_mflux_target_is_refused():
    # Bug caught: a DF11 matrix with nowhere to land, found only at the first matmul.
    with pytest.raises(DFloatIntegrationError, match=r"feed_forward\.w2"):
        name_map_from_mapping(BASE, matrix_subs={"layers": ("attention.to_q", "feed_forward.w2")})


def test_a_transform_on_a_block_matrix_is_refused():
    # Bug caught: a transformed block matrix decoded and installed raw (block matrices are pure renames).
    bad = T(
        "layers.{layer}.attention.to_q.weight",
        "layers.{layer}.attention.to_q.weight",
        transform=lambda a: a,
    )
    with pytest.raises(DFloatIntegrationError, match="not a pure rename"):
        name_map_from_mapping([bad, *BASE[1:]], matrix_subs=SUBS)


def test_a_block_matrix_built_from_several_sources_is_refused():
    # Bug caught: a fused or split block matrix (several checkpoint tensors into one weight) taken for a rename.
    fused = T(
        "layers.{layer}.attention.to_q.weight",
        "layers.{layer}.attention.to_q.weight",
        "layers.{layer}.attention.q2.weight",
    )
    with pytest.raises(DFloatIntegrationError, match="not a pure rename"):
        name_map_from_mapping([fused, *BASE[1:]], matrix_subs={"layers": ("attention.to_q",)})


def test_a_transform_on_a_block_extra_is_refused():
    # Bug caught: a transformed block extra (a norm scale) loaded untransformed.
    bad = T(
        "layers.{layer}.attention.norm_q.weight",
        "layers.{layer}.attention.norm_q.weight",
        transform=lambda a: a,
    )
    with pytest.raises(DFloatIntegrationError, match="transform"):
        name_map_from_mapping([*BASE[:2], bad], matrix_subs=SUBS)


def test_a_non_block_transform_is_recorded_under_the_target_name():
    # Bug caught: the transform recorded under the checkpoint name (load_extras looks it up by parameter).
    def t(a):
        return a

    m = name_map_from_mapping(
        [*BASE, T("x_embedder.weight", "x_embedder.proj.weight", transform=t)], matrix_subs=SUBS
    )
    assert m.transform_of("x_embedder.weight") is t
    assert m.transform_of("x_embedder.proj.weight") is None


def test_one_source_mapped_to_two_targets_is_refused():
    # Bug caught: one-to-many targets (mflux's T5 relative bias) silently keeping only the last.
    with pytest.raises(DFloatIntegrationError, match="two targets"):
        name_map_from_mapping(
            [*BASE, T("a.weight", "s.weight"), T("b.weight", "s.weight")], matrix_subs=SUBS
        )


def test_a_derived_placement_that_differs_from_mflux_is_refused():
    # Bug caught: a block rename the tables cannot express (here a norm moved to another path) loaded under the
    # checkpoint's own name instead of mflux's.
    with pytest.raises(DFloatIntegrationError, match="would load it into"):
        name_map_from_mapping(
            [*BASE, T("layers.{layer}.qnorm.weight", "layers.{layer}.attention.norm_k.weight")],
            matrix_subs=SUBS,
        )


def test_a_block_matrix_mflux_moves_out_of_its_block_list_is_refused_by_name():
    # Bug caught: `next(...)` without a default when mflux maps a block matrix to a target outside the same block
    # list (here layers.* -> blocks.*): a bare StopIteration instead of a package error naming the target.
    moved = T("blocks.{layer}.attention.to_q.weight", "layers.{layer}.attention.to_q.weight")
    with pytest.raises(DFloatIntegrationError, match=r"blocks\.\{layer\}\.attention\.to_q\.weight"):
        name_map_from_mapping([moved, *BASE[1:]], matrix_subs=SUBS)
