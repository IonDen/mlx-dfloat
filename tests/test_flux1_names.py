"""FLUX.1 naming: the map derived from mflux's own weight mapping, and the group-kind check."""

from types import SimpleNamespace

import pytest

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import matrix_names_for
from mlx_dfloat.mflux.flux1.names import DROPPED_EXTRAS, check_flux_groups


def _group(name: str) -> SimpleNamespace:
    """A checkpoint-shaped group with the given name and FLUX-correct matrix count."""
    n = 14 if name.startswith("transformer_blocks.") else 6
    return SimpleNamespace(name=name, matrix_names=tuple(f"{name}.m{i}.weight" for i in range(n)))


def _ckpt(names: list[str]) -> SimpleNamespace:
    """A checkpoint-shaped object with the given group names and FLUX matrix counts."""
    return SimpleNamespace(groups={n: _group(n) for n in names})


def test_flux_groups_counts_contiguous_blocks_of_both_kinds():
    # Bug caught: miscounting the double or single family (e.g. counting groups instead of the max
    # index + 1, or swapping which count goes with which kind).
    ckpt = _ckpt(["transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0"])
    assert check_flux_groups(ckpt) == (2, 1)


@pytest.mark.parametrize(
    ("names", "reason"),
    [
        (
            ["transformer_blocks.0", "transformer_blocks.2", "single_transformer_blocks.0"],
            "not contiguous",
        ),
        (["transformer_blocks.0"], "no single_transformer_blocks"),
        (
            ["transformer_blocks.0", "single_transformer_blocks.0", "model.layers.0"],
            "not a FLUX block",
        ),
    ],
)
def test_flux_groups_refuses_holes_missing_kinds_and_foreign_groups(names, reason):
    # Bug caught: a hole in the block index sequence accepted as if the download were complete, a
    # missing block kind treated as zero valid blocks instead of a refusal, or a group from a
    # different model's checkpoint (or a stray non-block tensor) passed straight through.
    with pytest.raises(DFloatFormatError, match=reason):
        check_flux_groups(_ckpt(names))


def test_flux_groups_refuses_a_group_with_the_wrong_matrix_count():
    # Bug caught: a group with the wrong pattern_dict subs count (a different FLUX variant, or a
    # corrupted/truncated config) treated as an ordinary block instead of refused.
    bad = SimpleNamespace(
        groups={
            "transformer_blocks.0": SimpleNamespace(
                name="transformer_blocks.0", matrix_names=("a.weight",) * 13
            ),
            "single_transformer_blocks.0": SimpleNamespace(
                name="single_transformer_blocks.0", matrix_names=("a.weight",) * 6
            ),
        }
    )
    with pytest.raises(DFloatFormatError, match="13 matrices"):
        check_flux_groups(bad)


def test_dropped_extras_is_the_one_bias_mflux_has_no_parameter_for():
    # Bug caught: DROPPED_EXTRAS growing to include a bias mflux actually loads, silently dropping
    # a real parameter from the build.
    assert frozenset({"norm_out.linear.bias"}) == DROPPED_EXTRAS


@pytest.mark.parametrize("pattern", [r"transformer_blocks.\d+", r"transformer_blocks\.\d+"])
def test_both_spellings_of_the_block_pattern_parse_to_the_same_groups(pattern):
    # Bug caught: an adapter comparing a config's pattern string against one hard-coded spelling.
    # schnell/dev's DF11 0.2.0 config (version 0.2.0, verified against the local snapshots) leaves
    # the block pattern's dot unescaped (`transformer_blocks.\d+`); Krea-dev's 0.3.1 config escapes
    # it (`transformer_blocks\.\d+`). Both must parse to the same matrix names and the same group
    # counts through check_flux_groups, which never looks at the pattern at all.
    pattern_dict = {pattern: ("attn.to_q",), r"single_transformer_blocks\.\d+": ("attn.to_q",)}
    assert matrix_names_for("transformer_blocks.0", pattern_dict) == (
        "transformer_blocks.0.attn.to_q.weight",
    )
    ckpt = _ckpt(["transformer_blocks.0", "single_transformer_blocks.0"])
    assert check_flux_groups(ckpt) == (1, 1)


@pytest.mark.mflux
def test_flux_name_map_derived_from_mflux_equals_the_measured_table():
    from tests._flux_fakes import EXPECTED_PATHS, FLUX_TABLE

    from mlx_dfloat.integrate.names import Placement
    from mlx_dfloat.mflux.flux1.names import flux_name_map

    derived = flux_name_map()
    assert derived.kinds == FLUX_TABLE.kinds
    for kind in FLUX_TABLE.kinds:
        # Bug caught: a matrix missing from (or a norm scale leaking into) the derived table for this
        # kind. attrs_of's own order is the map's construction order (mflux's WeightTarget declaration
        # order), not the checkpoint's concatenation order, so this compares as sets — length-equal too,
        # so a duplicate can't hide a missing name — and leaves the real per-matrix check to place()
        # below.
        derived_attrs, table_attrs = derived.attrs_of(kind), FLUX_TABLE.attrs_of(kind)
        assert len(derived_attrs) == len(table_attrs)
        assert set(derived_attrs) == set(table_attrs)
    for matrix_name, (block, attr) in EXPECTED_PATHS:
        # Bug caught: a matrix name mapped to the wrong block or the wrong attribute path.
        assert derived.place(matrix_name) == Placement(block=block, attr=attr)
