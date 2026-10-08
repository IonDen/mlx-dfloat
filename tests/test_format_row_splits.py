"""Fused matrices: the split plan, the inserted split points, and a decode cut at the seam (offline)."""

import dataclasses

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.decode import decode_group, split_matrices
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import (
    insert_row_splits,
    load_group_mx,
    open_checkpoint,
    parse_df11_config,
    row_split_plan,
    with_row_splits,
)

# S0 (2026-10-07): every Qwen-Image 2.1 block's stored split_positions and element count.
QWEN_SPLITS = np.array(
    [16_777_216, 33_554_432, 50_331_648, 67_108_864, 167_772_160], dtype=np.int64
)
QWEN_BLOCK_ELEMENTS = 218_103_808
QWEN_RAW = {
    "version": "0.3.1",
    "threads_per_block": [512],
    "bytes_per_thread": 8,
    "pattern_dict": {
        r"modulation\.1": [],
        r"transformer_blocks\.\d+": [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.gate_up",
            "img_mlp.out",
        ],
    },
}
FUSED = {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}


def test_the_gate_up_segment_is_cut_in_half_at_the_s0_seam():
    # Bug caught: the seam taken from the wrong segment (index off by one) or not halved. S0 measured the proj rows
    # (gate_up row 12288) starting at element 117,440,512.
    got = insert_row_splits(
        QWEN_SPLITS, QWEN_BLOCK_ELEMENTS, ((4, 2),), name="transformer_blocks.0"
    )
    assert got.tolist() == [
        16_777_216,
        33_554_432,
        50_331_648,
        67_108_864,
        117_440_512,
        167_772_160,
    ]
    assert got.dtype == np.int64


def test_two_fused_segments_are_cut_from_the_original_bounds():
    # Bug caught: the first insert shifting the second segment's index. Bounds 0|4|10|16|25: segment 1 = [4, 10) in
    # 2 parts -> 7; segment 3 = [16, 25) in 3 parts -> 19, 22.
    got = insert_row_splits(np.array([4, 10, 16], dtype=np.int64), 25, ((1, 2), (3, 3)), name="g")
    assert got.tolist() == [4, 7, 10, 16, 19, 22]


def test_the_first_and_the_last_segment_can_be_split():
    # Bug caught: the bounds missing 0 or n_elements (the first or last matrix unsplittable).
    assert insert_row_splits(np.array([6], dtype=np.int64), 10, ((0, 2),), name="g").tolist() == [
        3,
        6,
    ]
    assert insert_row_splits(np.array([6], dtype=np.int64), 10, ((1, 2),), name="g").tolist() == [
        6,
        8,
    ]
    assert insert_row_splits(np.array([], dtype=np.int64), 8, ((0, 2),), name="g").tolist() == [4]


def test_an_empty_plan_returns_the_stored_splits_unchanged():
    # Bug caught: every config.json checkpoint (no fused matrices) getting its splits rewritten.
    got = insert_row_splits(QWEN_SPLITS, QWEN_BLOCK_ELEMENTS, (), name="g")
    assert got.tolist() == QWEN_SPLITS.tolist()


@pytest.mark.parametrize(
    ("splits", "n", "plan", "match"),
    [
        ([5], 10, ((0, 2),), "5 elements"),  # [0, 5) is not divisible by 2
        ([4], 10, ((2, 2),), "segment 2"),  # only segments 0 and 1 exist
        ([4], 10, ((0, 1),), "2 parts"),  # one part is no split
        ([4], 10, ((0, 2), (0, 2)), "twice"),  # the same segment planned twice
        ([4], 10, ((-1, 2),), "segment -1"),  # a negative index must not wrap to the last segment
        ([4], 10, ((-2, 2),), "segment -2"),  # wrapped, -2 would cut the real segment [4, 10) at 7
    ],
)
def test_an_unsplittable_segment_is_a_format_error(splits, n, plan, match):
    # Bug caught: an odd segment cut at a truncated midpoint (one matrix a row short, the next a row long), a seam
    # outside every stored matrix, a duplicate seam (a zero-length matrix), or a negative index read from the end.
    with pytest.raises(DFloatFormatError, match=match):
        insert_row_splits(np.array(splits, dtype=np.int64), n, plan, name="g")


@pytest.mark.parametrize(
    ("splits", "n"),
    [
        ([10, 4], 16),  # a reversed pair outside the fused segment [0, 10)
        ([4, 4], 16),  # a repeated point (a zero-length matrix)
        ([4, 16], 16),  # a point on the upper bound
    ],
)
def test_a_disordered_stored_split_table_is_refused_before_planning(splits, n):
    # Bug caught: the planned cuts sorted together with the stored points, which turns a corrupt stored table
    # ([10, 4] -> [4, 5, 10]) into a valid-looking one that decodes matrices under the wrong names.
    with pytest.raises(DFloatFormatError, match=rf"strictly increasing inside \(0, {n}\)"):
        insert_row_splits(np.array(splits, dtype=np.int64), n, ((0, 2),), name="g")


def test_the_plan_expands_gate_up_into_gate_layer_then_proj():
    # Bug caught: the halves swapped (S0: rows 0..12287 are gate_layer), or the fused name left in the list.
    config = with_row_splits(parse_df11_config(QWEN_RAW, source="t"), FUSED, source="t")
    stored, names, plan = row_split_plan("transformer_blocks.7", config)
    head = "transformer_blocks.7."
    assert stored == tuple(
        head + s + ".weight"
        for s in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.gate_up",
            "img_mlp.out",
        )
    )
    assert names == tuple(
        head + s + ".weight"
        for s in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.gate_layer",
            "img_mlp.proj",
            "img_mlp.out",
        )
    )
    assert plan == ((4, 2),)
    assert row_split_plan("modulation.1", config) == (
        ("modulation.1.weight",),
        ("modulation.1.weight",),
        (),
    )


def test_a_config_without_row_splits_plans_nothing():
    # Bug caught: a config.json checkpoint (no row_splits) getting a plan or renamed matrices.
    config = parse_df11_config(QWEN_RAW, source="t")
    stored, names, plan = row_split_plan("transformer_blocks.0", config)
    assert names == stored
    assert plan == ()


def test_a_downloaded_config_never_carries_row_splits():
    # Bug caught: a crafted config.json declaring its own split (the reader would cut real matrices anywhere).
    raw = dict(QWEN_RAW, row_splits={"img_mlp.gate_up": ["a", "b"]})
    assert parse_df11_config(raw, source="t").row_splits == {}


@pytest.mark.parametrize(
    ("splits", "match"),
    [
        ({"img_mlp.nope": ("a", "b")}, "img_mlp.nope"),  # not a sub of any pattern
        ({"img_mlp.gate_up": ("only",)}, "2 parts"),
        ({"img_mlp.gate_up": ("img_mlp.out", "x")}, "img_mlp.out"),  # collides with a stored sub
        ({"img_mlp.gate_up": ("a", "a")}, "'a'"),
        ({"img_mlp.gate_up": ("../x", "y")}, r"\.\./x"),
        ({"img_mlp.gate_up": ("a.", "y")}, r"'a\.'"),  # an empty name segment
    ],
)
def test_a_bad_row_split_table_is_refused(splits, match):
    # Bug caught: a layout table typo producing duplicate or path-like parameter names.
    with pytest.raises(DFloatFormatError, match=match):
        with_row_splits(parse_df11_config(QWEN_RAW, source="t"), splits, source="t")


def test_the_split_table_cannot_be_changed_after_validation():
    # Bug caught: a caller editing config.row_splits (or the input mapping) after with_row_splits checked it, so an
    # unchecked part name or a missing split reaches the decoder.
    table = {"img_mlp.gate_up": ["img_mlp.gate_layer", "img_mlp.proj"]}
    config = with_row_splits(parse_df11_config(QWEN_RAW, source="t"), table, source="t")
    with pytest.raises(TypeError):
        config.row_splits["img_mlp.out"] = ("../x", "y")  # type: ignore[index]
    table["img_mlp.gate_up"].reverse()
    del table["img_mlp.gate_up"]
    assert config.row_splits == {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}
    with pytest.raises(TypeError):
        parse_df11_config(QWEN_RAW, source="t").row_splits["x"] = ("a", "b")  # type: ignore[index]


def test_a_fused_sub_listed_by_two_patterns_is_refused():
    # Bug caught: one table entry silently splitting matrices in two group families.
    raw = {**QWEN_RAW, "pattern_dict": {r"a\.\d+": ["gate_up"], r"b\.\d+": ["gate_up", "out"]}}
    with pytest.raises(DFloatFormatError, match="more than one pattern"):
        with_row_splits(parse_df11_config(raw, source="t"), {"gate_up": ("g", "u")}, source="t")


def _fused_tiny(tmp_path):
    rng = np.random.default_rng(5)
    q, g, u, o = (random_bf16(rng, s) for s in [(4, 4), (6, 4), (6, 4), (4, 6)])
    write_checkpoint(
        tmp_path,
        groups={"blocks.0": [q, np.concatenate([g, u]), o]},
        pattern=r"blocks\.\d+",
        sub_paths=("q", "gate_up", "out"),
        single_file=True,
    )
    group = open_checkpoint(tmp_path).groups["blocks.0"]
    names = (
        "blocks.0.q.weight",
        "blocks.0.gate.weight",
        "blocks.0.up.weight",
        "blocks.0.out.weight",
    )
    return group, names, (q, g, u, o)


def test_a_group_with_a_row_plan_decodes_into_the_two_original_halves(tmp_path):
    # Bug caught (end to end): the inserted split not reaching load()/to_mx(), so the fused matrix arrives whole.
    # Oracle: G and U are the inputs; the checkpoint stores concat(G, U) as one matrix.
    group, names, originals = _fused_tiny(tmp_path)
    fused = dataclasses.replace(group, matrix_names=names, row_plan=((1, 2),))
    mxg = load_group_mx(fused)
    assert mxg.split_positions == (16, 40, 64)  # 4*4 | 16 + 6*4 | 16 + 12*4
    parts = split_matrices(decode_group(mxg, backend="reference").bits, mxg.split_positions)
    for part, want in zip(parts, originals, strict=True):
        assert np.array_equal(np.array(part).ravel(), want.ravel())


def test_a_group_without_a_row_plan_loads_its_stored_splits(tmp_path):
    # Bug caught: load() rewriting split_positions for every group (an empty plan must leave the stored array).
    group, _names, _originals = _fused_tiny(tmp_path)
    assert group.row_plan == ()
    assert group.load().split_positions.tolist() == [16, 64]  # 4*4 | 16 + 12*4


def test_names_that_disagree_with_the_split_points_are_refused_at_load(tmp_path):
    # Bug caught: expanded names loaded without their row plan, so three stored matrices decode under four names.
    group, names, _originals = _fused_tiny(tmp_path)
    with pytest.raises(DFloatFormatError, match="3 matrices but 4 names"):
        dataclasses.replace(group, matrix_names=names).load()
