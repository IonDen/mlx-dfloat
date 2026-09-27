"""Coverage bookkeeping: the resident set, the extras plan, coverage checks, full-block decode."""

import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint
from tests._flux_fakes import FF, FLUX_TABLE, D, FakeTransformer, Recorder, block_lists, df11_groups

from mlx_dfloat.decode import STATUS_INVALID_CODE, decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import MxGroup, open_checkpoint
from mlx_dfloat.integrate import seam
from mlx_dfloat.integrate.coverage import (
    check_extras_cover,
    decode_resident,
    extras_plan,
    load_resident_set,
    read_extra,
)
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.integrate.providers import DF11Provider


def _shapes(n_double, n_single):
    tf = FakeTransformer(Recorder(), n_double=n_double, n_single=n_single)
    return install_placeholders(block_lists(tf), FLUX_TABLE)


# --- decode_resident -------------------------------------------------------------------------------


def test_decode_resident_evaluates_each_block_before_decoding_the_next(monkeypatch):
    # Bug caught: one lazy eval over every block's decode (all decoded groups allocated at once: the
    # run-ahead the per-block eval policy exists to prevent), a block decoded twice, or the deferred
    # status words never checked.
    shapes = _shapes(2, 1)
    groups, names, source = df11_groups(shapes, np.random.default_rng(12))
    calls, evals = [], []

    def counting_decode(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    real_eval = seam._eval
    monkeypatch.setattr(seam, "_eval", lambda x: (evals.append(("eval", len(calls))), real_eval(x)))
    provider = DF11Provider(groups, names, FLUX_TABLE, decode=counting_decode)
    per_block = decode_resident(provider, shapes)
    # The k-th eval happens after exactly k decodes: block k is evaluated before block k+1 is decoded.
    assert evals == [("eval", 1), ("eval", 2), ("eval", 3)]
    assert calls == list(shapes)
    assert provider.pending == []
    for block_name in shapes:
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert np.array_equal(np.array(per_block[block_name]["attn.to_q"].view(mx.uint16)), want)


def test_decode_resident_raises_on_a_flagged_block():
    # Bug caught: resident dicts built from a decode whose status reports an error, with nobody
    # reading it (decode_resident must call verify() after the last block).
    shapes = _shapes(1, 0)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(13))
    bad_status = mx.array([STATUS_INVALID_CODE], dtype=mx.uint32)
    provider = DF11Provider(
        groups,
        names,
        FLUX_TABLE,
        decode=lambda g: dataclasses.replace(
            decode_group(g, backend="reference"), status=bad_status
        ),
    )
    with pytest.raises(DFloatFormatError, match=r"transformer_blocks\.0: block 0: invalid code"):
        decode_resident(provider, shapes)


# --- resident set and extras ------------------------------------------------------------------------

PATTERN = r"transformer_blocks\.\d+"
SUBS = ("attn.to_q", "ff.net.0.proj")


def _ckpt(tmp_path, extras=None):
    rng = np.random.default_rng(0)
    groups = {
        f"transformer_blocks.{i}": [random_bf16(rng, (D, D)), random_bf16(rng, (FF, D))]
        for i in range(2)
    }
    root = write_checkpoint(
        tmp_path / "ckpt", groups=groups, pattern=PATTERN, sub_paths=SUBS, extras=extras
    )
    return open_checkpoint(root), groups


def test_load_resident_set_loads_every_group_evaluated_under_its_name(tmp_path):
    # Bug caught: a group stored under another block's name, or split positions lost on the way.
    ckpt, groups = _ckpt(tmp_path)
    resident = load_resident_set(ckpt)
    assert list(resident) == ["transformer_blocks.0", "transformer_blocks.1"]
    g = resident["transformer_blocks.1"]
    assert isinstance(g, MxGroup)
    assert g.name == "transformer_blocks.1"
    assert g.split_positions == (D * D,)
    assert g.n_elements == D * D + FF * D
    bits = np.array(decode_group(g, backend="reference").bits)
    assert np.array_equal(bits[: D * D], groups["transformer_blocks.1"][0].reshape(-1))


def test_load_resident_set_filters_by_name_and_refuses_an_unknown_one(tmp_path):
    # Bug caught: names= not actually filtering the resident set, or an unknown name silently
    # ignored instead of refused.
    ckpt, _ = _ckpt(tmp_path)
    assert list(load_resident_set(ckpt, names=["transformer_blocks.1"])) == ["transformer_blocks.1"]
    with pytest.raises(DFloatIntegrationError, match=r"single_transformer_blocks\.0"):
        load_resident_set(ckpt, names=["single_transformer_blocks.0"])


def test_extras_plan_renames_block_extras_skips_out_of_range_blocks_and_the_dropped_bias(tmp_path):
    # Bug caught: an extra of a block at or beyond the requested count planned for a block that does
    # not exist (with load_weights(strict=False) it would vanish silently), or a dropped name kept.
    rng = np.random.default_rng(1)
    extras = {
        "transformer_blocks.0.ff.net.0.proj.bias": random_bf16(rng, (FF,)),
        "transformer_blocks.0.attn.norm_q.weight": random_bf16(rng, (D,)),
        "transformer_blocks.1.attn.to_q.bias": random_bf16(rng, (D,)),
        "norm_out.linear.bias": random_bf16(rng, (2 * D,)),
        "x_embedder.weight": random_bf16(rng, (D, 3)),
    }
    ckpt, _ = _ckpt(tmp_path, extras=extras)
    plan = extras_plan(
        ckpt,
        FLUX_TABLE,
        counts={"transformer_blocks": 1, "single_transformer_blocks": 0},
        dropped=frozenset({"norm_out.linear.bias"}),
    )
    assert sorted(name for name, _path, _info in plan) == [
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.ff.linear1.bias",
        "x_embedder.weight",
    ]
    loaded = {name: read_extra(path, info) for name, path, info in plan}
    assert loaded["x_embedder.weight"].dtype == mx.bfloat16
    assert loaded["x_embedder.weight"].shape == (D, 3)
    assert np.array_equal(
        np.array(loaded["transformer_blocks.0.ff.linear1.bias"].view(mx.uint16)),
        extras["transformer_blocks.0.ff.net.0.proj.bias"],
    )
    wider = extras_plan(
        ckpt, FLUX_TABLE, counts={"transformer_blocks": 2, "single_transformer_blocks": 0}
    )
    assert [n for n, _p, _i in wider].count("transformer_blocks.1.attn.to_q.bias") == 1


def test_check_extras_cover_passes_only_when_every_non_matrix_parameter_has_an_extra():
    params = {
        "x_embedder.weight",
        "x_embedder.bias",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_q.bias",
    }
    matrices = {"transformer_blocks.0.attn.to_q.weight"}
    check_extras_cover(
        params,
        {"x_embedder.weight", "x_embedder.bias", "transformer_blocks.0.attn.to_q.bias"},
        matrices,
    )
    # Bug caught: a parameter nobody sets keeping the module's random init.
    with pytest.raises(DFloatIntegrationError, match=r"x_embedder\.bias"):
        check_extras_cover(
            params, {"x_embedder.weight", "transformer_blocks.0.attn.to_q.bias"}, matrices
        )
    # Bug caught: an extra with no target (a rename gone wrong) silently dropped by
    # load_weights(strict=False).
    with pytest.raises(DFloatIntegrationError, match=r"norm_out\.linear\.bias"):
        check_extras_cover(
            params,
            {
                "x_embedder.weight",
                "x_embedder.bias",
                "transformer_blocks.0.attn.to_q.bias",
                "norm_out.linear.bias",
            },
            matrices,
        )
