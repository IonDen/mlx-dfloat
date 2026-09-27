"""The FLUX rig core on fakes: name maps, placeholders, the block seam, providers, eval policies.

The fake transformer mirrors mflux 0.20.0's `Transformer` hooks exactly (keyword-only calls to
`_apply_joint_transformer_block(idx, block, hidden_states, encoder_hidden_states, text_embeddings,
image_rotary_embeddings, controlnet_block_samples)` and the single-block twin), so the seam mixin is
exercised the way mflux will drive it. No test here imports mflux.
"""

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from scripts import _flux_rig as rig
from scripts._flux_rig import (
    DF11Provider,
    ResidentProvider,
    ReuseProvider,
    RigError,
    SeamMixin,
    check_extras_cover,
    extras_plan,
    get_attr_path,
    install_placeholders,
    load_resident_set,
    mflux_param_name,
    mflux_path,
    set_attr_path,
)
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.decode import STATUS_INVALID_CODE, decode_group
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import MxGroup, open_checkpoint

D, FF = 4, 8  # hidden width and feed-forward width of the fakes

# The 20 DF11 matrix names of FLUX.1 (pattern_dict order) and their mflux 0.20.0 attribute paths,
# copied from the run-record tables, not derived from the maps under test.
DOUBLE_SUBS = (
    "norm1.linear",
    "norm1_context.linear",
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_out.0",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)
SINGLE_SUBS = ("norm.linear", "proj_mlp", "proj_out", "attn.to_q", "attn.to_k", "attn.to_v")
EXPECTED_PATHS = [
    ("transformer_blocks.4.norm1.linear.weight", ("transformer_blocks.4", "norm1.linear")),
    (
        "transformer_blocks.4.norm1_context.linear.weight",
        ("transformer_blocks.4", "norm1_context.linear"),
    ),
    ("transformer_blocks.4.attn.to_q.weight", ("transformer_blocks.4", "attn.to_q")),
    ("transformer_blocks.4.attn.to_k.weight", ("transformer_blocks.4", "attn.to_k")),
    ("transformer_blocks.4.attn.to_v.weight", ("transformer_blocks.4", "attn.to_v")),
    ("transformer_blocks.4.attn.add_k_proj.weight", ("transformer_blocks.4", "attn.add_k_proj")),
    ("transformer_blocks.4.attn.add_v_proj.weight", ("transformer_blocks.4", "attn.add_v_proj")),
    ("transformer_blocks.4.attn.add_q_proj.weight", ("transformer_blocks.4", "attn.add_q_proj")),
    ("transformer_blocks.4.attn.to_out.0.weight", ("transformer_blocks.4", "attn.to_out.0")),
    ("transformer_blocks.4.attn.to_add_out.weight", ("transformer_blocks.4", "attn.to_add_out")),
    ("transformer_blocks.4.ff.net.0.proj.weight", ("transformer_blocks.4", "ff.linear1")),
    ("transformer_blocks.4.ff.net.2.weight", ("transformer_blocks.4", "ff.linear2")),
    (
        "transformer_blocks.4.ff_context.net.0.proj.weight",
        ("transformer_blocks.4", "ff_context.linear1"),
    ),
    (
        "transformer_blocks.4.ff_context.net.2.weight",
        ("transformer_blocks.4", "ff_context.linear2"),
    ),
    (
        "single_transformer_blocks.37.norm.linear.weight",
        ("single_transformer_blocks.37", "norm.linear"),
    ),
    ("single_transformer_blocks.37.proj_mlp.weight", ("single_transformer_blocks.37", "proj_mlp")),
    ("single_transformer_blocks.37.proj_out.weight", ("single_transformer_blocks.37", "proj_out")),
    (
        "single_transformer_blocks.37.attn.to_q.weight",
        ("single_transformer_blocks.37", "attn.to_q"),
    ),
    (
        "single_transformer_blocks.37.attn.to_k.weight",
        ("single_transformer_blocks.37", "attn.to_k"),
    ),
    (
        "single_transformer_blocks.37.attn.to_v.weight",
        ("single_transformer_blocks.37", "attn.to_v"),
    ),
]


# --- fakes ----------------------------------------------------------------------------------------


class Recorder:
    """Plain object (not a dict/list/array), so an nn.Module keeps it out of its parameter tree."""

    def __init__(self):
        self.seen = []  # the `attn.to_q.weight` array each block saw while it ran
        self.events = []  # ("run" | "eval" | "async", id of the object the block returned)
        self.keep = []  # every recorded object, so no id can be reused by a later one


class _Sub(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def _attn(*, joint):
    layers = {
        "to_q": nn.Linear(D, D),
        "to_k": nn.Linear(D, D),
        "to_v": nn.Linear(D, D),
        "norm_q": nn.RMSNorm(D),  # a non-Linear parameter the placeholder step must leave alone
    }
    if joint:
        layers |= {
            "add_k_proj": nn.Linear(D, D),
            "add_v_proj": nn.Linear(D, D),
            "add_q_proj": nn.Linear(D, D),
            "to_out": [nn.Linear(D, D)],
            "to_add_out": nn.Linear(D, D),
        }
    return _Sub(**layers)


class FakeDoubleBlock(nn.Module):
    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self.norm1 = _Sub(linear=nn.Linear(D, 6 * D))
        self.norm1_context = _Sub(linear=nn.Linear(D, 6 * D))
        self.attn = _attn(joint=True)
        self.ff = _Sub(linear1=nn.Linear(D, FF), linear2=nn.Linear(FF, D))
        self.ff_context = _Sub(linear1=nn.Linear(D, FF), linear2=nn.Linear(FF, D))

    def __call__(self, hidden_states, encoder_hidden_states, text_embeddings, rotary_embeddings):
        self._recorder.seen.append(self.attn.to_q.weight)
        out = (encoder_hidden_states, self.attn.to_q(hidden_states))
        self._recorder.events.append(("run", id(out)))  # the whole tuple, as mflux returns it
        self._recorder.keep.append(out)
        return out


class FakeSingleBlock(nn.Module):
    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self.norm = _Sub(linear=nn.Linear(D, 3 * D))
        self.attn = _attn(joint=False)
        self.proj_mlp = nn.Linear(D, FF)
        self.proj_out = nn.Linear(D + FF, D)

    def __call__(self, hidden_states, text_embeddings, rotary_embeddings):
        self._recorder.seen.append(self.attn.to_q.weight)
        hidden = self.attn.to_q(hidden_states)
        self._recorder.events.append(("run", id(hidden)))
        self._recorder.keep.append(hidden)
        return hidden


class FakeTransformer(nn.Module):
    """mflux's Transformer shape: two block lists, plain loops, the two `_apply_*` hooks."""

    def __init__(self, recorder, *, n_double, n_single):
        super().__init__()
        self.transformer_blocks = [FakeDoubleBlock(recorder) for _ in range(n_double)]
        self.single_transformer_blocks = [FakeSingleBlock(recorder) for _ in range(n_single)]

    def __call__(
        self, hidden_states, encoder_hidden_states, text_embeddings, image_rotary_embeddings
    ):
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = self._apply_joint_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_block_samples=None,
            )
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = self._apply_single_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_single_block_samples=None,
            )
        return hidden_states

    def _apply_joint_transformer_block(
        self,
        idx,
        block,
        hidden_states,
        encoder_hidden_states,
        text_embeddings,
        image_rotary_embeddings,
        controlnet_block_samples,
    ):
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )

    def _apply_single_transformer_block(
        self,
        idx,
        block,
        hidden_states,
        encoder_hidden_states,
        text_embeddings,
        image_rotary_embeddings,
        controlnet_single_block_samples,
    ):
        return block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )


class FakeSeamTransformer(SeamMixin, FakeTransformer):
    """What `seam_transformer_class()` builds over mflux, composed over the fake instead."""


def _inputs():
    x = mx.ones((1, 2, D), dtype=mx.bfloat16)
    return x, x, x, x  # hidden, encoder hidden, text embeddings, rotary embeddings


def _resident_dicts(shapes):
    """One constant-filled bf16 matrix per mapped path, a different constant per block."""
    return {
        name: {attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()}
        for i, (name, per) in enumerate(shapes.items())
    }


def _all_block_weights(tf):
    """Every mapped block matrix weight of the fake, through the maps under test."""
    lists = (
        ("transformer_blocks", tf.transformer_blocks, DOUBLE_SUBS),
        ("single_transformer_blocks", tf.single_transformer_blocks, SINGLE_SUBS),
    )
    for prefix, blocks, subs in lists:
        for block in blocks:
            for sub in subs:
                yield get_attr_path(block, mflux_path(f"{prefix}.0.{sub}.weight")[1]).weight


def _df11_groups(shapes, rng):
    """Compress one real DF11 group per block (all 14 or 6 matrices, in pattern_dict order)."""
    groups, names, source = {}, {}, {}
    for block_name, per in shapes.items():
        subs = DOUBLE_SUBS if block_name.startswith("transformer_blocks.") else SINGLE_SUBS
        matrix_names = tuple(f"{block_name}.{sub}.weight" for sub in subs)
        mats = [random_bf16(rng, per[mflux_path(m)[1]]) for m in matrix_names]
        flat = np.concatenate([m.reshape(-1) for m in mats])
        splits = np.cumsum([m.size for m in mats])[:-1].tolist()
        groups[block_name] = encoder_group(flat, *splits).to_mx(name=block_name)
        names[block_name] = matrix_names
        source[block_name] = dict(zip(matrix_names, mats, strict=True))
    return groups, names, source


# --- name maps and attribute paths ---------------------------------------------------------------


@pytest.mark.parametrize(("matrix_name", "expected"), EXPECTED_PATHS)
def test_mflux_path_maps_every_flux_matrix_name(matrix_name, expected):
    # Bug caught: a missing or wrong rename (ff.net.0.proj must become ff.linear1, not ff.net.0.proj).
    assert mflux_path(matrix_name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "transformer_blocks.4.attn.to_z.weight",  # unknown sub-path
        "single_transformer_blocks.1.ff.net.2.weight",  # a double-block path on a single block
        "transformer_blocks.4.attn.to_q.bias",  # not a matrix
        "x_embedder.weight",  # not a block
    ],
)
def test_mflux_path_refuses_names_outside_the_block_matrix_grammar(name):
    # Bug caught: an unknown name passing through as itself and landing on a non-existent attribute.
    with pytest.raises(RigError, match=name.split(".weight")[0].split(".bias")[0][-6:]):
        mflux_path(name)


@pytest.mark.parametrize(
    ("df11_name", "expected"),
    [
        ("transformer_blocks.3.ff.net.0.proj.bias", "transformer_blocks.3.ff.linear1.bias"),
        (
            "transformer_blocks.3.ff_context.net.2.bias",
            "transformer_blocks.3.ff_context.linear2.bias",
        ),
        ("transformer_blocks.3.attn.to_out.0.bias", "transformer_blocks.3.attn.to_out.0.bias"),
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
    ],
)
def test_mflux_param_name_renames_only_the_feed_forward_block_extras(df11_name, expected):
    # Bug caught: a block bias that keeps its DF11 name would be dropped by load_weights(strict=False)
    # and leave mflux's random init in place; a non-block name that gets "renamed" would go missing.
    assert mflux_param_name(df11_name) == expected


def test_get_and_set_attr_path_walk_list_indices():
    # Bug caught: `attn.to_out.0` read with getattr on the list instead of indexing it.
    block = FakeDoubleBlock(Recorder())
    linear = get_attr_path(block, "attn.to_out.0")
    assert linear is block.attn.to_out[0]
    set_attr_path(block, "attn.to_out.0.weight", mx.zeros((2, 2)))
    assert block.attn.to_out[0].weight.shape == (2, 2)
    with pytest.raises(RigError, match=r"attn\.to_out\.7"):
        get_attr_path(block, "attn.to_out.7")


# --- the placeholder step --------------------------------------------------------------------------


def test_install_placeholders_replaces_every_block_matrix_and_records_its_shape():
    # Bug caught: one of the 20 mapped matrices left as mflux's lazy random init (it would be
    # evaluated at the first step and silently used instead of the decoded weight).
    tf = FakeTransformer(Recorder(), n_double=2, n_single=1)
    shapes = install_placeholders(tf)
    assert list(shapes) == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "single_transformer_blocks.0",
    ]
    assert shapes["transformer_blocks.1"]["ff.linear1"] == (FF, D)
    assert shapes["transformer_blocks.1"]["ff.linear2"] == (D, FF)
    assert shapes["single_transformer_blocks.0"]["proj_out"] == (D, D + FF)
    assert len(shapes["transformer_blocks.0"]) == 14
    assert len(shapes["single_transformer_blocks.0"]) == 6
    weights = list(_all_block_weights(tf))
    assert len(weights) == 2 * 14 + 6
    assert all(w.size == 0 and w.dtype == mx.bfloat16 for w in weights)
    # Biases and norm weights are extras loaded from the checkpoint, never placeholders.
    assert tf.transformer_blocks[0].attn.to_q.bias.shape == (D,)
    assert tf.transformer_blocks[0].attn.norm_q.weight.shape == (D,)


def test_install_placeholders_fails_when_a_mapped_matrix_is_missing():
    # Bug caught: a renamed or removed mflux attribute (say ff.linear2 -> ff.out) going unnoticed
    # until the first step crashes inside a matmul.
    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    del tf.transformer_blocks[0].ff["linear2"]
    with pytest.raises(RigError, match=r"transformer_blocks\.0.*ff\.linear2"):
        install_placeholders(tf)


def test_install_placeholders_fails_on_a_linear_the_maps_do_not_cover():
    # Bug caught: a new mflux Linear inside a block keeping its random init because no DF11 matrix
    # maps to it; the run would be "fine" and wrong.
    tf = FakeTransformer(Recorder(), n_double=0, n_single=1)
    tf.single_transformer_blocks[0].attn.extra = nn.Linear(D, D)
    with pytest.raises(RigError, match=r"single_transformer_blocks\.0.*attn\.extra"):
        install_placeholders(tf)


# --- the seam ---------------------------------------------------------------------------------------


def test_seam_runs_each_block_on_the_provider_weight_and_restores_the_placeholder():
    # Bug caught: the weight assigned to the wrong block or attribute, or the placeholder not
    # restored so decoded weights stay resident across blocks.
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(tf)
    per_block = _resident_dicts(shapes)
    provider = ResidentProvider(per_block)
    tf.attach(provider, shapes, eval_policy="per-block")
    out = tf(*_inputs())
    mx.eval(out)
    assert out.shape == (1, 2, D)
    assert [
        w is per_block[name]["attn.to_q"] for w, name in zip(rec.seen, shapes, strict=True)
    ] == [
        True,
        True,
        True,
    ]
    assert all(w.size == 0 for w in _all_block_weights(tf))
    assert provider.launches == 0


def test_seam_restores_the_placeholder_when_the_block_raises(monkeypatch):
    # Bug caught: a missing try/finally leaving a decoded weight assigned after a failed step, or
    # the failed step's depth2 `prev` (block 0's output, already queued with async_eval when block 1
    # raises) surviving into the next step, where the first block's eval would drain it.
    class BoomOnce(FakeDoubleBlock):
        def __init__(self, recorder):
            super().__init__(recorder)
            self._armed = True

        def __call__(self, **kwargs):
            if self._armed:
                self._armed = False
                self._recorder.seen.append(self.attn.to_q.weight)
                raise RuntimeError("boom")
            return super().__call__(**kwargs)

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=0)
    tf.transformer_blocks[1] = BoomOnce(rec)
    shapes = install_placeholders(tf)
    tf.attach(ResidentProvider(_resident_dicts(shapes)), shapes, eval_policy="depth2")
    _patch_eval_recorders(monkeypatch, rec)
    with pytest.raises(RuntimeError, match="boom"):
        tf(*_inputs())
    assert rec.seen[1].size > 0  # the weight was assigned when the raising block ran
    assert all(w.size == 0 for w in _all_block_weights(tf))
    failed = [key for kind, key in rec.events if kind == "run"]
    # Block 0 ran and was queued; block 1 raised before any eval.
    assert rec.events == [("run", failed[0]), ("async", failed[0])]
    # The second step starts clean: the depth2 sequence of a fresh step, and the failed step's queued
    # block-0 output is never evaluated (a surviving `prev` would put ("eval", failed[0]) first).
    tf(*_inputs())
    o = [key for kind, key in rec.events[2:] if kind == "run"]
    assert rec.events[2:] == [
        ("run", o[0]),
        ("async", o[0]),
        ("run", o[1]),
        ("async", o[1]),
        ("eval", o[0]),
        ("eval", o[1]),
    ]
    assert ("eval", failed[0]) not in rec.events


def test_seam_refuses_a_provider_dict_that_does_not_match_the_block():
    # Bug caught: a provider returning a subset (or a typo'd key) leaving a placeholder in place;
    # without the check the matmul would raise a shape error naming nothing useful.
    class Partial:
        launches = 0
        launching = False
        groups_evaluated = True

        def weights_for(self, block_name, shapes):
            return {k: mx.zeros(v, dtype=mx.bfloat16) for k, v in list(shapes.items())[:-1]}

        def verify(self):
            pass

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(tf)
    tf.attach(Partial(), shapes)
    with pytest.raises(RigError, match=r"transformer_blocks\.0.*ff_context\.linear2"):
        tf(*_inputs())
    assert all(w.size == 0 for w in _all_block_weights(tf))


def test_seam_refuses_an_unknown_eval_policy():
    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(tf)
    with pytest.raises(RigError, match="depth3"):
        tf.attach(ResidentProvider(_resident_dicts(shapes)), shapes, eval_policy="depth3")


class _Launching:
    """A provider that decodes (launches grow), like DF11Provider, without any groups."""

    launches = 0
    launching = True
    groups_evaluated = True

    def weights_for(self, block_name, shapes):
        self.launches += 1
        return {k: mx.zeros(v, dtype=mx.bfloat16) for k, v in shapes.items()}

    def verify(self):
        pass


def test_none_policy_is_refused_for_a_launching_provider_over_more_than_two_blocks():
    # Bug caught: "none" + DF11Provider over the full model keeps every decoded group alive until
    # the final eval (about 24 GB on FLUX.1); the refusal must fire before any block is decoded.
    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=1)
    shapes = install_placeholders(tf)
    with pytest.raises(RigError, match=r"none.*3 blocks|3 blocks.*none"):
        tf.attach(_Launching(), shapes, eval_policy="none")
    # Two blocks (the reduced-depth validation) and the non-launching control are fine.
    tf.attach(_Launching(), {k: shapes[k] for k in list(shapes)[:2]}, eval_policy="none")
    double = {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["transformer_blocks.0"].items()}
    single = {
        a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["single_transformer_blocks.0"].items()
    }
    tf.attach(ReuseProvider(double, single), shapes, eval_policy="none")
    mx.eval(tf(*_inputs()))


def test_df11_provider_defers_the_status_check_to_verify():
    # Bug caught: a host read of the status inside weights_for (a sync before every block, which
    # is what the deferral exists to remove), or verify() that never reads it / never clears it.
    class Unread:
        """A status that fails the moment anything tries to read it."""

        def __array__(self, *args, **kwargs):
            raise AssertionError("status was read")

    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(tf)
    groups, names, _source = _df11_groups(shapes, np.random.default_rng(9))
    unread = DF11Provider(
        groups,
        names,
        decode=lambda g: dataclasses.replace(decode_group(g, backend="reference"), status=Unread()),
    )
    unread.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    assert [name for name, _status in unread.pending] == ["transformer_blocks.0"]
    with pytest.raises(AssertionError, match="status was read"):
        unread.verify()

    bad_status = mx.array([STATUS_INVALID_CODE], dtype=mx.uint32)
    corrupt = DF11Provider(
        groups,
        names,
        decode=lambda g: dataclasses.replace(
            decode_group(g, backend="reference"), status=bad_status
        ),
    )
    corrupt.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    with pytest.raises(DFloatFormatError, match=r"transformer_blocks\.0: block 0: invalid code"):
        corrupt.verify()
    assert corrupt.pending == []
    corrupt.verify()  # nothing pending: a no-op, not a repeat of the error


def test_df11_provider_decodes_each_block_once_into_bit_exact_views():
    # Bug caught: a matrix cut at the wrong split, reshaped in the wrong order, or a block decoded
    # twice (launches would exceed the block count).
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(tf)
    groups, names, source = _df11_groups(shapes, np.random.default_rng(7))
    calls = []

    def counting_decode(group: MxGroup):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    provider = DF11Provider(groups, names, decode=counting_decode)
    assert provider.groups_evaluated is True
    assert provider.launching is True
    tf.attach(provider, shapes, eval_policy="per-block")
    mx.eval(tf(*_inputs()))
    assert [name for name, _status in provider.pending] == list(shapes)
    tf.verify_step()  # every status is clean; the seam's hook drains the pending list
    assert provider.pending == []
    assert provider.launches == 3
    assert calls == ["transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0"]
    for seen, block_name in zip(rec.seen, shapes, strict=True):
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert seen.dtype == mx.bfloat16
        assert seen.shape == want.shape
        assert np.array_equal(np.array(seen.view(mx.uint16)), want)
    assert all(w.size == 0 for w in _all_block_weights(tf))
    # Every matrix of a block, not only to_q: check the odd shapes through weights_for directly.
    w = provider.weights_for("transformer_blocks.1", shapes["transformer_blocks.1"])
    assert provider.launches == 4
    for sub in DOUBLE_SUBS:
        attr = mflux_path(f"transformer_blocks.1.{sub}.weight")[1]
        want = source["transformer_blocks.1"][f"transformer_blocks.1.{sub}.weight"]
        assert w[attr].shape == want.shape
        assert np.array_equal(np.array(w[attr].view(mx.uint16)), want)


@pytest.mark.metal
def test_df11_provider_with_the_metal_backend_feeds_the_seam_bit_exact_weights():
    # Bug caught: a Metal decode whose lazy views the seam mishandles (evaluated after the placeholder is
    # restored, or cut at the wrong split), or a status word the step never checks; the reference-backed
    # tests above cannot see either, since only this path launches the kernel through the seam.
    from functools import partial

    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=1)
    shapes = install_placeholders(tf)
    groups, names, source = _df11_groups(shapes, np.random.default_rng(11))
    provider = DF11Provider(groups, names, decode=partial(decode_group, backend="metal"))
    tf.attach(provider, shapes, eval_policy="per-block")
    mx.eval(tf(*_inputs()))
    tf.verify_step()
    assert provider.launches == 3
    assert provider.pending == []
    for seen, block_name in zip(rec.seen, shapes, strict=True):
        want = source[block_name][f"{block_name}.attn.to_q.weight"]
        assert seen.shape == want.shape
        assert np.array_equal(np.array(seen.view(mx.uint16)), want)
    assert all(w.size == 0 for w in _all_block_weights(tf))


def test_df11_provider_refuses_a_matrix_whose_size_does_not_match_the_shape():
    # Bug caught: a checkpoint/model mismatch (wrong n_single, a different FLUX variant) surfacing
    # as a reshape error inside MLX instead of a named refusal.
    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(tf)
    groups, names, _source = _df11_groups(shapes, np.random.default_rng(8))
    wrong = dict(shapes["transformer_blocks.0"])
    wrong["ff.linear1"] = (FF, D + 1)
    provider = DF11Provider(groups, names, decode=lambda g: decode_group(g, backend="reference"))
    with pytest.raises(
        RigError, match=r"ff\.net\.0\.proj\.weight: decoded 32 elements, expected \(8, 5\) = 40"
    ):
        provider.weights_for("transformer_blocks.0", wrong)


def test_df11_provider_names_a_block_it_does_not_hold():
    provider = DF11Provider({}, {}, decode=lambda g: decode_group(g, backend="reference"))
    with pytest.raises(RigError, match=r"transformer_blocks\.5"):
        provider.weights_for("transformer_blocks.5", {})


def test_reuse_and_resident_providers_never_launch_and_return_matching_shapes():
    # Bug caught: a control provider that decodes (launches > 0 would make the control a DF11 run)
    # or returns the double dict for a single block.
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=2, n_single=2)
    shapes = install_placeholders(tf)
    double = {
        a: mx.full(s, 2.0, dtype=mx.bfloat16) for a, s in shapes["transformer_blocks.0"].items()
    }
    single = {
        a: mx.full(s, 3.0, dtype=mx.bfloat16)
        for a, s in shapes["single_transformer_blocks.0"].items()
    }
    reuse = ReuseProvider(double, single)
    tf.attach(reuse, shapes, eval_policy="none")
    mx.eval(tf(*_inputs()))
    assert reuse.launches == 0
    assert [w is double["attn.to_q"] for w in rec.seen[:2]] == [True, True]
    assert [w is single["attn.to_q"] for w in rec.seen[2:]] == [True, True]

    resident = ResidentProvider(_resident_dicts(shapes))
    assert resident.launches == 0
    assert resident.groups_evaluated is True
    with pytest.raises(RigError, match=r"transformer_blocks\.9"):
        resident.weights_for("transformer_blocks.9", shapes["transformer_blocks.0"])


def test_reuse_provider_refuses_a_shape_that_does_not_match_the_block():
    # Bug caught: a reuse dict decoded from another model's block reaching the matmul.
    tf = FakeTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(tf)
    double = {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["transformer_blocks.0"].items()}
    double["ff.linear1"] = mx.zeros((FF, D + 1), dtype=mx.bfloat16)
    with pytest.raises(RigError, match=r"ff\.linear1"):
        ReuseProvider(double, {}).weights_for(
            "transformer_blocks.0", shapes["transformer_blocks.0"]
        )


# --- eval policies -----------------------------------------------------------------------------------


def _patch_eval_recorders(monkeypatch, rec):
    """Record which object the seam hands to mx.eval / mx.async_eval: for a joint block that must
    be the very tuple the block returned, not one of its elements."""
    real_eval, real_async = rig._eval, rig._async_eval

    def eval_(out):
        rec.events.append(("eval", id(out)))
        rec.keep.append(out)
        real_eval(out)

    def async_(out):
        rec.events.append(("async", id(out)))
        rec.keep.append(out)
        real_async(out)

    monkeypatch.setattr(rig, "_eval", eval_)
    monkeypatch.setattr(rig, "_async_eval", async_)


def _run_with_policy(monkeypatch, policy, *, n_double=2, n_single=2):
    rec = Recorder()
    tf = FakeSeamTransformer(rec, n_double=n_double, n_single=n_single)
    shapes = install_placeholders(tf)
    tf.attach(ResidentProvider(_resident_dicts(shapes)), shapes, eval_policy=policy)
    _patch_eval_recorders(monkeypatch, rec)
    out = tf(*_inputs())
    runs = [key for kind, key in rec.events if kind == "run"]
    return rec.events, runs, out


def test_per_block_policy_evaluates_each_block_before_the_next_runs(monkeypatch):
    # Bug caught: evaluating only the last block (all decoded weights resident at once), or
    # evaluating a tuple's second element only.
    events, o, _ = _run_with_policy(monkeypatch, "per-block")
    assert events == [
        ("run", o[0]),
        ("eval", o[0]),
        ("run", o[1]),
        ("eval", o[1]),
        ("run", o[2]),
        ("eval", o[2]),
        ("run", o[3]),
        ("eval", o[3]),
    ]


def test_depth2_policy_evaluates_block_i_minus_1_before_block_i_plus_1_runs(monkeypatch):
    # Bug caught: eval(prev) placed before async_eval(out) (no overlap), or prev never drained.
    events, o, _ = _run_with_policy(monkeypatch, "depth2")
    assert events == [
        ("run", o[0]),
        ("async", o[0]),
        ("run", o[1]),
        ("async", o[1]),
        ("eval", o[0]),
        ("run", o[2]),
        ("async", o[2]),
        ("eval", o[1]),
        ("run", o[3]),
        ("async", o[3]),
        ("eval", o[2]),
        ("eval", o[3]),  # the drain after the last block
    ]


def test_none_policy_never_evaluates(monkeypatch):
    # Bug caught: a stray eval in the seam turning the run-ahead control into a per-block run.
    events, o, out = _run_with_policy(monkeypatch, "none")
    assert events == [("run", k) for k in o]
    mx.eval(out)


# --- resident set and extras ---------------------------------------------------------------------------

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
    ckpt, _ = _ckpt(tmp_path)
    assert list(load_resident_set(ckpt, names=["transformer_blocks.1"])) == ["transformer_blocks.1"]
    with pytest.raises(RigError, match=r"single_transformer_blocks\.0"):
        load_resident_set(ckpt, names=["single_transformer_blocks.0"])


def test_extras_plan_renames_block_extras_skips_out_of_range_blocks_and_the_dropped_bias(tmp_path):
    # Bug caught: an extra of a block beyond n_double planned for a block that does not exist (with
    # load_weights(strict=False) it would vanish silently), or the DF11 feed-forward bias name kept.
    rng = np.random.default_rng(1)
    extras = {
        "transformer_blocks.0.ff.net.0.proj.bias": random_bf16(rng, (FF,)),
        "transformer_blocks.0.attn.norm_q.weight": random_bf16(rng, (D,)),
        "transformer_blocks.1.attn.to_q.bias": random_bf16(rng, (D,)),
        "norm_out.linear.bias": random_bf16(rng, (2 * D,)),
        "x_embedder.weight": random_bf16(rng, (D, 3)),
    }
    ckpt, _ = _ckpt(tmp_path, extras=extras)
    plan = extras_plan(ckpt, n_double=1, n_single=0)
    assert sorted(name for name, _path, _info in plan) == [
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.ff.linear1.bias",
        "x_embedder.weight",
    ]
    loaded = {name: rig.read_extra(path, info) for name, path, info in plan}
    assert loaded["x_embedder.weight"].dtype == mx.bfloat16
    assert loaded["x_embedder.weight"].shape == (D, 3)
    assert np.array_equal(
        np.array(loaded["transformer_blocks.0.ff.linear1.bias"].view(mx.uint16)),
        extras["transformer_blocks.0.ff.net.0.proj.bias"],
    )
    assert [n for n, _p, _i in extras_plan(ckpt, n_double=2, n_single=0)].count(
        "transformer_blocks.1.attn.to_q.bias"
    ) == 1


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
    # Bug caught: a parameter nobody sets keeping mflux's random init.
    with pytest.raises(RigError, match=r"x_embedder\.bias"):
        check_extras_cover(
            params, {"x_embedder.weight", "transformer_blocks.0.attn.to_q.bias"}, matrices
        )
    # Bug caught: an extra with no target (a rename gone wrong) dropped by load_weights(strict=False).
    with pytest.raises(RigError, match=r"norm_out\.linear\.bias"):
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
