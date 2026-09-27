"""Fake FLUX-shaped modules and the static name map used by the offline (non-mflux) tests.

The fake transformer mirrors mflux 0.20.0's ``Transformer`` hooks exactly (keyword-only calls to
``_apply_joint_transformer_block`` and the single-block twin), so the seam mixin is exercised the way
mflux will drive it. No test here imports mflux.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.integrate.placeholders import get_attr_path
from mlx_dfloat.mflux.flux1.transformer import SeamMixin

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
    """mflux's Transformer shape: two block lists, plain loops, the two `_apply_*` hooks.

    The hook signatures are mflux 0.20.0's exactly (no ``**kwargs``;
    mflux/models/flux/model/flux_transformer/transformer.py:82 and :105).
    """

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


def inputs():
    x = mx.ones((1, 2, D), dtype=mx.bfloat16)
    return x, x, x, x  # hidden, encoder hidden, text embeddings, rotary embeddings


def resident_dicts(shapes):
    """One constant-filled bf16 matrix per mapped path, a different constant per block."""
    return {
        name: {attr: mx.full(shape, float(i + 1), dtype=mx.bfloat16) for attr, shape in per.items()}
        for i, (name, per) in enumerate(shapes.items())
    }


def all_block_weights(tf):
    """Every mapped block matrix weight of the fake, through the maps under test."""
    lists = (
        ("transformer_blocks", tf.transformer_blocks, DOUBLE_SUBS),
        ("single_transformer_blocks", tf.single_transformer_blocks, SINGLE_SUBS),
    )
    for prefix, blocks, subs in lists:
        for block in blocks:
            for sub in subs:
                yield get_attr_path(block, FLUX_TABLE.place(f"{prefix}.0.{sub}.weight").attr).weight


def df11_groups(shapes, rng):
    """Compress one real DF11 group per block (all 14 or 6 matrices, in pattern_dict order)."""
    groups, names, source = {}, {}, {}
    for block_name, per in shapes.items():
        subs = DOUBLE_SUBS if block_name.startswith("transformer_blocks.") else SINGLE_SUBS
        matrix_names = tuple(f"{block_name}.{sub}.weight" for sub in subs)
        mats = [random_bf16(rng, per[FLUX_TABLE.place(m).attr]) for m in matrix_names]
        flat = np.concatenate([m.reshape(-1) for m in mats])
        splits = np.cumsum([m.size for m in mats])[:-1].tolist()
        groups[block_name] = encoder_group(flat, *splits).to_mx(name=block_name)
        names[block_name] = matrix_names
        source[block_name] = dict(zip(matrix_names, mats, strict=True))
    return groups, names, source


def write_flux_checkpoint(
    root, shapes, rng, *, double_pattern=r"transformer_blocks\.\d+", extras=None
):
    """A real DF11 checkpoint of one group per block of ``shapes`` (FLUX sub-paths, pattern_dict order).

    ``double_pattern`` is the double-block pattern's spelling (the 0.2.0 configs leave its dot
    unescaped); the single-block pattern is always the escaped one. Returns the matrices per group.
    """
    groups = {}
    for block_name, per in shapes.items():
        subs = DOUBLE_SUBS if block_name.startswith("transformer_blocks.") else SINGLE_SUBS
        groups[block_name] = [
            random_bf16(rng, per[FLUX_TABLE.place(f"{block_name}.{sub}.weight").attr])
            for sub in subs
        ]
    write_checkpoint(
        root,
        groups=groups,
        patterns={double_pattern: DOUBLE_SUBS, r"single_transformer_blocks\.\d+": SINGLE_SUBS},
        extras=extras,
    )
    return groups


# --- the static name map used by the offline tests -------------------------------------------------

DOUBLE_MAP = {
    "norm1.linear": "norm1.linear",
    "norm1_context.linear": "norm1_context.linear",
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.add_k_proj": "attn.add_k_proj",
    "attn.add_v_proj": "attn.add_v_proj",
    "attn.add_q_proj": "attn.add_q_proj",
    "attn.to_out.0": "attn.to_out.0",
    "attn.to_add_out": "attn.to_add_out",
    "ff.net.0.proj": "ff.linear1",
    "ff.net.2": "ff.linear2",
    "ff_context.net.0.proj": "ff_context.linear1",
    "ff_context.net.2": "ff_context.linear2",
}
SINGLE_MAP = {
    "norm.linear": "norm.linear",
    "proj_mlp": "proj_mlp",
    "proj_out": "proj_out",
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
}
FLUX_TABLE = StaticNameMap(
    {"transformer_blocks": DOUBLE_MAP, "single_transformer_blocks": SINGLE_MAP}
)


def block_lists(tf):
    """The two block lists of a FLUX-shaped transformer, in step order."""
    return [
        ("transformer_blocks", tf.transformer_blocks),
        ("single_transformer_blocks", tf.single_transformer_blocks),
    ]
