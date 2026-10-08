"""Z-Image-shaped fakes for the offline seam tests: three block lists called inline, a non-block embedder between
kinds (mflux 0.20.0 models/z_image/model/z_image_transformer/transformer.py:96-134). No mflux import."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16, write_checkpoint
from tests._flux_fakes import _Sub

from mlx_dfloat.integrate.blockseam import StepSeam
from mlx_dfloat.integrate.names import StaticNameMap

D = 4
KINDS = ("noise_refiner", "context_refiner", "layers")


class FakeRefinerBlock(nn.Module):
    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self.attention = _Sub(to_q=nn.Linear(D, D, bias=False))
        self.feed_forward = _Sub(w2=nn.Linear(D, D, bias=False))
        self.attention_norm1 = nn.RMSNorm(D)
        self.adaLN_modulation = [nn.Linear(D, 4 * D)]

    def __call__(self, x, t_emb):
        self._recorder.seen.append(self.attention.to_q.weight)
        out = x + self.feed_forward.w2(self.attention.to_q(x))
        self._recorder.events.append(("run", id(out)))
        self._recorder.keep.append(out)
        return out


class FakeContextBlock(nn.Module):
    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self.attention = _Sub(to_q=nn.Linear(D, D, bias=False))
        self.feed_forward = _Sub(w2=nn.Linear(D, D, bias=False))

    def __call__(self, x):
        self._recorder.seen.append(self.attention.to_q.weight)
        out = x + self.feed_forward.w2(self.attention.to_q(x))
        self._recorder.events.append(("run", id(out)))
        self._recorder.keep.append(out)
        return out


class FakeZImageTransformer(nn.Module):
    def __init__(self, recorder, *, n_refiner_layers=1, n_layers=2):
        super().__init__()
        self.noise_refiner = [FakeRefinerBlock(recorder) for _ in range(n_refiner_layers)]
        self.context_refiner = [FakeContextBlock(recorder) for _ in range(n_refiner_layers)]
        self.layers = [FakeRefinerBlock(recorder) for _ in range(n_layers)]
        self.cap_embedder = [nn.RMSNorm(D), nn.Linear(D, D)]

    def __call__(self, x, cap, t_emb):
        for layer in self.noise_refiner:
            x = layer(x, t_emb)
        cap = self.cap_embedder[1](self.cap_embedder[0](cap))  # resident, outside every block
        for layer in self.context_refiner:
            cap = layer(cap)
        unified = mx.concatenate([x, cap], axis=1)
        for layer in self.layers:
            unified = layer(unified, t_emb)
        return unified


class FakeSeamZImage(StepSeam, FakeZImageTransformer):
    """What the Z-Image adapter composes over mflux's class, over the fake instead."""


_REFINER = {
    "attention.to_q": "attention.to_q",
    "feed_forward.w2": "feed_forward.w2",
    "adaLN_modulation.0": "adaLN_modulation.0",
}
_CONTEXT = {"attention.to_q": "attention.to_q", "feed_forward.w2": "feed_forward.w2"}
ZIMAGE_TABLE = StaticNameMap(
    {"noise_refiner": _REFINER, "context_refiner": _CONTEXT, "layers": _REFINER}
)


def zimage_block_lists(tf):
    return [(kind, getattr(tf, kind)) for kind in KINDS]


def zimage_inputs():
    x = mx.ones((1, 2, D), dtype=mx.bfloat16)
    return x, x, mx.ones((1, D), dtype=mx.bfloat16)


def compress_blocks(shapes, rng):
    """One real DF11 group per block of ``shapes`` (sub-paths in table order); returns groups, names, sources."""
    groups, names, source = {}, {}, {}
    for block, per in shapes.items():
        subs = tuple(per)  # install_placeholders keeps the table's order
        matrix_names = tuple(f"{block}.{sub}.weight" for sub in subs)
        mats = [random_bf16(rng, per[sub]) for sub in subs]
        flat = np.concatenate([m.reshape(-1) for m in mats])
        splits = np.cumsum([m.size for m in mats])[:-1].tolist()
        groups[block] = encoder_group(flat, *splits).to_mx(name=block)
        names[block] = matrix_names
        source[block] = dict(zip(matrix_names, mats, strict=True))
    return groups, names, source


# --- Z-Image with its non-block parameters and full-size blocks (the build and extras-coverage tests) -----------
# Literals copied from mflux 0.20.0 (models/z_image/weights/z_image_weight_mapping.py:383-442) and the pattern_dict
# of the mingyi456 DF11 checkpoints; the blocks carry every compressed matrix so ``check_zimage_groups`` accepts them.

_FULL_SUBS = (
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
)
FULL_SUBS = {
    "noise_refiner": (*_FULL_SUBS, "adaLN_modulation.0"),
    "context_refiner": _FULL_SUBS,
    "layers": (*_FULL_SUBS, "adaLN_modulation.0"),
}
ZIMAGE_RENAMES = {
    "t_embedder.mlp.0.weight": "t_embedder.linear1.weight",
    "t_embedder.mlp.0.bias": "t_embedder.linear1.bias",
    "t_embedder.mlp.2.weight": "t_embedder.linear2.weight",
    "t_embedder.mlp.2.bias": "t_embedder.linear2.bias",
    "all_final_layer.2-1.adaLN_modulation.1.weight": "all_final_layer.2-1.adaLN_modulation.0.weight",
    "all_final_layer.2-1.adaLN_modulation.1.bias": "all_final_layer.2-1.adaLN_modulation.0.bias",
}
ZIMAGE_FULL_TABLE = StaticNameMap(
    {kind: {sub: sub for sub in subs} for kind, subs in FULL_SUBS.items()},
    renames=ZIMAGE_RENAMES,
)


class FullBlock(nn.Module):
    """A block with every matrix a Z-Image block has (``adaln`` adds the modulation Linear), plus a norm."""

    def __init__(self, *, adaln):
        super().__init__()
        self.attention = _Sub(
            to_q=nn.Linear(D, D, bias=False),
            to_k=nn.Linear(D, D, bias=False),
            to_v=nn.Linear(D, D, bias=False),
            to_out=[nn.Linear(D, D, bias=False)],
        )
        self.feed_forward = _Sub(
            w1=nn.Linear(D, D, bias=False),
            w2=nn.Linear(D, D, bias=False),
            w3=nn.Linear(D, D, bias=False),
        )
        self.attention_norm1 = nn.RMSNorm(D)
        if adaln:
            self.adaLN_modulation = [nn.Linear(D, 4 * D)]


class FakeZImageFull(FakeZImageTransformer):
    """Z-Image's non-block parameters (mflux names) around full-size blocks; not callable end to end."""

    def __init__(self, recorder, *, n_refiner_layers=1, n_layers=2):
        super().__init__(recorder, n_refiner_layers=n_refiner_layers, n_layers=n_layers)
        self.noise_refiner = [FullBlock(adaln=True) for _ in range(n_refiner_layers)]
        self.context_refiner = [FullBlock(adaln=False) for _ in range(n_refiner_layers)]
        self.layers = [FullBlock(adaln=True) for _ in range(n_layers)]
        self.t_embedder = _Sub(linear1=nn.Linear(D, D), linear2=nn.Linear(D, D))
        self.all_x_embedder = {"2-1": nn.Linear(D, D)}
        self.all_final_layer = {
            "2-1": _Sub(linear=nn.Linear(D, D), adaLN_modulation=[nn.Linear(D, D)])
        }
        self.x_pad_token = mx.zeros((1, D))
        self.cap_pad_token = mx.zeros((1, D))


class FakeSeamZImageFull(StepSeam, FakeZImageFull):
    """The seam class composed over ``FakeZImageFull``."""


def write_fake_zimage_checkpoint(root, tf_shapes, rng):
    """A DF11 checkpoint for ``FakeZImageFull``: a group per block, the ``cap_embedder`` group, BF16 extras.

    Every other parameter is an extra under its diffusers name (the inverse of ``ZIMAGE_RENAMES``), a distinct
    constant ``0x3F80 + i`` filled, so a mis-routed extra shows. Returns ``(matrices per group, {parameter: constant})``.
    """
    counts = {kind: sum(1 for b in tf_shapes if b.startswith(f"{kind}.")) for kind in KINDS}
    groups = {
        block: [random_bf16(rng, per[sub]) for sub in FULL_SUBS[block.partition(".")[0]]]
        for block, per in tf_shapes.items()
    }
    groups["cap_embedder"] = [random_bf16(rng, (D, D))]
    probe = FakeZImageFull(
        None, n_layers=counts["layers"], n_refiner_layers=counts["noise_refiner"]
    )
    matrices = {f"{b}.{a}.weight" for b, per in tf_shapes.items() for a in per}
    matrices.add("cap_embedder.1.weight")
    inverse = {param: ckpt for ckpt, param in ZIMAGE_RENAMES.items()}
    constants, extras = {}, {}
    for i, (param, array) in enumerate(sorted(tree_flatten(probe.parameters()))):
        if param in matrices:
            continue
        constants[param] = 0x3F80 + i
        extras[inverse.get(param, param)] = np.full(array.shape, 0x3F80 + i, dtype=np.uint16)
    write_checkpoint(
        root,
        groups=groups,
        patterns={
            r"noise_refiner\.\d+": FULL_SUBS["noise_refiner"],
            r"context_refiner\.\d+": FULL_SUBS["context_refiner"],
            r"layers\.\d+": FULL_SUBS["layers"],
            "cap_embedder": ["1"],
        },
        extras=extras,
    )
    return groups, constants
