"""The Qwen-Image 2.1 block seam packs Q/K/V and gate/up per block, as mflux 0.23 packs them per generation.

mflux 0.23 ``Qwen21Transformer.inference_projections()`` (qwen21_transformer.py:126-140) concatenates each block's
``to_q``/``to_k``/``to_v`` and ``gate_layer``/``proj`` weights once per generation; the blocks then run one matmul
against the packed matrix (qwen21_attention.py:208-212, qwen21_feed_forward.py:17-22). Our blocks hold zero-size
placeholders until their decode, so the seam packs each block's decoded weights itself. No mflux import here: the
block below has the 0.23 block's shape (the same attribute names and the same packed/unpacked branches), and the
packer is handed in, as ``build_transformer`` hands in mflux's own ``Qwen21Transformer._pack_linears``.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate.blockseam import SeamCell
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.integrate.placeholders import PLACEHOLDER, install_placeholders
from mlx_dfloat.integrate.providers import ResidentProvider
from mlx_dfloat.integrate.seam import attach_state
from mlx_dfloat.mflux.qwen21.transformer import PackSwitch, seam_qwen21_blocks

# Small M and K = 1024: the shape where a packed matmul's bits differ from the separate ones on this machine
# (measured 2026-10-23, mlx 0.32.2, M1 Max: M=16, K=1024 differs in 6 / 3 / 0 of 16384 Q/K/V outputs; at the real
# width K=4096 with 24 text tokens 34 / 32 / 36 of 98304). test_the_parity_shape_tells_packed_from_separate pins it.
TOKENS, DIM = 16, 1024
SUBS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)
NAME_MAP = StaticNameMap({"transformer_blocks": {sub: sub for sub in SUBS}})


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_k = nn.Linear(DIM, DIM, bias=False)
        self.to_v = nn.Linear(DIM, DIM, bias=False)
        self.to_out = [nn.Linear(DIM, DIM, bias=False)]
        self._qkv_weight = None

    def qkv(self, x):
        # mflux 0.23 Qwen21Attention._project_qkv
        if self._qkv_weight is not None:
            return mx.split(x @ self._qkv_weight.T, 3, axis=-1)
        return self.to_q(x), self.to_k(x), self.to_v(x)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)
        self.gate_layer = nn.Linear(DIM, DIM, bias=False)
        self._gate_proj_weight = None

    def __call__(self, x):
        # mflux 0.23 Qwen21SwiGLUFeedForward.__call__
        if self._gate_proj_weight is None:
            gate, proj = self.gate_layer(x), self.proj(x)
        else:
            gate, proj = mx.split(x @ self._gate_proj_weight.T, 2, axis=-1)
        return self.out(nn.silu(gate) * proj)


class Block(nn.Module):
    """The three entry points mflux 0.23's transformer calls on a block, with the 0.23 projections inside."""

    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.img_mlp = Mlp()

    def _qkv_out(self, x):
        q, k, v = self.attn.qkv(x)
        return q, k, v, self.attn.to_out[0](q * k + v)

    def __call__(self, x):
        *_qkv, attn = self._qkv_out(x)
        return x + attn + self.img_mlp(x)

    def text_forward(self, x):
        _q, k, v, attn = self._qkv_out(x)
        return x + attn + self.img_mlp(x), k, v

    def image_forward(self, x):
        *_qkv, attn = self._qkv_out(x)
        return x + attn + self.img_mlp(x)


def pack_linears(layers):
    """mflux 0.23 ``Qwen21Transformer._pack_linears`` (qwen21_transformer.py:159-175), for nn.Linear layers."""
    packed = mx.concatenate([layer.weight for layer in layers], axis=0)
    mx.eval(packed)
    for layer, weight in zip(layers, mx.split(packed, len(layers), axis=0), strict=True):
        layer.weight = weight
    return packed


def _weights(seed):
    keys = mx.random.split(mx.random.key(seed), len(SUBS))
    return {
        sub: (mx.random.normal((DIM, DIM), key=k) * 0.03).astype(mx.bfloat16)
        for sub, k in zip(SUBS, keys, strict=True)
    }


def _x(seed=11):
    return mx.random.normal((1, TOKENS, DIM), key=mx.random.key(seed)).astype(mx.bfloat16)


def _native(weights, *, packed):
    """mflux's own block with the BF16 weights, packed as ``inference_projections()`` leaves it (or not)."""
    block = Block()
    for sub, w in weights.items():
        node = block
        for part in sub.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        node.weight = w
    if packed:
        block.attn._qkv_weight = pack_linears((block.attn.to_q, block.attn.to_k, block.attn.to_v))
        block.img_mlp._gate_proj_weight = pack_linears(
            (block.img_mlp.gate_layer, block.img_mlp.proj)
        )
    return block


def _seamed(weights, *, on):
    blocks = [Block()]
    lists = [("transformer_blocks", blocks)]
    shapes = install_placeholders(lists, NAME_MAP)
    cell, switch = SeamCell(), PackSwitch()
    seam_qwen21_blocks(lists, cell, switch, pack=pack_linears)
    cell.state = attach_state(ResidentProvider({"transformer_blocks.0": weights}), shapes)
    switch.on = on
    return blocks[0]


def _bits(out):
    arrays = out if isinstance(out, tuple) else (out,)
    mx.eval(*arrays)
    return [a.view(mx.uint16).tolist() for a in arrays]


def test_the_parity_shape_tells_packed_from_separate():
    # Bug caught: a parity shape where packing changes no bit, so the seam tests below would pass with the packing
    # left out. Our oracle is mflux's own native block, packed vs unpacked, at TOKENS x DIM.
    weights, x = _weights(3), _x()
    assert _bits(_native(weights, packed=True)(x)) != _bits(_native(weights, packed=False)(x))


@pytest.mark.parametrize("entry", ["__call__", "text_forward", "image_forward"])
def test_each_entry_point_computes_the_native_packed_block_bit_for_bit(entry):
    # Bugs caught: the seam leaving an entry point unseamed (mflux 0.23's text-prefix cache calls text_forward, its
    # image step image_forward: a placeholder matmul raises), or running it on the separate matmuls (the bits of the
    # unpacked block, which the test above shows differ).
    weights, x = _weights(3), _x()
    want = _bits(getattr(_native(weights, packed=True), entry)(x))
    assert _bits(getattr(_seamed(weights, on=True), entry)(x)) == want


def test_outside_inference_projections_the_block_runs_unpacked():
    # Bug caught: packing with the switch off (mflux packs only inside inference_projections(), and only with
    # use_packed_projections set): the bits would be the packed block's, not the unpacked one mflux runs then.
    weights, x = _weights(3), _x()
    assert _bits(_seamed(weights, on=False)(x)) == _bits(_native(weights, packed=False)(x))


def test_the_packed_matrices_and_the_decoded_weights_are_released_after_each_block():
    # Bug caught: a packed matrix left on the block after its call (it holds 3 + 2 decoded matrices' worth of memory,
    # 288 MiB a block at the real width, until the next decode of the block), or a weight left in place of the
    # placeholder.
    block = _seamed(_weights(3), on=True)
    mx.eval(block.text_forward(_x()))
    assert block.attn._qkv_weight is None
    assert block.img_mlp._gate_proj_weight is None
    for sub in SUBS:
        node = block
        for part in sub.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        assert node.weight is PLACEHOLDER


def test_a_block_whose_call_raises_still_releases_the_packed_matrices():
    # Bug caught: clearing the packed matrices only on success, so a failed step leaves them pinned.
    block = _seamed(_weights(3), on=True)
    with pytest.raises(ValueError, match="matmul"):
        block(mx.zeros((1, TOKENS, DIM + 1), dtype=mx.bfloat16))
    assert block.attn._qkv_weight is None
    assert block.img_mlp._gate_proj_weight is None


def test_a_detached_block_refuses_every_entry_point():
    # Bug caught: an entry point that skips the attach check runs on the zero-size placeholders.
    blocks = [Block()]
    lists = [("transformer_blocks", blocks)]
    install_placeholders(lists, NAME_MAP)
    seam_qwen21_blocks(lists, SeamCell(), PackSwitch(), pack=pack_linears)
    for entry in ("__call__", "text_forward", "image_forward"):
        with pytest.raises(DFloatIntegrationError, match=r"transformer_blocks\.0: call attach"):
            getattr(blocks[0], entry)(_x())


def test_seaming_a_block_twice_is_refused():
    # Bug caught: a second swap stacking two seams, which would decode and pack the block twice per call.
    blocks = [Block()]
    lists = [("transformer_blocks", blocks)]
    seam_qwen21_blocks(lists, SeamCell(), PackSwitch(), pack=pack_linears)
    with pytest.raises(DFloatIntegrationError, match="already seamed"):
        seam_qwen21_blocks(lists, SeamCell(), PackSwitch(), pack=pack_linears)
