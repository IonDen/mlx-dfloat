"""build_transformer offline (a fake transformer class in mflux's place) and, in the mflux lane, one
real forward of mflux's FLUX.1 Transformer through the seam.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from tests._flux_fakes import (
    DOUBLE_MAP,
    FLUX_TABLE,
    SINGLE_MAP,
    FakeSeamTransformer,
    FakeTransformer,
    Recorder,
    all_block_weights,
    block_lists,
    write_flux_checkpoint,
)

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.mflux.flux1 import transformer
from mlx_dfloat.mflux.flux1.transformer import build_transformer

GIB = 1024**3


def _group(name: str) -> SimpleNamespace:
    n = 14 if name.startswith("transformer_blocks.") else 6
    return SimpleNamespace(name=name, matrix_names=tuple(f"{name}.m{i}.weight" for i in range(n)))


def _ckpt(names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(groups={n: _group(n) for n in names})


def test_build_transformer_refuses_a_depth_override_past_the_checkpoints_own_count():
    # Bug caught: an n_double/n_single override used to build a transformer deeper than the
    # checkpoint actually has groups for, which would leave the extra blocks' matrices as
    # placeholders forever (the checkpoint has no extras or DF11 groups for a block that does not
    # exist). Passing name_map explicitly skips the mflux-importing flux_name_map(), so this raises
    # before seam_transformer_class() ever needs mflux.
    ckpt = _ckpt(["transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0"])
    with pytest.raises(
        DFloatIntegrationError,
        match=r"asked for 3 double / 1 single blocks; the checkpoint has 2 / 1",
    ):
        build_transformer(None, ckpt, name_map=FLUX_TABLE, n_double=3)
    with pytest.raises(
        DFloatIntegrationError,
        match=r"asked for 2 double / 2 single blocks; the checkpoint has 2 / 1",
    ):
        build_transformer(None, ckpt, name_map=FLUX_TABLE, n_single=2)


@pytest.mark.parametrize(("n_double", "n_single"), [(-1, None), (None, -1)])
def test_build_transformer_refuses_a_negative_depth(n_double, n_single):
    # Bug caught: a negative override passing the "at most the checkpoint's count" check and reaching
    # mflux, which builds range(-1) == no blocks while the extras plan and the shapes disagree.
    ckpt = _ckpt(["transformer_blocks.0", "single_transformer_blocks.0"])
    with pytest.raises(DFloatIntegrationError, match="negative"):
        build_transformer(None, ckpt, name_map=FLUX_TABLE, n_double=n_double, n_single=n_single)


# --- the whole build over a fake transformer class -------------------------------------------------

_TABLES = {"transformer_blocks": DOUBLE_MAP, "single_transformer_blocks": SINGLE_MAP}


def _checkpoint_name(param: str) -> str:
    """A module parameter name back in checkpoint spelling, from the fakes' hand tables."""
    kind, _dot, rest = param.partition(".")
    if kind not in _TABLES:
        return param
    idx, _dot, rest = rest.partition(".")
    head, _dot, leaf = rest.rpartition(".")
    inverse = {attr: sub for sub, attr in _TABLES[kind].items()}
    return f"{kind}.{idx}.{inverse.get(head, head)}.{leaf}"


def _fake_seam_class():
    def build(model_config, *, num_transformer_blocks, num_single_transformer_blocks):
        del model_config
        return FakeSeamTransformer(
            Recorder(), n_double=num_transformer_blocks, n_single=num_single_transformer_blocks
        )

    return build


def _write_flux(tmp_path, *, n_double=2, n_single=1):
    """A DF11 checkpoint for the fake transformer: one group per block and every non-matrix extra."""
    rng = np.random.default_rng(51)
    fake = FakeTransformer(Recorder(), n_double=n_double, n_single=n_single)
    shapes = install_placeholders(block_lists(fake), FLUX_TABLE)
    extras = {
        _checkpoint_name(name): np.asarray(
            np.random.default_rng(len(name)).integers(0x3C00, 0x3F80, size=p.shape), dtype=np.uint16
        )
        for name, p in tree_flatten(fake.parameters())
        if p.size > 0
    }
    write_flux_checkpoint(tmp_path / "ckpt", shapes, rng, extras=extras)
    return open_checkpoint(tmp_path / "ckpt"), extras


def test_build_transformer_loads_every_extra_and_leaves_every_matrix_a_placeholder(
    tmp_path, monkeypatch
):
    # Bug caught: a renamed block extra (ff.net.0.proj.bias -> ff.linear1.bias) loaded under its
    # checkpoint name and dropped by load_weights(strict=False), an extra of a block past the
    # reduced depth planned anyway (the coverage check would refuse it), or a block matrix left as
    # the module's own init instead of a placeholder.
    monkeypatch.setattr(transformer, "seam_transformer_class", _fake_seam_class)
    ckpt, extras = _write_flux(tmp_path)
    tf, shapes = build_transformer(None, ckpt, name_map=FLUX_TABLE, n_double=1)
    assert list(shapes) == ["transformer_blocks.0", "single_transformer_blocks.0"]
    assert all(w.size == 0 for w in all_block_weights(tf))
    bias = tf.transformer_blocks[0].ff.linear1.bias
    assert np.array_equal(
        np.array(bias.view(mx.uint16)), extras["transformer_blocks.0.ff.net.0.proj.bias"]
    )
    scale = tf.single_transformer_blocks[0].attn.norm_q.weight
    assert np.array_equal(
        np.array(scale.view(mx.uint16)), extras["single_transformer_blocks.0.attn.norm_q.weight"]
    )


def test_build_transformer_names_the_active_memory_the_extras_added(tmp_path, monkeypatch):
    # Bug caught: the build's active-memory guard dropped (a resident block matrix or a float32
    # extras copy would go unnoticed), or its message blaming a block matrix, which a lazy init
    # never makes active.
    monkeypatch.setattr(transformer, "seam_transformer_class", _fake_seam_class)
    ckpt, _extras = _write_flux(tmp_path)
    readings = iter([0, 3 * GIB])
    monkeypatch.setattr(transformer.mx, "get_active_memory", lambda: next(readings))
    with pytest.raises(
        DFloatIntegrationError,
        match=r"extras added 3\.00 GiB of active memory \(limit 2 GiB\)",
    ):
        build_transformer(None, ckpt, name_map=FLUX_TABLE)


# --- the mflux lane --------------------------------------------------------------------------------


@pytest.mark.mflux
def test_mflux_runs_one_forward_through_the_seam_and_ends_on_placeholders():
    # Bug caught: an mflux point release renaming or re-signing a per-block hook (the seam's
    # override would never run and the blocks would matmul against zero-size placeholders), or
    # changing a block's Linear set (install_placeholders refuses it). Checked against mflux 0.20.0:
    # Transformer(model_config, num_transformer_blocks, num_single_transformer_blocks) and
    # __call__(t, config, hidden_states, prompt_embeds, pooled_prompt_embeds, ...) at
    # models/flux/model/flux_transformer/transformer.py:16 and :32; the image tokens are
    # (height // 16) * (width // 16) (transformer.py:160); the pooled embedding is 768 wide
    # (text_embedder.py:8); Config(model_config, num_inference_steps, height, width, guidance) at
    # models/common/config/config.py:17.
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    from mlx_dfloat.integrate.placeholders import get_attr_path
    from mlx_dfloat.integrate.providers import ResidentProvider
    from mlx_dfloat.mflux.flux1.names import flux_name_map
    from mlx_dfloat.mflux.flux1.transformer import seam_transformer_class

    model_config = ModelConfig.schnell()
    tf = seam_transformer_class()(
        model_config, num_transformer_blocks=1, num_single_transformer_blocks=1
    )
    names = flux_name_map()
    shapes = install_placeholders(
        [
            ("transformer_blocks", tf.transformer_blocks),
            ("single_transformer_blocks", tf.single_transformer_blocks),
        ],
        names,
    )
    assert [len(per) for per in shapes.values()] == [14, 6]
    zeros = {
        block: {attr: mx.zeros(shape, dtype=mx.bfloat16) for attr, shape in per.items()}
        for block, per in shapes.items()
    }
    tf.attach(ResidentProvider(zeros), shapes, eval_policy="per-block")
    size = 32  # 2 x 2 latent patches: four image tokens
    config = Config(model_config, num_inference_steps=4, height=size, width=size, guidance=0.0)
    keys = mx.random.split(mx.random.key(0), 3)
    hidden = mx.random.normal((1, (size // 16) ** 2, 64), key=keys[0])
    prompt = mx.random.normal((1, 256, 4096), key=keys[1])
    pooled = mx.random.normal((1, 768), key=keys[2])
    out = tf(
        t=0, config=config, hidden_states=hidden, prompt_embeds=prompt, pooled_prompt_embeds=pooled
    )
    mx.eval(out)
    tf.verify_step()
    assert out.shape == (1, (size // 16) ** 2, 64)
    assert bool(mx.isfinite(out).all().item())
    for block_name, per in shapes.items():
        kind, _dot, idx = block_name.partition(".")
        block = getattr(tf, kind)[int(idx)]
        for attr in per:
            assert get_attr_path(block, attr).weight.size == 0, f"{block_name}.{attr}"
