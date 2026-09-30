"""The q8 mode's real build: mflux's schnell transformer (1 + 1 blocks) quantized at load from the base.

Needs the mflux extra, ``--run-slow`` and ``MLX_DFLOAT_SCHNELL_BASE`` (the local schnell base
snapshot); skips otherwise. It never downloads: ``HF_HUB_OFFLINE=1`` on the command line below is the
outer guard (huggingface_hub reads the variable at import, so the test also patches
``huggingface_hub.constants.HF_HUB_OFFLINE``). mflux is imported inside the test function, so the module collects cleanly in the dev venv. Run it on the main thread (it loads
real weights on the GPU):

    MLX_DFLOAT_SCHNELL_BASE=<snapshot> HF_HUB_OFFLINE=1 uv run --group bench pytest --run-slow \
        tests/test_bench_q8_slow.py -q -p no:cacheprovider; uv sync --group dev
"""

import os
import weakref
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

GIB = 1024**3


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return Path(value)


def _linears(block: nn.Module) -> dict[str, nn.Module]:
    """Every Linear of a block, plain or quantized, by its path inside the block."""
    return {
        path: module
        for path, module in block.named_modules()
        if isinstance(module, nn.Linear | nn.QuantizedLinear)
    }


@pytest.mark.slow
@pytest.mark.mflux
def test_q8_build_quantizes_every_block_linear_and_steps_without_a_decode(monkeypatch):
    # Bug caught: the applier asked for another bit width or group size (or none, leaving plain BF16
    # Linears under the q8 name), a block Linear the predicate skips, a q8 step that still reaches
    # the DF11 Metal decode, mflux's real loader/applier path keeping the LoadedWeights alive into
    # the eval (the weakref below), or an eager loader that materialises the whole BF16 dict (the
    # MLX peak bound below).
    import huggingface_hub.constants
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from scripts import _q8_rig as q8
    from scripts.bench_flux_step import denoise_step
    from scripts.encode_prompt import synthetic_embeds

    import mlx_dfloat._metal_decode as metal_decode
    import mlx_dfloat.decode as decode_api

    base = _env("MLX_DFLOAT_SCHNELL_BASE")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    launches: list[str] = []
    real_metal, real_group = metal_decode.decode, decode_api.decode_group

    def count_metal(group, **kwargs):
        launches.append(group.name)
        return real_metal(group, **kwargs)

    def count_group(group, **kwargs):
        launches.append(group.name)
        return real_group(group, **kwargs)

    monkeypatch.setattr(metal_decode, "decode", count_metal)
    monkeypatch.setattr(decode_api, "decode_group", count_group)

    loaded_refs: list[weakref.ref] = []
    evaluated: list[bool] = []

    def loader(component, root):
        loaded = q8._mflux_loader(component, root)
        loaded_refs.append(weakref.ref(loaded))  # LoadedWeights is a plain dataclass
        return loaded

    def evaluate(transformer):
        assert [r() is None for r in loaded_refs] == [True], "LoadedWeights alive at the eval"
        evaluated.append(True)
        q8.evaluate_by_block(transformer)

    mx.reset_peak_memory()
    transformer = q8.build_q8_transformer(
        "schnell", base, n_double=1, n_single=1, loader=loader, evaluate=evaluate
    )
    assert evaluated == [True]
    # The bound, by hand from FLUX.1's shapes (hidden 3072, MLP 12288): a double block holds
    # 339.7M elements (the DF11 group of transformer_blocks.0), a single block 3072 x (3 x 3072 +
    # 12288 + 3 x 3072) + 15360 x 3072 = 141.6M, the parts outside the blocks about 54M (context and
    # x embedders, time/text embedders, norm_out, proj_out), so ~535M elements: ~1.07 GB of BF16
    # sources and ~0.57 GB of q8 (1 byte + scales/biases per 64). Block by block the MLX peak is the
    # q8 total plus one block's BF16, ~1.3 GB. An eager loader materialises every block of the 19 +
    # 38 dict, ~11.9G elements = ~22 GiB of BF16. 4 GiB sits well between the two.
    peak = int(mx.get_peak_memory())
    assert peak < 4 * GIB, f"MLX peak {peak / GIB:.2f} GiB during a 1 + 1 build"

    for block, prefixes in (
        (
            transformer.transformer_blocks[0],
            (
                "attn.to_q",
                "ff.linear1",
                "ff_context.linear2",
                "norm1.linear",
                "norm1_context.linear",
            ),
        ),
        (
            transformer.single_transformer_blocks[0],
            ("attn.to_q", "norm.linear", "proj_mlp", "proj_out"),
        ),
    ):
        linears = _linears(block)
        plain = sorted(p for p, m in linears.items() if not isinstance(m, nn.QuantizedLinear))
        assert plain == []
        for prefix in prefixes:
            assert any(p.startswith(prefix) for p in linears), prefix  # the brief's attn/ff/norm
        for path, module in linears.items():
            assert (module.bits, module.group_size) == (8, 64), path

    config = Config(
        ModelConfig.schnell(),
        num_inference_steps=1,
        height=256,
        width=256,
        guidance=3.5,
        scheduler="linear",
    )
    prompt, pooled = synthetic_embeds("schnell", 42)
    latents = FluxLatentCreator.create_noise(42, 256, 256)
    out, took, verify = denoise_step(transformer, config, latents, prompt, pooled, 0)
    assert launches == []
    assert verify == 0.0
    assert took > 0.0
    assert bool(mx.isfinite(out).all().item())
