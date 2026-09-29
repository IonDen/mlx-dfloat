"""The model's own encoder path against mflux's, on the real schnell encoders (slow: ~10 GB, twice, in turn)."""

import gc
import os
from pathlib import Path

import mlx.core as mx
import pytest


def _env(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} not set")
    return Path(value)


def _bits(array):
    return array.view(mx.uint32 if array.dtype == mx.float32 else mx.uint16)


@pytest.mark.slow
@pytest.mark.mflux
def test_dfloat_model_encodes_the_prompt_bit_identically_to_mflux_own_encoder_path():
    # Bug caught: a tokenizer length, a dtype cast, or a weight-mapping slip in our init path
    # (the encoders load through our definitions, not FluxInitializer's), giving embeddings that
    # differ from mflux's; the image would then differ from mflux's for a reason unrelated to DF11.
    from scripts.encode_prompt import encode_real

    from mlx_dfloat.mflux.flux1.model import DFloatFlux1

    df11 = _env("MLX_DFLOAT_SCHNELL_DF11")
    base = _env("MLX_DFLOAT_SCHNELL_BASE")
    prompt = "A stone lighthouse on a rocky shore at dawn"
    model = DFloatFlux1("schnell", df11_path=str(df11), base_path=str(base))
    model.encode(prompt)
    ours = model.prompt_cache[prompt]
    assert model.t5_text_encoder is None  # dropped after encoding
    del model
    gc.collect()
    mx.clear_cache()
    theirs = encode_real("schnell", prompt, base)
    mx.eval(*theirs)
    for mine, upstream in zip(ours, theirs, strict=True):
        assert mine.dtype == upstream.dtype
        assert mine.shape == upstream.shape
        assert bool(mx.array_equal(_bits(mine), _bits(upstream)))
