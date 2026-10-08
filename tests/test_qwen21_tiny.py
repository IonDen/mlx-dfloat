"""One real tiny Qwen-Image 2.1 generation through mflux's own loop: our prelude, the uncompiled forward pass, the
shared prompt cache, the seam, the VAE guard and mflux's image assembly composing. `@pytest.mark.mflux`."""

import mlx.core as mx
import numpy as np
import pytest
from tests._qwen21_tiny import fake_model

pytestmark = pytest.mark.mflux


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


@pytest.mark.parametrize(
    ("guidance", "negative", "calls"),
    [(4.0, "n", 2), (1.0, "n", 1), (4.0, "", 1)],
    ids=["cfg", "guidance-1", "empty-negative"],
)
def test_one_tiny_generation_decodes_once_per_block_per_transformer_call(
    tmp_path, monkeypatch, guidance, negative, calls
):
    # Bug caught: anything between our prelude and mflux's loop not composing (the eager forward pass, the pairs in
    # mflux's prompt cache, the VAE guard), or the negative branch run at guidance 1.0 or with an empty negative
    # (mflux's rule: guidance > 1.0 and bool(negative_prompt), qwen_image_21.py:89). Tiny transformer: 2 blocks; 2
    # steps: launches == 2 steps x calls x 2 blocks == 8 with CFG, 4 without.
    from mflux.utils.apple_silicon import AppleSiliconUtil

    # mflux compiles this transformer's forward on every chip; a Max chip shows no chip check is relied on.
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch)
    seen = {}

    class Probe:
        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            seen.setdefault("limits", []).append(_cache_limit_in_force())

        def call_after_loop(self, seed, prompt, latents, config):
            seen["latents"] = np.array(latents.astype(mx.float32))

    model.callbacks.register(Probe())
    before = _cache_limit_in_force()
    image = model.generate_image(
        seed=1,
        prompt="a",
        num_inference_steps=2,
        height=64,
        width=64,
        guidance=guidance,
        negative_prompt=negative,
    )
    assert np.asarray(image.image).shape == (64, 64, 3)
    # The denoised latents: 4 x 4 = 16 tokens of 64 channels at 64^2 (qwen21_latent_creator.py:11-15), finite.
    assert seen["latents"].shape == (1, 16, 64)
    assert np.isfinite(seen["latents"]).all()
    report = model.report()
    assert report["decode_launches"] == 2 * calls * 2
    assert report["cfg_calls_per_step"] == calls
    assert report["predict"] == "uncompiled"
    # The tiny block group (4 x 32 x 32 + 2 x 96 x 32 + 32 x 96 = 13_312 elements, 26_624 B) next to the allowance's
    # floor (500_000_000 B: 16 image tokens + 5 text tokens is far below the reference).
    assert report["cache_limit_in_force"] == 26_624 + 500_000_000
    assert seen["limits"] == [report["cache_limit_in_force"]] * 2  # in force in the loop
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert report["lifecycle"]["set_loads"] == 1
    assert _cache_limit_in_force() == before
