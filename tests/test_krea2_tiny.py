"""One real tiny Krea 2 generation through mflux's own loop: our prelude, the uncompiled predict, the per-prompt cache,
the seam, the VAE guard and mflux's image assembly composing. `@pytest.mark.mflux`."""

import mlx.core as mx
import numpy as np
import pytest
from tests._krea2_tiny import fake_model

pytestmark = pytest.mark.mflux


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


@pytest.mark.parametrize(
    ("model_name", "guidance", "calls"),
    [("krea-2", 3.5, 2), ("krea-2", 1.0, 1), ("krea-2", 0.5, 2), ("krea-2-raw", None, 1)],
    ids=["cfg", "guidance-1", "guidance-half", "raw-default"],
)
def test_one_tiny_generation_decodes_once_per_block_per_transformer_call(
    tmp_path, monkeypatch, model_name, guidance, calls
):
    # Bug caught: anything between our prelude and mflux's loop not composing (the uncompiled predict, the per-prompt
    # cache answering mflux's _encode_prompts, the VAE guard), the negative branch skipped below guidance 1.0 (mflux
    # runs CFG for any guidance other than 1.0), or Raw's default running CFG. Tiny transformer: 2
    # blocks; 2 steps: launches == 2 steps x calls x 2 blocks == 8 with CFG, 4 without. The chip check says "not
    # M1/M2", so stock mflux would compile the predict here.
    from mflux.utils.apple_silicon import AppleSiliconUtil

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch, model=model_name)
    seen = {}

    class Probe:
        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            seen.setdefault("limits", []).append(_cache_limit_in_force())

        def call_after_loop(self, seed, prompt, latents, config):
            seen["latents"] = np.array(latents)

    model.callbacks.register(Probe())
    before = _cache_limit_in_force()
    image = model.generate_image(
        seed=1, prompt="a", num_inference_steps=2, height=64, width=64, guidance=guidance
    )
    assert np.asarray(image.image).shape == (64, 64, 3)
    # The denoised latents: 16 channels at 64 / 8 = 8 x 8, float32 (krea2_latent_creator.py:6-10), finite.
    assert seen["latents"].shape == (1, 16, 8, 8)
    assert seen["latents"].dtype == np.float32
    assert np.isfinite(seen["latents"]).all()
    report = model.report()
    assert report["decode_launches"] == 2 * calls * 2
    assert report["cfg_calls_per_step"] == calls
    assert report["predict"] == "uncompiled"
    assert model.prompt_cache == {}
    # The tiny block group (16_384 elements, 32_768 B) next to the allowance's floor (500_000_000 B: 16 image tokens
    # and a few text tokens are far below the reference).
    assert report["cache_limit_in_force"] == 32_768 + 500_000_000
    assert seen["limits"] == [report["cache_limit_in_force"]] * 2  # in force in the loop
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert report["lifecycle"]["set_loads"] == 1
    assert _cache_limit_in_force() == before
