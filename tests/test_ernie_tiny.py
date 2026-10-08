"""One real tiny ERNIE-Image generation through mflux's own loop: our prelude, the uncompiled predict, the batch
cache, the seam, the VAE guard and mflux's image assembly composing. `@pytest.mark.mflux`."""

import mlx.core as mx
import numpy as np
import pytest
from tests._ernie_tiny import fake_model

pytestmark = pytest.mark.mflux


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


@pytest.mark.parametrize(("guidance", "batch"), [(4.0, 2), (1.0, 1)], ids=["cfg", "guidance-1"])
def test_one_tiny_generation_decodes_each_block_once_per_step(
    tmp_path, monkeypatch, guidance, batch
):
    # Bug caught: anything between our prelude and mflux's loop not composing (the uncompiled predict, the batch
    # cache, the VAE guard), or CFG split into two transformer calls (8 decodes instead of 4). The chip check answers
    # "not M1/M2" so a lost bypass is red on any host. Tiny transformer: 2 blocks; 2 steps: 2 x 2 = 4 decodes either
    # way (CFG is one batch-2 call, ernie_image.py:239-250).
    from mflux.utils.apple_silicon import AppleSiliconUtil

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
        seed=1, prompt="a", num_inference_steps=2, height=64, width=64, guidance=guidance
    )
    assert np.asarray(image.image).shape == (64, 64, 3)
    # The denoised latents at 64²: 128 channels x 4 x 4 (ernie_latent_creator.py:11-17), finite.
    assert seen["latents"].shape == (1, 128, 4, 4)
    assert np.isfinite(seen["latents"]).all()
    report = model.report()
    assert report["decode_launches"] == 4
    assert report["cfg_batch"] == batch
    assert report["predict"] == "uncompiled"
    assert seen["limits"] == [report["cache_limit_in_force"]] * 2  # in force in the loop
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert model.prompt_cache == {}
    assert _cache_limit_in_force() == before
