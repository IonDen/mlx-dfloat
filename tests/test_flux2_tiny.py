"""One real tiny FLUX.2 Klein generation through mflux's own loop: our prelude, the uncompiled predict, the
cached embeddings, the seam, the VAE guard and mflux's image assembly composing. `@pytest.mark.mflux`."""

import mlx.core as mx
import numpy as np
import pytest
from tests._flux2_tiny import fake_model

pytestmark = pytest.mark.mflux


def _cache_limit_in_force():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


@pytest.mark.parametrize(("guidance", "calls"), [(4.0, 2), (1.0, 1)])
def test_one_tiny_generation_decodes_once_per_block_per_transformer_call(
    tmp_path, monkeypatch, guidance, calls
):
    # Bug caught: anything between our prelude and mflux's loop not composing (the uncompiled predict, the cached
    # 4-tuple, the VAE guard), or the negative branch run at guidance 1.0. Tiny transformer: 1 double + 2 single = 3
    # blocks; 2 steps: launches == 2 steps x calls x 3 blocks == 12 at guidance 4.0 and 6 at 1.0.
    from mflux.utils.apple_silicon import AppleSiliconUtil

    # A Max chip: stock mflux compiles predict here, so a lost bypass fails this run end to end.
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: False))
    model = fake_model(tmp_path, monkeypatch, model="flux2-klein-base-4b")
    in_loop = []

    class LimitProbe:
        def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
            in_loop.append(_cache_limit_in_force())

    model.callbacks.register(LimitProbe())
    before = _cache_limit_in_force()
    image = model.generate_image(
        seed=1, prompt="a", num_inference_steps=2, height=64, width=64, guidance=guidance
    )
    pixels = np.asarray(image.image)
    assert pixels.shape == (64, 64, 3)
    assert np.isfinite(pixels).all()
    report = model.report()
    assert report["decode_launches"] == 2 * calls * 3
    assert report["cfg_calls_per_step"] == calls
    assert report["predict"] == "uncompiled"
    assert report["fit"]["label"] == "predicted"
    # The tiny groups' largest double (53_248 B) and single (26_624 B) block next to the allowance's floor
    # (500_000_000 B: 64^2 is 16 image tokens + 512 text tokens, 1.5e9 * 528 / 4352 is below it).
    assert report["cache_limit_in_force"] == 53_248 + 26_624 + 500_000_000
    assert in_loop == [report["cache_limit_in_force"]] * 2  # in force in the loop, not just planned
    assert set(report["peaks"]) >= {"encode", "set_load", "denoise", "vae"}
    assert report["lifecycle"]["set_loads"] == 1
    assert _cache_limit_in_force() == before
