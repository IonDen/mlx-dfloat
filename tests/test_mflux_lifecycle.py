"""The lifecycle moved from flux1/ to mflux/ so every family shares it; FLUX.1's old path stays importable."""

import logging

import mlx.core as mx
import pytest

from mlx_dfloat.mflux import lifecycle as shared
from mlx_dfloat.mflux.flux1 import lifecycle as old


def test_the_old_flux1_path_re_exports_the_same_classes():
    # Bug caught: flux1/lifecycle.py left as a second copy (two classes: isinstance and counters diverge), or
    # dropped (DFloatFlux1's import breaks).
    assert old.Lifecycle is shared.Lifecycle
    assert old.LifecycleCounters is shared.LifecycleCounters


def test_it_logs_under_the_family_neutral_logger_name(caplog):
    # Bug caught: the shared lifecycle still logging as "mlx_dfloat.mflux.flux1" (a Z-Image warning filed under FLUX.1).
    def encode(prompt):
        raise ValueError("encode broke")

    leaked = []

    def load_encoders():
        leaked.append(
            mx.ones((2048, 2048), dtype=mx.bfloat16)
        )  # kept alive: the reclaim check must complain
        mx.eval(leaked[0])

    bound = int(mx.get_active_memory())
    life = shared.Lifecycle(
        load_encoders=load_encoders,
        unload_encoders=lambda: None,
        encode=encode,
        load_set=lambda: None,
        unload_set=lambda: None,
        prompt_cache={},
        retained_bound=lambda: bound,
    )
    with (
        caplog.at_level(logging.WARNING, logger="mlx_dfloat.mflux"),
        pytest.raises(ValueError, match="encode broke"),
    ):
        life.ensure_embeddings("p")
    assert [r.name for r in caplog.records] == ["mlx_dfloat.mflux"]
