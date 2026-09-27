"""build_transformer: the part testable without mflux (the depth-override validation).

Everything past the count check needs mflux's real ``Transformer`` (``seam_transformer_class``), so
those steps stay untested here; the mflux-lane build is exercised where mflux is installed.
"""

from types import SimpleNamespace

import pytest
from tests._flux_fakes import FLUX_TABLE

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.mflux.flux1.transformer import build_transformer


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
