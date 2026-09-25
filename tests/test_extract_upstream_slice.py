import pytest
from scripts.extract_upstream_slice import subpaths_for

from mlx_dfloat.errors import DFloatFormatError


def test_subpaths_for_returns_the_matching_patterns_subpaths():
    pattern_dict = {r"lm_head": [], r"model\.layers\.\d+": ["self_attn.q_proj", "mlp.up_proj"]}
    assert subpaths_for(pattern_dict, "model.layers.0") == ["self_attn.q_proj", "mlp.up_proj"]


def test_subpaths_for_screens_remote_patterns_before_matching():
    # Bug caught: the extractor running a downloaded pattern through re.fullmatch unscreened; a
    # `.*` chain hangs it instead of failing with a clear error.
    with pytest.raises(DFloatFormatError, match="not allowed"):
        subpaths_for({".*" * 20 + "!": []}, "model.layers.0")
