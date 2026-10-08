"""Z-Image naming: the group-kind check (offline) and the map derived from mflux 0.20.0 (mflux lane)."""

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.format import open_checkpoint
from mlx_dfloat.integrate.names import Placement
from mlx_dfloat.mflux.zimage.names import MATRIX_SUBS, check_zimage_groups, zimage_name_map

REFINER_SUBS = MATRIX_SUBS["noise_refiner"]
CONTEXT_SUBS = MATRIX_SUBS["context_refiner"]
PATTERNS = {
    r"noise_refiner\.\d+": REFINER_SUBS,
    r"context_refiner\.\d+": CONTEXT_SUBS,
    r"layers\.\d+": REFINER_SUBS,
    "cap_embedder": ("1",),
}


def _mats(n, rng):
    return [random_bf16(rng, (2, 2)) for _ in range(n)]


def _ckpt(tmp_path, spec, patterns=PATTERNS):
    """``spec``: group name -> matrix count; a real DF11 checkpoint in one file."""
    rng = np.random.default_rng(0)
    groups = {name: _mats(n, rng) for name, n in spec.items()}
    root = write_checkpoint(tmp_path / "ckpt", groups=groups, patterns=patterns, single_file=True)
    return open_checkpoint(root)


GOOD = {
    "noise_refiner.0": 8,
    "context_refiner.0": 7,
    "layers.0": 8,
    "layers.1": 8,
    "cap_embedder": 1,
}


def test_the_good_layout_returns_the_block_counts_per_kind(tmp_path):
    # Bug caught: counts swapped between kinds, the non-block cap_embedder group counted as a block, or a good
    # checkpoint refused.
    assert check_zimage_groups(_ckpt(tmp_path, GOOD)) == {
        "noise_refiner": 1,
        "context_refiner": 1,
        "layers": 2,
    }


def test_the_non_block_group_is_optional(tmp_path):
    # Bug caught: cap_embedder required (a checkpoint that keeps it as an extra would be refused).
    spec = {k: v for k, v in GOOD.items() if k != "cap_embedder"}
    assert check_zimage_groups(_ckpt(tmp_path, spec))["layers"] == 2


def test_a_hole_in_an_index_sequence_is_refused(tmp_path):
    # Bug caught: a missing block (layers.1) accepted as if the download were complete.
    spec = {**GOOD}
    del spec["layers.1"]
    spec["layers.2"] = 8
    with pytest.raises(DFloatFormatError, match="not contiguous"):
        check_zimage_groups(_ckpt(tmp_path, spec))


def test_a_context_refiner_group_holding_a_modulation_matrix_is_refused(tmp_path):
    # Bug caught: a context block with adaLN_modulation (mflux's context blocks have none) treated as a valid
    # context block, so its extra matrix would have nowhere to land.
    patterns = {**PATTERNS, r"context_refiner\.\d+": REFINER_SUBS}
    spec = {**GOOD, "context_refiner.0": 8}
    with pytest.raises(DFloatFormatError, match=r"context_refiner\.0"):
        check_zimage_groups(_ckpt(tmp_path, spec, patterns))


def test_refiner_counts_that_differ_are_refused(tmp_path):
    # Bug caught: noise and context refiner lists of different lengths (mflux builds both from one count).
    spec = {**GOOD, "noise_refiner.1": 8}
    with pytest.raises(DFloatFormatError, match="refiner counts differ"):
        check_zimage_groups(_ckpt(tmp_path, spec))


def test_a_stray_group_is_refused(tmp_path):
    # Bug caught: a group of another model (foo.0) passed through.
    patterns = {**PATTERNS, r"foo\.\d+": ("a",)}
    with pytest.raises(DFloatFormatError, match="not a Z-Image block group"):
        check_zimage_groups(_ckpt(tmp_path, {**GOOD, "foo.0": 1}, patterns))


def test_a_missing_kind_is_refused(tmp_path):
    # Bug caught: a checkpoint without any layers.* treated as zero valid blocks instead of refused.
    spec = {k: v for k, v in GOOD.items() if not k.startswith("layers")}
    with pytest.raises(DFloatFormatError, match="no layers groups"):
        check_zimage_groups(_ckpt(tmp_path, spec))


def test_the_non_block_group_with_other_matrices_is_refused(tmp_path):
    # Bug caught: a cap_embedder group holding some other matrix accepted, then decoded into the wrong weight.
    patterns = {**PATTERNS, "cap_embedder": ("2",)}
    with pytest.raises(DFloatFormatError, match="cap_embedder"):
        check_zimage_groups(_ckpt(tmp_path, GOOD, patterns))


@pytest.mark.mflux
def test_the_map_places_what_mflux_places():
    # Bug caught: a rename of mflux 0.20.0's ZImageWeightMapping missed or misread. Literals from
    # models/z_image/weights/z_image_weight_mapping.py:384-441 (t_embedder mlp.0/mlp.2 -> linear1/linear2; the final
    # layer's adaLN_modulation.1 -> .0) and :555-614 (layers.{layer}.*).
    m = zimage_name_map()
    assert m.place("layers.29.feed_forward.w3.weight") == Placement(
        block="layers.29", attr="feed_forward.w3"
    )
    assert m.place("noise_refiner.1.adaLN_modulation.0.weight") == Placement(
        block="noise_refiner.1", attr="adaLN_modulation.0"
    )
    with pytest.raises(DFloatIntegrationError):
        m.place("context_refiner.0.adaLN_modulation.0.weight")  # context blocks have no modulation
    assert m.param_name("t_embedder.mlp.0.weight") == "t_embedder.linear1.weight"
    assert m.param_name("t_embedder.mlp.2.bias") == "t_embedder.linear2.bias"
    assert (
        m.param_name("all_final_layer.2-1.adaLN_modulation.1.weight")
        == "all_final_layer.2-1.adaLN_modulation.0.weight"
    )
    assert m.param_name("cap_embedder.1.weight") == "cap_embedder.1.weight"
    assert m.param_name("layers.3.attention.norm_k.weight") == "layers.3.attention.norm_k.weight"
    assert m.transform_of("t_embedder.linear1.weight") is None
