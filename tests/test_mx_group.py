import mlx.core as mx
import numpy as np
import pytest
from tests._decode_fixtures import encoder_group, short_form_group
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat.errors import DFloatBackendError
from mlx_dfloat.format import MxGroup, load_group_mx, open_checkpoint


def test_to_mx_carries_host_metadata_and_evaluated_arrays():
    # Bug caught: metadata computed from the wrong array, or positions left as bytes.
    arrays = encoder_group(random_bf16(np.random.default_rng(0), (3000,)))
    g = arrays.to_mx(name="blocks.0")
    assert isinstance(g, MxGroup)
    assert g.name == "blocks.0"
    assert (g.n_elements, g.n_launch, g.max_elements_per_block) == (3000, 1, 3000)
    assert g.positions.dtype == mx.uint32
    assert g.positions.shape == (2,)
    assert g.intervals.tolist() == [3000]
    assert g.luts.shape == tuple(arrays.luts.shape)
    assert g.n_luts == arrays.luts.shape[0]
    assert np.array_equal(np.array(g.encoded_exponent), arrays.encoded_exponent)


def test_short_form_group_launches_one_fewer_threadgroup_than_its_byte_blocks():
    # Bug caught: n_launch = n_blocks (the last code straddles into a code-free tail block).
    arrays, _ = short_form_group()
    assert (arrays.n_bytes, arrays.n_blocks, arrays.output_positions.tolist()) == (
        4097,
        2,
        [0, 32768],
    )
    assert arrays.to_mx().n_launch == 1


def test_oversized_elements_is_a_backend_error(monkeypatch):
    # Bug caught: the int32 element bound not enforced.
    import mlx_dfloat.format as fmt

    arrays = encoder_group(random_bf16(np.random.default_rng(2), (500,)))
    assert arrays.n_bytes < 400 < arrays.n_elements
    monkeypatch.setattr(fmt, "MAX_ARRAY_ELEMENTS", 400)
    with pytest.raises(DFloatBackendError, match="elements"):
        arrays.to_mx()


def test_oversized_bytes_is_a_backend_error(monkeypatch):
    # Bug caught: only the element bound being checked (bytes are checked first, so the message names bytes).
    import mlx_dfloat.format as fmt

    arrays = encoder_group(random_bf16(np.random.default_rng(2), (500,)))
    monkeypatch.setattr(fmt, "MAX_ARRAY_ELEMENTS", arrays.n_bytes - 1)
    with pytest.raises(DFloatBackendError, match="bytes"):
        arrays.to_mx()


def test_load_group_mx_from_a_written_checkpoint(tmp_path):
    # Bug caught: the read-only memmap -> mx.array path (never exercised elsewhere) breaking on real files.
    bits = random_bf16(np.random.default_rng(4), (6000,))
    root = write_checkpoint(
        tmp_path / "ckpt",
        groups={"blocks.0": [bits[:2000].reshape(40, 50), bits[2000:].reshape(80, 50)]},
        pattern=r"blocks\.\d+",
        sub_paths=["a", "b"],
    )
    g = load_group_mx(open_checkpoint(root).groups["blocks.0"])
    assert g.name == "blocks.0"
    assert g.n_elements == 6000
    assert g.split_positions == (2000,)
    assert np.array_equal(
        np.array(g.sign_mantissa),
        np.array(open_checkpoint(root).groups["blocks.0"].load().sign_mantissa),
    )
