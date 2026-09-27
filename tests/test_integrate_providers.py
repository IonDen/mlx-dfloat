import dataclasses

import mlx.core as mx
import numpy as np
import pytest
from tests._flux_fakes import (
    DOUBLE_SUBS,
    FLUX_TABLE,
    FakeTransformer,
    Recorder,
    block_lists,
    df11_groups,
    resident_dicts,
)

from mlx_dfloat.decode import STATUS_INVALID_CODE, decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.integrate.placeholders import install_placeholders
from mlx_dfloat.integrate.providers import (
    EVAL_POLICIES,
    DF11Provider,
    ResidentProvider,
    ReuseProvider,
)


def _shapes(n_double, n_single):
    tf = FakeTransformer(Recorder(), n_double=n_double, n_single=n_single)
    return install_placeholders(block_lists(tf), FLUX_TABLE)


def test_df11_provider_decodes_a_block_into_bit_exact_views_of_the_right_shape():
    # Bug caught: a matrix cut at the wrong split, reshaped in the wrong order, or a block decoded
    # twice (launches would exceed the calls).
    shapes = _shapes(2, 1)
    groups, names, source = df11_groups(shapes, np.random.default_rng(7))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=lambda g: decode_group(g, backend="reference")
    )
    assert provider.launching is True
    assert provider.policies == EVAL_POLICIES
    w = provider.weights_for("transformer_blocks.1", shapes["transformer_blocks.1"])
    assert provider.launches == 1
    for sub in DOUBLE_SUBS:
        attr = FLUX_TABLE.place(f"transformer_blocks.1.{sub}.weight").attr
        want = source["transformer_blocks.1"][f"transformer_blocks.1.{sub}.weight"]
        assert w[attr].dtype == mx.bfloat16
        assert w[attr].shape == want.shape
        assert np.array_equal(np.array(w[attr].view(mx.uint16)), want)


def test_df11_provider_defers_the_status_check_to_verify_and_reset_clears_it():
    # Bug caught: weights_for reading the pending status word on the host instead of deferring it
    # to verify (a sync before every block), or reset()/verify() not actually clearing or checking
    # the pending list (a stale or unread status word going undetected).
    class Unread:
        def __array__(self, *args, **kwargs):
            raise AssertionError("status was read")

    shapes = _shapes(1, 0)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(9))
    unread = DF11Provider(
        groups,
        names,
        FLUX_TABLE,
        decode=lambda g: dataclasses.replace(decode_group(g, backend="reference"), status=Unread()),
    )
    unread.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    assert [n for n, _s in unread.pending] == ["transformer_blocks.0"]
    unread.reset()  # after an interrupted step nothing stale is verified later
    assert unread.pending == []
    unread.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    with pytest.raises(AssertionError, match="status was read"):
        unread.verify()
    bad = mx.array([STATUS_INVALID_CODE], dtype=mx.uint32)
    corrupt = DF11Provider(
        groups,
        names,
        FLUX_TABLE,
        decode=lambda g: dataclasses.replace(decode_group(g, backend="reference"), status=bad),
    )
    corrupt.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    with pytest.raises(DFloatFormatError, match=r"transformer_blocks\.0: block 0: invalid code"):
        corrupt.verify()
    assert corrupt.pending == []


def test_df11_provider_refuses_a_block_without_a_resident_group_or_with_a_size_mismatch():
    # Bug caught: weights_for not checking block residency before decoding, or not validating a
    # decoded matrix's element count against its placeholder shape (either would surface later as
    # a confusing shape mismatch deep inside a matmul instead of a clear refusal here).
    shapes = _shapes(1, 1)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(3))
    provider = DF11Provider(
        groups, names, FLUX_TABLE, decode=lambda g: decode_group(g, backend="reference")
    )
    with pytest.raises(DFloatIntegrationError, match="no resident DF11 group"):
        provider.weights_for("transformer_blocks.9", shapes["transformer_blocks.0"])
    wrong = {**shapes["transformer_blocks.0"], "attn.to_q": (4, 5)}
    with pytest.raises(
        DFloatIntegrationError, match=r"decoded 16 elements, expected \(4, 5\) = 20"
    ):
        provider.weights_for("transformer_blocks.0", wrong)


def test_reuse_provider_hands_back_the_block_of_each_kind_and_never_launches():
    # Bug caught: ReuseProvider copying or re-decoding instead of returning the same per-kind dict
    # by identity for every block of that kind, or incrementing launches despite never decoding.
    shapes = _shapes(2, 2)
    dicts = resident_dicts(shapes)
    provider = ReuseProvider(
        {
            "transformer_blocks": dicts["transformer_blocks.0"],
            "single_transformer_blocks": dicts["single_transformer_blocks.0"],
        },
        FLUX_TABLE,
    )
    assert provider.launching is False
    assert (
        provider.weights_for("transformer_blocks.1", shapes["transformer_blocks.1"])
        is dicts["transformer_blocks.0"]
    )
    assert (
        provider.weights_for("single_transformer_blocks.1", shapes["single_transformer_blocks.1"])
        is dicts["single_transformer_blocks.0"]
    )
    assert provider.launches == 0
    provider.verify()
    provider.reset()


def test_reuse_provider_refuses_a_kind_it_was_not_given():
    # Bug caught: weights_for returning None, the wrong kind's dict, or otherwise not refusing a
    # block whose kind was never given to the provider.
    shapes = _shapes(1, 1)
    dicts = resident_dicts(shapes)
    provider = ReuseProvider({"transformer_blocks": dicts["transformer_blocks.0"]}, FLUX_TABLE)
    with pytest.raises(
        DFloatIntegrationError, match="no reusable block of kind 'single_transformer_blocks'"
    ):
        provider.weights_for("single_transformer_blocks.0", shapes["single_transformer_blocks.0"])


def test_resident_provider_hands_back_each_blocks_own_dict():
    # Bug caught: ResidentProvider returning a different block's dict (or none at all) instead of
    # the exact block's own resident dict, or not refusing a block that has no entry.
    shapes = _shapes(1, 1)
    dicts = resident_dicts(shapes)
    provider = ResidentProvider(dicts)
    assert (
        provider.weights_for("single_transformer_blocks.0", shapes["single_transformer_blocks.0"])
        is dicts["single_transformer_blocks.0"]
    )
    with pytest.raises(DFloatIntegrationError, match="not resident"):
        provider.weights_for("transformer_blocks.7", shapes["transformer_blocks.0"])
