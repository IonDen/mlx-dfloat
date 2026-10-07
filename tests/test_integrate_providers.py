import dataclasses
from functools import partial

import mlx.core as mx
import numpy as np
import pytest
from tests._flux_fakes import (
    DOUBLE_SUBS,
    FF,
    FLUX_TABLE,
    D,
    FakeSeamTransformer,
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
    # Bug caught: verify() reading the words without clearing them, so a second call raises the
    # same error again for a step that was already checked.
    corrupt.verify()  # nothing pending: a no-op, not a repeat of the error


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


@pytest.mark.metal
def test_df11_provider_decodes_on_the_metal_backend_by_default():
    # Bug caught: the default decode switched to the slow CPU reference (every bench and the
    # adapter would still be bit-exact, just ~1000x slower, so no parity test would notice).
    shapes = _shapes(1, 0)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(4))
    provider = DF11Provider(groups, names, FLUX_TABLE)
    assert isinstance(provider._decode, partial)
    assert provider._decode.func is decode_group
    assert provider._decode.keywords == {"backend": "metal"}


def _counting_reference():
    calls = []

    def decode(group):
        calls.append(group.name)
        return decode_group(group, backend="reference")

    return calls, decode


def test_df11_provider_refuses_a_block_without_matrix_names_before_decoding():
    # Bug caught: a block with a resident group but no matrix_names entry decoding first and then
    # escaping as a bare KeyError, with a status word already queued for a block that never ran.
    shapes = _shapes(1, 1)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(5))
    del names["single_transformer_blocks.0"]
    calls, decode = _counting_reference()
    provider = DF11Provider(groups, names, FLUX_TABLE, decode=decode)
    with pytest.raises(
        DFloatIntegrationError, match=r"single_transformer_blocks\.0: no matrix names"
    ):
        provider.weights_for("single_transformer_blocks.0", shapes["single_transformer_blocks.0"])
    assert calls == []
    assert provider.pending == []
    assert provider.launches == 0


def test_df11_provider_refuses_matrix_names_that_do_not_match_the_decoded_group():
    # Bug caught: dropping the count guard (zip(strict=True) would then raise a bare ValueError) or
    # the placement guard (another block's names, same count, would place this block's matrices
    # onto the right attributes of the wrong block and pass silently).
    shapes = _shapes(2, 0)
    groups, names, _source = df11_groups(shapes, np.random.default_rng(6))
    extra = {**names, "transformer_blocks.0": (*names["transformer_blocks.0"], "x.weight")}
    provider = DF11Provider(groups, extra, FLUX_TABLE, decode=_counting_reference()[1])
    with pytest.raises(DFloatIntegrationError, match="14 matrices decoded for 15 names"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])
    swapped = {**names, "transformer_blocks.0": names["transformer_blocks.1"]}
    provider = DF11Provider(groups, swapped, FLUX_TABLE, decode=_counting_reference()[1])
    with pytest.raises(DFloatIntegrationError, match=r"not a matrix of transformer_blocks\.0"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])


def _wrong_shape_and_missing(shapes):
    double = {a: mx.zeros(s, dtype=mx.bfloat16) for a, s in shapes["transformer_blocks.0"].items()}
    wrong = {**double, "ff.linear1": mx.zeros((FF, D + 1), dtype=mx.bfloat16)}
    missing = {a: w for a, w in double.items() if a != "ff.linear2"}
    return wrong, missing


def test_reuse_provider_refuses_a_shape_that_does_not_match_the_block():
    # Bug caught: a reuse dict decoded from another model's block (or missing a matrix) reaching the
    # matmul; ReuseProvider must refuse it itself, whatever the seam checks afterwards.
    shapes = _shapes(1, 0)
    wrong, missing = _wrong_shape_and_missing(shapes)
    with pytest.raises(DFloatIntegrationError, match=r"ff\.linear1: weight has shape"):
        ReuseProvider({"transformer_blocks": wrong}, FLUX_TABLE).weights_for(
            "transformer_blocks.0", shapes["transformer_blocks.0"]
        )
    with pytest.raises(DFloatIntegrationError, match=r"no weight for 'ff\.linear2'"):
        ReuseProvider({"transformer_blocks": missing}, FLUX_TABLE).weights_for(
            "transformer_blocks.0", shapes["transformer_blocks.0"]
        )


def test_resident_provider_refuses_a_shape_that_does_not_match_the_block():
    # Bug caught: ResidentProvider handing back a dict with a transposed or missing matrix without
    # checking it against the block's recorded shapes.
    shapes = _shapes(1, 0)
    wrong, missing = _wrong_shape_and_missing(shapes)
    with pytest.raises(DFloatIntegrationError, match=r"ff\.linear1: weight has shape"):
        ResidentProvider({"transformer_blocks.0": wrong}).weights_for(
            "transformer_blocks.0", shapes["transformer_blocks.0"]
        )
    with pytest.raises(DFloatIntegrationError, match=r"no weight for 'ff\.linear2'"):
        ResidentProvider({"transformer_blocks.0": missing}).weights_for(
            "transformer_blocks.0", shapes["transformer_blocks.0"]
        )


def test_streaming_bf16_provider_reads_each_block_bit_exactly_from_the_shards_and_keeps_nothing(
    tmp_path,
):
    # Bug caught: a shard read landing on the wrong attribute (name map bypassed), a dtype cast on
    # the way (bf16 -> f16 loses bits), or the provider caching the block dicts (every block resident).
    from tests._df11_fixtures import write_bf16_original

    from mlx_dfloat._safetensors import read_header
    from mlx_dfloat.integrate.providers import StreamingBF16Provider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    _groups, names, source = df11_groups(shapes, np.random.default_rng(5))
    bits = {m: arr for per in source.values() for m, arr in per.items()}
    root = write_bf16_original(tmp_path / "bf16", bits)
    index = {}
    for shard in sorted(root.glob("*.safetensors")):
        index.update({n: (shard, info) for n, info in read_header(shard).items()})
    provider = StreamingBF16Provider(index, names, FLUX_TABLE)
    for block, per in shapes.items():
        weights = provider.weights_for(block, per)
        assert weights.keys() == per.keys()
        for matrix_name in names[block]:
            attr = FLUX_TABLE.place(matrix_name).attr
            assert np.array_equal(np.array(weights[attr].view(mx.uint16)), bits[matrix_name]), (
                matrix_name
            )
    assert provider.reads == 20
    assert provider.launches == 0
    assert not provider.launching
    assert set(vars(provider)) == {"_index", "_matrix_names", "_name_map", "launches", "reads"}
    assert not any(isinstance(v, mx.array) for v in vars(provider).values())


def test_streaming_bf16_provider_refuses_a_block_without_matrix_names():
    # Bug caught: a block with no matrix_names entry reaching the index lookup loop instead of a
    # clear refusal (a bare TypeError iterating over None).
    from mlx_dfloat.integrate.providers import StreamingBF16Provider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=1)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    _groups, names, _source = df11_groups(shapes, np.random.default_rng(6))
    del names["single_transformer_blocks.0"]
    provider = StreamingBF16Provider({}, names, FLUX_TABLE)
    with pytest.raises(
        DFloatIntegrationError, match=r"single_transformer_blocks\.0: no matrix names"
    ):
        provider.weights_for("single_transformer_blocks.0", shapes["single_transformer_blocks.0"])
    assert provider.reads == 0


def test_streaming_bf16_provider_refuses_a_name_missing_from_the_index():
    # Bug caught: a missing matrix silently left at its placeholder (a zero-size matmul later).
    from mlx_dfloat.integrate.providers import StreamingBF16Provider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    _groups, names, _source = df11_groups(shapes, np.random.default_rng(6))
    provider = StreamingBF16Provider({}, names, FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match="not in the BF16 index"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])


def test_streaming_bf16_provider_refuses_a_name_whose_placement_is_another_block(tmp_path):
    # Bug caught: dropping the placement guard would let another block's matrix names (same count,
    # present in the index) place their weights onto this block's attributes and pass silently.
    from tests._df11_fixtures import write_bf16_original

    from mlx_dfloat._safetensors import read_header
    from mlx_dfloat.integrate.providers import StreamingBF16Provider

    tf = FakeSeamTransformer(Recorder(), n_double=2, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    _groups, names, source = df11_groups(shapes, np.random.default_rng(7))
    bits = {m: arr for per in source.values() for m, arr in per.items()}
    root = write_bf16_original(tmp_path / "bf16", bits)
    index = {}
    for shard in sorted(root.glob("*.safetensors")):
        index.update({n: (shard, info) for n, info in read_header(shard).items()})
    swapped = {**names, "transformer_blocks.0": names["transformer_blocks.1"]}
    provider = StreamingBF16Provider(index, swapped, FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match=r"not a matrix of transformer_blocks\.0"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])


def test_streaming_bf16_provider_refuses_a_shard_tensor_of_the_wrong_shape(tmp_path):
    # Bug caught: a shard tensor whose shape differs from the block's placeholder (e.g. transposed)
    # reaching the seam and the first matmul instead of being refused here, where it is still legible.
    from tests._df11_fixtures import write_bf16_original

    from mlx_dfloat._safetensors import read_header
    from mlx_dfloat.integrate.providers import StreamingBF16Provider

    tf = FakeSeamTransformer(Recorder(), n_double=1, n_single=0)
    shapes = install_placeholders(block_lists(tf), FLUX_TABLE)
    _groups, names, source = df11_groups(shapes, np.random.default_rng(7))
    bits = {m: arr for per in source.values() for m, arr in per.items()}
    root = write_bf16_original(tmp_path / "bf16", bits)
    index = {}
    for shard in sorted(root.glob("*.safetensors")):
        index.update({n: (shard, info) for n, info in read_header(shard).items()})
    matrix_name = "transformer_blocks.0.ff.net.0.proj.weight"
    attr = FLUX_TABLE.place(matrix_name).attr
    transposed = tuple(reversed(shapes["transformer_blocks.0"][attr]))
    wrong_path = tmp_path / "wrong.safetensors"
    mx.save_safetensors(str(wrong_path), {"x": mx.zeros(transposed, dtype=mx.bfloat16)})
    index[matrix_name] = (wrong_path, read_header(wrong_path)["x"])
    provider = StreamingBF16Provider(index, names, FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match="shard tensor has shape"):
        provider.weights_for("transformer_blocks.0", shapes["transformer_blocks.0"])


def test_streaming_bf16_provider_refuses_the_none_policy():
    # Bug caught: StreamingBF16Provider advertising every policy, so "none" is accepted and the
    # whole BF16 transformer (fresh arrays per block) is kept alive in one step's lazy graph.
    from mlx_dfloat.integrate.providers import StreamingBF16Provider
    from mlx_dfloat.integrate.seam import attach_state

    shapes = _shapes(1, 1)
    provider = StreamingBF16Provider({}, {}, FLUX_TABLE)
    with pytest.raises(DFloatIntegrationError, match="'none'"):
        attach_state(provider, shapes, eval_policy="none")
    attach_state(provider, shapes, eval_policy="per-block")
    attach_state(provider, shapes, eval_policy="depth2")
