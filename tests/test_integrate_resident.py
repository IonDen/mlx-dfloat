"""Non-block DF11 groups: placeholders, decode at set load, install and clear (no mflux)."""

from functools import partial
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from tests._decode_fixtures import encoder_group
from tests._df11_fixtures import random_bf16
from tests._flux_fakes import _Sub

from mlx_dfloat.decode import STATUS_BROKEN_CHAIN, DecodeResult, decode_group
from mlx_dfloat.errors import DFloatFormatError, DFloatIntegrationError
from mlx_dfloat.integrate.names import StaticNameMap
from mlx_dfloat.integrate.resident import (
    clear_nonblock,
    decode_nonblock,
    install_nonblock,
    install_nonblock_placeholders,
)

SHAPE = (4, 6)
MATRIX = "cap_embedder.1.weight"
TABLE = {"layers": {"attention.to_q": "attention.to_q"}}
REFERENCE = partial(decode_group, backend="reference")


def _module():
    return SimpleNamespace(
        cap_embedder=[nn.RMSNorm(6), nn.Linear(6, 4)], proj=_Sub(inner=nn.Linear(6, 4))
    )


def _group():
    source = random_bf16(np.random.default_rng(3), SHAPE)
    return encoder_group(source.reshape(-1)).to_mx(name="cap_embedder"), source


def _owner_weight(module, path):
    node = module
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node.weight


def test_a_nonblock_group_decodes_into_its_renamed_parameter_bit_exact_and_clears_back():
    # Bug caught: the matrix placed at its checkpoint name instead of the renamed parameter (the module has no
    # `cap_embedder.1` of the renamed kind to land on), bits shuffled by a wrong split or reshape, or clear_nonblock
    # leaving the decoded weight referenced after the set drops.
    names = StaticNameMap(TABLE, renames={MATRIX: "proj.inner.weight"})
    module = _module()
    group, source = _group()
    shapes = install_nonblock_placeholders(module, [MATRIX], names)
    assert shapes == {"proj.inner.weight": SHAPE}
    assert module.proj.inner.weight.size == 0
    weights = decode_nonblock(
        {"cap_embedder": group}, {"cap_embedder": (MATRIX,)}, shapes, names, decode=REFERENCE
    )
    install_nonblock(module, weights)
    assert np.array_equal(np.array(module.proj.inner.weight.view(mx.uint16)), source)
    assert module.cap_embedder[1].weight.size == 24  # the unrenamed module was never touched
    clear_nonblock(module, shapes)
    assert module.proj.inner.weight.size == 0


def test_a_nonblock_matrix_that_lands_on_a_non_linear_is_refused():
    # Bug caught: a compressed matrix assigned to a norm scale (a shape-compatible but wrong module).
    with pytest.raises(DFloatIntegrationError, match="not a matrix module"):
        install_nonblock_placeholders(_module(), ["cap_embedder.0.weight"], StaticNameMap(TABLE))


def test_a_nonblock_matrix_with_a_transform_is_refused():
    # Bug caught: a transformed matrix decoded and installed raw (the transform silently skipped).
    names = StaticNameMap(TABLE, transforms={MATRIX: lambda a: a})
    with pytest.raises(DFloatIntegrationError, match="a transform on a compressed matrix"):
        install_nonblock_placeholders(_module(), [MATRIX], names)


def test_a_nonblock_name_that_is_not_a_weight_is_refused():
    # Bug caught: a bias name taken for a matrix (its owner path would strip nothing and resolve a parameter).
    with pytest.raises(DFloatIntegrationError, match=r"must land on a \.weight"):
        install_nonblock_placeholders(_module(), ["cap_embedder.1.bias"], StaticNameMap(TABLE))


def test_a_nonblock_decode_error_is_raised_at_set_load_with_the_group_name():
    # Bug caught: a non-block status word deferred like a block's and never read (no step ever verifies it).
    names = StaticNameMap(TABLE)
    module = _module()
    group, _source = _group()
    shapes = install_nonblock_placeholders(module, [MATRIX], names)

    def flagged(g):
        ok = REFERENCE(g)
        status = mx.full(ok.status.shape, STATUS_BROKEN_CHAIN, dtype=mx.uint32)
        return DecodeResult(
            bits=ok.bits,
            status=status,
            backend=ok.backend,
            direct_blocks=0,
            threadgroup_bytes=0,
        )

    with pytest.raises(DFloatFormatError, match="cap_embedder"):
        decode_nonblock(
            {"cap_embedder": group}, {"cap_embedder": (MATRIX,)}, shapes, names, decode=flagged
        )


def test_a_group_that_is_not_resident_and_a_size_that_does_not_fit_are_refused():
    # Bug caught: a missing group skipped silently (the weight stays a placeholder), or a matrix decoded for a
    # parameter of another size reshaped into garbage.
    names = StaticNameMap(TABLE)
    module = _module()
    group, _source = _group()
    shapes = install_nonblock_placeholders(module, [MATRIX], names)
    with pytest.raises(DFloatIntegrationError, match="no resident DF11 group"):
        decode_nonblock({}, {"cap_embedder": (MATRIX,)}, shapes, names, decode=REFERENCE)
    with pytest.raises(DFloatIntegrationError, match="decoded 24 elements"):
        decode_nonblock(
            {"cap_embedder": group},
            {"cap_embedder": (MATRIX,)},
            {MATRIX: (5, 6)},
            names,
            decode=REFERENCE,
        )
    with pytest.raises(DFloatIntegrationError, match="1 matrices decoded for 2 names"):
        decode_nonblock(
            {"cap_embedder": group},
            {"cap_embedder": (MATRIX, "cap_embedder.2.weight")},
            shapes,
            names,
            decode=REFERENCE,
        )


def _two_groups(module, names):
    """Two one-matrix groups landing on proj.inner and cap_embedder.1 (renamed and plain)."""
    rng = np.random.default_rng(9)
    a, b = random_bf16(rng, SHAPE), random_bf16(rng, SHAPE)
    groups = {
        "proj": encoder_group(a.reshape(-1)).to_mx(name="proj"),
        "cap_embedder": encoder_group(b.reshape(-1)).to_mx(name="cap_embedder"),
    }
    matrices = {"proj": ("proj.inner.weight",), "cap_embedder": (MATRIX,)}
    shapes = install_nonblock_placeholders(module, ["proj.inner.weight", MATRIX], names)
    return groups, matrices, shapes, {"proj.inner.weight": a, MATRIX: b}


@pytest.mark.parametrize(("together", "evals"), [(False, [2, 2]), (True, [4])])
def test_the_nonblock_groups_evaluate_per_group_or_in_one_eval(monkeypatch, together, evals):
    # Bug caught: the per-call path (Krea 2 decodes its seven groups at every transformer call) paying one eval per
    # group, or the default changed for the families that decode once at set load (it stays one eval per group:
    # status word + matrix each). The bits are the same either way.
    from mlx_dfloat.integrate import resident

    seen = []
    real = resident._eval
    monkeypatch.setattr(resident, "_eval", lambda *a: (seen.append(len(a)), real(*a)))
    names = StaticNameMap(TABLE)
    module = _module()
    groups, matrices, shapes, sources = _two_groups(module, names)
    weights = decode_nonblock(
        groups, matrices, shapes, names, decode=REFERENCE, eval_together=together
    )
    assert seen == evals  # one status word and one matrix per group, per eval
    for param, source in sources.items():
        assert np.array_equal(np.array(weights[param].view(mx.uint16)), source), param


def test_a_decode_error_in_one_eval_still_names_its_group():
    # Bug caught: the one-eval path checking the status words before they are evaluated, or naming the first group
    # for an error in the second.
    names = StaticNameMap(TABLE)
    groups, matrices, shapes, _sources = _two_groups(_module(), names)

    def flag_cap(g):
        ok = REFERENCE(g)
        if g.name != "cap_embedder":
            return ok
        status = mx.full(ok.status.shape, STATUS_BROKEN_CHAIN, dtype=mx.uint32)
        return DecodeResult(
            bits=ok.bits, status=status, backend=ok.backend, direct_blocks=0, threadgroup_bytes=0
        )

    with pytest.raises(DFloatFormatError, match="cap_embedder"):
        decode_nonblock(groups, matrices, shapes, names, decode=flag_cap, eval_together=True)
