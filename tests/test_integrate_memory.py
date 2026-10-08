from types import SimpleNamespace

import pytest

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.integrate import memory
from mlx_dfloat.integrate.memory import (
    budget_bytes,
    budget_for,
    decoded_bytes,
    largest_decoded_bytes,
)


def _group(name, n_elements):
    return SimpleNamespace(name=name, tensors={"sign_mantissa": SimpleNamespace(nbytes=n_elements)})


def test_decoded_bytes_is_two_per_element_from_the_sign_mantissa_header():
    # Bug caught: using n_elements as the decoded byte count directly instead of doubling it (BF16 is
    # 2 bytes per element, and sign_mantissa is 1 byte per element).
    assert decoded_bytes(_group("g", 1000)) == 2000


def test_largest_decoded_bytes_per_kind():
    # Bug caught: taking the last group's size per kind instead of the max (or mixing kinds together).
    from tests._flux_fakes import FLUX_TABLE

    ckpt = SimpleNamespace(
        groups={
            # The larger double group first: "the last one seen" and "the max" must differ.
            "transformer_blocks.0": _group("transformer_blocks.0", 340),
            "transformer_blocks.1": _group("transformer_blocks.1", 300),
            "single_transformer_blocks.0": _group("single_transformer_blocks.0", 140),
        }
    )
    assert largest_decoded_bytes(ckpt, FLUX_TABLE) == {
        "transformer_blocks": 680,
        "single_transformer_blocks": 280,
    }


def test_budget_bytes_subtracts_the_reserve_from_the_devices_working_set(monkeypatch):
    # Bug caught: the reserve subtracted from the wrong figure (or ignored) or a hardcoded device size.
    monkeypatch.setattr(
        memory.mx, "device_info", lambda: {"max_recommended_working_set_size": 10 * 1024**3}
    )
    assert budget_bytes() == 8 * 1024**3
    assert budget_bytes(reserve_bytes=1) == 10 * 1024**3 - 1
    # Bug caught: a positional reserve accepted, so budget_bytes(total) reads a caller's budget as
    # the reserve and returns a nonsense figure instead of refusing the call.
    with pytest.raises(TypeError):
        budget_bytes(1)


def test_largest_decoded_bytes_skips_the_named_nonblock_groups():
    # Bug caught: kind_of("cap_embedder") raising for a checkpoint with a non-block group (every Z-Image plan_call
    # would fail), or a skipped group counted as a kind.
    from tests._family_fakes import ZIMAGE_TABLE

    ckpt = SimpleNamespace(
        groups={
            "layers.0": _group("layers.0", 100),
            "noise_refiner.0": _group("noise_refiner.0", 70),
            "cap_embedder": _group("cap_embedder", 999),
        }
    )
    assert largest_decoded_bytes(ckpt, ZIMAGE_TABLE, skip={"cap_embedder"}) == {
        "layers": 200,
        "noise_refiner": 140,
    }
    # The default still refuses a group of no kind: skipping is opt-in.
    with pytest.raises(DFloatIntegrationError):
        largest_decoded_bytes(ckpt, ZIMAGE_TABLE)


def test_budget_for_subtracts_a_2_gib_reserve_from_a_given_working_set():
    # Bug caught: the rule budget_bytes() applies on a device not available for a working set that
    # is not this device's (a CAPPED tier), or a reserve other than 2 GiB.
    # By hand: 11_453_246_122 - 2_147_483_648 = 9_305_762_474 (a 16 GB Mac at 2/3).
    assert budget_for(11_453_246_122) == 9_305_762_474
    assert budget_for(11_453_246_122, reserve_bytes=1) == 11_453_246_121
