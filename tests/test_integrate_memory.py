from types import SimpleNamespace

from mlx_dfloat.integrate.memory import decoded_bytes, largest_decoded_bytes


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
            "transformer_blocks.0": _group("transformer_blocks.0", 300),
            "transformer_blocks.1": _group("transformer_blocks.1", 340),
            "single_transformer_blocks.0": _group("single_transformer_blocks.0", 140),
        }
    )
    assert largest_decoded_bytes(ckpt, FLUX_TABLE) == {
        "transformer_blocks": 680,
        "single_transformer_blocks": 280,
    }
