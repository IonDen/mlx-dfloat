"""``_hub.load_into`` against mflux 0.23's loader, which refuses a component whose required weights are missing.

Offline: the two mflux modules ``load_into`` imports are replaced by fakes in ``sys.modules``.
"""

import sys
import types

import pytest

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.mflux import _hub


class ModelConfigError(ValueError):
    """mflux 0.23.0's class, as declared in ``mflux/utils/exceptions.py:44`` (a ``ValueError``)."""


MESSAGE = (
    "The vae weights in /base/vae do not fit this model: 2 required weights have no match "
    "(expected names like encoder.conv_in.weight; found none under the names this component reads)."
)


def _install_loader(monkeypatch, load):
    loader = types.ModuleType("mflux.models.common.weights.loading.weight_loader")
    loader.WeightLoader = types.SimpleNamespace(load=load)  # type: ignore[attr-defined]
    applier = types.ModuleType("mflux.models.common.weights.loading.weight_applier")
    applier.WeightApplier = types.SimpleNamespace(  # type: ignore[attr-defined]
        apply_and_quantize=lambda **kw: pytest.fail("applied after a refused load")
    )
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, applier.__name__, applier)


def _definition():
    return types.SimpleNamespace(
        get_components=lambda: [types.SimpleNamespace(name="vae")],
        get_download_patterns=lambda: ["vae/*.safetensors"],
    )


def test_mfluxs_missing_weights_refusal_becomes_a_format_error_naming_the_base(
    monkeypatch, tmp_path
):
    # Bug caught: the required-weights refusal (mflux 0.23's ModelConfigError) escaping untranslated, so the
    # command prints a traceback for a wrong base directory instead of naming it.
    def load(**kwargs):
        raise ModelConfigError(MESSAGE)

    _install_loader(monkeypatch, load)
    with pytest.raises(DFloatFormatError) as info:
        _hub.load_into(tmp_path, _definition(), {"vae": object()}, what="Z-Image base")
    text = str(info.value)
    assert text.startswith(f"{tmp_path}: not a Z-Image base with ['vae']")
    assert MESSAGE in text
    assert isinstance(info.value.__cause__, ModelConfigError)


# mflux 0.23.0's wording (weight_loader.py:500-505, WeightLoader._describe_missing_weights): at most three names, the
# total count, then what the directory holds instead.
ONE_MISSING = (
    "The vae weights in /base/vae do not fit this model: 1 required weight has no match "
    "(expected names like bn.running_mean; found bn.num_batches_tracked). The checkpoint was probably converted "
    "for another program or model. Use one in the original layout, or one written by mflux-save."
)
MANY_MISSING = (
    "The text_encoder weights in /base/text_encoder do not fit this model: 374 required weights have no match "
    "(expected names like model.language_model.layers.2.input_layernorm.weight, "
    "model.language_model.layers.2.post_attention_layernorm.weight, "
    "model.language_model.layers.2.self_attn.q_proj.weight; found none under the names this component reads). "
    "The checkpoint was probably converted for another program or model."
)


def _refused_with(monkeypatch, tmp_path, message, error=ModelConfigError):
    def load(**kwargs):
        raise error(message)

    _install_loader(monkeypatch, load)
    with pytest.raises(DFloatFormatError) as info:
        _hub.load_into(tmp_path, _definition(), {"vae": object()}, what="FLUX.2 Klein base")
    return str(info.value)


def test_a_missing_required_tensor_is_refused_by_its_name_before_mfluxs_wording(
    monkeypatch, tmp_path
):
    # Bug caught: mflux 0.23's required-weights refusal wrapped as a generic "not a … base", so the message never says
    # which tensor the base lacks (the name was only buried in mflux's own sentence, after ours).
    text = _refused_with(monkeypatch, tmp_path, ONE_MISSING)
    assert text.startswith(
        f"{tmp_path}: not a FLUX.2 Klein base with ['vae']: vae bn.running_mean not in the base"
    )
    assert ONE_MISSING in text


def test_more_missing_tensors_than_mflux_names_are_counted(monkeypatch, tmp_path):
    # Bug caught: the count dropped (a base missing 374 tensors read as missing three), or computed from the names
    # mflux quotes instead of the total it reports (off by the three shown).
    text = _refused_with(monkeypatch, tmp_path, MANY_MISSING)
    assert (
        "text_encoder model.language_model.layers.2.input_layernorm.weight, "
        "model.language_model.layers.2.post_attention_layernorm.weight, "
        "model.language_model.layers.2.self_attn.q_proj.weight and 371 more not in the base"
    ) in text


def test_another_value_error_keeps_the_generic_refusal(monkeypatch, tmp_path):
    # Bug caught: any ValueError from the loader (a bad shard header, an unreadable index) reported as missing tensors
    # it never named.
    text = _refused_with(monkeypatch, tmp_path, "Unsupported safetensors dtype F8_E4M3", ValueError)
    assert text.startswith(
        f"{tmp_path}: not a FLUX.2 Klein base with ['vae'] (a DFloat11 checkpoint"
    )
    assert "not in the base" not in text


# The contract: the message is built by mflux 0.23's own WeightLoader._describe_missing_weights, not copied. Its inputs
# are minimal: a component namespace with the three fields it reads (name, num_blocks, num_layers; the counts given, so
# mflux does not detect them from the weights), an empty mapping (every raw name is then unmapped) and raw weights of
# one tiny array.
@pytest.mark.mflux
@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        pytest.param(["bn.running_mean"], "vae bn.running_mean", id="one"),
        pytest.param(
            [f"decoder.up.{i}.weight" for i in range(5)],
            "vae decoder.up.0.weight, decoder.up.1.weight, decoder.up.2.weight and 2 more",
            id="five",
        ),
    ],
)
def test_the_parser_reads_mflux_023s_own_missing_weights_message(missing, expected):
    # Red when: mflux's wording drifts from the pattern (the singular "weight has", the count, the "found" clause), or
    # the count is dropped or taken from the names mflux shows.
    import mlx.core as mx

    loader = pytest.importorskip("mflux.models.common.weights.loading.weight_loader")
    component = types.SimpleNamespace(name="vae", num_blocks=0, num_layers=0)
    raw = {"bn.num_batches_tracked": mx.zeros((1,))}
    message = loader.WeightLoader._describe_missing_weights(
        component, "/base/vae", [], missing, raw
    )
    assert _hub.missing_weights(message) == expected
