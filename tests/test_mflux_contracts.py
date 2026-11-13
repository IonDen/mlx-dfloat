"""Contracts of the private mflux symbols the adapters reach, each pinned by its output on a fixed input.

Every expected value comes from the mflux 0.24 source (the file:line in each test), never from what an adapter
returns. A silent upstream change to one of these symbols turns this file red instead of shipping a wrong number.

Symbol (mflux 0.24 file:line)                                    Pinned by
WeightLoader._describe_missing_weights (weight_loader.py:490)     test_mflux_hub.py (text on a fixed bad checkpoint)
WeightLoader._load_component (weight_loader.py:120)               test_weight_loader_load_component_signature
AppleSiliconUtil.is_m1_or_m2 override (apple_silicon.py:9)        test_the_chip_check_override_reaches_a_stock_factory,
                                                                  test_mflux_compile.py (mechanics, a stand-in chip)
FluxInitializer._init_config (flux_initializer.py:169)            test_init_config_sets_exactly_these_attributes[flux1]
Flux2Initializer._init_config (flux2_initializer.py:75)           test_init_config_sets_exactly_these_attributes[flux2]
ZImageInitializer._init_config (z_image_initializer.py:52)        test_init_config_sets_exactly_these_attributes[zimage]
ErnieImageInitializer._init_config (ernie_image_initializer:42)   test_init_config_sets_exactly_these_attributes[ernie]
Krea2Initializer._init_config (krea2_initializer.py:234)          test_init_config_sets_exactly_these_attributes[krea2]
Qwen21Initializer.init_config (qwen21_initializer.py:119)         test_init_config_sets_exactly_these_attributes[qwen21]
Flux2Klein / QwenImage21 small_* class defaults                   test_the_class_defaults_our_constructors_rely_on
VAE decode_mode (stock initializers set LOW_MEMORY;              test_each_familys_load_vae_ends_on_the_legacy_decode,
load_vae pins LEGACY; vae_util.py:47, flux2 vae.py:78)           test_flux2_packed_decode_stays_legacy,
                                                                 test_zimage_vae_util_decode_stays_legacy,
                                                                 test_upstream_legacy_branch_is_the_023_computation_*
Transformer._apply_{joint,single}_transformer_block (flux1)       test_flux1_block_hooks_return_what_the_block_returns
Flux2Klein._predict, ZImage._predict, ErnieImage._predict,
Krea2._predict                                                    test_flux2_compile.py, test_mflux_compile.py,
                                                                  test_ernie_transformer.py, test_krea2_transformer.py
Flux2Klein._encode_prompt_pair, ZImage/ErnieImage/Krea2
._encode_prompts (overridden)                                    test_our_model_classes_override_only_the_listed_mflux_methods;
                                                                 bodies in each family's test_*_model.py
ErnieTransformer._pos_cache                                       test_ernie_model.py
Krea2._resolve_scheduler (krea2.py:209)                           test_krea2_resolve_scheduler_on_fixed_inputs
Krea2TextEncoder._template_end (text_encoder.py:117)              test_krea2_template_end_on_fixed_ids,
                                                                  test_krea2_init.py (a real tokenizer)
Qwen21PromptEncoder._system_prefix_length (prompt_encoder.py:36)  test_qwen21_system_prefix_length_counts_the_prefix_ids
Qwen21Transformer._pack_linears (qwen21_transformer.py:157)       test_qwen21_pack_linears_*, test_qwen21_mflux023.py
Qwen21Transformer._clear_inference_projections, _step_fn,
_image_step_fn, inference_projections                             test_qwen21_mflux023.py
QwenImage21._encode_prompt (qwen_image_21.py:192-207)            test_our_qwen21_class_never_defines_its_own_encode_prompt
"""

import inspect
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import pytest

pytestmark = pytest.mark.mflux

_THE_CONFIG = "the model config passed in"


def _described(obj: object, model_config: object) -> dict[str, Any]:
    """Every attribute ``obj`` holds: the passed config by identity, plain values as is, objects by type name."""
    from mflux.models.common.vae.tiling_config import TilingConfig

    out: dict[str, Any] = {}
    for name, value in vars(obj).items():
        if value is model_config:
            out[name] = _THE_CONFIG
        elif value is None or isinstance(value, dict | TilingConfig):
            out[name] = value
        else:
            out[name] = type(value).__name__
    return out


def _init_config_case(family: str) -> tuple[Any, str, dict[str, Any]]:
    from mflux.models.common.vae.tiling_config import TilingConfig

    base = {
        "prompt_cache": {},
        "model_config": _THE_CONFIG,
        "callbacks": "CallbackRegistry",
        "tiling_config": None,
    }
    if family == "flux1":
        from mflux.models.flux.flux_initializer import FluxInitializer

        return FluxInitializer._init_config, "schnell", base
    if family == "flux2":
        from mflux.models.flux2.flux2_initializer import Flux2Initializer

        return Flux2Initializer._init_config, "flux2-klein-4b", base
    if family == "zimage":
        from mflux.models.z_image.z_image_initializer import ZImageInitializer

        no_cache = {k: v for k, v in base.items() if k != "prompt_cache"}
        return ZImageInitializer._init_config, "z-image-turbo", no_cache
    if family == "ernie":
        from mflux.models.ernie_image.ernie_image_initializer import ErnieImageInitializer

        tiled = {**base, "tiling_config": TilingConfig(vae_decode_tiles_per_dim=None)}
        return ErnieImageInitializer._init_config, "ernie-image-turbo", tiled
    if family == "krea2":
        from mflux.models.krea2.krea2_initializer import Krea2Initializer

        with_lora = {**base, "lora_paths": None, "lora_scales": None}
        return Krea2Initializer._init_config, "krea-2", with_lora
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

    return Qwen21Initializer.init_config, "qwen-image-2.1", base


class _Bare:
    """A plain class: an instance starts with no attributes, so everything after the call was set by it."""


@pytest.mark.parametrize("family", ["flux1", "flux2", "zimage", "ernie", "krea2", "qwen21"])
def test_init_config_sets_exactly_these_attributes(family):
    # Bug caught: upstream's config step sets an attribute that generate_image reads (or drops/changes one, e.g. a
    # tiling_config default), and our constructors, which call this step and never mflux's full init, miss it.
    # Expected sets read from the 0.24 sources: flux_initializer.py:169-173, flux2_initializer.py:75-79,
    # z_image_initializer.py:52-55, ernie_image_initializer.py:42-46, krea2_initializer.py:234-240,
    # qwen21_initializer.py:119-123.
    from mflux.models.common.config.model_config import ModelConfig

    init_config, model_name, expected = _init_config_case(family)
    model_config = ModelConfig.from_name(model_name=model_name, base_model=None)
    model = object.__new__(_Bare)
    init_config(model, model_config)
    assert _described(model, model_config) == expected


def test_the_class_defaults_our_constructors_rely_on():
    # Bug caught: upstream flips a class default that only its own init overrides (flux2_klein.py:31-32,
    # qwen_image_21.py:30); our constructors bypass init, so the flipped default would route generate_image to the
    # small text encoder or decoder we never load.
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    assert (
        Flux2Klein.small_text_encoder,
        Flux2Klein.small_decoder,
        QwenImage21.small_text_encoder,
    ) == (False, False, False)


class _RecordingBlock:
    def __init__(self, returns: Any) -> None:
        self.returns = returns
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.returns


def _bits(a: mx.array) -> list[Any]:
    return a.view(mx.uint16).tolist()  # type: ignore[no-any-return]


_HIDDEN = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=mx.bfloat16)
_ENCODER = mx.array([[[7.0, 8.0]]], dtype=mx.bfloat16)


@pytest.mark.parametrize("kind", ["joint", "single"])
def test_flux1_block_hooks_return_what_the_block_returns(kind):
    # Bug caught: a hook renamed, re-signed or no longer passing the block's output through
    # (flux_transformer/transformer.py:82-127), so SeamMixin's override of it never runs or runs on other values.
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    rope = mx.array([0.5], dtype=mx.bfloat16)
    common = {
        "hidden_states": mx.zeros((1, 3, 2), dtype=mx.bfloat16),
        "encoder_hidden_states": mx.zeros((1, 1, 2), dtype=mx.bfloat16),
        "text_embeddings": mx.zeros((1, 2), dtype=mx.bfloat16),
        "image_rotary_embeddings": rope,
    }
    owner = SimpleNamespace(transformer_blocks=[None], single_transformer_blocks=[None])
    if kind == "joint":
        block = _RecordingBlock((_ENCODER, _HIDDEN))
        encoder, hidden = Transformer._apply_joint_transformer_block(
            owner, idx=0, block=block, controlnet_block_samples=None, **common
        )
        assert (_bits(encoder), _bits(hidden)) == (_bits(_ENCODER), _bits(_HIDDEN))
        names = ["encoder_hidden_states", "hidden_states", "rotary_embeddings", "text_embeddings"]
    else:
        block = _RecordingBlock(_HIDDEN)
        hidden = Transformer._apply_single_transformer_block(
            owner, idx=0, block=block, controlnet_single_block_samples=None, **common
        )
        assert _bits(hidden) == _bits(_HIDDEN)
        names = ["hidden_states", "rotary_embeddings", "text_embeddings"]
    assert sorted(block.kwargs) == names
    assert block.kwargs["rotary_embeddings"] is rope


@pytest.mark.parametrize(
    ("scheduler", "resolved"),
    [(None, "er_sde"), ("linear", "er_sde"), ("er_sde", "er_sde"), ("euler", "euler")],
)
def test_krea2_resolve_scheduler_on_fixed_inputs(scheduler, resolved):
    # Bug caught: upstream maps None/"linear" to another sampler (krea2.py:209-214); our DFloatKrea2 forwards the
    # user's value and would run a scheduler the user did not pick.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    assert Krea2._resolve_scheduler(scheduler) == resolved


def test_krea2_resolve_scheduler_refuses_an_unknown_name_with_value_error():
    # Bug caught: upstream stops raising ValueError (krea2.py:214), so our translation to DFloatUnsupportedError
    # (krea2/model.py:_resolve_scheduler) lets an unknown scheduler through.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    with pytest.raises(ValueError, match="Unknown Krea-2 scheduler 'ddim'"):
        Krea2._resolve_scheduler("ddim")


def test_weight_loader_load_component_signature():
    # Bug caught: a parameter added, dropped or reordered (weight_loader.py:120-125); scripts/encode_prompt.py:138
    # passes the four positionally, so a reorder loads with the wrong save_subdir silently.
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    params = list(inspect.signature(WeightLoader._load_component).parameters)
    assert params == ["root_path", "component", "raw_weights_cache", "save_subdir"]


def test_the_chip_check_override_reaches_a_stock_factory():
    # Bug caught: upstream's _predict factory reads the chip another way than AppleSiliconUtil.is_m1_or_m2
    # (krea2.py:203-205), so eager_factories() no longer turns compile off and the block seam runs inside mx.compile.
    from mflux.models.krea2.variants.txt2img.krea2 import Krea2

    from mlx_dfloat.mflux._compile import predict_mode

    embeds = mx.zeros((1, 1, 2))
    assert predict_mode(Krea2._predict, lambda *a: a[0], embeds, None, 1.0) == "uncompiled"


# Krea 2's template ids (text_encoder.py:12): <|im_start|> 151644, "user" 872, "\n" 198.
@pytest.mark.parametrize(
    ("ids", "end"),
    [
        # system turn, then the user turn: the prompt starts after "<|im_start|>user\n" at index 6 -> 9
        ([151644, 8948, 198, 100, 151645, 198, 151644, 872, 198, 101, 102], 9),
        # one user turn only: strip the three template tokens
        ([151644, 872, 198, 5, 6], 3),
        # no template: nothing is stripped
        ([5, 6, 7, 8], 0),
    ],
    ids=["system-then-user", "user-only", "no-template"],
)
def test_krea2_template_end_on_fixed_ids(ids, end):
    # Bug caught: the template end moves by one (text_encoder.py:117-128); krea2/memory.py subtracts it from the
    # token count that sizes the text phase, so the memory fit is off by the shift.
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    assert Krea2TextEncoder._template_end(mx.array([ids], dtype=mx.int32)) == end


class _PrefixTokenizer:
    """The wrapper ``_system_prefix_length`` reads: ``.tokenizer(text, add_special_tokens=...)`` gives 7 ids."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tokenizer = self._encode

    def _encode(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.calls.append((text, kwargs))
        return {"input_ids": [11, 12, 13, 14, 15, 16, 17]}


def test_qwen21_system_prefix_length_counts_the_prefix_ids():
    # Bug caught: the prefix count shifted, or taken with special tokens (qwen21_prompt_encoder.py:36-39);
    # qwen21/memory.py subtracts it from the prompt's tokens, so the text-phase memory fit moves.
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
        Qwen21PromptEncoder,
    )

    tokenizer = _PrefixTokenizer()
    assert Qwen21PromptEncoder._system_prefix_length(tokenizer) == 7
    [(text, kwargs)] = tokenizer.calls
    assert kwargs == {"add_special_tokens": False}
    assert text.startswith("<|im_start|>system\n")
    assert text.endswith("<|im_end|>\n")


def _linear(rows: list[list[float]], *, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(2, 2, bias=bias)
    layer.weight = mx.array(rows, dtype=mx.bfloat16)
    return layer


def test_qwen21_pack_linears_concatenates_rows_in_order_and_views_them_back():
    # Bug caught: the packed weight built in another order or axis (qwen21_transformer.py:157-174), so the fused
    # q/k/v matmul our seam runs no longer matches mflux's packed path.
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    layers = (_linear([[1, 2], [3, 4]]), _linear([[5, 6], [7, 8]]), _linear([[9, 10], [11, 12]]))
    packed = Qwen21Transformer._pack_linears(layers)
    assert packed is not None
    assert packed.dtype == mx.bfloat16
    assert packed.tolist() == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    assert [layer.weight.tolist() for layer in layers] == [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
    ]


def test_qwen21_pack_linears_leaves_a_biased_group_unpacked():
    # Bug caught: a group with a bias packed anyway (qwen21_transformer.py:159-160), which drops the bias.
    from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

    layers = (_linear([[1, 2], [3, 4]]), _linear([[5, 6], [7, 8]], bias=True))
    assert Qwen21Transformer._pack_linears(layers) is None


def _ours_and_stock(family: str) -> tuple[type, type]:
    if family == "flux1":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        from mlx_dfloat.mflux.flux1.model import DFloatFlux1

        return DFloatFlux1, Flux1
    if family == "flux2":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        from mlx_dfloat.mflux.flux2.model import DFloatFlux2Klein

        return DFloatFlux2Klein, Flux2Klein
    if family == "zimage":
        from mflux.models.z_image.variants.z_image import ZImage

        from mlx_dfloat.mflux.zimage.model import DFloatZImage

        return DFloatZImage, ZImage
    if family == "ernie":
        from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage

        from mlx_dfloat.mflux.ernie.model import DFloatErnieImage

        return DFloatErnieImage, ErnieImage
    if family == "krea2":
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2

        from mlx_dfloat.mflux.krea2.model import DFloatKrea2

        return DFloatKrea2, Krea2
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    return DFloatQwenImage21, QwenImage21


# What each of our classes overrides on purpose today; a name outside this list is an accidental shadow.
_OVERRIDES: dict[str, set[str]] = {
    "flux1": {"freeze", "from_name", "generate_image", "save_model"},
    "flux2": {"_encode_prompt_pair", "_predict", "freeze", "generate_image", "save_model"},
    "zimage": {"_encode_prompts", "_predict", "freeze", "generate_image", "save_model"},
    "ernie": {"_encode_prompts", "_predict", "freeze", "generate_image", "save_model"},
    "krea2": {
        "_encode_prompts",
        "_predict",
        "_resolve_scheduler",
        "freeze",
        "generate_image",
        "save_model",
    },
    "qwen21": {"freeze", "generate_image", "save_model"},
}


@pytest.mark.parametrize("family", sorted(_OVERRIDES))
def test_our_model_classes_override_only_the_listed_mflux_methods(family):
    # Bug caught: upstream adds a method with the name of one of ours, so our helper silently replaces it for every
    # inherited caller (as mflux 0.24's QwenImage21._encode_prompt, qwen_image_21.py:192-207, would be).
    ours, stock = _ours_and_stock(family)
    own = {name for name in vars(ours) if not (name.startswith("__") and name.endswith("__"))}
    assert own & set(dir(stock)) == _OVERRIDES[family]


def test_our_qwen21_class_never_defines_its_own_encode_prompt():
    # Bug caught: a method named _encode_prompt added to DFloatQwenImage21; mflux 0.24's generate_image calls
    # self._encode_prompt (qwen_image_21.py:105,109) and would run ours instead of the stock prompt encoder.
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    assert "_encode_prompt" not in vars(DFloatQwenImage21)


_TABLE_RIGHT_COLUMN = 65


def _tests_and_files() -> tuple[set[str], set[str]]:
    import ast
    from pathlib import Path

    here = Path(__file__).parent
    files = {p.name for p in here.glob("test_*.py")}
    names: set[str] = set()
    for path in here.glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                "test_"
            ):
                names.add(node.name)
    return names, files


def test_every_test_the_contract_table_names_exists():
    # Bug caught: the table citing a test by description or by a name that was never written (or renamed), so a
    # reader trusts a pin that does not exist.
    import fnmatch
    import re

    names, files = _tests_and_files()
    lines = __doc__.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Symbol (mflux"))
    assert lines[header].index("Pinned by") == _TABLE_RIGHT_COLUMN
    table = [line for line in lines[header + 1 :] if len(line) > _TABLE_RIGHT_COLUMN]
    missing = []
    for line in table:
        cell = line[_TABLE_RIGHT_COLUMN:].strip()
        refs = re.findall(r"test_[\w*]+(?:\.py)?", cell)
        if not refs:
            missing.append(f"no test named: {cell!r}")
        for ref in refs:
            pool = files if ref.endswith(".py") else names
            if not fnmatch.filter(pool, ref):
                missing.append(ref)
    assert missing == []
