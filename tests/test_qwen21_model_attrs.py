"""Class attributes `DFloatQwenImage21` must carry for mflux's own `QwenImage21` methods it inherits."""

import pytest

pytestmark = pytest.mark.mflux


def test_the_model_class_carries_a_none_processor_for_the_inherited_prompt_encoder():
    # Bug caught: mflux 0.24's `QwenImage21._encode_prompt` reads `self.processor` (on its small-text-encoder
    # branch) and our class never runs mflux's initializer, which is what sets it; without a class default an
    # inherited read raises AttributeError instead of seeing "no processor".
    from mlx_dfloat.mflux.qwen21.model import DFloatQwenImage21

    assert hasattr(DFloatQwenImage21, "processor")
    assert DFloatQwenImage21.processor is None
