"""``mflux_uncompiled`` gives back the compile state it found: off stays off, on comes back on."""

import os

import mlx.core as mx
import pytest
from tests._qwen21_tiny import mflux_uncompiled


def _compile_is_on() -> bool:
    # A compiled function traces its Python body once per input shape; uncompiled, it runs the body on every call.
    calls = []

    def body(x):
        calls.append(1)
        return x + 1

    fn = mx.compile(body)
    x = mx.zeros((1,))
    mx.eval(fn(x))
    mx.eval(fn(x))
    return len(calls) == 1


@pytest.fixture
def restore_compile():
    yield
    if "MLX_DISABLE_COMPILE" not in os.environ:
        mx.enable_compile()


def test_off_on_entry_from_the_env_var_stays_off_on_exit(restore_compile, monkeypatch):
    # Bug caught: the block switches compile back on at exit under MLX_DISABLE_COMPILE, so the rest of the session
    # runs compiled after the user turned compile off. restore_compile comes first so it tears down last, after
    # monkeypatch has removed the variable, and switches compile back on for the next test.
    monkeypatch.setenv("MLX_DISABLE_COMPILE", "1")
    mx.disable_compile()  # what MLX does at start when the variable is set
    with mflux_uncompiled():
        assert not _compile_is_on()
    assert not _compile_is_on()


def test_a_nested_block_leaves_the_outer_block_uncompiled(restore_compile):
    # Bug caught: an inner block's exit switches compile on while the outer block still needs the plain graph.
    with mflux_uncompiled():
        with mflux_uncompiled():
            pass
        assert not _compile_is_on()
    assert _compile_is_on()


def test_on_on_entry_is_on_again_on_exit(restore_compile):
    # Bug caught: the block never switches compile off, or leaves it off for every later test.
    assert _compile_is_on()
    with mflux_uncompiled():
        assert not _compile_is_on()
    assert _compile_is_on()
