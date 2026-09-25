import json
import os

import mlx.core as mx
import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat import format as fmt
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import matrix_names_for, open_checkpoint

PATTERN = r"blocks\.\d+"
SUBS = ("attn.to_q", "attn.to_k")


def _two_block_ckpt(tmp_path, **kwargs):
    rng = np.random.default_rng(0)
    groups = {
        f"blocks.{i}": [random_bf16(rng, (8, 16)), random_bf16(rng, (4, 16))] for i in range(2)
    }
    return write_checkpoint(
        tmp_path / "ckpt", groups=groups, pattern=PATTERN, sub_paths=SUBS, **kwargs
    )


def _rewrite(shard, transform):
    data = mx.load(str(shard))
    mx.eval(data)  # materialise before overwriting the file the lazy arrays read from
    mx.save_safetensors(str(shard), transform(data))


def test_matrix_names_follow_pattern_dict_order():
    assert matrix_names_for("blocks.3", {PATTERN: SUBS}) == (
        "blocks.3.attn.to_q.weight",
        "blocks.3.attn.to_k.weight",
    )


def test_single_matrix_group_is_named_group_weight():
    assert matrix_names_for("lm_head", {r"lm_head": ()}) == ("lm_head.weight",)


def test_unmatched_group_is_a_format_error():
    with pytest.raises(DFloatFormatError, match="no pattern"):
        matrix_names_for("other.0", {PATTERN: SUBS})


def test_matrix_names_for_screens_a_hand_built_pattern_dict():
    # Bug caught: matrix_names_for trusting patterns that never went through parse_df11_config
    # (a DF11Config built by hand) runs a catastrophic regex against the group name.
    with pytest.raises(DFloatFormatError, match="not allowed"):
        matrix_names_for("a" * 200, {".*" * 20 + "!": ()})


def test_ambiguous_group_is_a_format_error():
    with pytest.raises(DFloatFormatError, match="more than one pattern"):
        matrix_names_for("blocks.0", {PATTERN: SUBS, r"blocks\.0": ("x",)})


@pytest.mark.parametrize("single_file", [False, True])
def test_discovers_groups_and_extras(tmp_path, single_file):
    root = _two_block_ckpt(
        tmp_path,
        extras={"norm.weight": np.array([0x3F80, 0x4000], np.uint16)},
        single_file=single_file,
    )
    (root / "README.md").write_text("# hi")
    (root / ".gitattributes").write_text("*.safetensors filter=lfs")
    ckpt = open_checkpoint(root)
    assert sorted(ckpt.groups) == ["blocks.0", "blocks.1"]
    assert ckpt.groups["blocks.1"].matrix_names == (
        "blocks.1.attn.to_q.weight",
        "blocks.1.attn.to_k.weight",
    )
    assert list(ckpt.extras) == ["norm.weight"]
    arrays = ckpt.groups["blocks.0"].load()
    assert arrays.n_elements == 8 * 16 + 4 * 16
    assert arrays.output_positions.dtype == np.uint32


def test_discovery_reads_headers_only(tmp_path, monkeypatch):
    root = _two_block_ckpt(tmp_path)

    def _no_data(*_a, **_k):
        raise AssertionError("open_checkpoint must not read tensor data")

    monkeypatch.setattr(fmt, "read_array", _no_data)
    assert sorted(open_checkpoint(root).groups) == ["blocks.0", "blocks.1"]


def test_missing_group_field_is_a_format_error(tmp_path):
    root = _two_block_ckpt(tmp_path)
    _rewrite(
        root / "blocks_0.safetensors",
        lambda d: {k: v for k, v in d.items() if not k.endswith(".gaps")},
    )
    with pytest.raises(DFloatFormatError, match="missing gaps"):
        open_checkpoint(root)


def test_group_split_across_files_is_a_format_error(tmp_path):
    root = _two_block_ckpt(tmp_path)
    shard = root / "blocks_0.safetensors"
    data = mx.load(str(shard))
    mx.eval(data)
    mx.save_safetensors(
        str(root / "extra_luts.safetensors"), {"blocks.0.luts": data["blocks.0.luts"]}
    )
    mx.save_safetensors(str(shard), {k: v for k, v in data.items() if k != "blocks.0.luts"})
    with pytest.raises(DFloatFormatError, match="split across files"):
        open_checkpoint(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("luts", mx.zeros((3, 256), dtype=mx.float32), "dtype"),
        ("split_positions", mx.zeros((1, 1), dtype=mx.int64), "rank"),
        ("gaps", mx.zeros((320,), dtype=mx.int32), "dtype"),
    ],
)
def test_wrong_stored_dtype_or_rank_is_a_format_error(tmp_path, field, value, message):
    root = _two_block_ckpt(tmp_path)
    _rewrite(root / "blocks_0.safetensors", lambda d: {**d, f"blocks.0.{field}": value})
    with pytest.raises(DFloatFormatError, match=message):
        open_checkpoint(root)


def test_split_count_must_match_pattern_sub_paths(tmp_path):
    rng = np.random.default_rng(1)
    root = write_checkpoint(
        tmp_path / "c",
        groups={"blocks.0": [random_bf16(rng, (4, 4))] * 3},
        pattern=PATTERN,
        sub_paths=SUBS,
    )
    with pytest.raises(DFloatFormatError, match="holds 3 matrices"):
        open_checkpoint(root)


def test_duplicate_tensor_across_files_is_a_format_error(tmp_path):
    root = _two_block_ckpt(tmp_path)
    (root / "copy.safetensors").write_bytes((root / "blocks_0.safetensors").read_bytes())
    with pytest.raises(DFloatFormatError, match="appears in both"):
        open_checkpoint(root)


@pytest.mark.parametrize("bad", ["/etc/passwd_x", "../escape", "a" * 257, "sp ace", "blocks..0"])
def test_unsafe_group_names_are_refused(tmp_path, bad):
    # A group name becomes a result-file name in the verify scripts: never trust it.
    root = tmp_path / "c"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "dfloat11_config": {
                    "version": "0.5.0",
                    "threads_per_block": [512],
                    "bytes_per_thread": 8,
                    # Any in-grammar pattern: the name check runs before pattern matching.
                    "pattern_dict": {r"blocks\.\d+": []},
                }
            }
        )
    )
    mx.save_safetensors(
        str(root / "x.safetensors"), {f"{bad}.encoded_exponent": mx.zeros((1,), dtype=mx.uint8)}
    )
    with pytest.raises(DFloatFormatError, match="group name"):
        open_checkpoint(root)


def test_non_regular_shard_is_refused(tmp_path):
    root = _two_block_ckpt(tmp_path)
    os.mkfifo(root / "pipe.safetensors")
    with pytest.raises(DFloatFormatError, match="not a regular file"):
        open_checkpoint(root)


def test_accepts_str_path_with_user_home(tmp_path, monkeypatch):
    _two_block_ckpt(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert sorted(open_checkpoint("~/ckpt").groups) == ["blocks.0", "blocks.1"]
