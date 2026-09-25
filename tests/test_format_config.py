import json
import pickle
import time

import pytest

from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import DF11Config, parse_df11_config, read_df11_config

GOOD = {
    "version": "0.2.0",
    "threads_per_block": [512],
    "bytes_per_thread": 8,
    "pattern_dict": {r"transformer_blocks\.\d+": ["attn.to_q", "attn.to_k"]},
}


def test_parses_a_good_config():
    cfg = parse_df11_config(GOOD, source="config.json")
    assert cfg == DF11Config(
        version="0.2.0",
        threads_per_block=512,
        bytes_per_thread=8,
        pattern_dict={r"transformer_blocks\.\d+": ("attn.to_q", "attn.to_k")},
    )


@pytest.mark.parametrize("version", ["0.2.0", "0.3.1", "0.3.2", "0.5.0"])
def test_supported_versions_parse(version):
    assert parse_df11_config({**GOOD, "version": version}, source="c").version == version


def test_optional_group_pattern_is_allowed():
    # '(...)?' is not a quantified group in the ReDoS sense (')' followed by '?').
    cfg = parse_df11_config(
        {**GOOD, "pattern_dict": {r"(model\.)?layers\.\d+": ["mlp"]}}, source="c"
    )
    assert list(cfg.pattern_dict) == [r"(model\.)?layers\.\d+"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"version": "0.4.0"}, "unsupported DF11 format version"),
        ({"version": None}, "version"),
        ({"threads_per_block": [256]}, "threads_per_block"),
        ({"threads_per_block": 512}, "threads_per_block"),
        ({"bytes_per_thread": 4}, "bytes_per_thread"),
        ({"pattern_dict": {}}, "pattern_dict"),
        ({"pattern_dict": {"a": "not-a-list"}}, "pattern_dict"),
        ({"pattern_dict": {"(": []}}, "regular expression"),
        ({"pattern_dict": {"a" * 257: []}}, "too long"),
        ({"pattern_dict": {f"p{i}": [] for i in range(65)}}, "too many"),
    ],
)
def test_bad_configs_are_refused(patch, message):
    with pytest.raises(DFloatFormatError, match=message):
        parse_df11_config({**GOOD, **patch}, source="c")


@pytest.mark.parametrize("pattern", ["(a+)+b", "(a*)*", "(ab|a){2,}c"])
def test_catastrophic_patterns_are_refused_quickly(pattern):
    start = time.monotonic()
    with pytest.raises(DFloatFormatError, match="quantified group"):
        parse_df11_config({**GOOD, "pattern_dict": {pattern: []}}, source="c")
    assert time.monotonic() - start < 0.1


def test_reads_config_json_from_a_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "X", "dfloat11_config": GOOD}))
    assert read_df11_config(tmp_path).version == "0.2.0"


def test_missing_config_is_refused(tmp_path):
    with pytest.raises(DFloatFormatError, match=r"config\.json"):
        read_df11_config(tmp_path)


def test_config_without_dfloat11_block_is_refused(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "X"}))
    with pytest.raises(DFloatFormatError, match="dfloat11_config"):
        read_df11_config(tmp_path)


@pytest.mark.parametrize("with_base_config", [False, True])
def test_legacy_pickle_repo_is_refused_without_unpickling(tmp_path, monkeypatch, with_base_config):
    # The .pkl raises if unpickled (it references a missing module); the loaders are also trapped.
    (tmp_path / "model.pkl").write_bytes(b"cmlx_dfloat_never\nboom\n(tR.")
    (tmp_path / "decode.ptx").write_text("// ptx")
    if with_base_config:
        (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))

    def _trap(*_args, **_kwargs):
        raise AssertionError("pickle must never be used")

    monkeypatch.setattr(pickle, "load", _trap)
    monkeypatch.setattr(pickle, "loads", _trap)
    monkeypatch.setattr(pickle, "Unpickler", _trap)
    with pytest.raises(DFloatFormatError, match="legacy pickle"):
        read_df11_config(tmp_path)
