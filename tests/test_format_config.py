import json
import pickle

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
def test_catastrophic_patterns_are_refused(pattern):
    with pytest.raises(DFloatFormatError, match="quantified group"):
        parse_df11_config({**GOOD, "pattern_dict": {pattern: []}}, source="c")


# Pattern shapes taken from upstream DF11 configs (LLM layers, optional prefix, diffusion blocks,
# MoE experts, a single un-numbered matrix). All must keep parsing.
@pytest.mark.parametrize(
    "pattern",
    [
        r"model\.layers\.\d+",
        r"(model\.)?layers\.\d+",
        r"single_transformer_blocks\.\d+",
        r"model\.layers\.\d+\.mlp\.experts\.\d+",
        r"lm_head",
        r"blocks\.[0-9]+",
    ],
)
def test_upstream_pattern_shapes_are_accepted(pattern):
    assert list(
        parse_df11_config({**GOOD, "pattern_dict": {pattern: []}}, source="c").pattern_dict
    ) == [pattern]


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        # Bug caught: a guard that refuses only `)*`, `)+`, `){` lets a chain of bare `.*` through;
        # fullmatch against a 200-char group name then backtracks for minutes.
        (".*" * 20 + "!", "not allowed"),
        # Bug caught: verbose mode turns `) +` into `)+`, hiding a quantified group from the guard.
        ("(?x)(a+) +b", "not allowed"),
        ("(?i)a+", "not allowed"),
        # Nested counted repetition and backreferences are outside the grammar.
        ("a{1,99}", "not allowed"),
        (r"(a)\1", "not allowed"),
        (r"\s+", "not allowed"),
    ],
)
def test_patterns_outside_the_df11_grammar_are_refused(pattern, message):
    with pytest.raises(DFloatFormatError, match=message):
        parse_df11_config({**GOOD, "pattern_dict": {pattern: []}}, source="c")


@pytest.mark.parametrize(
    ("pattern", "ok"),
    [
        (r"a\d+", True),
        (r"a\d+\.\d+", True),
        # Bug caught: polynomial backtracking; three overlapping `\d+` against 200 digits and a
        # trailing `.` is O(n^3) per fullmatch.
        (r"\d+\d+\d+", False),
        (r"\d*\d*\d*x", False),
    ],
)
def test_unbounded_quantifiers_are_capped_at_two(pattern, ok):
    config = {**GOOD, "pattern_dict": {pattern: []}}
    if ok:
        parse_df11_config(config, source="c")
    else:
        with pytest.raises(DFloatFormatError, match="unbounded quantifiers"):
            parse_df11_config(config, source="c")


@pytest.mark.parametrize(
    ("pattern", "ok"),
    [
        (r"(model\.)?(layers\.)?\d+", True),
        # Bug caught: `x?` repeated backtracks as 2^k on a failing match, no `*` or `+` needed.
        (r"\d?\d?\d?\d", False),
        (r"(a|b)(c|d)(e|f)(g|h)", True),
        # Bug caught: k alternation groups in sequence backtrack as 2^k on a failing match.
        (r"(\d|\d)(\d|\d)(\d|\d)(\d|\d)(\d|\d)", False),
    ],
)
def test_optional_and_alternation_choice_points_are_capped(pattern, ok):
    config = {**GOOD, "pattern_dict": {pattern: []}}
    if ok:
        parse_df11_config(config, source="c")
    else:
        with pytest.raises(DFloatFormatError, match="too many"):
            parse_df11_config(config, source="c")


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


@pytest.mark.parametrize(
    "patch",
    [
        {"version": "9" * 10_000},
        {"threads_per_block": [1] * 10_000},
        {"bytes_per_thread": "x" * 10_000},
        {"pattern_dict": {"(" * 10_000: "not-a-list"}},
    ],
)
def test_untrusted_config_values_are_truncated_in_error_text(patch):
    # Bug caught: echoing config.json values uncapped into the error (and the parity summary).
    with pytest.raises(DFloatFormatError) as info:
        parse_df11_config({**GOOD, **patch}, source="c")
    assert len(str(info.value)) < 400
