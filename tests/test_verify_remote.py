import json
from pathlib import Path

import numpy as np
import pytest
from scripts.verify_checkpoint import VerifyError
from scripts.verify_remote_group import (
    HfRangeSource,
    LocalRangeSource,
    bf16_index,
    df11_index,
    main,
    pick_groups,
    verify_group,
)
from tests._df11_fixtures import random_bf16, write_bf16_original, write_checkpoint

from mlx_dfloat.errors import DFloatError
from mlx_dfloat.format import read_df11_config


def test_pick_groups_natural_order_block_and_code_selectors():
    stats = {
        "blocks.10": {"elements_per_block": 5, "max_code_length": 30},
        "blocks.2": {"elements_per_block": 9, "max_code_length": 12},
        "blocks.1": {"elements_per_block": 3, "max_code_length": 14},
    }
    assert pick_groups(stats, "first") == ["blocks.1"]
    assert pick_groups(stats, "last") == ["blocks.10"]
    assert pick_groups(stats, "max-block") == ["blocks.2"]
    assert pick_groups(stats, "max-code") == ["blocks.10"]
    assert pick_groups(stats, "first,last,first") == ["blocks.1", "blocks.10"]
    with pytest.raises(KeyError):
        pick_groups(stats, "nope")


def _pair(tmp_path, single_file):
    rng = np.random.default_rng(11)
    mats = {"blocks.0": [random_bf16(rng, (5, 9)), random_bf16(rng, (3, 9))]}
    df11 = write_checkpoint(
        tmp_path / "d",
        groups=mats,
        pattern=r"blocks\.\d+",
        sub_paths=("q", "k"),
        single_file=single_file,
    )
    originals = {"blocks.0.q.weight": mats["blocks.0"][0], "blocks.0.k.weight": mats["blocks.0"][1]}
    return df11, write_bf16_original(tmp_path / "b", originals), originals


def _triple(tmp_path):
    """Three single-matrix groups, each its own shard: enough shards for first != last."""
    rng = np.random.default_rng(23)
    mats = {f"blocks.{i}": [random_bf16(rng, (4, 6))] for i in range(3)}
    df11 = write_checkpoint(
        tmp_path / "d3", groups=mats, pattern=r"blocks\.\d+", sub_paths=(), single_file=False
    )
    originals = {f"blocks.{i}.weight": mats[f"blocks.{i}"][0] for i in range(3)}
    bf16 = write_bf16_original(tmp_path / "b3", originals)
    return df11, bf16, originals


def _run(df11, bf16, group="blocks.0"):
    d, b = LocalRangeSource(df11), LocalRangeSource(bf16)
    return verify_group(
        d, b, group, config=read_df11_config(df11), dindex=df11_index(d), bindex=bf16_index(b)
    )


def _local_factory(repo, revision, subdir):
    return LocalRangeSource(Path(repo))


def _argv(df11, bf16, out):
    return [
        "--df11-repo",
        str(df11),
        "--df11-revision",
        "x",
        "--bf16-repo",
        str(bf16),
        "--bf16-revision",
        "y",
        "--out",
        str(out),
        "--wall-budget",
        "60",
    ]


@pytest.mark.parametrize("single", [False, True])
def test_verify_group_equal_for_sharded_and_single_file(tmp_path, single):
    df11, bf16, _ = _pair(tmp_path, single)
    record = _run(df11, bf16)
    assert record["status"] == "equal", record
    assert [m["name"] for m in record["matrices"]] == ["blocks.0.q.weight", "blocks.0.k.weight"]
    assert record["max_code_length"] >= 1
    assert record["n_blocks"] == 1


def test_verify_group_reports_a_mismatch(tmp_path):
    df11, _, originals = _pair(tmp_path, False)
    bad = {**originals, "blocks.0.q.weight": originals["blocks.0.q.weight"] ^ np.uint16(0x8000)}
    assert _run(df11, write_bf16_original(tmp_path / "bad", bad))["status"] == "mismatch"


def test_missing_original_is_an_error_not_a_mismatch(tmp_path):
    df11, _, originals = _pair(tmp_path, False)
    fewer = {k: v for k, v in originals.items() if k != "blocks.0.k.weight"}
    assert _run(df11, write_bf16_original(tmp_path / "less", fewer))["status"] == "error"


def test_short_read_is_an_error(tmp_path):
    # A truncated shard must fail closed: df11_index raises so main() exits before group
    # selection ever runs, instead of silently dropping the group and letting the CLI
    # report success from the shards that survive.
    df11, _, _ = _pair(tmp_path, False)
    shard = df11 / "blocks_0.safetensors"
    shard.write_bytes(shard.read_bytes()[:-5])
    with pytest.raises((DFloatError, VerifyError)):
        df11_index(LocalRangeSource(df11))


def test_df11_index_raises_when_index_json_names_a_missing_shard(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    (df11 / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"blocks.0.encoded_exponent": "ghost.safetensors"}})
    )
    with pytest.raises(VerifyError, match=r"ghost\.safetensors"):
        df11_index(LocalRangeSource(df11))


def test_df11_index_raises_on_a_group_present_in_two_shards(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    shard = df11 / "blocks_0.safetensors"
    (df11 / "blocks_0_dup.safetensors").write_bytes(shard.read_bytes())
    with pytest.raises(VerifyError, match="duplicate group"):
        df11_index(LocalRangeSource(df11))


def test_verify_group_error_record_defaults_max_code_length_and_n_blocks_to_none(tmp_path):
    df11, _, originals = _pair(tmp_path, False)
    fewer = {k: v for k, v in originals.items() if k != "blocks.0.k.weight"}
    # blocks.0.q.weight compares fine, so the failure happens after arrays decode: those two
    # fields ARE populated here. A group missing from dindex fails before decoding starts, so
    # they stay None; exercise that path directly.
    d = LocalRangeSource(df11)
    record = verify_group(
        d,
        LocalRangeSource(write_bf16_original(tmp_path / "less", fewer)),
        "no-such-group",
        config=read_df11_config(df11),
        dindex=df11_index(d),
        bindex={},
    )
    assert record["status"] == "error"
    assert record["max_code_length"] is None
    assert record["n_blocks"] is None


def test_verify_group_without_a_bf16_source_is_structural_ok(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    d = LocalRangeSource(df11)
    record = verify_group(
        d, None, "blocks.0", config=read_df11_config(df11), dindex=df11_index(d), bindex={}
    )
    assert record["status"] == "structural-ok"
    assert [m["name"] for m in record["matrices"]] == ["blocks.0.q.weight", "blocks.0.k.weight"]


def test_bf16_index_falls_back_to_scanning_shards_without_an_index(tmp_path):
    _, _, originals = _pair(tmp_path, False)
    no_index = write_bf16_original(tmp_path / "no_index", originals, with_index=False)
    mapping = bf16_index(LocalRangeSource(no_index))
    assert set(mapping) == set(originals)


class _CountingSource:
    """Wraps a RangeSource and counts read() calls per path (proves per-shard header caching)."""

    def __init__(self, inner):
        self._inner = inner
        self.read_counts: dict[str, int] = {}

    def read(self, path, start, n):
        self.read_counts[path] = self.read_counts.get(path, 0) + 1
        return self._inner.read(path, start, n)

    def size(self, path):
        return self._inner.size(path)

    def text(self, path):
        return self._inner.text(path)

    def files(self):
        return self._inner.files()


def test_verify_group_fetches_each_bf16_shard_header_once(tmp_path):
    rng = np.random.default_rng(31)
    mats = {
        "blocks.0": [
            random_bf16(rng, (2, 4)),
            random_bf16(rng, (2, 4)),
            random_bf16(rng, (2, 4)),
        ]
    }
    df11 = write_checkpoint(
        tmp_path / "dh",
        groups=mats,
        pattern=r"blocks\.\d+",
        sub_paths=("a", "b", "c"),
        single_file=False,
    )
    originals = {
        "blocks.0.a.weight": mats["blocks.0"][0],
        "blocks.0.b.weight": mats["blocks.0"][1],
        "blocks.0.c.weight": mats["blocks.0"][2],
    }
    bf16 = write_bf16_original(tmp_path / "bh", originals)
    weight_map = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"]
    shard2 = "model-00002-of-00002.safetensors"
    assert weight_map["blocks.0.b.weight"] == shard2
    assert weight_map["blocks.0.c.weight"] == shard2

    d = LocalRangeSource(df11)
    counting = _CountingSource(LocalRangeSource(bf16))
    record = verify_group(
        d,
        counting,
        "blocks.0",
        config=read_df11_config(df11),
        dindex=df11_index(d),
        bindex=bf16_index(counting),
    )
    assert record["status"] == "equal", record
    # One header fetch for shard2 (2 reads: length + body) plus one data read per matrix that
    # lands there (2 matrices) = 4. A per-matrix (uncached) fetch would be 6.
    assert counting.read_counts[shard2] == 4


@pytest.mark.network
def test_hf_range_source_reads_a_seeked_range():
    # The only test that touches HfRangeSource over the network: pins that ranged reads at an
    # offset work on the installed huggingface_hub (a streaming file would raise "Cannot seek
    # streaming HF file").
    from huggingface_hub import HfApi

    repo = "DFloat11/Qwen3-4B-DF11"
    src = HfRangeSource(repo, HfApi().model_info(repo).sha)
    shard = next(f for f in src.files() if f.endswith(".safetensors"))
    (length,) = __import__("struct").unpack("<Q", src.read(shard, 0, 8))
    assert src.read(shard, 8, 1) == b"{"  # the JSON header starts right after the length
    assert 0 < length < src.size(shard)


def test_hf_range_source_path_guard_rejects_escapes_and_subdirs():
    # Network-free: constructing HfRangeSource and calling its path guard never touches the Hub.
    src = HfRangeSource("owner/repo", "0" * 40)
    for bad in ("../x.safetensors", "sub/x.safetensors", "x.bin"):
        with pytest.raises(VerifyError, match="unsafe path"):
            src._full(bad)
    rev = "0" * 40
    assert (
        src._full("model-00001-of-00002.safetensors")
        == f"owner/repo@{rev}/model-00001-of-00002.safetensors"
    )
    assert src._full("config.json") == f"owner/repo@{rev}/config.json"


def test_escaping_shard_name_in_index_is_refused(tmp_path):
    _, bf16, _ = _pair(tmp_path, False)
    index = bf16 / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["blocks.0.q.weight"] = "../x.safetensors"
    index.write_text(json.dumps(data))
    with pytest.raises(Exception, match="unsafe shard"):
        bf16_index(LocalRangeSource(bf16))


def test_main_all_equal_exits_zero(tmp_path):
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "out.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 0
    assert json.loads(out.read_text())["mode"] == "parity"


def test_main_a_mismatch_exits_one(tmp_path):
    df11, _, originals = _pair(tmp_path, False)
    bad = {**originals, "blocks.0.q.weight": originals["blocks.0.q.weight"] ^ np.uint16(0x8000)}
    bf16 = write_bf16_original(tmp_path / "bad_main", bad)
    out = tmp_path / "out.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 1


def test_main_returns_error_when_an_unselected_df11_shard_is_corrupt(tmp_path):
    # Reproduces the false-PASS this fix closes: truncating blocks.2 (not "first", "last" of the
    # surviving two, or otherwise selected under the old skip-and-continue df11_index) used to
    # still exit 0. It must now exit 2 regardless of which group --groups would have picked.
    df11, bf16, _ = _triple(tmp_path)
    shard = df11 / "blocks_2.safetensors"
    shard.write_bytes(shard.read_bytes()[:-5])
    out = tmp_path / "out.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 2


def test_main_a_missing_original_exits_two(tmp_path):
    df11, _, originals = _pair(tmp_path, False)
    fewer = {k: v for k, v in originals.items() if k != "blocks.0.k.weight"}
    bf16 = write_bf16_original(tmp_path / "less_main", fewer)
    out = tmp_path / "out.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 2


def test_main_unknown_group_selector_exits_two(tmp_path):
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "out.json"
    argv = [*_argv(df11, bf16, out), "--groups", "nope"]
    assert main(argv, source_factory=_local_factory) == 2


def test_main_requires_bf16_repo_or_structural_only(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    out = tmp_path / "out.json"
    argv = [
        "--df11-repo",
        str(df11),
        "--df11-revision",
        "x",
        "--out",
        str(out),
        "--wall-budget",
        "60",
    ]
    assert main(argv, source_factory=_local_factory) == 2


def test_main_structural_only_exits_zero_and_records_mode(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    out = tmp_path / "out.json"
    argv = [
        "--df11-repo",
        str(df11),
        "--df11-revision",
        "x",
        "--structural-only",
        "--out",
        str(out),
        "--wall-budget",
        "60",
    ]
    assert main(argv, source_factory=_local_factory) == 0
    written = json.loads(out.read_text())
    assert written["mode"] == "structural-only"
    assert all(g["status"] == "structural-ok" for g in written["groups"])
