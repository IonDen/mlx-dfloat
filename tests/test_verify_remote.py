import json

import numpy as np
import pytest
from scripts.verify_remote_group import (
    LocalRangeSource,
    bf16_index,
    df11_index,
    pick_groups,
    verify_group,
)
from tests._df11_fixtures import random_bf16, write_bf16_original, write_checkpoint

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


def _run(df11, bf16, group="blocks.0"):
    d, b = LocalRangeSource(df11), LocalRangeSource(bf16)
    return verify_group(
        d, b, group, config=read_df11_config(df11), dindex=df11_index(d), bindex=bf16_index(b)
    )


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
    df11, bf16, _ = _pair(tmp_path, False)
    shard = df11 / "blocks_0.safetensors"
    shard.write_bytes(shard.read_bytes()[:-5])
    assert _run(df11, bf16)["status"] == "error"


@pytest.mark.network
def test_hf_range_source_reads_a_seeked_range():
    # The only test that touches HfRangeSource: pins that ranged reads at an offset work on the
    # installed huggingface_hub (a streaming file would raise "Cannot seek streaming HF file").
    from huggingface_hub import HfApi
    from scripts.verify_remote_group import HfRangeSource

    repo = "DFloat11/Qwen3-4B-DF11"
    src = HfRangeSource(repo, HfApi().model_info(repo).sha)
    shard = next(f for f in src.files() if f.endswith(".safetensors"))
    (length,) = __import__("struct").unpack("<Q", src.read(shard, 0, 8))
    assert src.read(shard, 8, 1) == b"{"  # the JSON header starts right after the length
    assert 0 < length < src.size(shard)


def test_escaping_shard_name_in_index_is_refused(tmp_path):
    _, bf16, _ = _pair(tmp_path, False)
    index = bf16 / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["blocks.0.q.weight"] = "../x.safetensors"
    index.write_text(json.dumps(data))
    with pytest.raises(Exception, match="unsafe shard"):
        bf16_index(LocalRangeSource(bf16))
