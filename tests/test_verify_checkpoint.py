import json

import mlx.core as mx
import numpy as np
import pytest
import scripts.verify_checkpoint as vc
from scripts.verify_checkpoint import main, run_key, verify
from tests._df11_fixtures import random_bf16, write_bf16_original, write_checkpoint

KEY = {
    "df11_revision": "a" * 40,
    "bf16_revision": "b" * 40,
    "mode": "parity",
    "decoder": "reference",
    "source": "x",
    "git": "y",
    "mlx": "z",
}
SKEY = {**KEY, "mode": "structural-only"}
NORM = np.array([0x3F80], np.uint16)


@pytest.fixture
def pair(tmp_path):
    rng = np.random.default_rng(7)
    mats = {f"blocks.{i}": [random_bf16(rng, (6, 10)), random_bf16(rng, (4, 10))] for i in range(3)}
    df11 = write_checkpoint(
        tmp_path / "df11",
        groups=mats,
        pattern=r"blocks\.\d+",
        sub_paths=("q", "k"),
        extras={"norm.weight": NORM},
    )
    originals = {"norm.weight": NORM}
    for g, (q, k) in mats.items():
        originals[f"{g}.q.weight"], originals[f"{g}.k.weight"] = q, k
    return df11, write_bf16_original(tmp_path / "bf16", originals), originals


def _count_decodes(monkeypatch):
    calls = []
    real = vc.decode_matrices
    monkeypatch.setattr(
        vc, "decode_matrices", lambda g, **kw: calls.append(g.name) or real(g, **kw)
    )
    return calls


def test_equal_pair_exits_0_and_records_every_matrix(tmp_path, pair):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    results = [json.loads(p.read_text()) for p in sorted((out / "groups").glob("*.json"))]
    assert [len(r["matrices"]) for r in results] == [2, 2, 2]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["compared"] == 6
    assert summary["extras_compared"] == 1


def test_one_flipped_bit_exits_1(tmp_path, pair):
    df11, _, originals = pair
    bad = dict(originals)
    flipped = bad["blocks.1.k.weight"].copy()
    flipped.flat[17] ^= 0x0001
    bad["blocks.1.k.weight"] = flipped
    out = tmp_path / "out"
    assert verify(df11, write_bf16_original(tmp_path / "bad", bad), out, key=KEY) == 1
    mismatch = [
        m
        for m in json.loads((out / "groups" / "blocks.1.json").read_text())["matrices"]
        if not m["equal"]
    ]
    assert [(m["name"], m["first_mismatch_index"]) for m in mismatch] == [("blocks.1.k.weight", 17)]


def test_flipped_extra_exits_1(tmp_path, pair):
    df11, _, originals = pair
    bad = {**originals, "norm.weight": NORM ^ np.uint16(1)}
    assert verify(df11, write_bf16_original(tmp_path / "bad", bad), tmp_path / "out", key=KEY) == 1


def test_original_without_a_group_is_a_coverage_failure_unless_ignored(tmp_path, pair):
    df11, _, originals = pair
    more = write_bf16_original(
        tmp_path / "more", {**originals, "lm_head.weight": np.zeros((2, 2), np.uint16)}
    )
    assert verify(df11, more, tmp_path / "o1", key=KEY) == 2
    assert verify(df11, more, tmp_path / "o2", key=KEY, ignore_originals=("lm_head.weight",)) == 0


def test_missing_original_is_a_coverage_failure(tmp_path, pair):
    df11, _, originals = pair
    fewer = write_bf16_original(
        tmp_path / "less", {k: v for k, v in originals.items() if k != "blocks.2.q.weight"}
    )
    assert verify(df11, fewer, tmp_path / "out", key=KEY) == 2


def test_non_bf16_original_is_an_error_not_a_mismatch(tmp_path, pair):
    df11, _, originals = pair
    f32 = write_bf16_original(tmp_path / "f32", originals, dtype=mx.float32)
    assert verify(df11, f32, tmp_path / "out", key=KEY) == 2


def test_truncated_original_shard_is_an_error_not_a_mismatch(tmp_path, pair):
    df11, bf16, _ = pair
    shard = bf16 / "model-00001-of-00002.safetensors"
    shard.write_bytes(shard.read_bytes()[:-20])
    assert verify(df11, bf16, tmp_path / "out", key=KEY) == 2


def test_shard_path_escaping_the_directory_is_refused(tmp_path, pair):
    df11, bf16, _ = pair
    index = bf16 / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["blocks.0.q.weight"] = "../elsewhere.safetensors"
    index.write_text(json.dumps(data))
    assert verify(df11, bf16, tmp_path / "out", key=KEY) == 2


def test_resume_skips_only_results_with_the_same_key(tmp_path, pair, monkeypatch):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    calls = _count_decodes(monkeypatch)
    assert verify(df11, bf16, out, key=KEY) == 0
    assert calls == []
    assert verify(df11, bf16, out, key={**KEY, "source": "changed"}) == 0
    assert sorted(calls) == ["blocks.0", "blocks.1", "blocks.2"]


def test_resumed_mismatch_stays_a_mismatch(tmp_path, pair, monkeypatch):
    df11, _, originals = pair
    bad = dict(originals)
    bad["blocks.0.q.weight"] = bad["blocks.0.q.weight"] ^ np.uint16(0x8000)
    bf16 = write_bf16_original(tmp_path / "bad", bad)
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 1
    calls = _count_decodes(monkeypatch)
    assert verify(df11, bf16, out, key=KEY) == 1
    assert calls == []


def test_structural_results_are_never_reused_as_parity(tmp_path, pair):
    df11, _, originals = pair
    out = tmp_path / "out"
    assert verify(df11, None, out, key=SKEY) == 0
    bad = {k: (v ^ np.uint16(1) if k.endswith(".q.weight") else v) for k, v in originals.items()}
    assert verify(df11, write_bf16_original(tmp_path / "bad", bad), out, key=KEY) == 1


def test_unparsable_result_file_is_recomputed(tmp_path, pair, monkeypatch):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    verify(df11, bf16, out, key=KEY)
    (out / "groups" / "blocks.1.json").write_text("{truncated")
    calls = _count_decodes(monkeypatch)
    assert verify(df11, bf16, out, key=KEY) == 0
    assert calls == ["blocks.1"]


def test_unknown_revision_disables_resume(tmp_path, pair, monkeypatch):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    key = {**KEY, "df11_revision": "unknown"}
    assert verify(df11, bf16, out, key=key) == 0
    calls = _count_decodes(monkeypatch)
    verify(df11, bf16, out, key=key)
    assert len(calls) == 3


def test_structural_only_passes_on_a_good_checkpoint_and_fails_on_a_corrupt_one(tmp_path, pair):
    df11, _, _ = pair
    assert verify(df11, None, tmp_path / "o1", key=SKEY) == 0
    shard = df11 / "blocks_1.safetensors"
    data = mx.load(str(shard))
    mx.eval(data)
    gaps = np.array(data["blocks.1.gaps"])
    gaps[0] ^= 0xFF  # corrupt the first thread gaps -> continuity / structure breaks
    data["blocks.1.gaps"] = mx.array(gaps)
    mx.save_safetensors(str(shard), data)
    assert verify(df11, None, tmp_path / "o2", key=SKEY) == 2


def test_unknown_group_name_via_cli_exits_2(tmp_path, pair):
    df11, bf16, _ = pair
    argv = [
        "--df11",
        str(df11),
        "--bf16",
        str(bf16),
        "--out",
        str(tmp_path / "o"),
        "--groups",
        "nope",
        "--df11-revision",
        "a" * 40,
        "--bf16-revision",
        "b" * 40,
        "--no-watchdog",
    ]
    assert main(argv) == 2


def test_run_key_hashes_sources_and_records_mode():
    key = run_key(df11_revision="r1", bf16_revision="r2", mode="parity")
    assert key["mode"] == "parity"
    assert len(key["source"]) == 64
