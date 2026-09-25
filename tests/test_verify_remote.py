import json
import struct
from pathlib import Path

import mlx.core as mx
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


def test_main_requires_bf16_repo_or_structural_only(tmp_path, capsys):
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
    with pytest.raises(SystemExit) as info:
        main(argv, source_factory=_local_factory)
    assert info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_main_refuses_bf16_repo_together_with_structural_only(tmp_path, capsys):
    # Bug caught: --structural-only silently winning over --bf16-repo, so a run the caller meant
    # as parity records "structural-ok" and exits 0 without comparing a single bit.
    df11, bf16, _ = _pair(tmp_path, False)
    argv = [*_argv(df11, bf16, tmp_path / "out.json"), "--structural-only"]
    with pytest.raises(SystemExit) as info:
        main(argv, source_factory=_local_factory)
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "usage:" in err
    assert "not allowed with" in err


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


class _RecordingSource(_CountingSource):
    """Records every (path, start, n) read, to prove what was fetched before a check fired."""

    def __init__(self, inner):
        super().__init__(inner)
        self.reads: list[tuple[str, int, int]] = []

    def read(self, path, start, n):
        self.reads.append((path, start, n))
        return super().read(path, start, n)


def _raw_bf16_shard(path, entries, payload):
    body = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(body)) + body + payload)


def test_a_bf16_size_mismatch_is_an_error_not_a_mismatch(tmp_path):
    # Bug caught: `equal = original.size == got.size and ...` reports a size mismatch (a mapping
    # problem) as a bit mismatch: status "mismatch", exit 1, the kill signal.
    df11, _, originals = _pair(tmp_path, False)
    short = {**originals, "blocks.0.q.weight": originals["blocks.0.q.weight"].reshape(-1)[:-1]}
    bf16 = write_bf16_original(tmp_path / "short", short)
    record = _run(df11, bf16)
    assert record["status"] == "error"
    assert "blocks.0.q.weight" in record["error"]
    out = tmp_path / "out.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 2
    assert json.loads(out.read_text())["groups"][0]["status"] == "error"


@pytest.mark.parametrize(
    ("offsets", "shape"),
    [
        ([90, 0], [5, 9]),  # negative length: fsspec would read to EOF
        ([0, 10], [5, 9]),  # length disagrees with the shape
        ([-2, 88], [5, 9]),
        ([0, 90], [-5, -9]),  # a negative shape whose product still matches
    ],
)
def test_bf16_entry_bounds_are_checked_before_any_data_read(tmp_path, offsets, shape):
    # Bug caught: independent_header's entries used unchecked; a crafted original pulls a whole
    # multi-GB shard into RAM before anything notices the entry is nonsense.
    df11, _, originals = _pair(tmp_path, False)
    bf16 = tmp_path / "craft"
    bf16.mkdir()
    k = originals["blocks.0.k.weight"]
    q_bytes = originals["blocks.0.q.weight"].tobytes()
    entries = {
        "blocks.0.q.weight": {"dtype": "BF16", "shape": shape, "data_offsets": offsets},
        "blocks.0.k.weight": {
            "dtype": "BF16",
            "shape": list(k.shape),
            "data_offsets": [90, 90 + k.nbytes],
        },
    }
    _raw_bf16_shard(bf16 / "model.safetensors", entries, q_bytes + k.tobytes())
    d, b = LocalRangeSource(df11), _RecordingSource(LocalRangeSource(bf16))
    record = verify_group(
        d, b, "blocks.0", config=read_df11_config(df11), dindex=df11_index(d), bindex=bf16_index(b)
    )
    assert record["status"] == "error"
    assert "data_offsets" in record["error"] or "invalid shape" in record["error"]
    header_len = struct.unpack("<Q", (bf16 / "model.safetensors").read_bytes()[:8])[0]
    assert all(start < 8 + header_len for _, start, _ in b.reads), b.reads


def _rewrite(shard, transform):
    data = mx.load(str(shard))
    mx.eval(data)
    mx.save_safetensors(str(shard), transform(data))


def test_oversized_luts_are_refused_before_they_are_read(tmp_path):
    # Bug caught: the group-stats pass reads every group's luts before any validation, with no
    # size cap: a crafted header makes it download an arbitrarily large tensor.
    df11, bf16, _ = _pair(tmp_path, False)
    rows = 19  # one more than the 18 rows upstream can emit
    _rewrite(
        df11 / "blocks_0.safetensors",
        lambda d: {**d, "blocks.0.luts": mx.zeros((rows, 256), mx.uint8)},
    )
    recorders: list[_RecordingSource] = []

    def factory(repo, revision, subdir):
        recorders.append(_RecordingSource(LocalRangeSource(Path(repo))))
        return recorders[-1]

    assert main(_argv(df11, bf16, tmp_path / "out.json"), source_factory=factory) == 2
    assert all(n != rows * 256 for r in recorders for _, _, n in r.reads)


def test_oversized_output_positions_are_refused_before_they_are_read(tmp_path):
    # One 4096-byte block allows at most 2 uint32 entries (8 bytes); 3 entries must not be read.
    df11, bf16, _ = _pair(tmp_path, False)
    _rewrite(
        df11 / "blocks_0.safetensors",
        lambda d: {**d, "blocks.0.output_positions": mx.zeros((12,), mx.uint8)},
    )
    recorders: list[_RecordingSource] = []

    def factory(repo, revision, subdir):
        recorders.append(_RecordingSource(LocalRangeSource(Path(repo))))
        return recorders[-1]

    assert main(_argv(df11, bf16, tmp_path / "out.json"), source_factory=factory) == 2
    assert all(n != 12 for r in recorders for _, _, n in r.reads)


def test_group_arrays_refuses_oversized_luts_before_reading(tmp_path):
    df11, _, _ = _pair(tmp_path, False)
    _rewrite(
        df11 / "blocks_0.safetensors",
        lambda d: {**d, "blocks.0.luts": mx.zeros((19, 256), mx.uint8)},
    )
    d = _RecordingSource(LocalRangeSource(df11))
    record = verify_group(
        d, None, "blocks.0", config=read_df11_config(df11), dindex=df11_index(d), bindex={}
    )
    assert record["status"] == "error"
    assert "luts" in record["error"]
    assert all(n != 19 * 256 for _, _, n in d.reads)


def test_main_moves_a_stale_abort_and_result_aside(tmp_path):
    # Bug caught: a stale abort.json (the watchdog's artifact, in --out's parent) makes a clean run
    # look aborted, and a stale RESULT.json passes for this run's result.
    df11, bf16, _ = _pair(tmp_path, False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "abort.json").write_text('{"reason": "stale"}')
    (run_dir / "RESULT.json").write_text('{"stale": true}')
    assert main(_argv(df11, bf16, run_dir / "RESULT.json"), source_factory=_local_factory) == 0
    assert (run_dir / "abort.previous.json").read_text() == '{"reason": "stale"}'
    assert not (run_dir / "abort.json").exists()
    assert (run_dir / "RESULT.previous.json").read_text() == '{"stale": true}'
    assert json.loads((run_dir / "RESULT.json").read_text())["mode"] == "parity"


def test_a_failing_run_leaves_no_previous_result_in_place(tmp_path):
    # Bug caught: a run that exits 2 before writing its result leaves the previous (passing)
    # RESULT.json in place, so anything reading it sees a pass.
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "RESULT.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 0
    argv = [*_argv(df11, bf16, out), "--groups", "nope"]
    assert main(argv, source_factory=_local_factory) == 2
    assert not out.exists()
    assert json.loads((tmp_path / "RESULT.previous.json").read_text())["mode"] == "parity"


def test_main_creates_a_missing_output_directory(tmp_path):
    # Bug caught: a missing --out parent fails the final write after the whole decode (exit 2).
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "new" / "deeper" / "RESULT.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 0
    assert out.exists()


def test_an_error_after_a_mismatch_keeps_exit_2_but_records_the_mismatch(tmp_path):
    # Bug caught: an error in one sampled group hiding a real mismatch in another from the
    # result's top-level counts.
    df11, _, originals = _triple(tmp_path)
    bad = dict(originals)
    bad["blocks.0.weight"] = bad["blocks.0.weight"] ^ np.uint16(0x8000)
    del bad["blocks.2.weight"]  # blocks.2 has no original -> error
    bf16 = write_bf16_original(tmp_path / "bad3", bad)
    out = tmp_path / "RESULT.json"
    argv = [*_argv(df11, bf16, out), "--groups", "blocks.0,blocks.2"]
    assert main(argv, source_factory=_local_factory) == 2
    result = json.loads(out.read_text())
    assert result["mismatched"] == 1
    assert result["compared"] == 1


def test_main_sets_the_mlx_cache_limit_to_zero_and_records_caps(tmp_path):
    # Bug caught: this script keeping MLX's default cache pool (near device memory) while the
    # whole-checkpoint script bounds it; and a failed cap leaving no trace in the result.
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "RESULT.json"
    previous = mx.set_cache_limit(12345678)
    try:
        assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 0
        assert mx.set_cache_limit(previous) == 0
    finally:
        mx.set_cache_limit(previous)
    caps = json.loads(out.read_text())["memory_caps_gb"]
    assert len(caps) == 2
    assert all(isinstance(c, int) for c in caps)


def test_main_stops_the_watchdog_before_writing_the_result(tmp_path, monkeypatch):
    # Bug caught: stopping the watchdog only in `finally`, after RESULT.json is written, leaves a
    # window in which an abort (exit 70/71) contradicts a result already on disk.
    import scripts.verify_remote_group as vrg

    out = tmp_path / "RESULT.json"
    seen: list[bool] = []

    class _ProbeWatchdog:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            return self

        def stop(self):
            seen.append(out.exists())

    monkeypatch.setattr(vrg, "Watchdog", _ProbeWatchdog)
    df11, bf16, _ = _pair(tmp_path, False)
    assert main(_argv(df11, bf16, out), source_factory=_local_factory) == 0
    assert seen[0] is False
    assert out.exists()
