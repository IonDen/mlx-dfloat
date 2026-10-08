import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from scripts.verify_checkpoint import VerifyError
from scripts.verify_remote_group import (
    GROUP_FIELDS,
    HfRangeSource,
    LocalRangeSource,
    bf16_index,
    df11_index,
    fp32_to_bf16_rne,
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


def test_hf_range_source_reads_with_huggingface_hubs_seekable_default_only(monkeypatch):
    # Bug caught: a read-ahead knob reaching huggingface_hub (block_size=0 gives a streaming file whose seek() raises,
    # so every read past offset 0 fails) when no caller needs one. The fake file system mirrors that: a 0 block size
    # opens a stream that cannot seek; the default opens a seekable file.
    import io

    from scripts import verify_remote_group

    data = bytes(range(32))

    class Stream(io.BytesIO):
        def seek(self, *args):
            raise ValueError("Cannot seek streaming HF file")

    class FakeFs:
        def open(self, path, mode, **kwargs):
            return Stream(data) if kwargs.get("block_size") == 0 else io.BytesIO(data)

    monkeypatch.setattr(verify_remote_group, "HfFileSystem", FakeFs)
    with pytest.raises(TypeError):
        HfRangeSource("owner/repo", "0" * 40, block_size=0)
    src = HfRangeSource("owner/repo", "0" * 40)
    assert src.read("model.safetensors", 5, 4) == bytes([5, 6, 7, 8])


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


def test_an_unusable_output_path_exits_2_not_1(tmp_path, capsys):
    # Bug caught: the output-directory setup running outside main's error handling, so an OSError
    # (here: the --out parent is a regular file) escapes as exit 1, the bit-mismatch kill signal.
    df11, bf16, _ = _pair(tmp_path, False)
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    assert main(_argv(df11, bf16, blocker / "RESULT.json"), source_factory=_local_factory) == 2
    assert "error:" in capsys.readouterr().err


def test_an_existing_directory_as_out_is_refused_and_left_alone(tmp_path, capsys):
    # Bug caught: `args.out.replace(...)` renaming a directory the user named as --out to
    # `<name>.previous.json`.
    df11, bf16, _ = _pair(tmp_path, False)
    target = tmp_path / "results"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    assert main(_argv(df11, bf16, target), source_factory=_local_factory) == 2
    assert (target / "keep.txt").read_text() == "x"
    assert not (tmp_path / "results.previous.json").exists()
    assert "is a directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("f32", "bf16"),
    [
        (0x3F800000, 0x3F80),  # 1.0
        (0x3F808000, 0x3F80),  # exact tie, even neighbour below: stays
        (0x3F818000, 0x3F82),  # exact tie, odd 0x3F81: rounds up to even
        (0x3F807FFF, 0x3F80),  # just below half
        (0x3F808001, 0x3F81),  # just above half
        (0x7F7FFFFF, 0x7F80),  # largest finite float32 rounds to +inf
        (0xFF800000, 0xFF80),  # -inf
        (0x80000000, 0x8000),  # -0.0 keeps its sign
        (0x00000001, 0x0000),  # smallest denormal rounds to +0
        (0x7FC00001, 0x7FC0),  # NaN -> torch's canonical NaN
        (0xFFFFFFFF, 0x7FC0),  # negative NaN too
        (0x7F800001, 0x7FC0),  # the smallest NaN: plain rounding would turn it into +inf (0x7F80)
    ],
)
def test_fp32_to_bf16_rounds_to_nearest_even_like_torch(f32, bf16):
    # Bug caught: truncation (0x3F808001 -> 0x3F80), ties rounded away from zero (0x3F808000 -> 0x3F81), overflow
    # wrapping into NaN, or a NaN payload kept (torch writes 0x7FC0).
    assert int(fp32_to_bf16_rne(np.array([f32], np.uint32))[0]) == bf16


def test_finite_values_agree_with_mlx_cast():
    # Bug caught: an off-by-one in the rounding bias on some exponent range (an independent oracle over 100 000 values).
    bits = (
        np.random.default_rng(3).integers(0, 2**32, size=100_000, dtype=np.uint64).astype(np.uint32)
    )
    bits = bits[(bits & 0x7F800000) != 0x7F800000]  # finite only
    ours = fp32_to_bf16_rne(bits)
    theirs = np.array(mx.array(bits.view(np.float32)).astype(mx.bfloat16).view(mx.uint16))
    assert np.array_equal(ours, theirs)


def _fp32_pair(tmp_path, *, truncate):
    """A DF11 checkpoint and an FP32 original of the same two matrices.

    The first two elements are ties / near-ties where rounding and truncation differ.
    """
    rng = np.random.default_rng(31)
    f32 = {
        "blocks.0.q.weight": rng.integers(0x3F000000, 0x40000000, size=(5, 9)).astype(np.uint32),
        "blocks.0.k.weight": rng.integers(0x3F000000, 0x40000000, size=(3, 9)).astype(np.uint32),
    }
    f32["blocks.0.q.weight"].reshape(-1)[:2] = (0x3F818000, 0x3F808001)
    pick = (lambda b: (b >> 16).astype(np.uint16)) if truncate else fp32_to_bf16_rne
    mats = {"blocks.0": [pick(f32["blocks.0.q.weight"]), pick(f32["blocks.0.k.weight"])]}
    df11 = write_checkpoint(
        tmp_path / "d", groups=mats, pattern=r"blocks\.\d+", sub_paths=("q", "k")
    )
    root = tmp_path / "b"
    root.mkdir()
    mx.save_safetensors(
        str(root / "model.safetensors"),
        {n: mx.array(v.view(np.float32)) for n, v in f32.items()},
    )
    return df11, root


def test_an_fp32_original_passes_only_with_the_cast_and_a_truncated_checkpoint_fails(tmp_path):
    # Bug caught: the cast applied on the wrong side, or F32 accepted silently without the flag. Originals are FP32
    # tie values (0x3F818000 ...); a DF11 built from their RNE bits exits 0 with the flag, 2 without it; a DF11 built
    # from truncated bits exits 1 with it.
    df11, bf16 = _fp32_pair(tmp_path / "rne", truncate=False)
    out = tmp_path / "rne" / "out.json"
    argv = [*_argv(df11, bf16, out), "--groups", "first"]
    assert main(argv, source_factory=_local_factory) == 2
    refusal = json.loads(out.read_text())["groups"][0]["error"]
    assert "F32" in refusal
    assert "--cast-fp32-to-bf16" in refusal
    assert main([*argv, "--cast-fp32-to-bf16"], source_factory=_local_factory) == 0
    written = json.loads(out.read_text())
    assert written["control"] == "fp32 rounded to bf16 (nearest even)"
    assert [m["original_dtype"] for m in written["groups"][0]["matrices"]] == ["F32", "F32"]
    assert written["compared"] == 2
    df11, bf16 = _fp32_pair(tmp_path / "trunc", truncate=True)
    out = tmp_path / "trunc" / "out.json"
    argv = [*_argv(df11, bf16, out), "--groups", "first", "--cast-fp32-to-bf16"]
    assert main(argv, source_factory=_local_factory) == 1
    matrices = json.loads(out.read_text())["groups"][0]["matrices"]
    assert not next(m for m in matrices if m["name"] == "blocks.0.q.weight")["equal"]


def test_a_bf16_original_with_the_cast_flag_stays_bf16_and_records_it(tmp_path):
    # Bug caught: the flag casting an already-BF16 original (reading it as 4-byte floats), or the control label
    # claiming a cast that did not happen.
    df11, bf16, _ = _pair(tmp_path, False)
    out = tmp_path / "out.json"
    argv = [*_argv(df11, bf16, out), "--groups", "first", "--cast-fp32-to-bf16"]
    assert main(argv, source_factory=_local_factory) == 0
    written = json.loads(out.read_text())
    assert [m["original_dtype"] for m in written["groups"][0]["matrices"]] == ["BF16", "BF16"]
    assert written["control"] == "bf16"


def test_main_refuses_the_cast_flag_with_structural_only(tmp_path, capsys):
    # Bug caught: a structural run accepting the flag and claiming an FP32 control it never used.
    df11, _, _ = _pair(tmp_path, False)
    argv = [
        "--df11-repo", str(df11), "--df11-revision", "x", "--structural-only",
        "--cast-fp32-to-bf16", "--out", str(tmp_path / "o.json"),
    ]  # fmt: skip
    with pytest.raises(SystemExit) as info:
        main(argv, source_factory=_local_factory)
    assert info.value.code == 2
    assert "--cast-fp32-to-bf16" in capsys.readouterr().err


# --- config-less repositories, read through a pinned layout ----------------------------------------------------------

_FUSED_PATTERN = {r"blocks\.\d+": ("q", "gate_up")}
_FUSED_SPLITS = {"gate_up": ("gate", "up")}
_NORM = np.full((4,), 0x3F80, dtype=np.uint16)


def _config_less_fused(tmp_path, *, layout_groups=1, layout_extras=1):
    """A config-less single file storing q (4x4) and concat(gate, up) (12x4), its originals, its pinned layout."""
    from tests._df11_fixtures import pin_layout

    rng = np.random.default_rng(17)
    q, g, u = (random_bf16(rng, s) for s in [(4, 4), (6, 4), (6, 4)])
    df11 = write_checkpoint(
        tmp_path / "df11",
        groups={"blocks.0": [q, np.concatenate([g, u])]},
        pattern=r"blocks\.\d+",
        sub_paths=("q", "gate_up"),
        single_file=True,
        extras={"norm.weight": _NORM},
        write_config=False,
    )
    originals = {"blocks.0.q.weight": q, "blocks.0.gate.weight": g, "blocks.0.up.weight": u}
    bf16 = write_bf16_original(tmp_path / "bf16", originals)
    layout = pin_layout(
        df11 / "model.safetensors",
        pattern_dict=_FUSED_PATTERN,
        row_splits=_FUSED_SPLITS,
        groups=layout_groups,
        extras=layout_extras,
    )
    return df11, bf16, layout


def test_a_config_less_repository_is_verified_through_its_layout(tmp_path):
    # Bug caught: the remote verifier stops at the missing config.json, or compares the stored gate_up whole (no
    # original of that name: exit 2), or the halves against each other's originals (exit 1).
    df11, bf16, layout = _config_less_fused(tmp_path)
    out = tmp_path / "out" / "RESULT.json"
    code = main(_argv(df11, bf16, out), source_factory=_local_factory, layouts=(layout,))
    result = json.loads(out.read_text())
    assert code == 0, result
    assert result["config_source"] == "layout tiny"
    got = [(m["name"], m["n"], m["equal"]) for m in result["groups"][0]["matrices"]]
    assert got == [  # 4x4, then the 6x4 halves of the stored 12x4 matrix
        ("blocks.0.q.weight", 16, True),
        ("blocks.0.gate.weight", 24, True),
        ("blocks.0.up.weight", 24, True),
    ]
    assert (result["compared"], result["mismatched"]) == (3, 0)


def test_a_config_json_repository_records_where_its_config_came_from(tmp_path):
    # Bug caught: a repository with a config.json sent through the layout path (or the field left out, so a result
    # cannot say which config its matrix names came from).
    df11, bf16, _ = _pair(tmp_path, single_file=True)
    out = tmp_path / "out" / "RESULT.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory, layouts=()) == 0
    assert json.loads(out.read_text())["config_source"] == "config.json"


def _flip_probed_byte(df11, layout):
    probe = layout.probes[0]
    file = df11 / "model.safetensors"
    from mlx_dfloat._safetensors import read_header

    at = read_header(file)[probe.tensor].offset + probe.offset
    data = bytearray(file.read_bytes())
    data[at] ^= 0x01
    file.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        # Bug caught: an unknown config-less file verified with a guessed matrix order.
        ("no layout", "matches no known layout"),
        # Bug caught: the spot checks skipped on the remote path, so a re-saved file with the same header and other
        # bytes is read with the pinned order.
        ("probe byte flipped", "spot check"),
        # Bug caught: the layout's group and extra counts not checked, so a file holding more or fewer groups passes.
        ("counts differ", "expects 2 groups and 1 extras"),
    ],
)
def test_a_config_less_repository_is_refused_before_any_decode(tmp_path, capsys, case, message):
    df11, bf16, layout = _config_less_fused(
        tmp_path, layout_groups=2 if case == "counts differ" else 1
    )
    if case == "probe byte flipped":
        _flip_probed_byte(df11, layout)
    out = tmp_path / "out" / "RESULT.json"
    layouts = () if case == "no layout" else (layout,)
    assert main(_argv(df11, bf16, out), source_factory=_local_factory, layouts=layouts) == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_a_diffusers_config_without_a_dfloat11_block_is_an_error_not_the_layout_path(tmp_path):
    # Bug caught: a config.json without dfloat11_config (a diffusers transformer config) treated as "no config" and
    # the file read through a layout that happens to match.
    df11, bf16, layout = _config_less_fused(tmp_path)
    (df11 / "config.json").write_text(json.dumps({"_class_name": "QwenImageTransformer2DModel"}))
    out = tmp_path / "out" / "RESULT.json"
    assert main(_argv(df11, bf16, out), source_factory=_local_factory, layouts=(layout,)) == 2
    assert not out.exists()


def test_a_config_less_repository_must_be_one_file(tmp_path):
    # Bug caught: the first of several safetensors files identified and the rest ignored (their groups never seen).
    from scripts.verify_remote_group import config_from_source

    df11, _bf16, layout = _config_less_fused(tmp_path)
    (df11 / "other.safetensors").write_bytes((df11 / "model.safetensors").read_bytes())
    with pytest.raises(VerifyError, match="2 safetensors files"):
        config_from_source(LocalRangeSource(df11), layouts=(layout,))


def test_a_stored_count_that_disagrees_with_the_pattern_is_an_error_not_a_crash(tmp_path):
    # Bug caught: a pattern listing three stored matrices for a group that stores two reaching the row-split cut and
    # the zip (a confusing crash message, or the wrong segment cut) instead of an error record that says so.
    from mlx_dfloat.format import parse_df11_config, with_row_splits

    df11, bf16, _layout = _config_less_fused(tmp_path)
    raw = {
        "version": "0.5.0",
        "threads_per_block": [512],
        "bytes_per_thread": 8,
        "pattern_dict": {r"blocks\.\d+": ["q", "gate_up", "extra"]},
    }
    config = with_row_splits(parse_df11_config(raw, source="t"), _FUSED_SPLITS, source="t")
    d, b = LocalRangeSource(df11), LocalRangeSource(bf16)
    record = verify_group(
        d, b, "blocks.0", config=config, dindex=df11_index(d), bindex=bf16_index(b)
    )
    assert record["status"] == "error"
    assert "stores 2 matrices" in record["error"]
    assert record["matrices"] == []


@pytest.mark.network
def test_the_published_qwen_image_21_file_is_identified_by_range_reads():
    # Bug caught: the remote path reading the header or the spot checks at the wrong offsets (or not at all), so the
    # one published config-less file is refused, or accepted without its content checked. Reads the header (28,408
    # bytes) and nine 4096-byte spot checks, each with huggingface_hub's default read-ahead.
    from scripts.verify_remote_group import config_from_source

    from mlx_dfloat._layouts import KNOWN_LAYOUTS, QWEN_IMAGE_21_COMFYUI

    layout = QWEN_IMAGE_21_COMFYUI
    src = HfRangeSource(layout.repo_id, layout.revision)
    config, source = config_from_source(src, layouts=KNOWN_LAYOUTS)
    assert source == "layout qwen-image-2.1-comfyui"
    assert dict(config.row_splits) == {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}
    assert config.pattern_dict[r"transformer_blocks\.\d+"][4] == "img_mlp.gate_up"


@pytest.mark.parametrize(
    ("probes", "message"),
    [
        # Bug caught: a layout with no spot check accepted on its header alone (the content never checked).
        ((), "pins no spot check"),
        # Bug caught: a spot check of a tensor the file lacks skipped instead of refused.
        ("missing", "which the file lacks"),
        # Bug caught: a spot check past its tensor's end read from the next tensor's bytes (or short-read) silently.
        ("outside", "lies outside its"),
    ],
)
def test_a_layout_whose_spot_checks_cannot_run_is_refused(tmp_path, probes, message):
    import dataclasses

    from scripts.verify_remote_group import config_from_source

    df11, _bf16, layout = _config_less_fused(tmp_path)
    probe = layout.probes[0]
    if probes == "missing":
        probes = (dataclasses.replace(probe, tensor="blocks.9.sign_mantissa"),)
    elif probes == "outside":
        probes = (dataclasses.replace(probe, offset=10**6),)
    with pytest.raises(VerifyError, match=message):
        config_from_source(
            LocalRangeSource(df11), layouts=(dataclasses.replace(layout, probes=probes),)
        )


# --- extras: the uncompressed tensors compared by name with their originals ------------------------------------------

# Three BF16 extras with literal bit patterns (1.0, 2.0, -1.0, the smallest denormal; 0.25, -3.0; 1.0078125 ...).
_EXTRAS = {
    "first.bias": np.array([0x3F80, 0x4000, 0xBF80, 0x0001], dtype=np.uint16),
    "norm.scale": np.array([0x3E80, 0xC040], dtype=np.uint16),
    "mod.lin": np.array([[0x3F81, 0x3F82], [0x3F83, 0x8000]], dtype=np.uint16),
}
# An FP32 original of one extra: the two ties and the two near-ties where nearest-even and truncation differ in two
# of four places. By hand: 0x3F80_8000 (tie, even 0x3F80 below) -> 0x3F80; 0x3F81_8000 (tie, odd 0x3F81) -> 0x3F82;
# 0x3F80_8001 (just above half) -> 0x3F81; 0x3F80_7FFF (just below half) -> 0x3F80. Truncation: 0x3F80, 0x3F81,
# 0x3F80, 0x3F80.
_F32_ORIGINAL = np.array([0x3F808000, 0x3F818000, 0x3F808001, 0x3F807FFF], dtype=np.uint32)
_F32_AS_DF11 = np.array([0x3F80, 0x3F82, 0x3F81, 0x3F80], dtype=np.uint16)


def _extras_pair(tmp_path, *, df11_extras=None, originals=None, single_file=True):
    """A DF11 checkpoint (one group, `_EXTRAS`) and one original file holding the group's matrix and the extras."""
    rng = np.random.default_rng(41)
    q = random_bf16(rng, (4, 6))
    df11 = write_checkpoint(
        tmp_path / "d",
        groups={"blocks.0": [q]},
        pattern=r"blocks\.\d+",
        sub_paths=(),
        extras=_EXTRAS if df11_extras is None else df11_extras,
        single_file=single_file,
    )
    root = tmp_path / "b"
    root.mkdir()
    tensors = {"blocks.0.weight": q, **(_EXTRAS if originals is None else originals)}
    mx.save_safetensors(
        str(root / "model.safetensors"),
        {
            n: mx.array(v).view(mx.float32 if v.dtype == np.uint32 else mx.bfloat16)
            for n, v in tensors.items()
        },
    )
    return df11, root


def _extras_run(tmp_path, df11, bf16, *extra_args):
    out = tmp_path / "out" / "RESULT.json"
    code = main([*_argv(df11, bf16, out), *extra_args], source_factory=_local_factory)
    return code, json.loads(out.read_text())


@pytest.mark.parametrize("single_file", [True, False])
def test_extras_equal_to_bf16_originals_pass_and_are_counted(tmp_path, single_file):
    # Bug caught: extras never read (a pass with 0 compared), or an extras-only shard of a sharded checkpoint (no group
    # in it, so absent from the group index) skipped.
    df11, bf16 = _extras_pair(tmp_path, single_file=single_file)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 0, result
    assert (result["compared"], result["extras_compared"], result["extras_mismatched"]) == (0, 3, 0)
    got = [(e["name"], e["n"], e["equal"], e["original_dtype"]) for e in result["extras"]["extras"]]
    assert got == [
        ("first.bias", 4, True, "BF16"),
        ("mod.lin", 4, True, "BF16"),
        ("norm.scale", 2, True, "BF16"),
    ]
    assert result["control"] == "bf16"


def test_an_fp32_extra_is_compared_rounded_to_nearest_even(tmp_path):
    # Bug caught: truncation instead of RNE (two of four differ), or a silent cast without the flag. The control label
    # must name fp32 although no group matrix was compared (the extras count in it).
    df11, bf16 = _extras_pair(
        tmp_path,
        df11_extras={**_EXTRAS, "mod.lin": _F32_AS_DF11},
        originals={**_EXTRAS, "mod.lin": _F32_ORIGINAL},
    )
    code, result = _extras_run(
        tmp_path, df11, bf16, "--groups", "none", "--extras", "--cast-fp32-to-bf16"
    )
    assert code == 0, result
    assert result["control"] == "fp32 rounded to bf16 (nearest even)"
    dtypes = {e["name"]: e["original_dtype"] for e in result["extras"]["extras"]}
    assert dtypes == {"first.bias": "BF16", "mod.lin": "F32", "norm.scale": "BF16"}
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["status"] == "error"
    assert "--cast-fp32-to-bf16" in result["extras"]["error"]


def test_one_flipped_extra_bit_is_a_mismatch_exit_1(tmp_path):
    # Bug caught: any / all swapped, or the exit code computed before the extras (the group matrix is equal, so only
    # the extras can make this run exit 1).
    flipped = {**_EXTRAS, "norm.scale": _EXTRAS["norm.scale"] ^ np.uint16(0x0001)}
    df11, bf16 = _extras_pair(tmp_path, df11_extras=flipped)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "first", "--extras")
    assert code == 1, result
    assert (result["compared"], result["mismatched"]) == (1, 0)
    assert (result["extras_compared"], result["extras_mismatched"]) == (3, 1)
    assert result["extras"]["status"] == "mismatch"
    assert [e["name"] for e in result["extras"]["extras"] if not e["equal"]] == ["norm.scale"]


def test_an_extra_without_an_original_is_an_error_not_a_pass(tmp_path):
    # Bug caught: an extra with no original skipped (the run passes on the extras that happen to exist).
    df11, bf16 = _extras_pair(
        tmp_path, originals={k: v for k, v in _EXTRAS.items() if k != "mod.lin"}
    )
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["status"] == "error"
    assert result["extras"]["error"] == "no original for extra mod.lin"


def test_an_extra_whose_original_holds_another_element_count_is_an_error_not_a_mismatch(tmp_path):
    # Bug caught: a size disagreement (a mapping problem) reported as a bit mismatch, or compared by a prefix.
    longer = {**_EXTRAS, "norm.scale": np.array([0x3E80, 0xC040, 0x3F80], dtype=np.uint16)}
    df11, bf16 = _extras_pair(tmp_path, originals=longer)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["error"] == "extra norm.scale: shape (2,), original has (3,)"


def test_a_df11_extra_that_is_not_bf16_is_an_error(tmp_path):
    # Bug caught: an F32 extra in the DF11 file read as two-byte elements and reported as a mismatch (exit 1), or
    # passed through.
    df11, bf16 = _extras_pair(tmp_path, df11_extras={}, single_file=False)
    mx.save_safetensors(
        str(df11 / "extra.safetensors"), {"first.bias": mx.array([1.0, 2.0], dtype=mx.float32)}
    )
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["error"] == "extra first.bias is F32, not BF16"


def test_without_the_extras_flag_the_result_keeps_its_old_shape(tmp_path):
    # Bug caught: extras keys written on every run (a merge reading them would count a run that compared none).
    df11, bf16 = _extras_pair(tmp_path)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "first")
    assert code == 0, result
    assert {"extras", "extras_compared", "extras_mismatched"}.isdisjoint(result)


def test_extras_with_structural_only_is_refused(tmp_path, capsys):
    # Bug caught: a structural run accepting --extras and claiming a comparison it cannot make (no original).
    df11, _ = _extras_pair(tmp_path)
    argv = [
        "--df11-repo", str(df11), "--df11-revision", "x", "--structural-only",
        "--extras", "--out", str(tmp_path / "o.json"),
    ]  # fmt: skip
    with pytest.raises(SystemExit) as info:
        main(argv, source_factory=_local_factory)
    assert info.value.code == 2
    assert "--extras" in capsys.readouterr().err


def test_groups_none_without_extras_is_refused(tmp_path, capsys):
    # Bug caught: a run that compares nothing at all exiting 0 (a vacuous pass recorded as parity).
    df11, bf16 = _extras_pair(tmp_path)
    with pytest.raises(SystemExit) as info:
        main(
            [*_argv(df11, bf16, tmp_path / "o.json"), "--groups", "none"],
            source_factory=_local_factory,
        )
    assert info.value.code == 2
    assert "--groups none" in capsys.readouterr().err


def test_groups_none_selects_nothing_and_mixed_with_a_group_is_refused():
    # Bug caught: `none` silently dropping the other selectors, or read as a group name.
    stats = {"blocks.0": {"elements_per_block": 5, "max_code_length": 30}}
    assert pick_groups(stats, "none") == []
    for selector in ("none,first", "first,none"):
        with pytest.raises(KeyError, match="none"):
            pick_groups(stats, selector)


def test_an_extras_only_run_on_a_repository_without_extras_is_an_error(tmp_path):
    # Bug caught: `--groups none --extras` on a checkpoint with no extras exiting 0 having compared nothing.
    df11, bf16 = _extras_pair(tmp_path, df11_extras={}, single_file=False)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["status"] == "error"
    assert result["extras"]["error"] == (
        "the checkpoint holds no extras: --groups none --extras compares nothing"
    )
    # With a group selected the same checkpoint passes: the group is compared, and no extras is not an error then.
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "first", "--extras")
    assert code == 0, result
    assert (result["compared"], result["extras_compared"]) == (1, 0)


def test_an_extra_whose_original_has_another_shape_is_an_error_before_its_data_is_read(tmp_path):
    # Bug caught: a transposed or reshaped original (same element count) compared byte for byte, a mapping
    # problem passed off as a pass or a mismatch. mod.lin is 2 x 2 in the checkpoint; its original here is 1 x 4.
    reshaped = {**_EXTRAS, "mod.lin": _EXTRAS["mod.lin"].reshape(1, 4)}
    df11, bf16 = _extras_pair(tmp_path, originals=reshaped)
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 2
    assert result["extras"]["error"] == "extra mod.lin: shape (2, 2), original has (1, 4)"


def test_a_shard_outside_the_checkpoints_index_is_not_an_extra(tmp_path):
    # Bug caught: every safetensors file in the repository read as the checkpoint's (a DF11 repository that also
    # hosts an encoder file would fail with "no original for extra ..."). The index names the group shard and the
    # extras shard; vae.safetensors sits next to them, unlisted.
    df11, bf16 = _extras_pair(tmp_path, single_file=False)
    weight_map = {
        **{f"blocks.0.{f}": "blocks_0.safetensors" for f in GROUP_FIELDS},
        **dict.fromkeys(_EXTRAS, "model.safetensors"),
    }
    (df11 / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    mx.save_safetensors(
        str(df11 / "vae.safetensors"), {"decoder.conv.weight": mx.ones((2,), dtype=mx.bfloat16)}
    )
    code, result = _extras_run(tmp_path, df11, bf16, "--groups", "none", "--extras")
    assert code == 0, result
    assert [e["name"] for e in result["extras"]["extras"]] == [
        "first.bias",
        "mod.lin",
        "norm.scale",
    ]
