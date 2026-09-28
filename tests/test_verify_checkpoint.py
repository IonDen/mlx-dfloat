import json
import threading

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


def _summary(out):
    return json.loads((out / "summary.json").read_text())


def _corrupt_gaps(df11, group):
    shard = df11 / f"{group.replace('.', '_')}.safetensors"
    data = mx.load(str(shard))
    mx.eval(data)
    gaps = np.array(data[f"{group}.gaps"])
    gaps[0] ^= 0xFF  # thread continuity breaks -> DFloatFormatError while decoding this group
    data[f"{group}.gaps"] = mx.array(gaps)
    mx.save_safetensors(str(shard), data)


def _count_decodes(monkeypatch):
    calls = []
    real = vc.decode_matrices
    monkeypatch.setattr(
        vc, "decode_matrices", lambda g, **kw: calls.append(g.name) or real(g, **kw)
    )
    return calls


def _hf_symlinked_layout(dest, tensors):
    """A snapshots/<sha>/ dir whose *.safetensors shard files are symlinks into a sibling blobs/ dir.

    Mirrors the real Hugging Face cache layout: ``<repo>/snapshots/<revision>/<shard>`` is a
    symlink into ``<repo>/blobs/<hash>``.
    """
    staging = write_bf16_original(dest / "_staging", tensors)
    snapshot = dest / "snapshots" / ("c" * 40)
    blobs = dest / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    for path in sorted(staging.iterdir()):
        if path.suffix == ".safetensors":
            blob = blobs / f"{path.stem}.blob"
            blob.write_bytes(path.read_bytes())
            (snapshot / path.name).symlink_to(blob)
        else:
            (snapshot / path.name).write_bytes(path.read_bytes())
    return snapshot


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


def test_an_extra_size_mismatch_is_an_error_not_a_mismatch(tmp_path, pair):
    # Bug caught: an extra whose element count differs from its same-named original goes through
    # `np.array_equal` (False for mismatched shapes) and is counted as a bit mismatch (exit 1)
    # instead of the mapping/format problem it actually is (exit 2), same bug as `_compare`'s
    # matrix-size check above but on the extras path.
    df11, _, originals = pair
    bad = dict(originals)
    bad["norm.weight"] = np.concatenate([NORM, NORM])
    assert verify(df11, write_bf16_original(tmp_path / "bad", bad), tmp_path / "out", key=KEY) == 2


def test_verify_with_an_empty_groups_list_is_an_error(tmp_path, pair):
    # Bug caught: `groups or ckpt.groups` treats `[]` the same as `None` (falls back to every
    # group) while `groups is None` (used to gate the coverage check) is False for `[]`, so an
    # empty list silently runs the full checkpoint with the coverage check disabled.
    df11, bf16, _ = pair
    assert verify(df11, bf16, tmp_path / "out", key=KEY, groups=[]) == 2


def test_a_size_mismatch_is_an_error_not_a_mismatch(tmp_path, pair):
    # Bug caught: treating a size mismatch (a mapping/format problem) as `equal=False` reports it
    # as a bit mismatch (exit 1) instead of the tool error it actually is (exit 2).
    df11, _, originals = pair
    bad = dict(originals)
    bad["blocks.1.k.weight"] = bad["blocks.1.k.weight"].reshape(-1)[:-1]
    assert verify(df11, write_bf16_original(tmp_path / "bad", bad), tmp_path / "out", key=KEY) == 2


def test_extra_with_a_non_bf16_dtype_and_a_same_named_original_is_an_error(tmp_path, pair):
    # Bug caught: silently skipping an extra whose dtype isn't BF16 lets a mismatched-dtype extra
    # count as "covered" even though it was never bit-compared against its original.
    df11, bf16, _ = pair
    mx.save_safetensors(
        str(df11 / "model.safetensors"), {"norm.weight": mx.zeros((1,), mx.float32)}
    )
    assert verify(df11, bf16, tmp_path / "out", key=KEY) == 2


def test_extra_without_a_matching_original_does_not_block_success(tmp_path):
    rng = np.random.default_rng(11)
    mats = {"blocks.0": [random_bf16(rng, (4, 4)), random_bf16(rng, (4, 4))]}
    df11 = write_checkpoint(
        tmp_path / "df11",
        groups=mats,
        pattern=r"blocks\.\d+",
        sub_paths=("q", "k"),
        extras={"norm.weight": NORM, "scale.weight": np.array([0x3F80], np.uint16)},
    )
    q, k = mats["blocks.0"]
    originals = {"norm.weight": NORM, "blocks.0.q.weight": q, "blocks.0.k.weight": k}
    bf16 = write_bf16_original(tmp_path / "bf16", originals)
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["extras_compared"] == 1


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
    # The first group with an original in the truncated shard is the one that fails.
    error = _summary(tmp_path / "out")["error"]
    assert error.startswith("blocks.0: cannot load originals from model-00001-of-00002"), error


def test_shard_path_escaping_the_directory_is_refused(tmp_path, pair):
    df11, bf16, _ = pair
    index = bf16 / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    data["weight_map"]["blocks.0.q.weight"] = "../elsewhere.safetensors"
    index.write_text(json.dumps(data))
    assert verify(df11, bf16, tmp_path / "out", key=KEY) == 2
    assert "unsafe shard name '../elsewhere.safetensors'" in _summary(tmp_path / "out")["error"]


def test_hf_cache_shard_symlinks_into_a_sibling_blobs_dir_are_followed(tmp_path, pair):
    # Bug caught: rejecting a shard whose resolved parent differs from `root` (the real HF cache
    # layout, where every shard is a symlink into ../../blobs/<hash>) instead of following it.
    df11, _, originals = pair
    hf_root = _hf_symlinked_layout(tmp_path / "hf", originals)
    assert verify(df11, hf_root, tmp_path / "out", key=KEY) == 0


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
    assert _summary(tmp_path / "o2")["error"].startswith("blocks.1: ")


def test_checkpoint_with_zero_groups_is_an_error_in_both_modes(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    config = {
        "dfloat11_config": {
            "version": "0.5.0",
            "threads_per_block": [512],
            "bytes_per_thread": 8,
            "pattern_dict": {r"blocks\.\d+": ["q", "k"]},
        }
    }
    (root / "config.json").write_text(json.dumps(config))
    assert verify(root, None, tmp_path / "o1", key=SKEY) == 2
    assert verify(root, root, tmp_path / "o2", key=KEY) == 2


def test_summary_json_reports_exit_code_2_on_an_early_error(tmp_path, pair):
    # Bug caught: only writing summary.json on the success path leaves a stale (or absent)
    # summary.json behind after a run that errors early, misleading anything reading it.
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    assert verify(df11, bf16, out, key=KEY, groups=["nope"]) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["exit_code"] == 2
    assert "error" in summary


def test_success_summary_records_partial_and_selected_groups(tmp_path, pair):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY, groups=["blocks.0", "blocks.1"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["partial"] is True
    assert summary["selected_groups"] == ["blocks.0", "blocks.1"]


@pytest.mark.parametrize(
    "edited",
    [
        "src/mlx_dfloat/sub/deep.py",
        "src/mlx_dfloat/_watchdog.py",
        "scripts/verify_checkpoint.py",
        "scripts/_bench_common.py",
        "scripts/_flux_rig.py",
        "scripts/bench_flux_step.py",
    ],
)
def test_source_hash_covers_package_watchdog_and_script(tmp_path, monkeypatch, edited):
    # Bug caught: hashing only src/mlx_dfloat/*.py (top level) lets an edit to a subpackage,
    # verify_checkpoint.py, _watchdog.py or the FLUX rig / step bench go unnoticed, so a stale
    # resumed result is reused. Runs on a throwaway copy: the tracked files are never touched.
    for rel in ["src/mlx_dfloat/format.py", *[e for e in [edited] if e.endswith(".py")]]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")
    for rel in [
        "scripts/verify_checkpoint.py",
        "scripts/_bench_common.py",
        "scripts/_flux_rig.py",
        "scripts/bench_flux_step.py",
    ]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")
    monkeypatch.setattr(vc, "_SRC", tmp_path / "src" / "mlx_dfloat")
    monkeypatch.setattr(vc, "_SCRIPTS", tmp_path / "scripts")
    monkeypatch.setattr(vc, "_REPO", tmp_path)
    before = vc.source_hash()
    (tmp_path / edited).write_text("# edited\n")
    assert vc.source_hash() != before


def test_resume_recomputes_when_stored_matrix_names_do_not_match_the_group(
    tmp_path, pair, monkeypatch
):
    # Bug caught: trusting a stored result whose matrix names no longer match the group's current
    # matrix_names (e.g. after a pattern_dict rename) silently reuses data for the wrong matrices.
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    path = out / "groups" / "blocks.1.json"
    stored = json.loads(path.read_text())
    stored["matrices"][0]["name"] = "blocks.1.renamed.weight"
    path.write_text(json.dumps(stored))
    calls = _count_decodes(monkeypatch)
    assert verify(df11, bf16, out, key=KEY) == 0
    assert calls == ["blocks.1"]


def test_resumed_run_still_detects_a_missing_original(tmp_path, pair, monkeypatch):
    # Bug caught: not persisting the per-group `missing` list means a resumed run (which skips
    # decoding) forgets the coverage gap the first run found, and wrongly reports success.
    df11, _, originals = pair
    fewer = write_bf16_original(
        tmp_path / "less", {k: v for k, v in originals.items() if k != "blocks.2.q.weight"}
    )
    out = tmp_path / "out"
    assert verify(df11, fewer, out, key=KEY) == 2
    calls = _count_decodes(monkeypatch)
    assert verify(df11, fewer, out, key=KEY) == 2
    assert calls == []
    summary = json.loads((out / "summary.json").read_text())
    assert "blocks.2.q.weight" in summary["missing_originals"]


def test_a_mismatch_and_a_coverage_gap_together_the_mismatch_wins(tmp_path, pair):
    df11, _, originals = pair
    bad = dict(originals)
    bad["blocks.0.q.weight"] = bad["blocks.0.q.weight"] ^ np.uint16(0x8000)
    del bad["blocks.2.q.weight"]
    bf16 = write_bf16_original(tmp_path / "bad", bad)
    assert verify(df11, bf16, tmp_path / "out", key=KEY) == 1


def test_cli_runs_the_real_watchdog_and_cleans_up_its_thread(tmp_path, pair):
    df11, bf16, _ = pair
    out = tmp_path / "out"
    out.mkdir()
    (out / "abort.json").write_text('{"reason": "stale"}')
    baseline = threading.active_count()
    argv = [
        "--df11",
        str(df11),
        "--bf16",
        str(bf16),
        "--out",
        str(out),
        "--df11-revision",
        "a" * 40,
        "--bf16-revision",
        "b" * 40,
        "--wall-budget",
        "3600",
    ]
    assert main(argv) == 0
    assert threading.active_count() == baseline
    assert (out / "abort.previous.json").read_text() == '{"reason": "stale"}'
    assert not (out / "abort.json").exists()


def test_main_returns_2_on_an_unexpected_exception(tmp_path, pair, monkeypatch):
    df11, bf16, _ = pair

    def _boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(vc, "run_key", _boom)
    argv = [
        "--df11",
        str(df11),
        "--bf16",
        str(bf16),
        "--out",
        str(tmp_path / "o"),
        "--df11-revision",
        "a" * 40,
        "--bf16-revision",
        "b" * 40,
        "--no-watchdog",
    ]
    assert main(argv) == 2


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


def test_index_without_weight_map_is_an_error_with_a_summary(tmp_path, pair):
    # Bug caught: json.loads(...)["weight_map"] raising KeyError past verify's except tuple exits
    # 2 through main's catch-all with NO summary.json (the previous one was already moved aside).
    df11, bf16, _ = pair
    (bf16 / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 2
    assert "weight_map" in _summary(out)["error"]


@pytest.mark.parametrize(
    "index", [[1, 2], {"weight_map": ["a"]}, {"weight_map": {"blocks.0.q.weight": 7}}]
)
def test_malformed_index_shapes_are_errors_with_a_summary(tmp_path, pair, index):
    df11, bf16, _ = pair
    (bf16 / "model.safetensors.index.json").write_text(json.dumps(index))
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 2
    assert _summary(out)["exit_code"] == 2


def test_an_error_after_a_mismatch_keeps_exit_2_but_records_the_mismatch(tmp_path, pair):
    # Bug caught: the error summary for a later group dropping `mismatched`/`compared`, so a real
    # bit mismatch already found (the kill signal) is visible only in that group's own JSON.
    df11, _, originals = pair
    bad = dict(originals)
    bad["blocks.0.q.weight"] = bad["blocks.0.q.weight"] ^ np.uint16(0x8000)
    bf16 = write_bf16_original(tmp_path / "bad", bad)
    _corrupt_gaps(df11, "blocks.1")
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 2
    summary = _summary(out)
    assert summary["error"].startswith("blocks.1: ")
    assert summary["mismatched"] == 1
    assert summary["compared"] == 2


class _StopProbe:
    """A watchdog stand-in that records whether summary.json already existed when stopped."""

    def __init__(self, out):
        self._out = out
        self.summary_existed_at_stop = None

    def stop(self):
        self.summary_existed_at_stop = (self._out / "summary.json").exists()


@pytest.mark.parametrize("groups", [None, ["nope"]])
def test_the_watchdog_is_stopped_before_the_summary_is_written(tmp_path, pair, groups):
    # Bug caught: stopping the watchdog only after verify() returns leaves a window in which a
    # verdict fires after summary.json already says the run passed (or failed) cleanly.
    df11, bf16, _ = pair
    out = tmp_path / "out"
    probe = _StopProbe(out)
    verify(df11, bf16, out, key=KEY, groups=groups, watchdog=probe)
    assert probe.summary_existed_at_stop is False
    assert (out / "summary.json").exists()


def test_summary_records_the_installed_memory_caps(tmp_path, pair, monkeypatch):
    # Bug caught: a cap that failed to install (see install_memory_caps) leaves no trace in the
    # run's own record.
    monkeypatch.setattr(vc, "install_memory_caps", lambda: (7, 9))
    df11, bf16, _ = pair
    out = tmp_path / "out"
    assert verify(df11, bf16, out, key=KEY) == 0
    assert _summary(out)["memory_caps_gb"] == [7, 9]
    assert verify(df11, bf16, out, key=KEY, groups=["nope"]) == 2
    assert _summary(out)["memory_caps_gb"] == [7, 9]


# --- the Metal decoder (--decoder metal) --------------------------------------------------------


def _metal_argv(df11, out, *extra):
    return ["--df11", str(df11), "--out", str(out), "--decoder", "metal", "--no-watchdog", *extra]


@pytest.mark.parametrize(
    ("bf16_given", "decoder", "mode"),
    [
        (True, "reference", "parity"),
        (True, "metal", "parity"),
        (False, "metal", "kernel-vs-reference"),
        (False, "reference", "structural-only"),
    ],
)
def test_run_mode_picks_the_comparison_from_bf16_and_decoder(bf16_given, decoder, mode):
    # Bug caught: a Metal run with no BF16 falling back to structural-only (no oracle at all), or a
    # Metal run with BF16 compared against the reference instead of the originals.
    assert vc.run_mode(bf16_given=bf16_given, decoder=decoder) == mode


@pytest.mark.metal
def test_metal_decoder_runs_once_per_group_and_matches_the_bf16_original(
    pair, tmp_path, monkeypatch
):
    # Bug caught: --decoder metal silently decoding with the reference.
    calls = []
    real = vc._decode_with_metal
    monkeypatch.setattr(
        vc, "_decode_with_metal", lambda group, **kw: calls.append(group.name) or real(group, **kw)
    )
    df11, bf16, _ = pair
    out = tmp_path / "o"
    rc = main(_metal_argv(df11, out, "--bf16", str(bf16)))
    summary = _summary(out)
    assert rc == 0
    assert summary["mismatched"] == 0
    assert summary["key"]["decoder"] == "metal"
    assert summary["key"]["mode"] == "parity"
    assert summary["compared"] == 6
    assert sorted(calls) == sorted(summary["selected_groups"])
    record = json.loads((out / "groups" / "blocks.0.json").read_text())
    assert record["decoder"] == "metal"
    assert {"max_elements_per_block", "direct_blocks"} <= record.keys()
    assert record["max_elements_per_block"] == 100  # one 4096-byte block holds the whole group
    assert record["direct_blocks"] == 0


@pytest.mark.metal
def test_metal_without_bf16_compares_against_the_reference(pair, tmp_path):
    # Bug caught: a structural-only exit 0 hiding a kernel-vs-oracle divergence when no BF16 is
    # available.
    out = tmp_path / "o"
    rc = main(_metal_argv(pair[0], out))
    summary = _summary(out)
    assert rc == 0
    assert summary["key"]["mode"] == "kernel-vs-reference"
    assert summary["compared"] == summary["expected"] == 6


@pytest.mark.metal
def test_a_kernel_bit_flip_is_exit_1(pair, tmp_path, monkeypatch):
    # Bug caught: a kernel-vs-reference mismatch reported as 2 instead of the kill signal 1.
    real = vc._decode_with_metal

    def flipped(group, **kw):
        bits = real(group, **kw)
        first = next(k for k in bits if k != "__meta__")
        bits[first] = bits[first].copy()
        bits[first][0] ^= 1
        return bits

    monkeypatch.setattr(vc, "_decode_with_metal", flipped)
    out = tmp_path / "o"
    assert main(_metal_argv(pair[0], out)) == 1
    record = json.loads((out / "groups" / "blocks.0.json").read_text())
    assert record["status"] == "mismatch"
    assert record["matrices"][0]["name"] == "blocks.0.q.weight"
    assert record["matrices"][0]["first_mismatch_index"] == 0
    assert record["matrices"][1]["equal"] is True


@pytest.mark.metal
def test_a_nonzero_kernel_status_is_exit_2_not_1(pair, tmp_path, monkeypatch):
    # Bug caught: a corrupt group's status surfacing as the bit-mismatch kill signal.
    from mlx_dfloat.errors import DFloatFormatError

    def refuse(group, **kw):
        raise DFloatFormatError("blocks.0: block 0: thread chain broken")

    monkeypatch.setattr(vc, "_decode_with_metal", refuse)
    out = tmp_path / "o"
    assert main(_metal_argv(pair[0], out)) == 2
    assert _summary(out)["error"].startswith("blocks.0: ")


@pytest.mark.metal
def test_metal_on_a_corrupt_group_with_bf16_is_exit_2_not_1(pair, tmp_path):
    # Bug caught: a corrupt group reaching the compare (the kernel's garbage against the real
    # original) and reading as a false kill signal instead of the format error it is.
    df11, bf16, _ = pair
    _corrupt_gaps(df11, "blocks.1")
    out = tmp_path / "o"
    assert main(_metal_argv(df11, out, "--bf16", str(bf16))) == 2
    summary = _summary(out)
    assert summary["error"].startswith("blocks.1: ")
    assert summary["mismatched"] == 0


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("zip() argument 2 is shorter")])
def test_a_runtime_or_value_error_inside_a_group_decode_is_exit_2_with_a_summary(
    pair, tmp_path, monkeypatch, exc
):
    # Bug caught: the group loop catching only DFloatError and VerifyError, so a Metal command-buffer
    # failure at mx.eval (RuntimeError) or a zip(strict=True) mismatch (ValueError) escapes to main's
    # catch-all: exit 2, but no summary.json, and the previous one is already moved aside.
    monkeypatch.setattr(vc, "available_backends", lambda: ("reference", "metal"))

    def boom(group, **kw):
        raise exc

    monkeypatch.setattr(vc, "_decode_with_metal", boom)
    out = tmp_path / "o"
    assert main(_metal_argv(pair[0], out)) == 2
    summary = _summary(out)
    assert summary["exit_code"] == 2
    assert summary["error"] == f"blocks.0: {exc}"


def test_main_installs_the_memory_caps_before_probing_the_metal_backend(
    pair, tmp_path, monkeypatch
):
    # Bug caught: available_backends() (the kernel warm-up, the first GPU work of the process) running
    # before install_memory_caps(), so the warm-up dispatches with no wired cap in place.
    order = []
    monkeypatch.setattr(vc, "install_memory_caps", lambda: order.append("caps") or (7, 9))
    monkeypatch.setattr(
        vc, "available_backends", lambda: order.append("backends") or ("reference", "metal")
    )
    monkeypatch.setattr(
        vc,
        "_decode_with_metal",
        lambda group, **kw: {**vc.decode_matrices(group), "__meta__": np.array([100, 0])},
    )
    assert main(_metal_argv(pair[0], tmp_path / "o")) == 0
    assert order[:2] == ["caps", "backends"]


def test_metal_absent_is_exit_2(pair, tmp_path, monkeypatch, capsys):
    # Bug caught: --decoder metal on a machine without the kernel quietly running something else.
    monkeypatch.setattr(vc, "available_backends", lambda: ("reference",))
    out = tmp_path / "o"
    assert main(_metal_argv(pair[0], out)) == 2
    assert "metal" in capsys.readouterr().err
    assert not (out / "summary.json").exists()


def _rate_file(tmp_path, payload):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.metal
@pytest.mark.parametrize(
    ("gbps", "code"),
    # Each group decodes 100 elements = 200 bytes out; the per-dispatch limit is 0.25 s.
    [(1e-6, 0), (7e-7, 2)],  # 1000 B/s -> 0.2 s passes; 700 B/s -> 0.29 s is refused
)
def test_rate_from_converts_gbps_and_guards_each_dispatch(pair, tmp_path, gbps, code):
    # Bug caught: --rate-from read in the wrong unit (GB/s taken as B/s, or 1e6 for 1e9), or the
    # guard's refusal escaping as a crash with no summary.
    out = tmp_path / "o"
    rate = _rate_file(tmp_path, {"gbps": gbps, "exit_code": 0, "other": "ignored"})
    assert main(_metal_argv(pair[0], out, "--rate-from", str(rate))) == code
    summary = _summary(out)
    assert summary["exit_code"] == code
    if code:
        assert "blocks.0" in summary["error"]
        assert "0.25" in summary["error"]


@pytest.mark.parametrize(
    "payload",
    [
        {"exit_code": 0},
        {"gbps": "fast", "exit_code": 0},
        {"gbps": 0, "exit_code": 0},
        {"gbps": -1.0, "exit_code": 0},
        [1.0],
    ],
)
def test_rate_from_without_a_positive_gbps_is_exit_2(pair, tmp_path, monkeypatch, payload, capsys):
    # Bug caught: a bench JSON with no usable rate silently disabling the per-dispatch guard.
    monkeypatch.setattr(vc, "available_backends", lambda: ("reference", "metal"))
    out = tmp_path / "o"
    rate = _rate_file(tmp_path, payload)
    assert main(_metal_argv(pair[0], out, "--rate-from", str(rate))) == 2
    assert "gbps" in capsys.readouterr().err
    assert not (out / "summary.json").exists()


@pytest.mark.parametrize(
    "payload", [{"gbps": 10.0, "exit_code": 1}, {"gbps": 10.0, "exit_code": 2}, {"gbps": 10.0}]
)
def test_rate_from_refuses_a_bench_that_did_not_pass(pair, tmp_path, monkeypatch, payload, capsys):
    # Bug caught: a bench JSON whose kernel failed parity (exit 1), errored (exit 2) or never recorded
    # a verdict sizing the per-dispatch guard with a rate a wrong kernel produced.
    monkeypatch.setattr(vc, "available_backends", lambda: ("reference", "metal"))
    out = tmp_path / "o"
    rate = _rate_file(tmp_path, payload)
    assert main(_metal_argv(pair[0], out, "--rate-from", str(rate))) == 2
    assert "exit_code" in capsys.readouterr().err
    assert not (out / "summary.json").exists()


def test_compare_arrays_records_the_first_mismatch_and_refuses_a_size_mismatch():
    # Bug caught: an in-memory compare that reports only equal/unequal (no index), or that
    # counts a length difference as a bit mismatch (exit 1) instead of an error (exit 2).
    a = np.array([1, 2, 3, 4], np.uint16)
    records = vc._compare_arrays({"m": a, "n": a}, {"m": np.array([1, 2, 9, 4], np.uint16), "n": a})
    assert records == [
        {"name": "m", "n": 4, "equal": False, "first_mismatch_index": 2},
        {"name": "n", "n": 4, "equal": True, "first_mismatch_index": None},
    ]
    with pytest.raises(vc.VerifyError, match="decoded 4 elements, oracle has 3"):
        vc._compare_arrays({"m": a}, {"m": a[:3]})


@pytest.mark.parametrize("text", [None, "{not json"])
def test_rate_from_an_unreadable_bench_json_is_exit_2(pair, tmp_path, monkeypatch, capsys, text):
    # Bug caught: a missing or truncated bench JSON crashing out with a traceback (or running with
    # no guard) instead of a clear refusal before any decode.
    monkeypatch.setattr(vc, "available_backends", lambda: ("reference", "metal"))
    rate = tmp_path / "bench.json"
    if text is not None:
        rate.write_text(text)
    out = tmp_path / "o"
    assert main(_metal_argv(pair[0], out, "--rate-from", str(rate))) == 2
    assert "cannot read the bench JSON" in capsys.readouterr().err


def test_rate_from_with_the_reference_decoder_is_a_usage_error(pair, tmp_path, capsys):
    # The per-dispatch guard only exists on the Metal path. Bug caught: --rate-from accepted and silently
    # ignored under --decoder reference, so the caller believes a guard is armed that never runs.
    out = tmp_path / "o"
    rate = _rate_file(tmp_path, {"gbps": 10.0})
    argv = ["--df11", str(pair[0]), "--out", str(out), "--no-watchdog", "--rate-from", str(rate)]
    assert main(argv) == 2
    assert "--rate-from needs --decoder metal" in capsys.readouterr().err
    assert not (out / "summary.json").exists()
