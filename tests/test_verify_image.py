import json
import sys

import mlx.core as mx
from scripts import verify_image as vi


def test_nondegenerate_names_each_failed_check():
    # Bug caught: a NaN latent or an untouched noise tensor passing as "an image".
    noise = mx.random.normal((1, 16, 64), key=mx.random.key(0))
    assert vi.nondegenerate(noise * 2 + 1, noise) == []
    assert vi.nondegenerate(noise, noise) == ["moved_from_noise"]
    assert vi.nondegenerate(mx.zeros_like(noise), noise) == ["variance"]
    nan = mx.array([float("nan")]) * mx.ones_like(noise)
    assert "finite" in vi.nondegenerate(nan, noise)


def test_compare_latents_is_bitwise_and_distinguishes_signed_zero():
    # Bug caught: float == (which equates +0.0 and -0.0 and NaN-mismatches silently) instead of a bit view.
    a = mx.array([0.0, 1.5, -2.0], dtype=mx.float32)
    assert vi.compare_latents(a, mx.array([0.0, 1.5, -2.0], dtype=mx.float32))
    assert not vi.compare_latents(a, mx.array([-0.0, 1.5, -2.0], dtype=mx.float32))
    assert not vi.compare_latents(a, mx.array([0.0, 1.5], dtype=mx.float32))


def test_verdict_uses_1_only_for_a_real_mismatch():
    # Bug caught: a degenerate image (nothing compared, really) reported as the 0/1 verdict.
    assert vi.verdict(equal=True, degenerate=[]) == 0
    assert vi.verdict(equal=False, degenerate=[]) == 1
    assert vi.verdict(equal=True, degenerate=["variance"]) == 2


def test_child_command_runs_one_side_without_orchestrate(tmp_path):
    # Bug caught: the child re-entering --orchestrate (fork bomb), or losing a setting.
    args = vi.parse_args(
        [
            "--orchestrate",
            "--df11",
            "d",
            "--base",
            "b",
            "--out",
            str(tmp_path),
            "--steps",
            "3",
            "--size",
            "512",
            "--seed",
            "9",
            "--prompt",
            "p",
            "--eval-policy",
            "depth2",
        ]
    )
    cmd = vi.child_command(args, "bf16")
    assert cmd[:3] == [sys.executable, "-m", "scripts.verify_image"]
    assert "--orchestrate" not in cmd
    assert cmd[cmd.index("--side") + 1] == "bf16"
    for flag, value in (
        ("--steps", "3"),
        ("--size", "512"),
        ("--seed", "9"),
        ("--prompt", "p"),
        ("--eval-policy", "depth2"),
    ):
        assert cmd[cmd.index(flag) + 1] == value


def test_a_side_is_complete_only_with_a_result_of_the_same_key(tmp_path):
    # Bug caught: resuming on a result from another prompt, seed or source hash.
    key = {
        "model": "schnell",
        "seed": 42,
        "steps": 4,
        "size": 1024,
        "prompt": "p",
        "source": "x",
        "mlx": "y",
    }
    side = tmp_path / "df11"
    side.mkdir()
    assert not vi.side_complete(side, key)
    (side / "result.json").write_text(json.dumps({"key": key, "exit_code": 0}))
    assert vi.side_complete(side, key)
    (side / "result.json").write_text(json.dumps({"key": {**key, "seed": 1}, "exit_code": 0}))
    assert not vi.side_complete(side, key)
    (side / "result.json").write_text(json.dumps({"key": key, "exit_code": 2}))
    assert not vi.side_complete(side, key)


def _write_result(side_dir, key, exit_code):
    side_dir.mkdir(exist_ok=True)
    (side_dir / "result.json").write_text(json.dumps({"key": key, "exit_code": exit_code}))


def test_sides_ready_names_a_missing_result_json(tmp_path):
    # Bug caught: comparing before a side has even run (no result.json yet).
    key = {
        "model": "schnell",
        "seed": 42,
        "steps": 4,
        "size": 1024,
        "prompt": "p",
        "source": "x",
        "mlx": "y",
    }
    assert vi.sides_ready(tmp_path, key) == ["df11: no result.json", "bf16: no result.json"]


def test_sides_ready_names_a_side_that_did_not_exit_ok(tmp_path):
    # Bug caught: comparing a side that aborted or errored (exit_code != 0) as if it had succeeded.
    key = {
        "model": "schnell",
        "seed": 42,
        "steps": 4,
        "size": 1024,
        "prompt": "p",
        "source": "x",
        "mlx": "y",
    }
    _write_result(tmp_path / "df11", key, 2)
    _write_result(tmp_path / "bf16", key, 0)
    assert vi.sides_ready(tmp_path, key) == ["df11: exit_code 2, not 0"]


def test_sides_ready_names_a_side_with_a_different_seed(tmp_path):
    # Bug caught: comparing latents from a stale bf16 side left over from a different run's seed.
    key = {
        "model": "schnell",
        "seed": 42,
        "steps": 4,
        "size": 1024,
        "prompt": "p",
        "source": "x",
        "mlx": "y",
    }
    _write_result(tmp_path / "df11", key, 0)
    _write_result(tmp_path / "bf16", {**key, "seed": 1}, 0)
    assert vi.sides_ready(tmp_path, key) == ["bf16: key differs in ['seed']"]


def test_sides_ready_is_empty_when_both_sides_are_complete_and_matching(tmp_path):
    # Bug caught: refusing to compare a perfectly good, matching pair (a false positive).
    key = {
        "model": "schnell",
        "seed": 42,
        "steps": 4,
        "size": 1024,
        "prompt": "p",
        "source": "x",
        "mlx": "y",
    }
    _write_result(tmp_path / "df11", key, 0)
    _write_result(tmp_path / "bf16", key, 0)
    assert vi.sides_ready(tmp_path, key) == []


def test_run_key_carries_every_setting_that_changes_the_latents(tmp_path, monkeypatch):
    # Bug caught: guidance or the eval policy missing from the key (a stale side reused after a change).
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    args = vi.parse_args(
        [
            "--side",
            "df11",
            "--df11",
            str(tmp_path),
            "--base",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--guidance",
            "2.0",
        ]
    )
    key = vi.run_key(args)
    assert set(key) >= {
        "model",
        "seed",
        "steps",
        "size",
        "prompt",
        "guidance",
        "eval_policy",
        "df11",
        "base",
        "source",
        "mlx",
    }
    assert key["guidance"] == 2.0
    assert key["source"] == "src"
