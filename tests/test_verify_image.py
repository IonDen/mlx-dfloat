import json
import sys

import mlx.core as mx
import pytest
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


def _args(tmp_path, *extra):
    return vi.parse_args(
        [
            "--side",
            "compare",
            "--df11",
            str(tmp_path / "d"),
            "--base",
            str(tmp_path / "b"),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def _write_side(out, side, key, latents, *, exit_code=0, degenerate=()):
    side_dir = out / side
    side_dir.mkdir(parents=True, exist_ok=True)
    (side_dir / "result.json").write_text(
        json.dumps({"key": key, "exit_code": exit_code, "degenerate": list(degenerate)})
    )
    mx.save_safetensors(str(side_dir / "latents.safetensors"), {"latents": latents})
    (side_dir / "image.png").write_bytes(side.encode())  # bytes differ; the pixels are faked equal


def _compare_setup(tmp_path, monkeypatch, bf16_latents):
    monkeypatch.setattr(vi, "source_hash", lambda: "src-a")
    monkeypatch.setattr(vi, "pixels_identical", lambda a, b: True)
    args = _args(tmp_path)
    key = vi.run_key(args)
    latents = mx.array([0.0, 1.5, -2.0], dtype=mx.float32)
    _write_side(args.out, "df11", key, latents)
    _write_side(args.out, "bf16", key, bf16_latents)
    return args


def test_run_compare_is_0_for_bit_identical_latents_and_records_the_pixel_verdict(
    tmp_path, monkeypatch
):
    # Bug caught: the compare reading the wrong files (or the keys) and never reaching 0 on a
    # matching pair, or the pixel verdict missing beside the two informational hashes.
    args = _compare_setup(tmp_path, monkeypatch, mx.array([0.0, 1.5, -2.0], dtype=mx.float32))
    summary = vi.run_compare(args)
    assert summary["exit_code"] == 0
    assert summary["equal"] is True
    assert summary["pixels_identical"] is True
    assert summary["df11_png_sha256"] != summary["bf16_png_sha256"]


def test_run_compare_is_1_for_latents_that_differ_in_one_bit(tmp_path, monkeypatch):
    # Bug caught: a float `==` that equates -0.0 and +0.0, or the verdict ignoring the comparison.
    args = _compare_setup(tmp_path, monkeypatch, mx.array([-0.0, 1.5, -2.0], dtype=mx.float32))
    assert vi.run_compare(args)["exit_code"] == 1


def test_run_compare_refuses_with_2_when_this_run_has_another_key(tmp_path, monkeypatch):
    # Bug caught: comparing two sides produced by an older source tree as if they were this run's.
    args = _compare_setup(tmp_path, monkeypatch, mx.array([0.0, 1.5, -2.0], dtype=mx.float32))
    monkeypatch.setattr(vi, "source_hash", lambda: "src-b")
    summary = vi.run_compare(args)
    assert summary["exit_code"] == 2
    assert "source" in summary["error"]


def test_a_stale_result_is_moved_aside_and_a_matching_one_is_kept(tmp_path):
    # Bug caught: a result.json from other settings left in place (resumed as this run's), or a
    # matching one moved away (a finished side rerun for nothing).
    key = {"model": "schnell", "seed": 42}
    stale, fresh = tmp_path / "stale", tmp_path / "fresh"
    _write_result(stale, {**key, "seed": 1}, 0)
    _write_result(fresh, key, 0)
    vi._move_stale_result_aside(stale, key)
    vi._move_stale_result_aside(fresh, key)
    assert not (stale / "result.json").exists()
    assert json.loads((stale / "result.previous.json").read_text())["key"]["seed"] == 1
    assert (fresh / "result.json").exists()
    assert not (fresh / "result.previous.json").exists()


def test_orchestrate_stops_with_2_when_a_side_fails_and_never_compares(tmp_path, monkeypatch):
    # Bug caught: a failed child ignored, so the compare runs on stale or missing outputs.
    import subprocess

    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    commands = []

    def failing_run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(vi.subprocess, "run", failing_run)
    monkeypatch.setattr(
        vi, "_write_compare", lambda args: (_ for _ in ()).throw(AssertionError("compared"))
    )
    args = vi.parse_args(
        [
            "--orchestrate",
            "--df11",
            str(tmp_path / "d"),
            "--base",
            str(tmp_path / "b"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert vi.orchestrate(args) == 2
    assert len(commands) == 1
    assert commands[0][commands[0].index("--side") + 1] == "df11"


def test_parse_args_makes_the_paths_absolute_for_the_children(tmp_path, monkeypatch):
    # Bug caught: a relative --df11/--base/--out handed to a child that runs with the repository
    # root as its cwd (it then reads another directory, or none).
    monkeypatch.chdir(tmp_path)
    args = vi.parse_args(["--orchestrate", "--df11", "d", "--base", "~/b", "--out", "o"])
    assert args.df11 == str(tmp_path.resolve() / "d")
    assert args.base == str((vi.Path.home() / "b").resolve())
    assert args.out == tmp_path.resolve() / "o"
    cmd = vi.child_command(args, "df11")
    assert cmd[cmd.index("--df11") + 1] == str(tmp_path.resolve() / "d")


def test_the_df11_peak_is_the_largest_per_phase_peak():
    # Bug caught: reading the process counter at the end (reset at every phase boundary, so it
    # holds only the VAE phase's peak) instead of the largest phase peak.
    peaks = {
        "label": "sampled at phase boundaries",
        "encode": {"mlx_peak": 10},
        "denoise": {"mlx_peak": 30},
        "vae": {"mlx_peak": 20},
    }
    assert vi.max_phase_peak(peaks) == 30
    assert vi.max_phase_peak({"label": "x"}) == 0


@pytest.mark.mflux  # Pillow comes with mflux
def test_pixels_identical_decodes_the_images_and_ignores_embedded_metadata(tmp_path):
    # Bug caught: comparing file bytes (a PNG with mflux's metadata chunk differs from the same
    # pixels without it), or declaring different pixels identical.
    from PIL import Image, PngImagePlugin

    pixels = Image.new("RGB", (4, 4), (10, 20, 30))
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", "a lighthouse")
    pixels.save(tmp_path / "a.png", pnginfo=info)
    pixels.save(tmp_path / "b.png")
    Image.new("RGB", (4, 4), (10, 20, 31)).save(tmp_path / "c.png")
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()
    assert vi.pixels_identical(tmp_path / "a.png", tmp_path / "b.png")
    assert not vi.pixels_identical(tmp_path / "a.png", tmp_path / "c.png")
