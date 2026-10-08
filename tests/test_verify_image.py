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


def _parse(tmp_path, *extra):
    return vi.parse_args(
        [
            "--side",
            "df11",
            "--df11",
            str(tmp_path),
            "--base",
            str(tmp_path),
            "--out",
            str(tmp_path),
            *extra,
        ]
    )


def test_compare_latents_views_an_odd_length_bf16_array_and_tells_signed_zeros_apart():
    # Bug caught: a uint32 view of an odd-length bf16 array raises, or float == equates -0.0 with 0.0.
    pos = mx.array([0.0, 1.0, 2.0], dtype=mx.bfloat16)
    neg = mx.array([-0.0, 1.0, 2.0], dtype=mx.bfloat16)
    assert vi.compare_latents(pos, mx.array([0.0, 1.0, 2.0], dtype=mx.bfloat16))
    assert not vi.compare_latents(pos, neg)
    assert not vi.compare_latents(pos, pos.astype(mx.float32))  # another dtype is never equal


def test_compare_latents_still_compares_float32_bits():
    # Bug caught: the dtype-sized view breaking the 4-byte (FLUX.1 latents, Z-Image latents that end float32) path.
    a = mx.array([0.0, 1.5, -2.0], dtype=mx.float32)
    assert vi.compare_latents(a, mx.array([0.0, 1.5, -2.0], dtype=mx.float32))
    assert not vi.compare_latents(a, mx.array([-0.0, 1.5, -2.0], dtype=mx.float32))


def test_guidance_resolves_per_model_before_the_key(tmp_path, monkeypatch):
    # Bug caught: FLUX's 3.5 forced on Z-Image (base would run CFG at 3.5), or FLUX losing it.
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    z = _parse(tmp_path, "--model", "z-image")
    assert z.guidance is None
    assert "guidance" in vi.run_key(z)
    assert vi.run_key(z)["guidance"] is None
    assert _parse(tmp_path, "--model", "z-image", "--guidance", "4").guidance == 4.0
    flux = _parse(tmp_path, "--model", "schnell")
    assert vi.run_key(flux)["guidance"] == 3.5


def test_the_flux_key_has_exactly_its_old_fields(tmp_path, monkeypatch):
    # Bug caught: every stored FLUX.1 side invalidated by a new key field (negative_prompt) or a changed default.
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    key = vi.run_key(_parse(tmp_path, "--model", "dev"))
    assert set(key) == {
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
    assert key["guidance"] == 3.5


def test_the_negative_prompt_is_in_the_key_when_given(tmp_path, monkeypatch):
    # Bug caught: a stored base-model side reused after the negative prompt changed.
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    key = vi.run_key(_parse(tmp_path, "--model", "z-image", "--negative-prompt", "blurry"))
    assert key["negative_prompt"] == "blurry"


def test_the_model_choices_are_the_registry_names_with_a_bf16_original(tmp_path):
    # Bug caught: --model still limited to the FLUX.1 names.
    for name in ("schnell", "dev", "krea-dev", "z-image"):
        assert _parse(tmp_path, "--model", name).model == name


def test_turbo_is_refused_at_parse_time_because_its_original_is_fp32(tmp_path, capsys):
    # Bug caught: --model z-image-turbo accepted, so the df11 side runs for minutes before the bf16 side finds no
    # BF16 transformer to stream (Turbo's original weights are FP32), or fails later with an unrelated error.
    with pytest.raises(SystemExit) as info:
        _parse(tmp_path, "--model", "z-image-turbo")
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "z-image-turbo" in err
    assert "FP32" in err


def test_child_command_carries_the_negative_prompt_and_omits_an_unset_guidance(tmp_path):
    # Bug caught: a negative prompt lost between the orchestrator and its children, or the literal "None"
    # passed to --guidance (argparse float error in every Z-Image child).
    z = _parse(tmp_path, "--model", "z-image", "--negative-prompt", "blurry")
    cmd = vi.child_command(z, "bf16")
    assert cmd[cmd.index("--negative-prompt") + 1] == "blurry"
    assert "--guidance" not in cmd
    plain = vi.child_command(_parse(tmp_path, "--model", "z-image"), "df11")
    assert "--negative-prompt" not in plain
    flux = vi.child_command(_parse(tmp_path, "--model", "schnell"), "df11")
    assert flux[flux.index("--guidance") + 1] == "3.5"


def test_each_family_has_its_own_pair_of_side_runners():
    # Bug caught: Z-Image run through the FLUX.1 sides (or the reverse), a wrong-model identity verdict.
    assert vi.runner_for("schnell", "df11") is vi.run_df11
    assert vi.runner_for("krea-dev", "bf16") is vi.run_bf16
    assert vi.runner_for("z-image", "df11") is vi.run_df11_zimage
    assert vi.runner_for("z-image-turbo", "bf16") is vi.run_bf16_zimage


KLEIN = ("flux2-klein-base-4b", "flux2-klein-4b", "flux2-klein-base-9b", "flux2-klein-9b")


def _klein_base(tmp_path, *, distilled):
    """A base snapshot's model_index.json as BFL publishes it: the distilled repos add ``"is_distilled": true``."""
    root = tmp_path / ("distilled" if distilled else "base")
    root.mkdir(exist_ok=True)
    index = {"_class_name": "Flux2KleinPipeline", "_diffusers_version": "0.37.0.dev0"}
    if distilled:
        index["is_distilled"] = True
    (root / "model_index.json").write_text(json.dumps(index))
    return root


def _parse_klein(tmp_path, model, base, *extra):
    return vi.parse_args(
        [
            "--side",
            "df11",
            "--model",
            model,
            "--df11",
            str(tmp_path),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


@pytest.mark.parametrize("model", KLEIN)
def test_runner_for_dispatches_klein_names_to_the_flux2_sides(model):
    # Bug caught: Klein routed to the FLUX.1 or Z-Image sides (a wrong-model identity verdict), or no flux2 row
    # (a KeyError after the df11 side already ran for minutes).
    assert vi.runner_for(model, "df11") is vi.run_df11_flux2
    assert vi.runner_for(model, "bf16") is vi.run_bf16_flux2


@pytest.mark.parametrize(
    ("model", "distilled"),
    [
        ("flux2-klein-base-4b", False),
        ("flux2-klein-4b", True),
        ("flux2-klein-base-9b", False),
        ("flux2-klein-9b", True),
    ],
)
def test_every_klein_model_has_a_bf16_side_with_its_own_variants_base(tmp_path, model, distilled):
    # Bug caught: a Klein model refused like Z-Image-Turbo although all four BF16 originals exist (the distilled
    # repos ship BF16 transformers of the base repos' exact sizes), or the guidance left None in the key instead of
    # mflux's 1.0 (flux2_generate.py:63-64), which the bf16 side's Config cannot take.
    args = _parse_klein(tmp_path, model, _klein_base(tmp_path, distilled=distilled))
    assert args.model == model
    assert args.guidance == 1.0
    command = vi.child_command(args, "bf16")
    assert command[command.index("--guidance") + 1] == "1.0"


@pytest.mark.parametrize(
    ("model", "distilled", "repo"),
    [
        ("flux2-klein-4b", False, "black-forest-labs/FLUX.2-klein-4B"),
        ("flux2-klein-9b", False, "black-forest-labs/FLUX.2-klein-9B"),
        ("flux2-klein-base-4b", True, "black-forest-labs/FLUX.2-klein-base-4B"),
        ("flux2-klein-base-9b", True, "black-forest-labs/FLUX.2-klein-base-9B"),
    ],
)
def test_a_klein_base_of_the_other_variant_is_refused_at_parse_time(
    tmp_path, capsys, model, distilled, repo
):
    # Bug caught: a distilled model checked against the base repo's transformer (the encoder and VAE are the same,
    # so `generate` accepts it, but the bf16 side would stream the other transformer): exit 1, a false mismatch
    # verdict after minutes of compute, instead of a usage error before any side runs.
    with pytest.raises(SystemExit) as info:
        _parse_klein(tmp_path, model, _klein_base(tmp_path, distilled=distilled))
    assert info.value.code == 2
    assert repo in capsys.readouterr().err


def test_a_klein_base_without_its_model_index_is_refused_and_flux_needs_none(tmp_path, capsys):
    # Bug caught: a Klein run on a directory whose variant cannot be told (the false-mismatch case above slips
    # through), or the check applied to FLUX.1 / Z-Image bases (which never needed the file).
    with pytest.raises(SystemExit) as info:
        _parse_klein(tmp_path, "flux2-klein-base-4b", tmp_path)
    assert info.value.code == 2
    err = capsys.readouterr().err
    # Bug caught (review 2026-10-08): the message naming only the exception class, not what to pass instead (the
    # repository's full snapshot: the bf16 side streams its transformer/ too).
    assert "missing model_index.json" in err
    assert "pass the full snapshot of black-forest-labs/FLUX.2-klein-base-4B" in err
    assert "transformer/" in err
    assert _parse(tmp_path, "--model", "schnell").model == "schnell"
    assert _parse(tmp_path, "--model", "z-image").model == "z-image"


def test_a_negative_prompt_is_refused_for_klein(tmp_path, capsys):
    # Bug caught: --negative-prompt accepted for Klein, stored in the run key and ignored by both sides (mflux's
    # Flux2Klein always encodes its own blank negative " ", flux2_klein.py:75-79): a key that lies about the run.
    with pytest.raises(SystemExit) as info:
        _parse_klein(
            tmp_path,
            "flux2-klein-base-4b",
            _klein_base(tmp_path, distilled=False),
            "--negative-prompt",
            "blurry",
        )
    assert info.value.code == 2
    assert "--negative-prompt" in capsys.readouterr().err


# --- Qwen-Image 2.1 -------------------------------------------------------------------------------------------

QWEN = "qwen-image-2.1"


def _qwen_base(
    tmp_path,
    shards=(
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ),
    *,
    present=None,
):
    """A base snapshot's transformer/ as diffusers publishes it: an index naming its shards, the shards present."""
    root = tmp_path / "qbase"
    transformer = root / "transformer"
    transformer.mkdir(parents=True, exist_ok=True)
    weight_map = {f"transformer_blocks.{i}.attn.to_q.weight": s for i, s in enumerate(shards)}
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )
    for s in shards if present is None else present:
        (transformer / s).write_bytes(b"\0")
    return root


def _parse_qwen(tmp_path, base, *extra, side=("--side", "df11")):
    return vi.parse_args(
        [
            *side,
            "--model",
            QWEN,
            "--df11",
            str(tmp_path),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


def test_runner_for_dispatches_qwen_to_its_sides():
    # Bug caught: Qwen-Image 2.1 routed to the FLUX.1 sides (a wrong-model identity verdict), or no qwen21 row (a
    # KeyError after the df11 side already ran for minutes).
    assert vi.runner_for(QWEN, "df11") is vi.run_df11_qwen21
    assert vi.runner_for(QWEN, "bf16") is vi.run_bf16_qwen21


def test_qwen_base_problem_is_none_for_an_index_with_its_shards(tmp_path):
    # Bug caught: a complete transformer/ refused (the identity check could never run).
    assert vi.qwen21_base_problem(QWEN, _qwen_base(tmp_path)) is None


@pytest.mark.parametrize("layout", ["no_dir", "empty_dir", "no_index"])
def test_qwen_base_problem_names_the_missing_transformer(tmp_path, layout):
    # Bug caught: the bf16 side launched on a base without its BF16 transformer (the encoder and VAE are enough for
    # generate, and the transformer is the last, 14 GB download), failing after the df11 side's minutes.
    base = tmp_path / "qbase"
    base.mkdir()
    if layout != "no_dir":
        (base / "transformer").mkdir()
    if layout == "no_index":
        (base / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(
            b"\0"
        )
    problem = vi.qwen21_base_problem(QWEN, base)
    assert problem is not None
    assert f"--base {base}: the bf16 side streams the BF16 transformer from transformer/" in problem
    assert "download Qwen/Qwen-Image-2.1's transformer/* first" in problem


def test_qwen_base_problem_names_a_shard_the_index_lists_but_the_snapshot_lacks(tmp_path):
    # Bug caught: only the index checked, so a download stopped between the two shards passes the check and the bf16
    # side fails at the first block of the second shard.
    second = "diffusion_pytorch_model-00002-of-00002.safetensors"
    base = _qwen_base(tmp_path, present=("diffusion_pytorch_model-00001-of-00002.safetensors",))
    problem = vi.qwen21_base_problem(QWEN, base)
    assert problem is not None
    assert second in problem


def test_qwen_base_problem_reports_an_unreadable_index_instead_of_raising(tmp_path):
    # Bug caught: a truncated index JSON raising a traceback out of parse_args instead of a usage message.
    base = _qwen_base(tmp_path)
    (base / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text("{")
    problem = vi.qwen21_base_problem(QWEN, base)
    assert problem is not None
    assert "diffusion_pytorch_model.safetensors.index.json" in problem


def test_qwen_base_problem_refuses_an_index_that_names_no_shards(tmp_path):
    # Bug caught: an empty weight map passing as "every named shard present" (nothing to stream; the bf16 side fails
    # at the first block).
    base = _qwen_base(tmp_path)
    (base / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}})
    )
    assert vi.qwen21_base_problem(QWEN, base) is not None


@pytest.mark.parametrize("shard", ["../outside.safetensors", "sub/inner.safetensors"])
def test_qwen_base_problem_refuses_a_shard_name_that_is_not_a_plain_file_name(tmp_path, shard):
    # Bug caught: an index naming a path outside transformer/ (or below it) accepted because the file happens to exist
    # there, so the bf16 side streams another file's tensors; the streaming reader's index check refuses the same names
    # (Path(name).name must equal name).
    base = _qwen_base(tmp_path)
    target = base / "transformer" / shard
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0")
    index = base / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a.weight": shard}}))
    problem = vi.qwen21_base_problem(QWEN, base)
    assert problem is not None
    assert f"not a plain file name: {shard!r}" in problem


def test_qwen_base_problem_is_none_for_the_other_families(tmp_path):
    # Bug caught: the transformer/ check applied to FLUX.1 or Z-Image bases (their own checks differ).
    assert vi.qwen21_base_problem("schnell", tmp_path) is None
    assert vi.qwen21_base_problem("z-image", tmp_path) is None


@pytest.mark.parametrize("side", [("--side", "bf16"), ("--orchestrate",)])
def test_a_qwen_run_that_streams_the_bf16_side_is_refused_without_its_transformer(
    tmp_path, capsys, side
):
    # Bug caught: the check not wired into parse_args for the runs that stream the BF16 transformer (the bf16 side,
    # and --orchestrate, whose bf16 child starts after the df11 side's minutes).
    base = tmp_path / "qbase"
    (base / "transformer").mkdir(parents=True)
    with pytest.raises(SystemExit) as info:
        _parse_qwen(tmp_path, base, side=side)
    assert info.value.code == 2
    assert "transformer/" in capsys.readouterr().err


@pytest.mark.parametrize("side", [("--side", "df11"), ("--side", "compare")])
def test_a_qwen_df11_side_or_compare_needs_no_bf16_transformer(tmp_path, side):
    # Bug caught: the df11 side refused before the BF16 transformer is downloaded (it needs only the encoder, the VAE
    # and the DF11 file; the identity's df11 side runs first, before the 14 GB transformer download).
    base = tmp_path / "qbase"
    base.mkdir()
    assert _parse_qwen(tmp_path, base, side=side).model == QWEN


def test_qwen_negative_prompt_is_accepted_and_keyed_with_mfluxs_guidance(tmp_path, monkeypatch):
    # Bug caught: --negative-prompt refused for Qwen as for FLUX.2 Klein (Qwen's CFG needs it), dropped from the key
    # (a stored side reused after it changed) or from the children; or the guidance left None instead of mflux's 1.0
    # (qwen21_generate.py:60), which the bf16 side's Config cannot take.
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    args = _parse_qwen(
        tmp_path, _qwen_base(tmp_path), "--negative-prompt", " ", side=("--orchestrate",)
    )
    assert vi.run_key(args)["negative_prompt"] == " "
    assert args.guidance == 1.0
    command = vi.child_command(args, "bf16")
    assert command[command.index("--negative-prompt") + 1] == " "
    assert command[command.index("--guidance") + 1] == "1.0"


def test_qwen_embeds_hold_the_positive_pair_and_the_negative_one_when_cfg_ran():
    # Bug caught: the negative pair saved under the positive names (the bf16 side would run CFG against the prompt
    # itself), a mask not saved (the bf16 side's attention would see padding), or a negative saved when CFG did not
    # run (the bf16 side would then run two calls per step while the df11 side ran one).
    pos = (mx.array([[[1.0]]]), mx.array([[1]], dtype=mx.int32))
    neg = (mx.array([[[2.0]]]), mx.array([[0]], dtype=mx.int32))
    cache = {"p": pos, " ": neg}
    one = vi.qwen21_embeds(cache, ("p",))
    assert sorted(one) == ["prompt_embeds", "prompt_mask"]
    assert one["prompt_embeds"] is pos[0]
    assert one["prompt_mask"] is pos[1]
    two = vi.qwen21_embeds(cache, ("p", " "))
    assert sorted(two) == [
        "negative_prompt_embeds",
        "negative_prompt_mask",
        "prompt_embeds",
        "prompt_mask",
    ]
    assert two["negative_prompt_embeds"] is neg[0]
    assert two["negative_prompt_mask"] is neg[1]
    assert two["prompt_embeds"] is pos[0]


# --- ERNIE-Image ----------------------------------------------------------------------------------------------

ERNIE = "ernie-image"
ERNIE_TURBO = "ernie-image-turbo"


def _parse_ernie(tmp_path, base, *extra, model=ERNIE, side=("--side", "df11")):
    return vi.parse_args(
        [
            *side,
            "--model",
            model,
            "--df11",
            str(tmp_path),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


@pytest.mark.parametrize("model", [ERNIE, ERNIE_TURBO])
def test_runner_for_dispatches_ernie_to_its_sides(model):
    # Bug caught: ERNIE-Image routed to the FLUX.1 sides (a wrong-model identity verdict), or no ernie row (a KeyError
    # after the df11 side already ran for minutes).
    assert vi.runner_for(model, "df11") is vi.run_df11_ernie
    assert vi.runner_for(model, "bf16") is vi.run_bf16_ernie


@pytest.mark.parametrize("layout", ["no_dir", "empty_dir", "no_index"])
def test_ernie_base_problem_names_the_missing_transformer(tmp_path, layout):
    # Bug caught: the identity launched on a base without its BF16 transformer, failing after the df11 side's minutes.
    base = tmp_path / "qbase"
    base.mkdir()
    if layout != "no_dir":
        (base / "transformer").mkdir()
    if layout == "no_index":
        (base / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(
            b"\0"
        )
    problem = vi.ernie_base_problem(ERNIE, base)
    assert problem is not None
    assert f"--base {base}: the bf16 side streams the BF16 transformer from transformer/" in problem
    assert "download baidu/ERNIE-Image's transformer/* first" in problem


def test_ernie_base_problem_is_none_for_an_index_with_its_shards(tmp_path):
    # Bug caught: a complete transformer/ refused (the identity check could never run).
    assert vi.ernie_base_problem(ERNIE, _qwen_base(tmp_path)) is None


@pytest.mark.parametrize("shard", ["../outside.safetensors", "sub/inner.safetensors"])
def test_ernie_base_problem_keeps_the_shard_name_check(tmp_path, shard):
    # Bug caught: the extraction of the shared check losing the plain-file-name guard (an index naming a path outside
    # transformer/ accepted, so the bf16 side streams another file's tensors).
    base = _qwen_base(tmp_path)
    target = base / "transformer" / shard
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0")
    index = base / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a.weight": shard}}))
    problem = vi.ernie_base_problem(ERNIE, base)
    assert problem is not None
    assert f"not a plain file name: {shard!r}" in problem


def test_ernie_base_problem_is_none_for_the_other_families(tmp_path):
    # Bug caught: the ERNIE check applied to other families' bases (Qwen keeps its own; FLUX.1 needs no transformer/).
    assert vi.ernie_base_problem("schnell", tmp_path) is None
    assert vi.ernie_base_problem("qwen-image-2.1", tmp_path) is None


@pytest.mark.parametrize("side", [("--side", "bf16"), ("--orchestrate",)])
def test_an_ernie_run_that_streams_the_bf16_side_is_refused_without_its_transformer(
    tmp_path, capsys, side
):
    # Bug caught: the check not wired into parse_args for the runs that stream the BF16 transformer.
    base = tmp_path / "qbase"
    (base / "transformer").mkdir(parents=True)
    with pytest.raises(SystemExit) as info:
        _parse_ernie(tmp_path, base, side=side)
    assert info.value.code == 2
    assert "transformer/" in capsys.readouterr().err


def test_ernie_turbo_negative_prompt_is_refused(tmp_path, capsys):
    # Bug caught: a negative prompt accepted for Turbo (it runs at guidance 1.0 only: no negative branch), so the key
    # would carry a setting that changes nothing, or the bf16 side would run CFG the df11 side never ran.
    with pytest.raises(SystemExit) as info:
        _parse_ernie(tmp_path, tmp_path, "--negative-prompt", "blurry", model=ERNIE_TURBO)
    assert info.value.code == 2
    assert "runs at guidance 1.0 only; there is no negative branch" in capsys.readouterr().err


def test_ernie_negative_prompt_is_accepted_and_keyed_with_mfluxs_guidance(tmp_path, monkeypatch):
    # Bug caught: --negative-prompt refused for the base (its CFG uses it), dropped from the key (a stored side reused
    # after it changed) or from the children; or the guidance left None instead of mflux's 4.0
    # (ernie_image_generate.py:22), which the bf16 side's Config would read as 0.0 (no CFG).
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    args = _parse_ernie(
        tmp_path, _qwen_base(tmp_path), "--negative-prompt", "blurry", side=("--orchestrate",)
    )
    assert vi.run_key(args)["negative_prompt"] == "blurry"
    assert args.guidance == 4.0
    command = vi.child_command(args, "bf16")
    assert command[command.index("--negative-prompt") + 1] == "blurry"
    assert command[command.index("--guidance") + 1] == "4.0"


@pytest.mark.parametrize("guidance", ["4", "1.5"])
def test_ernie_turbo_guidance_other_than_1_is_refused(tmp_path, capsys, guidance):
    # Bug caught: an identity run of Turbo at a guidance its command refuses (mflux's turbo command errors on any
    # but 1.0, ernie_image_turbo_generate.py:42-45), so the check would cover a path no user runs.
    with pytest.raises(SystemExit) as info:
        _parse_ernie(tmp_path, tmp_path, "--guidance", guidance, model=ERNIE_TURBO)
    assert info.value.code == 2
    assert "runs at guidance 1.0 only" in capsys.readouterr().err


def test_ernie_turbo_guidance_1_is_accepted(tmp_path):
    # Bug caught: the refusal firing on Turbo's own 1.0 (or on the default, which resolves to it).
    assert _parse_ernie(tmp_path, tmp_path, "--guidance", "1", model=ERNIE_TURBO).guidance == 1.0
    assert _parse_ernie(tmp_path, tmp_path, model=ERNIE_TURBO).guidance == 1.0


# --- Krea 2 ------------------------------------------------------------------------------------------------------

KREA_RAW, KREA_TURBO = "krea-2-raw", "krea-2"


def _krea_base(tmp_path, native="raw.safetensors"):
    """A base snapshot with the native single-file transformer at its root (a link into a blob, as hf lays it out)."""
    base = tmp_path / "kbase"
    (base / "blobs").mkdir(parents=True)
    blob = base / "blobs" / "f99bb0ff"
    blob.write_bytes(b"\0")
    (base / native).symlink_to(blob)
    return base


def _parse_krea(tmp_path, base, *extra, model=KREA_RAW, side=("--side", "df11")):
    return vi.parse_args(
        [
            *side,
            "--model",
            model,
            "--df11",
            str(tmp_path),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ]
    )


@pytest.mark.parametrize("model", [KREA_RAW, KREA_TURBO])
def test_runner_for_dispatches_krea2_to_its_sides(model):
    # Bug caught: Krea 2 routed to another family's sides (a wrong-model identity verdict), or no krea2 row (a
    # KeyError after the df11 side already ran for minutes).
    assert vi.runner_for(model, "df11") is vi.run_df11_krea2
    assert vi.runner_for(model, "bf16") is vi.run_bf16_krea2


@pytest.mark.parametrize(
    ("model", "native", "repo"),
    [
        (KREA_RAW, "raw.safetensors", "krea/Krea-2-Raw"),
        (KREA_TURBO, "turbo.safetensors", "krea/Krea-2-Turbo"),
    ],
)
@pytest.mark.parametrize("layout", ["missing", "dangling", "directory", "other_model"])
def test_krea2_base_problem_names_the_missing_native_file(tmp_path, model, native, repo, layout):
    # Bug caught: the identity launched on a base without the model's own native transformer file (the bf16 side would
    # fail after the df11 side's minutes), a dangling link or a directory taken for the file, or Turbo's file accepted
    # as Raw's reference (the two bases share their encoder and VAE, not the transformer).
    base = tmp_path / "kbase"
    base.mkdir()
    if layout == "dangling":
        (base / native).symlink_to(base / "gone")
    elif layout == "directory":
        (base / native).mkdir()
    elif layout == "other_model":
        other = "turbo.safetensors" if native == "raw.safetensors" else "raw.safetensors"
        (base / other).write_bytes(b"\0")
    problem = vi.krea2_base_problem(model, base)
    assert problem == (
        f"--base {base}: the bf16 side streams the transformer from {native} at the base's root (download "
        f"{repo}'s {native} first)"
    )


def test_krea2_base_problem_is_none_with_the_native_file_and_for_the_other_families(tmp_path):
    # Bug caught: a complete base refused (the identity check could never run), or the Krea check applied to another
    # family's base.
    assert vi.krea2_base_problem(KREA_RAW, _krea_base(tmp_path)) is None
    assert vi.krea2_base_problem("qwen-image-2.1", tmp_path) is None
    assert vi.krea2_base_problem("schnell", tmp_path) is None


@pytest.mark.parametrize("side", [("--side", "bf16"), ("--orchestrate",)])
def test_a_krea_run_that_streams_the_bf16_side_is_refused_without_its_native_file(
    tmp_path, capsys, side
):
    # Bug caught: the check not wired into parse_args for the runs that read the base's transformer.
    base = tmp_path / "kbase"
    base.mkdir()
    with pytest.raises(SystemExit) as info:
        _parse_krea(tmp_path, base, side=side)
    assert info.value.code == 2
    assert "raw.safetensors" in capsys.readouterr().err


def test_a_krea_df11_side_needs_no_native_file(tmp_path):
    # Bug caught: the df11 side refused for a base that holds only the encoder, the VAE and the tokenizer (all it
    # reads).
    base = tmp_path / "kbase"
    base.mkdir()
    assert _parse_krea(tmp_path, base).model == KREA_RAW


def test_krea_raw_cfg_run_is_keyed_with_its_guidance_and_negative_prompt(tmp_path, monkeypatch):
    # Bug caught: the identity's CFG run (guidance 3.5, two batch-1 calls per step, mflux's " " negative) keyed or
    # passed to the children without the guidance or the negative prompt, so a stored side of another run is reused;
    # or the default guidance left None instead of mflux's 1.0 for Raw.
    monkeypatch.setattr(vi, "source_hash", lambda: "src")
    base = _krea_base(tmp_path)
    args = _parse_krea(
        tmp_path, base, "--guidance", "3.5", "--negative-prompt", " ", side=("--orchestrate",)
    )
    key = vi.run_key(args)
    assert (key["guidance"], key["negative_prompt"]) == (3.5, " ")
    command = vi.child_command(args, "bf16")
    assert command[command.index("--guidance") + 1] == "3.5"
    assert command[command.index("--negative-prompt") + 1] == " "
    assert _parse_krea(tmp_path, base).guidance == 1.0


@pytest.mark.parametrize("guidance", ["3.5", "0.5"])
def test_krea_turbo_guidance_other_than_1_is_refused(tmp_path, capsys, guidance):
    # Bug caught: an identity run of Turbo at a guidance off its distilled recipe (CFG at any value but 1.0, 0.5
    # included), which the check does not cover.
    with pytest.raises(SystemExit) as info:
        _parse_krea(tmp_path, tmp_path, "--guidance", guidance, model=KREA_TURBO)
    assert info.value.code == 2
    assert "krea-2 runs at guidance 1.0 only in this check" in capsys.readouterr().err


def test_krea_turbo_negative_prompt_is_refused_and_guidance_1_accepted(tmp_path, capsys):
    # Bug caught: a negative prompt keyed for Turbo at 1.0 (no negative branch there), or the refusal firing on 1.0.
    assert _parse_krea(tmp_path, tmp_path, "--guidance", "1", model=KREA_TURBO).guidance == 1.0
    with pytest.raises(SystemExit) as info:
        _parse_krea(tmp_path, tmp_path, "--negative-prompt", " ", model=KREA_TURBO)
    assert info.value.code == 2
    assert "there is no negative branch" in capsys.readouterr().err


def _tiny_krea_native(path, ckpt, matrices, extras):
    """The tiny checkpoint's source as a base's native file, under the checkpoint's names: the block matrices BF16,
    the non-block matrices FP32 holding the same BF16 values (the published file stores five of them that way), the
    extras BF16."""
    import numpy as np

    from mlx_dfloat.mflux.krea2.names import NONBLOCK_GROUPS, krea2_name_map

    names = krea2_name_map()
    tensors = {}
    for group, per in matrices.items():
        for name in ckpt.groups[group].matrix_names:
            if group in NONBLOCK_GROUPS:
                bits = per[names.param_name(name)].astype(np.uint32) << np.uint32(16)
                tensors[name] = mx.array(bits).view(mx.float32)
            else:
                sub = name.removeprefix(f"{group}.").removesuffix(".weight")
                tensors[name] = mx.array(per[sub]).view(mx.bfloat16)
    for name in ckpt.extras:
        tensors[name] = mx.array(extras[names.param_name(name)]).view(mx.bfloat16)
    mx.save_safetensors(str(path), tensors)


@pytest.mark.mflux
@pytest.mark.parametrize(
    ("guidance", "negative"), [(1.0, None), (3.5, "n")], ids=["guidance-1", "cfg"]
)
def test_the_krea2_bf16_side_gives_stock_mfluxs_latents_on_a_tiny_checkpoint(
    tmp_path, monkeypatch, guidance, negative
):
    # Bug caught: the bf16 side's copy of mflux's loop body drifting from mflux's own (the CFG formula, the
    # stepper's seed, the sigma passed as the timestep, the embeddings swapped), a native tensor not replacing the DF11
    # value it stands for, or the streamed blocks read under other names. Stock Krea2 over the same BF16 weights, the
    # same embeddings and seed; equality is bit for bit on the float32 latents, before the heavy identity run.
    import numpy as np
    from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
    from mflux.utils.apple_silicon import AppleSiliconUtil
    from tests._krea2_tiny import (
        TINY,
        StubTextEncoder,
        StubTokenizer,
        StubVAE,
        stock_krea2,
        write_tiny_checkpoint,
    )

    from mlx_dfloat import _layouts
    from mlx_dfloat.format import open_checkpoint
    from mlx_dfloat.mflux.krea2 import init as kinit
    from mlx_dfloat.mflux.krea2 import transformer as ktf

    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", classmethod(lambda cls: True))
    rng = np.random.default_rng(11)
    df11 = tmp_path / "df11"
    matrices, extras, layout = write_tiny_checkpoint(df11, rng, random_extras=rng)
    monkeypatch.setattr(_layouts, "KNOWN_LAYOUTS", (layout,))
    ckpt = open_checkpoint(df11)
    base = tmp_path / "base"
    base.mkdir()
    _tiny_krea_native(base / "raw.safetensors", ckpt, matrices, extras)
    real_build = ktf.build_transformer
    monkeypatch.setattr(
        ktf, "build_transformer", lambda c, **kw: real_build(c, transformer_kwargs=TINY, **kw)
    )
    ours_vae, stock_vae = StubVAE(), StubVAE()
    monkeypatch.setattr(kinit, "load_vae", lambda root: ours_vae)
    tokenizer = StubTokenizer({"n": 12})
    embeds, neg = Krea2PromptEncoder.encode_prompt_pair(
        prompt="p",
        negative_prompt=negative,
        guidance=guidance,
        tokenizer=tokenizer,
        text_encoder=StubTextEncoder(),
        prompt_cache={},
    )
    out = tmp_path / "out"
    (out / "df11").mkdir(parents=True)
    saved = {"embeds": embeds} if neg is None else {"embeds": embeds, "neg_embeds": neg}
    mx.save_safetensors(str(out / "df11" / "embeds.safetensors"), saved)
    args = vi.argparse.Namespace(
        out=out,
        df11=str(df11),
        base=str(base),
        model=KREA_RAW,
        eval_policy="per-block",
        seed=3,
        steps=2,
        size=64,
        guidance=guidance,
    )
    result = vi.run_bf16_krea2(args, vi.argparse.Namespace(peak_footprint=0))
    ours = mx.load(str(out / "bf16" / "latents.safetensors"))["latents"]
    stock = stock_krea2(tokenizer, StubTextEncoder(), stock_vae, matrices, extras)
    stock.generate_image(
        seed=3,
        prompt="p",
        num_inference_steps=2,
        height=64,
        width=64,
        guidance=guidance,
        negative_prompt=negative,
    )
    (theirs,) = stock_vae.seen
    assert result["calls_per_step"] == (2 if negative else 1)
    assert result["replaced_from_base"] == len(ckpt.extras) + 37  # every extra and non-block matrix
    assert ours.dtype == theirs.dtype == mx.float32
    assert ours.shape == theirs.shape == (1, 16, 8, 8)
    assert bool(mx.all(mx.isfinite(theirs)))
    assert np.array_equal(np.array(ours.view(mx.uint32)), np.array(theirs.view(mx.uint32)))
    (decoded,) = ours_vae.seen  # the VAE decodes the side's own final latents
    assert np.array_equal(np.array(decoded.view(mx.uint32)), np.array(ours.view(mx.uint32)))
