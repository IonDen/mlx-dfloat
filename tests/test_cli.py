import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_dfloat import DFloatDependencyError, DFloatResourceError
from mlx_dfloat.cli import main
from mlx_dfloat.mflux.flux1 import cli as gen

REPO = Path(__file__).resolve().parents[1]


class _Image:
    def __init__(self, log):
        self.log = log

    def save(self, path, export_json_metadata=False):
        self.log.append(("save", str(path), export_json_metadata))
        Path(path).write_bytes(b"png")


class _Model:
    def __init__(self, log, **kwargs):
        self.log, self.kwargs = log, kwargs

    def generate_image(self, **kwargs):
        self.log.append(("generate", kwargs))
        return _Image(self.log)

    def report(self):
        return {"model": self.kwargs["model"], "decode_launches": 57}


class _Watchdog:
    def __init__(self, out_dir, *, ceiling, budget):
        self.out_dir, self.ceiling, self.budget, self.stopped = out_dir, ceiling, budget, False

    def start(self):
        return self

    def stop(self):
        self.stopped = True


def _run(argv, log, tmp_path, factory=None):
    watchdogs = []

    def watchdog_factory(out_dir, **kw):
        watchdogs.append(_Watchdog(out_dir, **kw))
        return watchdogs[-1]

    args = gen.build_parser().parse_args(argv)
    code = gen.run(
        args,
        model_factory=factory or (lambda **kw: _Model(log, **kw)),
        install_caps=lambda: (20, 22),
        watchdog_factory=watchdog_factory,
    )
    return code, watchdogs


@pytest.mark.parametrize(
    "flag",
    [
        ["--quantize", "8"],
        ["-q", "4"],
        ["--lora-paths", "a.safetensors"],
        ["--lora-scales", "1.0"],
        ["--image-path", "in.png"],
        ["--image-strength", "0.5"],
        ["--pid-decode"],
        ["--controlnet-image-path", "c.png"],
    ],
)
def test_refused_mflux_flags_exit_2_naming_the_flag_before_anything_loads(flag, capsys, tmp_path):
    # Bug caught: an unsupported flag silently ignored (the user thinks the LoRA applied), or
    # argparse rejecting it as unknown (no reason given), or the model built before the refusal.
    log = []
    code, watchdogs = _run(
        ["--prompt", "p", "--output", str(tmp_path / "o.png"), *flag],
        log,
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
    )
    assert code == 2
    assert watchdogs == []
    assert log == []
    assert flag[0] in capsys.readouterr().err


def test_generate_builds_the_model_from_the_flags_writes_the_image_and_the_report(tmp_path):
    # Bug caught: a flag not forwarded (e.g. --cache-limit dropped), the image saved elsewhere,
    # the watchdog not stopped, or the report missing the exit code and the model's report.
    log = []
    out = tmp_path / "new" / "dir" / "o.png"  # Review Focus 5: the directory does not exist yet
    code, watchdogs = _run(
        [
            "--model",
            "dev",
            "--prompt",
            "a lighthouse",
            "--seed",
            "7",
            "--steps",
            "3",
            "--height",
            "512",
            "--width",
            "768",
            "--guidance",
            "2.0",
            "--output",
            str(out),
            "--df11",
            "/d",
            "--base",
            "/b",
            "--eval-policy",
            "depth2",
            "--cache-limit",
            "2.5e9",
            "--no-fit-check",
            "--report",
            str(tmp_path / "r.json"),
            "--wall-budget",
            "60",
        ],
        log,
        tmp_path,
    )
    assert code == 0
    assert out.read_bytes() == b"png"
    assert log[0] == (
        "generate",
        {
            "seed": 7,
            "prompt": "a lighthouse",
            "num_inference_steps": 3,
            "height": 512,
            "width": 768,
            "guidance": 2.0,
            "scheduler": "linear",
        },
    )
    assert log[1] == ("save", str(out), False)
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["exit_code"] == 0
    assert report["output"] == str(out)
    assert report["decode_launches"] == 57
    assert report["memory_caps_gb"] == [20, 22]
    assert report["model_kwargs"]["cache_limit"] == 2_500_000_000
    assert report["model_kwargs"] == {
        "model": "dev",
        "df11_path": "/d",
        "base_path": "/b",
        "eval_policy": "depth2",
        "cache_limit": 2_500_000_000,
        "fit_check": False,
    }
    assert watchdogs[0].out_dir == out.parent
    assert watchdogs[0].budget == 60
    assert watchdogs[0].stopped


def test_defaults_follow_mflux_per_model(tmp_path):
    # Bug caught: dev running 4 steps (schnell's default) or guidance None reaching mflux.
    log = []
    _run(["--model", "dev", "--prompt", "p", "--output", str(tmp_path / "o.png")], log, tmp_path)
    assert log[0][1]["num_inference_steps"] == 25
    assert log[0][1]["guidance"] == 3.5
    log.clear()
    _run(["--prompt", "p", "--output", str(tmp_path / "o.png")], log, tmp_path)
    assert log[0][1]["num_inference_steps"] == 4


def test_a_package_error_exits_2_with_its_name_and_the_report_records_it(tmp_path, capsys):
    # Bug caught: a DFloatResourceError (does not fit) surfacing as a traceback and exit 1 (the
    # bit-mismatch code), or the report not written on failure.
    def failing(**kw):
        raise DFloatResourceError("predicted peak 25.1 GiB exceeds the budget")

    code, watchdogs = _run(
        [
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.png"),
            "--report",
            str(tmp_path / "r.json"),
        ],
        [],
        tmp_path,
        factory=failing,
    )
    assert code == 2
    assert "DFloatResourceError" in capsys.readouterr().err
    assert json.loads((tmp_path / "r.json").read_text())["exit_code"] == 2
    assert watchdogs[0].stopped


def test_missing_mflux_is_a_dependency_error_with_the_install_hint(tmp_path, capsys, monkeypatch):
    # Bug caught: a bare ModuleNotFoundError for mflux from the default factory.
    monkeypatch.setattr(
        gen,
        "_model_class",
        lambda: (_ for _ in ()).throw(DFloatDependencyError("install mlx-dfloat[mflux]")),
    )
    args = gen.build_parser().parse_args(["--prompt", "p", "--output", str(tmp_path / "o.png")])
    code = gen.run(
        args, install_caps=lambda: (0, 0), watchdog_factory=lambda d, **k: _Watchdog(d, **k)
    )
    assert code == 2
    assert "install mlx-dfloat[mflux]" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["0", "-5", "abc"])
def test_cache_limit_must_be_a_positive_byte_count(text):
    # Bug caught (Review Focus 4): a zero or negative cache limit reaching mx.set_cache_limit.
    # A non-positive value and a non-numeric string raise ValueError with different messages
    # (the explicit check vs. float()'s own conversion error), so this shared parametrized
    # case checks the common ValueError contract rather than one specific message.
    with pytest.raises(ValueError):  # noqa: PT011
        gen.cache_limit_bytes(text)
    assert gen.cache_limit_bytes("2.5e9") == 2_500_000_000
    assert gen.cache_limit_bytes("1400000000") == 1_400_000_000


def test_negative_prompt_is_accepted_with_a_warning(tmp_path, capsys):
    # Bug caught: refusing --negative-prompt (mflux accepts and ignores it; pasted commands must work).
    log = []
    code, _ = _run(
        ["--prompt", "p", "--negative-prompt", "blur", "--output", str(tmp_path / "o.png")],
        log,
        tmp_path,
    )
    assert code == 0
    assert "ignored" in capsys.readouterr().err


def test_top_level_parser_routes_generate_and_reports_the_version(capsys):
    # Bug caught: the subcommand not wired to run(), or --version missing.
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mlx-dfloat" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([])  # no subcommand
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--help"], ["generate", "--help"]])
def test_help_works_without_mflux(argv, tmp_path):
    # Bug caught: the CLI module importing mflux (or the model) at import time.
    shadow = tmp_path / "mflux"
    shadow.mkdir()
    (shadow / "__init__.py").write_text('raise ImportError("no mflux in this test")\n')
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "mlx_dfloat.cli", *argv],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage:" in result.stdout, result.stderr[-800:]


def test_the_console_script_is_declared():
    # Bug caught: the entry point missing from the wheel.
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert data["project"]["scripts"] == {"mlx-dfloat": "mlx_dfloat.cli:main"}
