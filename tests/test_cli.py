import builtins
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_dfloat import DFloatDependencyError, DFloatResourceError, DFloatUnsupportedError
from mlx_dfloat.cli import main
from mlx_dfloat.mflux import generate as gen

REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3


class _Image:
    def __init__(self, log, *, writes=True):
        self.log, self.writes = log, writes

    def save(self, path, export_json_metadata=False, overwrite=False):
        self.log.append(("save", str(path), export_json_metadata, overwrite))
        if self.writes:
            Path(path).write_bytes(b"png")


class _Model:
    def __init__(self, log, *, image_writes=True, **kwargs):
        self.log, self.kwargs, self.image_writes = log, kwargs, image_writes

    def generate_image(self, **kwargs):
        self.log.append(("generate", kwargs))
        return _Image(self.log, writes=self.image_writes)

    def report(self):
        return {"model": self.kwargs["model"], "decode_launches": 57}


class _Watchdog:
    def __init__(self, out_dir, *, ceiling, budget, context=None, live_context=None):
        self.out_dir, self.ceiling, self.budget, self.stopped = out_dir, ceiling, budget, False
        self.context = context
        self.live_context = live_context
        # Far above any real footprint, and distinct from one another: the report must carry each
        # peak from its own counter, not the final sample and not another counter's value.
        self.peak_footprint = 10**15
        self.peak_watched = 2 * 10**15
        self.peak_mlx = 3 * 10**14

    def start(self):
        return self

    def stop(self):
        self.stopped = True


def _resolve(path):
    """Like mflux's ImageUtil.resolve_output_path: an existing name gets a suffix."""
    path = Path(path)
    return path.with_name(f"{path.stem}-1{path.suffix}") if path.exists() else path


# The 32 GB M1 Max: 32 GiB of RAM, a 24.96 GiB recommended working set (budget_bytes()'s 22.96 GiB
# plus its 2 GiB reserve), and default_ceiling() = RAM - 4 GiB.
HOST_RAM = 32 * GIB
HOST_RECOMMENDED = 26_800_603_136
HOST_CEILING = 28 * GIB
EFFECTIVE = {
    "memory": 111,
    "cache": 222,
    "wired": 333,
}  # what the fake limit reader "finds in force"


def _host_kwargs(**overrides):
    """The injectables that pin the host and keep MLX's real limits untouched."""
    kwargs = {
        "host_facts": lambda: (HOST_RAM, HOST_RECOMMENDED),
        "ceiling_default": lambda: HOST_CEILING,
        "apply_limits": lambda limits: pytest.fail("the tier limits applied on the host path"),
        "read_limits": lambda: dict(EFFECTIVE),
    }
    kwargs.update(overrides)
    return kwargs


def _run(argv, log, tmp_path, factory=None, install_caps=lambda: (20, 22), **run_kwargs):
    watchdogs = []

    def watchdog_factory(out_dir, **kw):
        watchdogs.append(_Watchdog(out_dir, **kw))
        return watchdogs[-1]

    args = gen.build_parser().parse_args(argv)
    code = gen.run(
        args,
        model_factory=factory or (lambda **kw: _Model(log, **kw)),
        install_caps=install_caps,
        watchdog_factory=watchdog_factory,
        resolve_output=_resolve,
        **_host_kwargs(**run_kwargs),
    )
    return code, watchdogs


def _limits_run(extra, tmp_path):
    """A run with ``extra`` flags whose installers record into ``calls``; returns the report too."""
    calls = []
    log = []
    report_path = tmp_path / "r.json"

    def install_caps():
        calls.append("install_caps")
        return (20, 22)

    def apply_limits(limits):
        calls.append(("apply", limits))
        return {"memory": 0, "cache": 0, "wired": 0}

    code, watchdogs = _run(
        [
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.png"),
            "--report",
            str(report_path),
            *extra,
        ],
        log,
        tmp_path,
        install_caps=install_caps,
        apply_limits=apply_limits,
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return code, watchdogs, calls, report


@pytest.mark.parametrize("flag", [[f, "x"] for f in gen.REFUSED] + [["-q", "4"]])
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
    assert log[1] == ("save", str(out), False, True)
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
    # Bug caught: the abort artifact not naming the run it stopped (the harness proof reads the
    # aborted run's size from it) or a context with the flag's None for steps.
    assert watchdogs[0].context == {
        "model": "dev",
        "height": 512,
        "width": 768,
        "seed": 7,
        "steps": 3,
        "tier_gb": 32,
        "label": "MEASURED",
        "limits": {"memory": 111, "cache": 222, "wired": 333},  # what read_limits found
    }
    assert report["footprint_peak_bytes"] == 10**15
    assert (
        report["footprint_peak_label"] == "OS phys_footprint, sampled every 0.05 s by the watchdog"
    )


def test_the_watchdog_context_carries_the_models_default_steps(tmp_path):
    # Bug caught: the context recording the --steps flag (None) instead of the steps that ran.
    _, watchdogs = _run(["--prompt", "p", "--output", str(tmp_path / "o.png")], [], tmp_path)
    assert watchdogs[0].context == {
        "model": "schnell",
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "steps": 4,
        "tier_gb": 32,
        "label": "MEASURED",
        "limits": {"memory": 111, "cache": 222, "wired": 333},  # what read_limits found
    }


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


def _hide_mflux(monkeypatch):
    """No mflux, whatever this venv holds: both the import and the ``find_spec`` probe miss it."""
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def is_mflux(name):
        return name == "mflux" or name.startswith("mflux.")

    def no_mflux(name, *args, **kwargs):
        if is_mflux(name):
            raise ImportError("no mflux here")
        return real_import(name, *args, **kwargs)

    def find_spec(name, *args, **kwargs):
        return None if is_mflux(name) else real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mflux)
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_missing_mflux_is_a_dependency_error_with_the_install_hint(tmp_path, capsys, monkeypatch):
    # Bug caught: the default model lookup (or the output-name lookup) importing mflux in a way
    # that surfaces a bare ModuleNotFoundError instead of the install hint, and exit 1.
    _hide_mflux(monkeypatch)
    monkeypatch.delitem(sys.modules, "mlx_dfloat.mflux.flux1.model", raising=False)
    args = gen.build_parser().parse_args(["--prompt", "p", "--output", str(tmp_path / "o.png")])
    code = gen.run(
        args,
        install_caps=lambda: (0, 0),
        watchdog_factory=lambda d, **k: _Watchdog(d, **k),
        **_host_kwargs(),
    )
    assert code == 2
    assert "install mlx-dfloat[mflux]" in capsys.readouterr().err


def test_an_existing_output_is_kept_and_the_report_names_the_file_actually_written(tmp_path):
    # Bug caught: the report (and the `ok:` line) naming the requested file while mflux wrote a
    # suffixed one next to it, so a script reads the old image as the new one.
    log = []
    out = tmp_path / "o.png"
    out.write_bytes(b"old")
    code, _ = _run(
        ["--prompt", "p", "--output", str(out), "--report", str(tmp_path / "r.json")],
        log,
        tmp_path,
    )
    assert code == 0
    assert out.read_bytes() == b"old"
    written = tmp_path / "o-1.png"
    assert written.read_bytes() == b"png"
    assert log[1] == ("save", str(written), False, True)
    assert json.loads((tmp_path / "r.json").read_text())["output"] == str(written)


def test_an_image_that_was_not_written_is_an_error_not_a_success(tmp_path, capsys):
    # Bug caught: trusting mflux's save (it logs and swallows a write failure), so the command
    # prints `ok:` and exits 0 with no image on disk.
    log = []
    code, _ = _run(
        [
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.png"),
            "--report",
            str(tmp_path / "r.json"),
        ],
        log,
        tmp_path,
        factory=lambda **kw: _Model(log, image_writes=False, **kw),
    )
    assert code == 2
    assert "not written" in capsys.readouterr().err
    assert json.loads((tmp_path / "r.json").read_text())["exit_code"] == 2


def test_an_output_directory_that_cannot_be_created_exits_2_with_a_report(tmp_path, capsys):
    # Bug caught: `mkdir` for the output's directory running outside the guarded block, so an
    # OSError escapes as a traceback (exit 1) and no report is written.
    (tmp_path / "file").write_bytes(b"")
    code, watchdogs = _run(
        [
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "file" / "o.png"),
            "--report",
            str(tmp_path / "r.json"),
        ],
        [],
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
    )
    assert code == 2
    assert watchdogs == []
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["exit_code"] == 2
    assert "Error" in report["error"]


def test_a_cap_install_failure_exits_2_with_a_report(tmp_path):
    # Bug caught: `install_caps()` running outside the guarded block (an OSError from the device
    # query escaping as exit 1 with no report).
    def failing_caps():
        raise OSError("no device")

    code, _ = _run(
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
        factory=lambda **kw: pytest.fail("built"),
        install_caps=failing_caps,
    )
    assert code == 2
    assert "no device" in json.loads((tmp_path / "r.json").read_text())["error"]


def test_a_report_that_cannot_be_written_exits_2_on_stderr(tmp_path, capsys):
    # Bug caught: an OSError from the report write escaping `_finish` as a traceback (exit 1), or
    # a run whose report was lost still exiting 0.
    (tmp_path / "file").write_bytes(b"")
    code, _ = _run(
        [
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "o.png"),
            "--report",
            str(tmp_path / "file" / "r.json"),
        ],
        [],
        tmp_path,
    )
    assert code == 2
    assert "report" in capsys.readouterr().err


def test_a_watchdog_construction_failure_still_writes_the_report(tmp_path, capsys):
    # Bug caught: the report skipped on the watchdog-failure path (an early return before the
    # write), and a non-DFloatError from watchdog construction escaping uncaught (exit 1 instead
    # of the tool-error exit 2).
    def failing_watchdog_factory(out_dir, **kw):
        raise DFloatDependencyError("psutil missing")

    args = gen.build_parser().parse_args(
        ["--prompt", "p", "--output", str(tmp_path / "o.png"), "--report", str(tmp_path / "r.json")]
    )
    code = gen.run(
        args,
        model_factory=lambda **kw: pytest.fail("built"),
        install_caps=lambda: (20, 22),
        watchdog_factory=failing_watchdog_factory,
        **_host_kwargs(),
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "DFloatDependencyError" in err
    assert "psutil missing" in err
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["exit_code"] == 2
    assert "DFloatDependencyError" in report["error"]


@pytest.mark.parametrize("text", ["0", "-5", "abc", "inf", "-inf", "nan"])
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


def test_top_level_parser_routes_generate_and_reports_the_version(capsys, monkeypatch, tmp_path):
    # Bug caught: the subcommand not wired to run() (main() never actually reaching args.run),
    # or --version missing.
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mlx-dfloat" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([])  # no subcommand
    assert exc.value.code == 2

    seen = {}

    def recorder(args):
        seen["prompt"] = args.prompt
        return 7

    # Patched before main() runs: build_parser() (called inside main()) binds `run` from this
    # module's globals at that later call time, so the subparser picks up the recorder.
    monkeypatch.setattr(gen, "run", recorder)
    code = main(["generate", "--prompt", "p", "--output", str(tmp_path / "o.png")])
    assert code == 7
    assert seen["prompt"] == "p"


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


# --- --tier and --memory-ceiling ---------------------------------------------------------------
# Worked by hand from the north star §4.3 (not from the code): a 24 GB tier's recommended working
# set is 2/3 of 24 GiB = 16 GiB exactly; its reserve is 1.5 GiB; its ceiling 16 - 1.5 = 14.5 GiB.
TIER_24_CEILING = 15_569_256_448  # int(14.5 * GIB)
# The fit budget a real 24 GB Mac's generate applies, budget_bytes(): 17_179_869_184 - 2 GiB.
TIER_24_FIT_BUDGET = 15_032_385_536
# A 16 GB tier: 16 GiB * 2 // 3 = 11_453_246_122 bytes, minus 1.5 GiB (1_610_612_736).
TIER_16_CEILING = 9_842_633_386


def _args(*extra):
    return gen.build_parser().parse_args(["--prompt", "p", *extra])


def _ceiling_for(*extra):
    return gen.ceiling_for(
        _args(*extra),
        host_ram_bytes=HOST_RAM,
        host_recommended_bytes=HOST_RECOMMENDED,
        default_ceiling_bytes=HOST_CEILING,
    )


@pytest.mark.parametrize(
    ("extra", "ceiling", "tier_gb", "is_host", "label"),
    [
        # Bug caught: a plain run getting a tier ceiling instead of the host's default.
        ((), HOST_CEILING, 32, True, "MEASURED"),
        # Bug caught: the proof ceiling ignored, or the proof labelled MEASURED (it is not a tier row).
        (("--memory-ceiling", "12345"), 12345, 32, True, "PROOF"),
        # Bug caught: the tier's budget used without the reserve (16 GiB instead of 14.5).
        (("--tier", "24"), TIER_24_CEILING, 24, False, "CAPPED"),
        # Bug caught: the 2/3 ratio applied as 3/4 at 16 GB (a 12 GiB budget).
        (("--tier", "16"), TIER_16_CEILING, 16, False, "CAPPED"),
        # Bug caught (Review Focus 2): the host's own tier treated as a cap (the ratio's 24 GiB
        # budget, 22 GiB ceiling) instead of keeping the host's default ceiling.
        (("--tier", "32"), HOST_CEILING, 32, True, "MEASURED"),
    ],
)
def test_ceiling_for_resolves_the_ceiling_the_limits_and_the_label(
    extra, ceiling, tier_gb, is_host, label
):
    got_ceiling, limits, got_label = _ceiling_for(*extra)
    assert (got_ceiling, limits.tier_gb, limits.is_host, got_label) == (
        ceiling,
        tier_gb,
        is_host,
        label,
    )


def test_ceiling_for_refuses_both_flags_and_a_tier_above_the_host():
    # Bug caught (Review Focus 6): --tier and --memory-ceiling together silently picking one of them,
    # or a 48 GB "tier" on a 32 GB host setting limits above its RAM and looking like a pass.
    with pytest.raises(ValueError, match="--tier") as info:
        _ceiling_for("--tier", "24", "--memory-ceiling", "12345")
    assert "--memory-ceiling" in str(info.value)
    with pytest.raises(DFloatUnsupportedError, match="48 GB"):
        _ceiling_for("--tier", "48")


@pytest.mark.parametrize("extra", [["--tier", "0"], ["--tier", "-8"], ["--memory-ceiling", "0"]])
def test_a_non_positive_tier_or_ceiling_is_a_usage_error(extra):
    # Bug caught: a 0 GB tier (a negative ceiling: the watchdog aborts at once) or a zero ceiling
    # reaching the watchdog instead of being refused by the parser.
    with pytest.raises(SystemExit) as exc:
        _args(*extra)
    assert exc.value.code == 2


def test_tier_24_applies_the_tier_limits_not_the_host_caps_and_budgets_the_fit_check(tmp_path):
    # Bug caught: the host caps installed under a tier (the row is not the tier's), the watchdog at
    # the host ceiling, or the fit check still on the host budget (a call refused, or planned, for
    # 32 GB while the row claims 24); or the fit check handed the watchdog ceiling (1.5 GiB
    # reserve), a looser budget than the 2 GiB-reserve one a real 24 GB Mac's generate applies.
    code, watchdogs, calls, report = _limits_run(["--tier", "24"], tmp_path)
    assert code == 0
    assert "install_caps" not in calls
    applied = [c[1] for c in calls if c[0] == "apply"]
    assert [lim.tier_gb for lim in applied] == [24]
    assert watchdogs[0].ceiling == TIER_24_CEILING
    assert report["model_kwargs"]["budget_bytes"] == TIER_24_FIT_BUDGET
    assert report["label"] == "CAPPED"
    assert report["tier_gb"] == 24
    assert report["watchdog_ceiling_bytes"] == TIER_24_CEILING
    assert report["memory_caps_gb"] is None
    assert report["limits"]["tier"]["tier_gb"] == 24
    assert report["limits"]["tier"]["ceiling_bytes"] == TIER_24_CEILING
    assert report["limits"]["applied"] == "tier-caps"
    assert report["limits"]["effective"] == {
        "memory_limit_bytes": 111,
        "cache_limit_bytes": 222,
        "wired_limit_bytes": 333,
    }


def test_tier_32_on_a_32_gb_host_keeps_the_host_caps_and_is_measured(tmp_path):
    # Bug caught (Review Focus 2): `--tier <host>` installing a smaller tier's caps instead of the
    # host caps, so MEASURED rows run under other limits than a plain generate.
    code, watchdogs, calls, report = _limits_run(["--tier", "32"], tmp_path)
    assert code == 0
    assert calls == ["install_caps"]  # apply_limits would have failed the test
    assert watchdogs[0].ceiling == HOST_CEILING
    assert "budget_bytes" not in report["model_kwargs"]
    assert report["label"] == "MEASURED"
    assert report["memory_caps_gb"] == [20, 22]
    assert report["limits"]["applied"] == "host-caps"
    assert report["limits"]["tier"]["tier_gb"] == 32


def test_memory_ceiling_sets_the_watchdog_alone_under_the_host_caps(tmp_path):
    # Bug caught: the proof ceiling not reaching the watchdog (the abort case would run to the host
    # ceiling), a tier's limits installed for it, or the proof labelled as a tier row.
    code, watchdogs, calls, report = _limits_run(["--memory-ceiling", "12345"], tmp_path)
    assert code == 0
    assert calls == ["install_caps"]
    assert watchdogs[0].ceiling == 12345
    assert "budget_bytes" not in report["model_kwargs"]
    assert report["label"] == "PROOF"
    assert report["memory_ceiling_bytes"] == 12345
    assert report["watchdog_ceiling_bytes"] == 12345
    assert report["limits"]["applied"] == "host-caps"


def test_a_plain_run_records_the_host_tiers_limits(tmp_path):
    # Bug caught (spec amendment 7): the limits record written only under --tier, so a plain
    # generate report cannot show which limits it ran under.
    code, watchdogs, calls, report = _limits_run([], tmp_path)
    assert code == 0
    assert calls == ["install_caps"]
    assert watchdogs[0].ceiling == HOST_CEILING
    assert report["label"] == "MEASURED"
    assert report["tier_gb"] == 32
    assert report["memory_ceiling_bytes"] is None
    assert report["limits"]["tier"]["tier_gb"] == 32
    assert report["limits"]["applied"] == "host-caps"


def test_both_flags_exit_2_naming_both_before_anything_is_installed(tmp_path, capsys):
    # Bug caught (Review Focus 6): argparse or run() silently preferring one flag, or the caps
    # installed / the watchdog started before the refusal.
    code, watchdogs, calls, report = _limits_run(
        ["--tier", "24", "--memory-ceiling", "12345"], tmp_path
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "--tier" in err
    assert "--memory-ceiling" in err
    assert calls == []
    assert watchdogs == []
    assert report["exit_code"] == 2


def test_a_tier_above_the_host_exits_2_before_anything_is_installed(tmp_path, capsys):
    # Bug caught: a 48 GB tier on a 32 GB host installing limits above RAM (Review Focus 2), or the
    # refusal surfacing as a traceback.
    code, watchdogs, calls, report = _limits_run(["--tier", "48"], tmp_path)
    assert code == 2
    assert "DFloatUnsupportedError" in capsys.readouterr().err
    assert calls == []
    assert watchdogs == []
    assert report["exit_code"] == 2


def test_the_report_carries_the_watched_and_mlx_peaks_and_the_size(tmp_path):
    # Bug caught: watched_peak_bytes read from the footprint counter (the tier rows and the proof
    # compare against the enforced number, max(footprint, active + cache)), mlx_peak_bytes missing,
    # or the size not recorded (the proof paragraph names the pass size).
    code, _, _, report = _limits_run(["--height", "512", "--width", "768"], tmp_path)
    assert code == 0
    assert report["watched_peak_bytes"] == 2 * 10**15
    assert report["mlx_peak_bytes"] == 3 * 10**14
    assert report["footprint_peak_bytes"] == 10**15
    assert (report["height"], report["width"]) == (512, 768)


def test_the_watched_peak_includes_the_final_footprint_sample(tmp_path):
    # Bug caught: the watched peak taken from the sampler alone, missing a peak after its last poll
    # (the footprint peak already takes that final sample; both must use the same one).
    class Idle(_Watchdog):
        def __init__(self, out_dir, **kw):
            super().__init__(out_dir, **kw)
            self.peak_footprint = self.peak_watched = 0
            self.peak_mlx = None  # what a watchdog that never sampled holds

    args = _args("--output", str(tmp_path / "o.png"), "--report", str(tmp_path / "r.json"))
    code = gen.run(
        args,
        model_factory=lambda **kw: _Model([], **kw),
        install_caps=lambda: (20, 22),
        watchdog_factory=Idle,
        resolve_output=_resolve,
        **_host_kwargs(),
    )
    assert code == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["watched_peak_bytes"] > 0
    assert report["watched_peak_bytes"] == report["footprint_peak_bytes"]
    # Bug caught: an unsampled MLX peak reported as 0 bytes instead of "not sampled".
    assert report["mlx_peak_bytes"] is None


def test_host_facts_reads_ram_and_the_recommended_working_set_from_the_device(monkeypatch):
    # Bug caught: a misspelt device-info key read as 0 (every --tier then refused as above a
    # "0 GB host"), or RAM and the working set swapped (a 25 GB host tier on a 32 GB Mac).
    monkeypatch.setattr(
        gen.mx,
        "device_info",
        lambda: {"memory_size": HOST_RAM, "max_recommended_working_set_size": HOST_RECOMMENDED},
    )
    assert gen._host_facts() == (HOST_RAM, HOST_RECOMMENDED)


def test_ceiling_for_refuses_a_proof_ceiling_above_the_host_ceiling():
    # Bug caught: a typo such as 1.8e11 for 1.8e10 accepted as the watchdog ceiling, above physical
    # RAM, so the run has no working memory guard (the paging-storm panic the watchdog exists for).
    with pytest.raises(ValueError, match="--memory-ceiling") as info:
        _ceiling_for("--memory-ceiling", str(HOST_CEILING + 1))
    assert "28.0 GiB" in str(info.value)  # the host ceiling, 32 GiB - 4 GiB
    # The host ceiling itself is the largest allowed value (only a raise above it is refused).
    assert _ceiling_for("--memory-ceiling", str(HOST_CEILING))[0] == HOST_CEILING


def test_a_proof_ceiling_above_the_host_ceiling_installs_nothing_and_starts_no_watchdog(
    tmp_path, capsys
):
    # Bug caught: the refusal raised after the caps are installed or the watchdog started, or
    # surfacing as a traceback instead of exit 2 naming the flag.
    code, watchdogs, calls, report = _limits_run(["--memory-ceiling", "1.8e11"], tmp_path)
    assert code == 2
    assert "--memory-ceiling" in capsys.readouterr().err
    assert calls == []
    assert watchdogs == []
    assert report["exit_code"] == 2


def test_finish_writes_the_report_with_the_home_directory_as_a_tilde(tmp_path):
    # Bug caught: the report file carrying absolute home paths (snapshot roots, paths inside an
    # error string) into a result JSON that gets committed to a public repo.
    home = str(Path.home())
    report = {
        "exit_code": 2,
        "output": "x.png",
        "df11": {"root": f"{home}/.cache/huggingface/hub/models--a/snapshots/s"},
        "error": f"cannot read {home}/.cache/x: missing",
    }
    path = tmp_path / "r.json"
    code = gen._finish(SimpleNamespace(report=path), report)
    assert code == 2
    written = json.loads(path.read_text())
    assert written["df11"]["root"] == "~/.cache/huggingface/hub/models--a/snapshots/s"
    assert written["error"] == "cannot read ~/.cache/x: missing"
    assert home not in path.read_text()


@pytest.mark.parametrize(
    ("model", "steps"),
    [
        ("schnell", 4),
        ("dev", 25),
        ("z-image", 50),
        ("z-image-turbo", 9),
        ("flux2-klein-base-4b", 50),
        ("flux2-klein-4b", 4),
        ("flux2-klein-base-9b", 50),
        ("flux2-klein-9b", 4),
        ("qwen-image-2.1", 40),
    ],
)
def test_default_steps_follow_mflux_per_model(model, steps):
    # Bug caught: Z-Image falling back to FLUX's 25 (mflux 0.20 cli/defaults/defaults.py:55-56: 50 and 9), or a
    # Klein base on the distilled 4 steps (defaults.py:41-45: distilled 4, base 50).
    assert gen._steps(gen.build_parser().parse_args(["--model", model, "--prompt", "p"])) == steps


def test_the_model_choices_are_the_registry_names():
    # Bug caught: --model still limited to the FLUX.1 names (z-image refused by argparse).
    for name in (
        "schnell",
        "dev",
        "krea-dev",
        "z-image",
        "z-image-turbo",
        "flux2-klein-base-4b",
        "flux2-klein-4b",
        "flux2-klein-base-9b",
        "flux2-klein-9b",
        "qwen-image-2.1",
    ):
        assert gen.build_parser().parse_args(["--model", name, "--prompt", "p"]).model == name
    with pytest.raises(SystemExit):
        gen.build_parser().parse_args(["--model", "z-image-edit", "--prompt", "p"])


def _generate_kwargs(model, tmp_path, *extra):
    log = []
    code, _ = _run(
        ["--model", model, "--prompt", "p", "--output", str(tmp_path / "o.png"), *extra],
        log,
        tmp_path,
    )
    assert code == 0
    return log[0][1]


def test_zimage_gets_no_guidance_or_scheduler_unless_asked_and_flux_keeps_its_defaults(tmp_path):
    # Bug caught: FLUX's 3.5 / "linear" forced on Z-Image (base would run CFG at 3.5 with the wrong scheduler), or
    # FLUX losing them.
    z = _generate_kwargs("z-image", tmp_path)
    assert z["guidance"] is None
    assert z["scheduler"] is None
    asked = _generate_kwargs("z-image", tmp_path, "--guidance", "4", "--scheduler", "euler")
    assert (asked["guidance"], asked["scheduler"]) == (4.0, "euler")
    f = _generate_kwargs("dev", tmp_path)
    assert (f["guidance"], f["scheduler"]) == (3.5, "linear")


def test_the_negative_prompt_reaches_zimage_and_flux_does_not_get_it(tmp_path):
    # Bug caught: a Z-Image base user's negative prompt dropped, or an unexpected kwarg sent to FLUX.1.
    assert (
        _generate_kwargs("z-image", tmp_path, "--negative-prompt", "blurry")["negative_prompt"]
        == "blurry"
    )
    assert "negative_prompt" not in _generate_kwargs(
        "schnell", tmp_path, "--negative-prompt", "blurry"
    )


def test_the_negative_prompt_warnings_follow_the_model_and_the_guidance(tmp_path, capsys):
    # Bug caught: telling a Z-Image base user their negative prompt is ignored when it is used (guidance 4), or
    # staying silent when it has no effect (guidance at its default 0, or exactly 1.0: mflux runs CFG only above 1).
    _generate_kwargs("z-image", tmp_path, "--negative-prompt", "blurry", "--guidance", "4")
    assert "warning" not in capsys.readouterr().err
    for extra in ([], ["--guidance", "1.0"]):
        _generate_kwargs("z-image", tmp_path, "--negative-prompt", "blurry", *extra)
        err = capsys.readouterr().err
        assert "warning: --negative-prompt has no effect" in err
        assert "Pass --guidance above 1.0 to enable it" in err
    _generate_kwargs(
        "z-image", tmp_path, "--guidance", "1.0"
    )  # no negative prompt: nothing to warn about
    assert "warning" not in capsys.readouterr().err
    _generate_kwargs("schnell", tmp_path, "--negative-prompt", "blurry")
    assert "--negative-prompt is ignored" in capsys.readouterr().err


@pytest.mark.parametrize("model", ["flux2-klein-4b", "flux2-klein-9b"])
def test_a_distilled_klein_refuses_another_guidance(model, tmp_path, capsys):
    # Bug caught: --guidance 4 reaching a distilled Klein (mflux's own CLI exits 2 there, flux2_generate.py:74-75),
    # or the refusal coming after the caps or the model build (minutes and gigabytes spent before the error).
    built = []
    args = gen.build_parser().parse_args(
        ["--model", model, "--prompt", "p", "--guidance", "4", "--output", str(tmp_path / "o.png")]
    )
    code = gen.run(
        args,
        model_factory=lambda **kw: built.append(kw),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
        watchdog_factory=lambda *a, **k: pytest.fail("watchdog started before the refusal"),
        resolve_output=_resolve,
        **_host_kwargs(),
    )
    assert code == 2
    assert built == []
    label = {"flux2-klein-4b": "FLUX.2-klein-4B", "flux2-klein-9b": "FLUX.2-klein-9B"}[model]
    assert f"error: --guidance: {label} is a distilled model and runs at guidance 1.0 only" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize("extra", [[], ["--guidance", "1.0"], ["--guidance", "1"]])
def test_a_distilled_klein_accepts_guidance_one_and_the_default(extra, tmp_path):
    # Bug caught: the refusal firing on the default (no --guidance) or on an explicit 1.0, or the distilled model
    # getting a guidance other than mflux's 1.0 (flux2_generate.py:63-64).
    call = _generate_kwargs("flux2-klein-4b", tmp_path, *extra)
    assert call["guidance"] == 1.0
    assert call["num_inference_steps"] == 4
    assert call["scheduler"] == "flow_match_euler_discrete"


def test_a_base_klein_passes_guidance_through_without_a_negative_prompt(tmp_path, capsys):
    # Bug caught: a base Klein's --guidance 4 refused like a distilled one, or a negative_prompt keyword sent to
    # Flux2Klein.generate_image (which has none: flux2_klein.py:50-63, a TypeError at the call).
    call = _generate_kwargs(
        "flux2-klein-base-9b", tmp_path, "--guidance", "4", "--negative-prompt", "blurry"
    )
    assert call["guidance"] == 4.0
    assert call["num_inference_steps"] == 50
    assert "negative_prompt" not in call
    assert "--negative-prompt is ignored" in capsys.readouterr().err


@pytest.mark.parametrize("model", ["flux2-klein-base-4b", "flux2-klein-9b"])
def test_a_klein_refuses_a_scheduler_before_anything_runs(model, tmp_path, capsys):
    # Bug caught (review 2026-10-08): --scheduler linear reaching a Klein model, whose mflux command has no scheduler
    # flag and always runs flow_match_euler_discrete (models/flux2/cli/flux2_generate.py): LinearScheduler.step ignores
    # the sigmas Klein passes (linear_scheduler.py:56-58), a sampling path with no identity evidence.
    built = []
    args = gen.build_parser().parse_args(
        [
            "--model",
            model,
            "--prompt",
            "p",
            "--scheduler",
            "linear",
            "--output",
            str(tmp_path / "o.png"),
        ]
    )
    code = gen.run(
        args,
        model_factory=lambda **kw: built.append(kw),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
        watchdog_factory=lambda *a, **k: pytest.fail("watchdog started before the refusal"),
        resolve_output=_resolve,
        **_host_kwargs(),
    )
    assert code == 2
    assert built == []
    assert (
        "error: --scheduler: mflux's FLUX.2 Klein command runs flow_match_euler_discrete only"
        in capsys.readouterr().err
    )


def test_the_scheduler_refusal_is_klein_only(tmp_path):
    # Bug caught: the Klein refusal put in the common list (FLUX.1's --scheduler, which mflux's FLUX.1 command takes,
    # refused too).
    assert _generate_kwargs("dev", tmp_path, "--scheduler", "linear")["scheduler"] == "linear"


def test_a_klein_negative_prompt_warning_says_it_takes_no_custom_negative(tmp_path, capsys):
    # Bug caught (review 2026-10-08): telling a base Klein user "this model has no negative branch" when guidance 4
    # does run one (mflux's blank " ", flux2_klein.py:78-80); what Klein lacks is a custom negative prompt.
    _generate_kwargs("flux2-klein-base-4b", tmp_path, "--guidance", "4", "--negative-prompt", "x")
    err = capsys.readouterr().err
    assert (
        "warning: --negative-prompt is ignored: this model takes no custom negative prompt" in err
    )
    assert "no negative branch" not in err
    _generate_kwargs("schnell", tmp_path, "--negative-prompt", "x")  # FLUX.1's wording unchanged
    assert "this model has no negative branch" in capsys.readouterr().err


def test_existing_models_ignore_the_fixed_guidance_check(tmp_path):
    # Bug caught: the check firing for fixed_guidance=None (schnell's or Z-Image's --guidance refused).
    assert _generate_kwargs("schnell", tmp_path, "--guidance", "7")["guidance"] == 7.0
    assert _generate_kwargs("z-image-turbo", tmp_path, "--guidance", "2")["guidance"] == 2.0


def test_the_watchdog_context_names_the_tier_and_label(tmp_path):
    # Bug caught: a --tier 16 abort artifact that cannot say which tier it was (the README row needs it).
    _, watchdogs, _, _ = _limits_run(["--tier", "16", "--model", "z-image-turbo"], tmp_path)
    context = watchdogs[0].context
    assert context["tier_gb"] == 16
    assert context["label"] == "CAPPED"
    assert context["model"] == "z-image-turbo"
    assert context["steps"] == 9
    _, watchdogs, _, _ = _limits_run([], tmp_path)
    assert (watchdogs[0].context["tier_gb"], watchdogs[0].context["label"]) == (32, "MEASURED")


def _stateful_limits_run(extra, tmp_path):
    """A run whose installers write a fake MLX limit state that ``read_limits`` reads back."""
    # Distinct from every cap below: a context read before the install would carry these.
    state = {"memory": 1, "cache": 2, "wired": 3}

    def install_caps():
        state.update(memory=22 * GIB, wired=20 * GIB)  # the host caps leave the cache limit alone
        return (20, 22)

    def apply_limits(limits):
        previous = dict(state)
        state.update(
            memory=limits.memory_limit_bytes,
            cache=limits.cache_limit_bytes,
            wired=limits.wired_limit_bytes,
        )
        return previous

    _, watchdogs = _run(
        ["--prompt", "p", "--output", str(tmp_path / "o.png"), *extra],
        [],
        tmp_path,
        install_caps=install_caps,
        apply_limits=apply_limits,
        read_limits=lambda: dict(state),
    )
    return watchdogs[0].context


def test_the_watchdog_context_carries_the_limits_in_force_after_the_install(tmp_path):
    # Bug caught: a CAPPED abort artifact that cannot say which limits the run was stopped under
    # (no `limits`), limits read before the install (1/2/3 here), or the tier's numbers swapped.
    # The 16 GB tier's caps, worked from its 2/3 ratio: a 10.67 GiB recommended working set gives
    # an 8 GiB wired cap and a 10 GiB memory limit; the cache limit is the smaller of the memory
    # limit and 95 % of the recommended set (10.13 GiB), so 10 GiB.
    context = _stateful_limits_run(["--tier", "16"], tmp_path)
    assert context["limits"] == {
        "memory": 10_737_418_240,
        "cache": 10_737_418_240,
        "wired": 8_589_934_592,
    }
    # Bug caught: the host run's context without limits, or carrying the tier path's values.
    assert _stateful_limits_run([], tmp_path)["limits"] == {
        "memory": 22 * GIB,
        "cache": 2,
        "wired": 20 * GIB,
    }


def test_family_refused_flags_are_refused_beyond_the_common_ones(monkeypatch):
    # Bug caught: a family's own refusal list ignored (the flag reaching the model) or applied to the other family.
    from dataclasses import replace

    from mlx_dfloat.mflux import families

    monkeypatch.setitem(
        families.FAMILIES,
        "zimage",
        replace(families.FAMILIES["zimage"], refused_flags={"--shift": "not on this family"}),
    )
    args = gen.build_parser().parse_args(["--model", "z-image", "--prompt", "p", "--shift", "3"])
    assert gen.refused_option(args) == "--shift: not on this family"
    flux = gen.build_parser().parse_args(["--model", "schnell", "--prompt", "p", "--shift", "3"])
    assert gen.refused_option(flux) is None


def test_the_model_class_comes_from_the_registry_per_model_name(monkeypatch):
    # Bug caught: the FLUX.1 class built for every name (z-image run through DFloatFlux1).
    from dataclasses import replace

    from mlx_dfloat.mflux import families

    for fam in ("flux1", "zimage"):
        monkeypatch.setitem(
            families.FAMILIES,
            fam,
            replace(families.FAMILIES[fam], load_model_class=lambda fam=fam: fam),
        )
    assert gen._model_class("dev") == "flux1"
    assert gen._model_class("z-image") == "zimage"


def test_the_help_says_one_image_not_one_flux_image():
    # Bug caught: the help still describing a FLUX.1-only command.
    text = gen.build_parser().format_help()
    assert "one image" in text
    assert "FLUX.1 image" not in text


def test_the_generate_help_reads_as_plain_text_and_explains_every_choice(capsys):
    # Bug caught: the module docstring's RST double backticks printed literally in `--help`, the
    # --eval-policy choices left unexplained, or the help still saying "the host caps".
    standalone = gen.build_parser().format_help()
    with pytest.raises(SystemExit):
        main(["generate", "--help"])
    texts = [standalone, capsys.readouterr().out]
    for text in texts:
        flat = " ".join(text.split())
        assert "``" not in text
        assert "host caps" not in flat
        assert "per-block: evaluate after each block (default); depth2: one block behind" in flat
        assert "the host's own limits" in flat


def test_the_abort_context_reads_the_phase_open_now_build_before_the_model_exists(tmp_path):
    # Bug caught: an abort artifact that cannot say where the run was (the 16 GB stop happened in the encode
    # phase, and the README states it), or a phase frozen at watchdog start.
    log, phases = [], []
    watchdogs = []

    class Phased(_Model):
        open_phase = None

        def generate_image(self, **kwargs):
            self.open_phase = "encode"
            phases.append(watchdogs[0].live_context["phase"]())
            self.open_phase = "vae"
            phases.append(watchdogs[0].live_context["phase"]())
            return super().generate_image(**kwargs)

    def factory(**kw):
        phases.append(watchdogs[0].live_context["phase"]())
        return Phased(log, **kw)

    def watchdog_factory(out_dir, **kw):
        watchdogs.append(_Watchdog(out_dir, **kw))
        return watchdogs[-1]

    args = gen.build_parser().parse_args(["--prompt", "p", "--output", str(tmp_path / "o.png")])
    code = gen.run(
        args,
        model_factory=factory,
        install_caps=lambda: (20, 22),
        watchdog_factory=watchdog_factory,
        resolve_output=_resolve,
        **_host_kwargs(),
    )
    assert code == 0
    assert phases == ["build", "encode", "vae"]


def test_the_report_records_the_wall_clock_of_the_whole_run_including_the_build(tmp_path):
    # Bug caught: no elapsed time in the report (README rows then quote times from memory), or one measured from
    # after the model build (the set load and encoder load would vanish from it).
    ticks = iter([100.0, 172.5])
    order = []

    def clock():
        order.append("clock")
        return next(ticks)

    def factory(**kw):
        order.append("build")
        return _Model([], **kw)

    code = gen.run(
        _args("--output", str(tmp_path / "o.png"), "--report", str(tmp_path / "r.json")),
        model_factory=factory,
        install_caps=lambda: (20, 22),
        watchdog_factory=_Watchdog,
        resolve_output=_resolve,
        clock=clock,
        **_host_kwargs(),
    )
    assert code == 0
    assert order == ["clock", "build", "clock"]
    assert json.loads((tmp_path / "r.json").read_text())["elapsed_seconds"] == 72.5


# --- Qwen-Image 2.1 ---------------------------------------------------------------------------------------------


def test_qwen_gets_mfluxs_defaults(tmp_path, capsys):
    # Bug caught: Qwen-Image 2.1 on FLUX's 3.5 guidance (a guidance mflux never passes; with a negative prompt it would
    # run CFG at 3.5) or on Z-Image's None. mflux 0.20.0: 40 steps (cli/defaults/defaults.py:52), guidance 1.0
    # (qwen21_generate.py:60), scheduler "linear" (cli/parser/parsers.py:180).
    call = _generate_kwargs("qwen-image-2.1", tmp_path)
    assert (call["num_inference_steps"], call["guidance"], call["scheduler"]) == (40, 1.0, "linear")
    assert call["negative_prompt"] is None
    assert "warning" not in capsys.readouterr().err


def test_qwen_passes_the_negative_prompt_through(tmp_path, capsys):
    # Bug caught: the negative prompt dropped for Qwen-Image 2.1 (CFG never runs: mflux needs both a guidance above
    # 1.0 and a negative prompt, qwen_image_21.py:89), or the "ignored" warning printed when it is used.
    call = _generate_kwargs(
        "qwen-image-2.1", tmp_path, "--guidance", "4", "--negative-prompt", "blurry"
    )
    assert (call["guidance"], call["negative_prompt"]) == (4.0, "blurry")
    assert "warning" not in capsys.readouterr().err


def test_qwen_guidance_without_a_negative_prompt_warns(tmp_path, capsys):
    # Bug caught (Review Focus 4): a Qwen user's --guidance 4 silently running no CFG (mflux's rule: guidance above
    # 1.0 AND a negative prompt, qwen_image_21.py:89), or the call changed instead of warned (the call still runs at
    # 4.0 without a negative prompt, as mflux would).
    call = _generate_kwargs("qwen-image-2.1", tmp_path, "--guidance", "4")
    assert (call["guidance"], call["negative_prompt"]) == (4.0, None)
    err = capsys.readouterr().err
    assert (
        "warning: --guidance above 1.0 runs no classifier-free guidance for Qwen-Image-2.1 without "
        "--negative-prompt (as mflux); pass --negative-prompt to enable it"
    ) in err


@pytest.mark.parametrize("guidance", ["1.0", "1"])
def test_qwen_guidance_at_one_without_a_negative_prompt_does_not_warn(guidance, tmp_path, capsys):
    # Bug caught: the CFG warning on `>= 1.0` (guidance 1.0 is mflux's default and runs no CFG by design).
    _generate_kwargs("qwen-image-2.1", tmp_path, "--guidance", guidance)
    assert "warning" not in capsys.readouterr().err


def test_the_cfg_warning_is_qwen_only(tmp_path, capsys):
    # Bug caught: the warning shown for a Z-Image base (mflux runs its CFG with an empty negative, so --guidance 4
    # alone does run it) or a base Klein (mflux's blank negative).
    _generate_kwargs("z-image", tmp_path, "--guidance", "4")
    _generate_kwargs("flux2-klein-base-4b", tmp_path, "--guidance", "4")
    assert "classifier-free guidance" not in capsys.readouterr().err


def test_qwen_negative_prompt_at_default_guidance_warns_with_its_default(tmp_path, capsys):
    # Bug caught: the "no effect" warning quoting Z-Image's default 0 for Qwen-Image 2.1, whose default is 1.0
    # (qwen21_generate.py:60).
    _generate_kwargs("qwen-image-2.1", tmp_path, "--negative-prompt", "blurry")
    err = capsys.readouterr().err
    assert "warning: --negative-prompt has no effect" in err
    assert "(the default is 1.0)" in err
    assert "(the default is 0)" not in err


def test_zimage_negative_prompt_warning_text_is_unchanged(tmp_path, capsys):
    # Bug caught: the Z-Image warning changed by the per-model default (its default is None, mflux's own rule: 0).
    _generate_kwargs("z-image", tmp_path, "--negative-prompt", "blurry")
    assert (
        "warning: --negative-prompt has no effect: classifier-free guidance runs only above "
        "guidance 1.0 (the default is 0). Pass --guidance above 1.0 to enable it."
    ) in capsys.readouterr().err


def test_qwen_takes_a_scheduler(tmp_path):
    # Bug caught: Klein's scheduler refusal applied to Qwen-Image 2.1 (mflux's Qwen command takes --scheduler,
    # cli/parser/parsers.py:180).
    assert (
        _generate_kwargs("qwen-image-2.1", tmp_path, "--scheduler", "linear")["scheduler"]
        == "linear"
    )


@pytest.mark.parametrize(
    "flag", [["--quantize", "8"], ["--lora-paths", "a"], ["--image-path", "x.png"], ["-q", "4"]]
)
def test_qwen_refuses_the_common_flags_before_anything_loads(flag, tmp_path, capsys):
    # Bug caught: a new family whose refusal list replaced the common one (quantize or LoRA reaching
    # DFloatQwenImage21, which refuses them only after the caps and the watchdog).
    log = []
    code, watchdogs = _run(
        ["--model", "qwen-image-2.1", "--prompt", "p", "--output", str(tmp_path / "o.png"), *flag],
        log,
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
    )
    assert (code, watchdogs, log) == (2, [], [])
    assert flag[0] in capsys.readouterr().err


# --- ERNIE-Image ------------------------------------------------------------------------------------------------


def test_ernie_steps_follow_mflux_per_variant():
    # Bug caught: an ERNIE model on another model's steps (mflux 0.20.0 cli/defaults/defaults.py:35-36: ernie-image 50,
    # ernie-image-turbo 8).
    steps = {
        m: gen._steps(gen.build_parser().parse_args(["--model", m, "--prompt", "p"]))
        for m in ("ernie-image", "ernie-image-turbo")
    }
    assert steps == {"ernie-image": 50, "ernie-image-turbo": 8}


def test_turbo_guidance_other_than_1_is_refused_before_the_build(tmp_path, capsys):
    # Bug caught (Review Focus 1, command half): --guidance 4 reaching ERNIE-Image-Turbo (mflux's turbo command errors,
    # ernie_image_turbo_generate.py:42-45), or the refusal coming after the caps or the build.
    built = []
    args = gen.build_parser().parse_args(
        [
            "--model",
            "ernie-image-turbo",
            "--prompt",
            "p",
            "--guidance",
            "4",
            "--output",
            str(tmp_path / "o.png"),
        ]
    )
    code = gen.run(
        args,
        model_factory=lambda **kw: built.append(kw),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
        watchdog_factory=lambda *a, **k: pytest.fail("watchdog started before the refusal"),
        resolve_output=_resolve,
        **_host_kwargs(),
    )
    assert (code, built) == (2, [])
    assert (
        "error: --guidance: ERNIE-Image-Turbo is a distilled model and runs at guidance 1.0 only"
        in capsys.readouterr().err
    )


def test_turbo_negative_prompt_warns_it_is_ignored(tmp_path, capsys):
    # Bug caught: a negative_prompt keyword sent to Turbo (mflux's turbo command ignores it with a warning,
    # ernie_image_turbo_generate.py:10-12, 40), or the warning missing.
    call = _generate_kwargs("ernie-image-turbo", tmp_path, "--negative-prompt", "blurry")
    assert "negative_prompt" not in call
    assert (call["guidance"], call["num_inference_steps"], call["scheduler"]) == (1.0, 8, "linear")
    assert (
        "warning: --negative-prompt is ignored: this model has no negative branch"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("extra", "negative"), [([], None), (["--negative-prompt", "blurry"], "blurry")]
)
def test_ernie_base_passes_guidance_4_and_the_negative_prompt_by_default(
    tmp_path, capsys, extra, negative
):
    # Bug caught: the base run at mflux's class default 1.0 (no CFG: not mflux's ernie-image command,
    # ernie_image_generate.py:22, 30-31), the negative prompt dropped, or a Qwen-style "no CFG without
    # --negative-prompt" warning (ERNIE runs CFG with mflux's " " negative).
    call = _generate_kwargs("ernie-image", tmp_path, *extra)
    assert (call["guidance"], call["num_inference_steps"], call["scheduler"]) == (4.0, 50, "linear")
    assert call["negative_prompt"] == negative
    assert "warning" not in capsys.readouterr().err


def test_ernie_base_negative_prompt_at_guidance_1_warns_with_its_default_4(tmp_path, capsys):
    # Bug caught: the "no effect" warning quoting another model's default (Z-Image's 0, Qwen's 1.0).
    _generate_kwargs("ernie-image", tmp_path, "--guidance", "1", "--negative-prompt", "blurry")
    err = capsys.readouterr().err
    assert "warning: --negative-prompt has no effect" in err
    assert "(the default is 4.0)" in err


@pytest.mark.parametrize("model", ["ernie-image", "ernie-image-turbo"])
@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_empty_ernie_prompt_is_refused_before_anything_loads(model, prompt, tmp_path, capsys):
    # Bug caught: an empty or blank prompt reaching ERNIE-Image (its tokenizer gives "" no tokens: a zero-length text
    # through the transformer), or the refusal coming after the caps, the watchdog or the build.
    log = []
    code, watchdogs = _run(
        ["--model", model, "--prompt", prompt, "--output", str(tmp_path / "o.png")],
        log,
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
    )
    assert (code, watchdogs, log) == (2, [], [])
    assert "an empty or blank prompt is refused as a user error" in capsys.readouterr().err


def test_an_empty_prompt_is_still_passed_to_the_other_families(tmp_path):
    # Bug caught: the ERNIE refusal applied to every model (mflux encodes an empty prompt for the others).
    log = []
    code, _ = _run(
        ["--model", "schnell", "--prompt", "", "--output", str(tmp_path / "o.png")], log, tmp_path
    )
    assert code == 0
    assert log[0][1]["prompt"] == ""


# --- Krea 2 ------------------------------------------------------------------------------------------------------


def test_krea_negative_prompt_at_guidance_half_does_not_warn_it_has_no_effect(tmp_path, capsys):
    # Bug caught (Review Focus 1): the "runs only above guidance 1.0" warning for Krea 2 at 0.5, where mflux does run
    # classifier-free guidance (any guidance other than 1.0, prompt_encoder.py:33), so the negative prompt is used.
    call = _generate_kwargs("krea-2", tmp_path, "--guidance", "0.5", "--negative-prompt", "blurry")
    assert (call["guidance"], call["negative_prompt"]) == (0.5, "blurry")
    assert "has no effect" not in capsys.readouterr().err


def test_krea_negative_prompt_at_guidance_1_warns_with_the_default(tmp_path, capsys):
    # Bug caught: the warning missing at exactly 1.0 (mflux runs no negative branch there), or worded with the other
    # families' "only above 1.0" (wrong for Krea 2, where 0.5 runs it too).
    _generate_kwargs("krea-2", tmp_path, "--guidance", "1", "--negative-prompt", "blurry")
    err = capsys.readouterr().err
    assert (
        "warning: --negative-prompt has no effect: classifier-free guidance runs only for a guidance other than "
        "1.0 (the default is 1.0)"
    ) in err
    assert "only above" not in err


def test_flux_dev_negative_prompt_at_guidance_half_still_warns(tmp_path, capsys):
    # Bug caught: the Krea rule applied to every family (FLUX.1 dev takes no negative prompt; Z-Image at 0.5 runs no
    # CFG, so its warning must stay).
    _generate_kwargs("z-image", tmp_path, "--guidance", "0.5", "--negative-prompt", "blurry")
    assert (
        "warning: --negative-prompt has no effect: classifier-free guidance runs only above guidance 1.0"
        in capsys.readouterr().err
    )
    _generate_kwargs("dev", tmp_path, "--guidance", "0.5", "--negative-prompt", "blurry")
    assert "warning: --negative-prompt is ignored" in capsys.readouterr().err


def test_an_unknown_krea_scheduler_is_refused_before_the_build(tmp_path, capsys):
    # Bug caught (Review Focus 3): flow_match_euler_discrete (another family's scheduler) reaching the model, where
    # mflux would raise only after the encoder and the set loaded, or the refusal after the caps.
    log = []
    code, watchdogs = _run(
        [
            "--model",
            "krea-2",
            "--prompt",
            "p",
            "--scheduler",
            "flow_match_euler_discrete",
            "--output",
            str(tmp_path / "o.png"),
        ],
        log,
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
    )
    assert (code, watchdogs, log) == (2, [], [])
    assert "error: --scheduler: Krea 2 Turbo runs er_sde, euler, linear" in capsys.readouterr().err


@pytest.mark.parametrize("scheduler", ["er_sde", "euler", "linear"])
def test_the_krea_schedulers_pass_through(scheduler, tmp_path):
    # Bug caught: "linear" refused although mflux maps it to er_sde (krea2.py:206-212), or euler refused.
    assert (
        _generate_kwargs("krea-2-raw", tmp_path, "--scheduler", scheduler)["scheduler"] == scheduler
    )


def test_the_scheduler_allow_list_is_krea_only():
    # Bug caught: another family's --scheduler checked against Krea's list (Qwen's "linear" passes, but Z-Image's
    # "euler" or a custom dotted scheduler would be refused).
    for model, scheduler in (("z-image", "euler"), ("qwen-image-2.1", "my.module.Scheduler")):
        args = gen.build_parser().parse_args(
            ["--model", model, "--prompt", "p", "--scheduler", scheduler]
        )
        assert gen.scheduler_refusal(args) is None


def test_krea_raw_passes_its_defaults(tmp_path, capsys):
    # Bug caught: the card's 52 / 3.5 left as Raw's default (a CFG run nobody asked for), or Turbo's 8 steps. Literals:
    # 25 steps (mflux defaults.py:19,111-118), guidance 1.0 (krea2.py:58), er_sde (krea2.py:206-212).
    call = _generate_kwargs("krea-2-raw", tmp_path, "--negative-prompt", "n")
    assert (call["num_inference_steps"], call["guidance"], call["scheduler"]) == (25, 1.0, "er_sde")
    assert call["negative_prompt"] == "n"
    turbo = _generate_kwargs("krea-2", tmp_path)
    assert (turbo["num_inference_steps"], turbo["guidance"], turbo["scheduler"]) == (
        8,
        1.0,
        "er_sde",
    )


def test_krea_raw_negative_prompt_at_its_default_guidance_warns(tmp_path, capsys):
    # Bug caught: the warning keyed on Raw's former 3.5 (silent at the real default 1.0, where the negative prompt has
    # no effect).
    _generate_kwargs("krea-2-raw", tmp_path, "--negative-prompt", "n")
    err = capsys.readouterr().err
    assert "warning: --negative-prompt has no effect" in err
    assert "(the default is 1.0)" in err


def test_krea_raw_card_recipe_flags_pass_through(tmp_path, capsys):
    # Bug caught: the model card's recipe clamped to the defaults, or --guidance 3.5 refused.
    call = _generate_kwargs("krea-2-raw", tmp_path, "--steps", "52", "--guidance", "3.5")
    assert (call["num_inference_steps"], call["guidance"]) == (52, 3.5)


@pytest.mark.parametrize(
    ("model", "extra", "note"),
    [
        (
            "krea-2",
            ["--guidance", "0.5"],
            "info: guidance 0.5 on Krea 2 Turbo runs classifier-free guidance: two transformer calls per step "
            '(negative prompt " ")',
        ),
        (
            "krea-2-raw",
            ["--guidance", "3.5", "--negative-prompt", "n"],
            "info: guidance 3.5 on Krea 2 Raw runs classifier-free guidance: two transformer calls per step",
        ),
        ("krea-2-raw", [], None),
        ("dev", ["--guidance", "3.5"], None),
        ("qwen-image-2.1", ["--guidance", "4", "--negative-prompt", "n"], None),
    ],
)
def test_cfg_cost_note_names_two_calls_for_krea_off_guidance_one(
    model, extra, note, tmp_path, capsys
):
    # Bug caught: the note on every family, missing below 1.0 (the rule copied as `> 1.0`), or claiming mflux's " "
    # when the user gave a negative prompt. The exit code is unchanged (an info line, not a refusal).
    args = gen.build_parser().parse_args(["--model", model, "--prompt", "p", *extra])
    expected = None if note is None else note.removeprefix("info: ")
    assert gen.cfg_cost_note(args) == expected
    _generate_kwargs(model, tmp_path, *extra)  # asserts exit 0
    err = capsys.readouterr().err
    if note is None:
        assert "info: guidance" not in err
    else:
        assert note in err


@pytest.mark.parametrize(
    ("model", "guidance", "expected"),
    [
        ("krea-2", None, False),
        ("krea-2", 1.0, False),
        ("krea-2", 0.5, True),
        ("krea-2", 1.0001, True),
        ("qwen-image-2.1", 0.5, False),
        ("qwen-image-2.1", 1.0, False),
        ("qwen-image-2.1", 4.0, True),
    ],
)
def test_runs_cfg_follows_each_familys_rule(model, guidance, expected):
    # Bug caught: Krea's != 1.0 rule applied to every family, or the other families' > 1.0 rule applied to Krea.
    from mlx_dfloat.mflux import families

    assert gen.runs_cfg(families.entry(model), guidance) is expected


@pytest.mark.parametrize("model", ["krea-2", "krea-2-raw"])
@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_empty_krea_prompt_is_refused_before_anything_loads(model, prompt, tmp_path, capsys):
    # Bug caught: an empty prompt reaching Krea 2 (its tokenizer gives "" no ids: an empty context through the
    # transformer), or the refusal coming after the caps, the watchdog or the build.
    log = []
    code, watchdogs = _run(
        ["--model", model, "--prompt", prompt, "--output", str(tmp_path / "o.png")],
        log,
        tmp_path,
        factory=lambda **kw: pytest.fail("built"),
        install_caps=lambda: pytest.fail("caps installed before the refusal"),
    )
    assert (code, watchdogs, log) == (2, [], [])
    assert (
        "an empty or blank prompt is refused as a user error (an empty prompt gives Krea 2's tokenizer"
        in (capsys.readouterr().err)
    )


def test_the_help_names_the_krea_defaults_and_the_cards_recipe():
    # Bug caught: --help still listing only the older families' defaults, or Raw's default shown as the card's 3.5.
    text = " ".join(gen.build_parser().format_help().split())
    assert "krea-2-raw 25, krea-2 8" in text
    assert (
        "Krea 2 Raw and Turbo 1.0 (Krea 2 Raw's model card uses --steps 52 --guidance 3.5); Krea 2 runs "
        "classifier-free guidance, two transformer calls per step, for any value other than 1.0"
    ) in text
    assert "Krea 2: er_sde (default), euler" in text
