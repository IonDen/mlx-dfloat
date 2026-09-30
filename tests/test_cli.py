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
from mlx_dfloat.mflux.flux1 import cli as gen

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
    def __init__(self, out_dir, *, ceiling, budget):
        self.out_dir, self.ceiling, self.budget, self.stopped = out_dir, ceiling, budget, False
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
    assert report["footprint_peak_bytes"] == 10**15
    assert (
        report["footprint_peak_label"] == "OS phys_footprint, sampled every 0.05 s by the watchdog"
    )


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
    # 32 GB while the row claims 24).
    code, watchdogs, calls, report = _limits_run(["--tier", "24"], tmp_path)
    assert code == 0
    assert "install_caps" not in calls
    applied = [c[1] for c in calls if c[0] == "apply"]
    assert [lim.tier_gb for lim in applied] == [24]
    assert watchdogs[0].ceiling == TIER_24_CEILING
    assert report["model_kwargs"]["budget_bytes"] == TIER_24_CEILING
    assert report["label"] == "CAPPED"
    assert report["tier_gb"] == 24
    assert report["watchdog_ceiling_bytes"] == TIER_24_CEILING
    assert report["memory_caps_gb"] is None
    assert report["limits"]["tier"]["tier_gb"] == 24
    assert report["limits"]["tier"]["ceiling_bytes"] == TIER_24_CEILING
    assert report["limits"]["applied"] == "tier-defaults"
    assert report["limits"]["effective"] == {
        "memory_limit_bytes": 111,
        "cache_limit_bytes": 222,
        "wired_limit_bytes": 333,
    }


def test_tier_32_on_a_32_gb_host_keeps_the_host_caps_and_is_measured(tmp_path):
    # Bug caught (Review Focus 2): `--tier <host>` installing the stock tier defaults instead of the
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
            self.peak_footprint = self.peak_watched = self.peak_mlx = 0

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
    assert report["mlx_peak_bytes"] == 0


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
