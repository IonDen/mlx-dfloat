"""`mlx-dfloat generate` against mflux 0.23's image save, which raises when the image file cannot be written.

Offline: the command's own fakes (``tests.test_cli``) stand in for the model, the watchdog and the host.
"""

import json
import platform

import pytest
from tests.test_cli import _Image, _Model, _run


class _RaisingImage(_Image):
    """mflux 0.23's ``GeneratedImage.save``: a failed image write propagates (``ImageUtil.save_image``)."""

    def save(self, path, export_json_metadata=False, overwrite=False):
        self.log.append(("save", str(path), export_json_metadata, overwrite))
        raise PermissionError(13, "Permission denied", str(path))


class _RaisingModel(_Model):
    def generate_image(self, **kwargs):
        self.log.append(("generate", kwargs))
        return _RaisingImage(self.log)


def test_a_save_that_raises_is_reported_as_the_image_not_written(tmp_path, capsys):
    # Bug caught: the write failure mflux 0.23 now raises falling through to the generic handler, which prints a
    # traceback instead of naming the file that was not written (exit 2 either way).
    log = []
    out = tmp_path / "o.png"
    code, _ = _run(
        ["--prompt", "p", "--output", str(out), "--report", str(tmp_path / "r.json")],
        log,
        tmp_path,
        factory=lambda **kw: _RaisingModel(log, **kw),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert f"{out}: the image was not written" in err
    assert "Traceback" not in err
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["exit_code"] == 2
    assert "Permission denied" in report["error"]


@pytest.mark.parametrize("model_class", [_Model, _RaisingModel])
def test_the_report_records_the_macos_version_of_the_run(tmp_path, monkeypatch, model_class):
    # Bug caught: a generate report with no OS field (the tier records could not say which macOS a MEASURED row ran
    # on), or the field written only when the run succeeds. The value is the bench provenance's: platform.mac_ver()[0].
    monkeypatch.setattr(platform, "mac_ver", lambda: ("26.9.1", ("", "", ""), "arm64"))
    log = []
    _run(
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
        factory=lambda **kw: model_class(log, **kw),
    )
    assert json.loads((tmp_path / "r.json").read_text())["macos"] == "26.9.1"


def test_the_watchdogs_run_context_records_the_versions_and_macos_as_the_report_does():
    # Red when: the context generate hands its watchdog (all a stopped run's abort artifact keeps) lacks the mlx or
    # mflux version or the macOS version, so the tier table cannot say what a stopped run ran on.
    from importlib import metadata

    import mlx.core as mx

    from mlx_dfloat.mflux import generate as gen

    try:
        mflux = metadata.version("mflux")
    except metadata.PackageNotFoundError:
        mflux = None
    args = gen.build_parser().parse_args(["--prompt", "p"])
    context = gen._run_context(
        args, tier_gb=16, label="CAPPED", limits={"memory": 1, "cache": 2, "wired": 3}
    )
    assert context["versions"] == {"mlx": mx.__version__, "mflux": mflux}
    assert context["macos"] == platform.mac_ver()[0]
