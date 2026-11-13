"""The source fingerprint of the mflux code the measurements depend on.

The VAE, text-encoder and transformer modules and the initializers of every supported family are hashed and compared
with the row recorded for the mflux release the memory constants and timings were measured on. A new mflux that changes
one of them turns the mflux lane red, so the change is read before the constants are trusted again.
"""

import importlib.util
import json
from importlib import metadata
from pathlib import Path

import pytest
from scripts._mflux_fingerprint import SCOPE, compare, fingerprint, row_to_compare, unmatched_globs

from mlx_dfloat.mflux._versions import NEWEST_TESTED_MFLUX

FIXTURE = Path(__file__).parent / "fixtures" / "mflux_fingerprint.json"

# sha256 of b"x\n" (`printf 'x\n' | shasum -a 256`) and of the empty input (the published empty-message digest).
SHA256_X_NEWLINE = "73cb3858a687a8494ca3323053016282f3dad39d42cf62ca4e79dda2aac7d9ac"
SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _recorded() -> dict:
    return json.loads(FIXTURE.read_text())


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_fingerprint_hashes_each_file_with_normalised_newlines(tmp_path):
    # Bug caught: CRLF not normalised (a checkout with Windows line endings reads as a change), paths keyed by the
    # platform separator or absolute, or files outside the scope (another extension, another directory) hashed.
    _write(tmp_path / "a.py", b"x\r\n")
    _write(tmp_path / "b" / "c.py", b"")
    _write(tmp_path / "b" / "notes.txt", b"not python")
    _write(tmp_path / "other" / "d.py", b"out of scope")
    assert fingerprint(tmp_path, ("a.py", "b/**/*.py")) == {
        "a.py": SHA256_X_NEWLINE,
        "b/c.py": SHA256_EMPTY,
    }


def test_a_scope_glob_that_matches_no_file_is_named(tmp_path):
    # Bug caught: a glob over a package mflux moved or renamed matches nothing and silently leaves the scope.
    _write(tmp_path / "kept" / "x.py", b"")
    assert unmatched_globs(tmp_path, ("kept/**/*.py", "moved/**/*.py")) == ["moved/**/*.py"]
    assert unmatched_globs(tmp_path, ("kept/**/*.py",)) == []


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("0.24.0", "0.24.0"),  # a recorded release: its own row
        ("0.26.0", "0.25.0"),  # newer than every row (the scheduled job): the newest row
        ("0.24.5", "0.24.0"),  # between two rows: the newest row not after it
        ("0.25.0rc1", "0.25.0"),  # a pre-release reads as its release
        ("0.23.9", None),  # older than every row: nothing to compare with
    ],
)
def test_the_row_compared_is_the_newest_recorded_release_not_after_the_installed_one(
    installed, expected
):
    # Bug caught: a newer mflux compared with the oldest row, or with no row (the scheduled job then proves
    # nothing), or an older mflux silently compared with a newer release's row.
    assert row_to_compare(installed, ["0.25.0", "0.24.0"]) == expected


def test_the_comparison_names_added_removed_and_changed_paths():
    # Bug caught: a failure message that hides which upstream file moved, so the review cannot start from it.
    recorded = {"same.py": "1", "gone.py": "2", "edited.py": "3"}
    measured = {"same.py": "1", "edited.py": "4", "new.py": "5"}
    assert compare(recorded, measured) == {
        "added": ["new.py"],
        "removed": ["gone.py"],
        "changed": ["edited.py"],
    }
    assert compare(recorded, dict(recorded)) == {"added": [], "removed": [], "changed": []}


def test_the_newest_fingerprint_row_is_the_newest_tested_release():
    # Bug caught: NEWEST_TESTED_MFLUX bumped without recording the new release's row (the scheduled job then
    # compares a newer mflux with an older baseline than the one the release was tested on), or the reverse.
    rows = list(_recorded()["versions"])
    assert row_to_compare("999.0.0", rows) == NEWEST_TESTED_MFLUX


def test_the_recorded_scope_is_the_hasher_scope():
    # Bug caught: a glob added to or dropped from SCOPE without re-recording, so the rows no longer describe the
    # files the comparison hashes.
    assert _recorded()["scope"] == list(SCOPE)


def _installed_mflux() -> tuple[str, Path]:
    spec = importlib.util.find_spec("mflux")
    assert spec is not None
    assert spec.submodule_search_locations
    return metadata.version("mflux"), Path(next(iter(spec.submodule_search_locations)))


@pytest.mark.mflux
def test_every_scope_glob_matches_a_file():
    # Bug caught: an mflux release moving a family's package, so its glob matches nothing and its files leave the
    # fingerprint without a single changed digest.
    _version, root = _installed_mflux()
    assert unmatched_globs(root, SCOPE) == []


@pytest.mark.mflux
def test_the_measured_mflux_code_is_unchanged():
    # Bug caught: an mflux release changing a VAE, text encoder, transformer block or initializer the memory
    # constants and timings were measured on, and nobody noticing.
    installed, root = _installed_mflux()
    rows = _recorded()["versions"]
    row = row_to_compare(installed, list(rows))
    if row is None:
        pytest.fail(
            f"mflux {installed} is older than every recorded fingerprint row ({sorted(rows)})"
        )
    delta = compare(rows[row], fingerprint(root, SCOPE))
    assert delta == {"added": [], "removed": [], "changed": []}, (
        f"mflux {installed} differs from the fingerprint recorded for {row} in {root}: {delta}"
    )


# Per family: the pipeline classes the adapters subclass, the weight definitions they load, and the latent creators.
_FAMILIES = ("flux", "z_image", "flux2", "ernie_image", "qwen21", "krea2")


@pytest.mark.parametrize(
    "glob",
    [
        *(f"models/{family}/variants/**/*.py" for family in _FAMILIES),
        *(f"models/{family}/weights/*_weight_definition.py" for family in _FAMILIES),
        *(f"models/{family}/latent_creator/**/*.py" for family in _FAMILIES),
        "models/krea2/model/krea2_sampler.py",
        "models/krea2/model/krea2_scheduler/**/*.py",
        # The weight mappings each family's names.py imports, and Qwen-Image's (Krea 2 loads its VAE through it)
        "models/flux/weights/flux_weight_mapping.py",
        "models/z_image/weights/z_image_weight_mapping.py",
        "models/flux2/weights/flux2_weight_mapping.py",
        "models/ernie_image/weights/ernie_weight_mapping.py",
        "models/qwen21/weights/qwen21_weight_mapping.py",
        "models/krea2/weights/krea2_weight_mapping.py",
        "models/qwen/weights/qwen_weight_mapping.py",
    ],
)
def test_the_scope_covers_the_pipelines_weights_latents_and_krea2_sampler(glob):
    # Bug caught: a variant pipeline (QwenImage21's `processor` read sits in variants/), a weight definition or the
    # Krea 2 sampler changing upstream without the fingerprint noticing. Paths from the mflux 0.24.0 tree.
    assert glob in SCOPE


def test_record_refuses_a_fixture_with_a_scope_but_no_versions(tmp_path, monkeypatch, capsys):
    # Bug caught: a hand-trimmed fixture dying with a bare KeyError: 'versions' instead of a message naming the file.
    from scripts import record_mflux_fingerprint as record

    out = tmp_path / "fp.json"
    out.write_text(json.dumps({"scope": list(SCOPE)}))
    monkeypatch.setattr(record, "_installed", lambda: ("0.24.0", tmp_path))
    monkeypatch.setattr(record, "unmatched_globs", lambda root, scope: [])
    assert record.main(["--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert str(out) in err
    assert "versions" in err
    assert json.loads(out.read_text()) == {"scope": list(SCOPE)}


def test_record_writes_atomically_and_keeps_the_old_file_on_a_failed_write(tmp_path):
    # Bug caught: the fixture truncated in place, so an error half-way through the write leaves an empty or partial
    # file instead of the previous rows (and a temp file left behind).
    from scripts import record_mflux_fingerprint as record

    out = tmp_path / "fp.json"
    out.write_text("old\n")
    with pytest.raises(UnicodeEncodeError):
        record.write_atomically(out, "new \ud800\n")
    assert out.read_text() == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fp.json"]
    record.write_atomically(out, "new\n")
    assert out.read_text() == "new\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fp.json"]
