"""Config-less checkpoints: layouts pinned to their header bytes plus spot checks of stored data.

The fixture ``fixtures/qwen-image-2.1-df11-header.bin`` is the first 28,416 bytes of
``qwen_image_2.1_bf16-DF11.safetensors`` in ``mingyi456/Qwen-Image-2.1-DF11-ComfyUI`` at revision
``1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc``, exactly as on disk: the u64 little-endian header
length (28,408) followed by the JSON header. It was range-read on 2026-10-08 (file bytes 0..28415)
and is byte-identical to the header saved by the 2026-10-07 range reads.

The spot-check digests were range-read at the same revision on 2026-10-08, 4096 bytes each (data
starts at file byte 28,416; ``transformer_blocks.0.sign_mantissa`` at tensor data offset
308,886,370, ``transformer_blocks.31.sign_mantissa`` at 7,730,108,779, ``modulation.1`` at
163,695,879). One per matrix start of block 0, element offsets from the stored split points
(16,777,216 elements = one 4096 x 4096 matrix), the gate/up seam, block 31's first matrix, and
``modulation.1``:

- block 0 element 0 (``attn.to_q``): file bytes 308,914,786..308,918,881.
- block 0 element 16,777,216 (``attn.to_k``): 325,692,002..325,696,097.
- block 0 element 33,554,432 (``attn.to_v``): 342,469,218..342,473,313.
- block 0 element 50,331,648 (``attn.to_out.0``): 359,246,434..359,250,529.
- block 0 element 67,108,864 (``img_mlp.gate_up`` row 0, ``gate_layer``): 376,023,650..376,027,745.
- block 0 element 117,440,512 (``gate_up`` row 12288, ``proj``): 426,355,298..426,359,393.
- block 0 element 167,772,160 (``img_mlp.out``): 476,686,946..476,691,041.
- block 31 element 0 (``attn.to_q``): 7,730,137,195..7,730,141,290.
- ``modulation.1`` element 0: 163,724,295..163,728,390.

A 2026-10-07 byte comparison matched the sign/mantissa bytes at each of block 0's offsets against the BF16
base (``Qwen/Qwen-Image-2.1``): 4096 of 4096 equal for the expected tensor, at most 29 for any other.
"""

import dataclasses
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from tests._df11_fixtures import random_bf16, write_checkpoint

from mlx_dfloat import _layouts
from mlx_dfloat._layouts import (
    KNOWN_LAYOUTS,
    QWEN_IMAGE_21_COMFYUI,
    ContentProbe,
    SynthesizedLayout,
    identify_layout,
)
from mlx_dfloat._safetensors import parse_header, read_header_bytes
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import config_for_layout, group_headers, open_checkpoint

FIXTURE = Path(__file__).parent / "fixtures" / "qwen-image-2.1-df11-header.bin"
QWEN_REVISION = "1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc"
QWEN_FILE = "qwen_image_2.1_bf16-DF11.safetensors"
QWEN_FILE_SIZE = 9_724_491_411  # the 2026-10-07 range reads, and the Hub's file size
QWEN_HEADER_SHA256 = "fa5b70707d9d5e08d1c92e40dcdeedd64f72508f88ca942217c6fd419ff5ba5e"
QWEN_FILE_SHA256 = (
    "fc07ef38e3609b1a38dc8f6ec585ef9921a4d4b5e6f74b744899e4dea91b4e2e"  # the Hub's LFS oid
)
QWEN_PROBES = (  # (tensor, byte offset into its data, sha256 of 4096 bytes), see the module docstring
    (
        "transformer_blocks.0.sign_mantissa",
        0,
        "f960e614c0abeee3fc6b4d345887288841b97f7e07e6e14ca2f0bcb3d4ec9a09",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        16_777_216,
        "9674d29df78034e5b53974310e5d72545004db3116e97d4f97c883a78b6d21cf",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        33_554_432,
        "7c9eb5ca6ba99dfd5cbe65a44d332492cc2b461a8be9ca1145253896eac28db6",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        50_331_648,
        "1a117be0dca8deb6ebedab5835f1612d91f02f9881697d8ee21e046ee34b94e8",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        67_108_864,
        "89b64f185893b34b7dc1256194c5335789b4f8540d04cfc7fac60f91222d2e08",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        117_440_512,
        "022e33a91b869d7570a3115e3d2a2414ae16deb595b7858cde4071cfbb81ea19",
    ),
    (
        "transformer_blocks.0.sign_mantissa",
        167_772_160,
        "732127bdf5ddadce246331844f1312b61f2a335d6634779e1ee594e42d7ea437",
    ),
    (
        "transformer_blocks.31.sign_mantissa",
        0,
        "c4fb1affbfaa3f620b087719a00f0d46d25a40e7857eee26abe150c116f99c8f",
    ),
    (
        "modulation.1.sign_mantissa",
        0,
        "83a4184a43758b55bf98e4457aa2e77f4a09e85517c2ab50f894f088818adace",
    ),
)
BLOCK_SUBS = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
)
ZERO_PAGE_SHA256 = hashlib.sha256(bytes(4096)).hexdigest()


def _fixture_header() -> bytes:
    data = FIXTURE.read_bytes()
    (length,) = struct.unpack("<Q", data[:8])
    assert length == len(data) - 8
    return data[8:]


def _qwen_groups():
    raw = _fixture_header()
    infos = parse_header(raw, data_start=8 + len(raw), file_size=QWEN_FILE_SIZE, source="qwen")
    return group_headers(
        {Path("qwen.safetensors"): infos}, config_for_layout(QWEN_IMAGE_21_COMFYUI)
    )


def test_the_fixture_is_the_published_header_as_on_disk():
    # Bug caught: a fixture edited or re-serialised (key order, padding), so the pin below tests another file.
    data = FIXTURE.read_bytes()
    assert len(data) == 28_416
    assert struct.unpack("<Q", data[:8]) == (
        28_408,
    )  # header length 28,408, data at 28,416 (the 2026-10-07 range reads)
    assert hashlib.sha256(data[8:]).hexdigest() == QWEN_HEADER_SHA256


def test_the_published_header_is_the_qwen_image_21_layout():
    # Bug caught: the pinned fingerprint not the published file's (the layout would never apply to the real file).
    assert identify_layout(_fixture_header(), known=KNOWN_LAYOUTS) is QWEN_IMAGE_21_COMFYUI


def test_one_changed_header_byte_matches_no_layout():
    # Bug caught: matching on the file name or size (a re-upload with another matrix order read with this order).
    raw = bytearray(_fixture_header())
    raw[200] ^= 0x01
    assert identify_layout(bytes(raw), known=KNOWN_LAYOUTS) is None


def test_the_published_header_groups_as_counted_by_hand():
    # Bug caught: the grouping or the synthesized patterns losing a group, or gate_up not expanded at open time.
    groups, extras = _qwen_groups()
    assert sorted(groups) == sorted(
        ["modulation.1", *[f"transformer_blocks.{i}" for i in range(32)]]
    )
    assert groups["modulation.1"].matrix_names == ("modulation.1.weight",)
    assert groups["modulation.1"].row_plan == ()
    for i in (0, 15, 31):
        g = groups[f"transformer_blocks.{i}"]
        assert g.matrix_names == tuple(f"transformer_blocks.{i}.{s}.weight" for s in BLOCK_SUBS)
        assert g.row_plan == ((4, 2),)
    assert len(extras) == 72
    assert {info.dtype for _p, info in extras.values()} == {"BF16"}
    assert {
        "img_in.weight",
        "proj_out.weight",
        "norm_out.linear.weight",
        "txt_in.in_layer.weight",
        "txt_in.out_layer.weight",
        "txt_in.text_norm.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "time_text_embed.timestep_embedder.linear_2.weight",
    } <= set(extras)
    assert (
        sum(info.nbytes for _p, info in extras.values()) == 137_388_032
    )  # 68,694,016 BF16 elements


def test_the_qwen_layout_config_is_the_synthesized_config():
    # Bug caught: a typo in the pattern list or the split table (one matrix would decode under the wrong name).
    layout = QWEN_IMAGE_21_COMFYUI
    assert json.loads(json.dumps(layout.raw_config)) == {
        "version": "0.3.1",
        "threads_per_block": [512],
        "bytes_per_thread": 8,
        "pattern_dict": {
            r"modulation\.1": [],
            r"transformer_blocks\.\d+": [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "img_mlp.gate_up",
                "img_mlp.out",
            ],
        },
    }
    assert dict(layout.row_splits) == {"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")}
    assert (layout.repo_id, layout.revision, layout.file_name) == (
        "mingyi456/Qwen-Image-2.1-DF11-ComfyUI",
        QWEN_REVISION,
        QWEN_FILE,
    )
    assert (layout.header_sha256, layout.groups, layout.extras) == (QWEN_HEADER_SHA256, 33, 72)
    assert layout.file_sha256 == QWEN_FILE_SHA256


def test_the_qwen_layout_spot_checks_every_matrix_start_and_the_seam():
    # Bug caught: a matrix start without a spot check, so a re-export that permutes the equal-shaped to_q / to_k /
    # to_v / to_out.0 (same header) or swaps the gate/up halves passes and installs one matrix under another's name.
    got = tuple((p.tensor, p.offset, p.length, p.sha256) for p in QWEN_IMAGE_21_COMFYUI.probes)
    assert got == tuple((t, off, 4096, digest) for t, off, digest in QWEN_PROBES)


def _probe(root: Path, tensor: str, offset: int, length: int) -> ContentProbe:
    raw, start, size = read_header_bytes(root / "model.safetensors")
    info = parse_header(raw, data_start=start, file_size=size, source="t")[tensor]
    with (root / "model.safetensors").open("rb") as fh:
        fh.seek(info.offset + offset)
        data = fh.read(length)
    return ContentProbe(
        tensor=tensor, offset=offset, length=length, sha256=hashlib.sha256(data).hexdigest()
    )


TINY_SEAM = 16 + 24  # q is 4x4 = 16 elements, gate 6x4 = 24: up starts at element 40


def _tiny(tmp_path, *, write_config=False):
    rng = np.random.default_rng(8)
    q, g, u = (random_bf16(rng, s) for s in [(4, 4), (6, 4), (6, 4)])
    root = write_checkpoint(
        tmp_path / "ckpt",
        groups={"blocks.0": [q, np.concatenate([g, u])]},
        pattern=r"blocks\.\d+",
        sub_paths=("q", "gate_up"),
        single_file=True,
        extras={"norm.weight": random_bf16(rng, (4,))},
        write_config=write_config,
    )
    raw, _start, _size = read_header_bytes(root / "model.safetensors")
    layout = SynthesizedLayout(
        key="tiny",
        label="tiny",
        repo_id="t/t",
        revision="0" * 40,
        file_name="model.safetensors",
        header_sha256=hashlib.sha256(raw).hexdigest(),
        file_sha256=hashlib.sha256((root / "model.safetensors").read_bytes()).hexdigest(),
        groups=1,
        extras=1,
        raw_config={
            "version": "0.5.0",
            "threads_per_block": [512],
            "bytes_per_thread": 8,
            "pattern_dict": {r"blocks\.\d+": ["q", "gate_up"]},
        },
        row_splits={"gate_up": ("gate", "up")},
        probes=(
            _probe(root, "blocks.0.sign_mantissa", 0, 8),
            _probe(root, "blocks.0.sign_mantissa", TINY_SEAM, 8),
        ),
    )
    return root, layout


def test_a_config_less_file_opens_through_its_layout(tmp_path):
    # Bug caught: the config-less path not taken, or the checkpoint not saying where its config came from.
    root, layout = _tiny(tmp_path)
    ckpt = open_checkpoint(root, layouts=(layout,))
    assert ckpt.groups["blocks.0"].matrix_names == (
        "blocks.0.q.weight",
        "blocks.0.gate.weight",
        "blocks.0.up.weight",
    )
    assert ckpt.groups["blocks.0"].row_plan == ((1, 2),)
    assert set(ckpt.extras) == {"norm.weight"}
    assert (
        ckpt.config_source == ("header and spot checks match layout tiny (t/t@" + "0" * 40 + ")")
    )  # a plain file name, not a cache blob: the whole-file pin could not be checked, so it is not claimed


HEADER_ONLY_SOURCE = "header and spot checks match layout tiny (t/t@" + "0" * 40 + ")"


def _move_behind_link(root: Path, target: Path) -> None:
    """Move the checkpoint's file to ``target`` and leave a symlink to it in its place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").rename(target)
    (root / "model.safetensors").symlink_to(target)


def _as_hf_blob(root: Path, blob_name: str) -> None:
    """Lay the file out like the classic (huggingface_hub < 2.0) cache: ``models--t--t/blobs/<name>``."""
    _move_behind_link(root, root.parent / "hub" / "models--t--t" / "blobs" / blob_name)


def _as_hf_shared_blob(tmp_path: Path, root: Path, pin: str, stored_name: str) -> Path:
    """Lay the file out like the hf 2.0 shared store and return the snapshot directory to open.

    The real chain (Qwen-Image 2.1 DF11, 2026-10-08): ``snapshots/<rev>/<file>`` links to
    ``../../blobs/<LFS sha256>``, which links to ``../../blobs/<2 chars>/<name>`` in the shared
    store, a regular file whose name is not the file's sha256. The store carries a marker file.
    """
    hub = tmp_path / "hub"
    store = hub / "blobs"
    store.mkdir(parents=True)
    (store / ".huggingface-shared-blobs").write_text("1\n")
    stored = store / stored_name[:2] / stored_name
    stored.parent.mkdir()
    (root / "model.safetensors").rename(stored)
    repo = hub / "models--t--t"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / pin).symlink_to(Path("..", "..", "blobs", stored_name[:2], stored_name))
    snapshot = repo / "snapshots" / ("1" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(Path("..", "..", "blobs", pin))
    return snapshot


def test_a_cache_blob_named_by_the_pinned_sha256_is_reported_as_a_whole_file_match(tmp_path):
    # Bug caught: the blob name ignored, so config_source cannot tell a whole-file match from a header-only one.
    root, layout = _tiny(tmp_path)
    _as_hf_blob(root, layout.file_sha256)
    ckpt = open_checkpoint(root, layouts=(layout,))
    assert ckpt.config_source == HEADER_ONLY_SOURCE + "; file sha256 matches"


def test_a_file_in_the_hf_2_shared_blob_store_opens_without_a_whole_file_claim(tmp_path):
    # Bug caught: the shared store's name (not the file's sha256) compared with the pin, so the right file is
    # refused for every hf >= 2.0 user; or a whole-file match claimed when no whole-file check ran.
    root, layout = _tiny(tmp_path)
    stored_name = "b2769418f6600232718aff11dbf291cf11e2aff7a06f41667bfc8e778ecfab0b"
    assert stored_name != layout.file_sha256
    snapshot = _as_hf_shared_blob(tmp_path, root, layout.file_sha256, stored_name)
    ckpt = open_checkpoint(snapshot, layouts=(layout,))
    assert ckpt.config_source == HEADER_ONLY_SOURCE


@pytest.mark.parametrize(
    "where",
    [
        Path("elsewhere"),  # neither part of the classic layout
        Path("notarepo", "blobs"),  # a blobs/ directory outside a models--* repo
        Path("models--t--t", "other"),  # inside a models--* repo, but not its blobs/
    ],
)
def test_a_64_hex_name_outside_the_classic_cache_layout_is_not_checked(tmp_path, where):
    # Bug caught: a 64-hex name anywhere taken for an LFS sha256 (one half of the classic-layout test dropped), so
    # a --local-dir copy or a renamed file under a hex name is refused.
    root, layout = _tiny(tmp_path)
    _move_behind_link(root, tmp_path / where / ("ab" * 32))
    assert open_checkpoint(root, layouts=(layout,)).config_source == HEADER_ONLY_SOURCE


def test_a_cache_blob_named_by_another_sha256_is_refused(tmp_path):
    # Bug caught: a downloaded file whose LFS sha256 differs from the pinned one (a re-upload with the same header
    # and the same bytes at the spot checks, changed elsewhere) read through the layout.
    root, layout = _tiny(tmp_path)
    other = "ab" * 32
    _as_hf_blob(root, other)
    with pytest.raises(DFloatFormatError) as err:
        open_checkpoint(root, layouts=(layout,))
    assert str(err.value) == (
        "model.safetensors: header matches the known layout tiny, but the file's sha256 (from "
        f"its Hugging Face cache name) is {other}, not the {layout.file_sha256} it was checked "
        "against; re-download the file, or report it at https://github.com/IonDen/mlx-dfloat/issues."
    )


@pytest.mark.parametrize("name", ["AB" * 32, "ab" * 31, "ab" * 32 + ".safetensors"])
def test_a_link_to_a_name_that_is_not_a_cache_blob_is_not_checked_as_one(tmp_path, name):
    # Bug caught: any 64-character or upper-case name taken for an LFS sha256 (a renamed local copy refused).
    root, layout = _tiny(tmp_path)
    _as_hf_blob(root, name)
    assert not open_checkpoint(root, layouts=(layout,)).config_source.endswith("matches")


def test_a_config_less_file_with_an_unknown_header_is_refused_naming_its_digest(tmp_path):
    # Bug caught (Review Focus 1): an unknown file read with a guessed order, or a refusal that cannot be acted on.
    root, _layout = _tiny(tmp_path)
    digest = hashlib.sha256(read_header_bytes(root / "model.safetensors")[0]).hexdigest()
    with pytest.raises(DFloatFormatError) as err:
        open_checkpoint(root)  # the default table holds only published files, never this tiny one
    # Bug caught too: a refusal that sends users to a maintainer script, or prints the caller's full path.
    assert str(err.value) == (
        f"model.safetensors: this config-less file is not one this version of mlx-dfloat knows "
        f"(header sha256 {digest}). Use a copy with a config.json that has a dfloat11_config "
        "block, or open an issue at https://github.com/IonDen/mlx-dfloat/issues with the repo, "
        "file name and sha256."
    )


def test_a_flipped_byte_at_the_seam_is_refused_by_the_content_probe(tmp_path):
    # Bug caught: the header pinning the layout but not the content (a re-saved file with new bytes and the same
    # header read as verified).
    root, layout = _tiny(tmp_path)
    file = root / "model.safetensors"
    raw, start, size = read_header_bytes(file)
    seam = (
        parse_header(raw, data_start=start, file_size=size, source="t")[
            "blocks.0.sign_mantissa"
        ].offset
        + TINY_SEAM
    )
    data = bytearray(file.read_bytes())
    data[seam] ^= 0x01
    file.write_bytes(bytes(data))
    seen = hashlib.sha256(data[seam : seam + 8]).hexdigest()  # the tiny spot check reads 8 bytes
    with pytest.raises(DFloatFormatError) as err:
        open_checkpoint(root, layouts=(layout,))
    assert str(err.value) == (
        "model.safetensors: header matches the known layout tiny, but the stored data differs from "
        f"the copy it was checked against (sha256 {seen} at byte {TINY_SEAM} of "
        "blocks.0.sign_mantissa); re-download the file, or report it at "
        "https://github.com/IonDen/mlx-dfloat/issues."
    )


@pytest.mark.parametrize(
    ("probes", "match"),
    [
        ((), "^model\\.safetensors: layout tiny .* pins no spot check"),
        ((ContentProbe(tensor="nope", offset=0, length=8, sha256="0" * 64),), "nope"),
        (
            (ContentProbe(tensor="blocks.0.sign_mantissa", offset=60, length=8, sha256="0" * 64),),
            "outside",
        ),
        (
            (ContentProbe(tensor="blocks.0.sign_mantissa", offset=0, length=0, sha256="0" * 64),),
            "outside",
        ),
        (
            (ContentProbe(tensor="blocks.0.sign_mantissa", offset=-1, length=8, sha256="0" * 64),),
            "outside",
        ),
    ],
)
def test_a_layout_without_a_usable_probe_is_refused(tmp_path, probes, match):
    # Bug caught: a layout pinned by its header alone, or a probe outside the tensor read from its neighbour.
    # The tiny sign_mantissa holds 64 bytes, so [60, 68) runs past its end and [-1, 7) starts before it.
    root, layout = _tiny(tmp_path)
    with pytest.raises(DFloatFormatError, match=match):
        open_checkpoint(root, layouts=(dataclasses.replace(layout, probes=probes),))


def test_a_spot_check_ending_on_the_tensors_last_byte_is_read(tmp_path):
    # Bug caught: an off-by-one bound (offset + length < nbytes) that refuses a spot check of a tensor's last bytes.
    root, layout = _tiny(tmp_path)
    last = (_probe(root, "blocks.0.sign_mantissa", 56, 8),)  # bytes [56, 64) of 64
    ckpt = open_checkpoint(root, layouts=(dataclasses.replace(layout, probes=last),))
    assert ckpt.config_source.startswith("header and spot checks match layout tiny")


def test_a_present_config_json_wins_and_no_layout_is_consulted(tmp_path):
    # Bug caught: a layout applied over a real config (FLUX.1 / Z-Image / Klein reads would change).
    root, layout = _tiny(tmp_path, write_config=True)  # same header bytes, config.json beside them
    ckpt = open_checkpoint(root, layouts=(layout,))
    assert ckpt.groups["blocks.0"].matrix_names == ("blocks.0.q.weight", "blocks.0.gate_up.weight")
    assert ckpt.groups["blocks.0"].row_plan == ()
    assert ckpt.config_source == "config.json"


def test_a_config_json_without_a_df11_block_keeps_its_own_error(tmp_path):
    # Bug caught: a diffusers config.json beside one shard falling through to the layout table (a BF16 or foreign
    # file reported as "matches no known layout", or worse, read through one).
    root, layout = _tiny(tmp_path)
    (root / "config.json").write_text(json.dumps({"_class_name": "QwenImageTransformer2DModel"}))
    with pytest.raises(DFloatFormatError, match="no dfloat11_config block"):
        open_checkpoint(root, layouts=(layout,))


def test_a_legacy_pickle_checkpoint_without_a_config_keeps_its_own_error(tmp_path):
    # Bug caught: the config-less branch swallowing the legacy refusal (a .pkl repo told "no known layout").
    root, layout = _tiny(tmp_path)
    (root / "model.pkl").write_bytes(b"x")
    with pytest.raises(DFloatFormatError, match="legacy pickle-format"):
        open_checkpoint(root, layouts=(layout,))


def test_a_layout_whose_counts_disagree_is_refused(tmp_path):
    # Bug caught: a group dropped from a file that otherwise matches (the counts are the layout's own check).
    root, layout = _tiny(tmp_path)
    with pytest.raises(
        DFloatFormatError,
        match=r"^model\.safetensors: 1 groups and 1 extras; layout tiny expects 2 and 1",
    ):
        open_checkpoint(root, layouts=(dataclasses.replace(layout, groups=2),))
    with pytest.raises(
        DFloatFormatError, match=r"1 groups and 1 extras; layout tiny expects 1 and 2"
    ):
        open_checkpoint(root, layouts=(dataclasses.replace(layout, extras=2),))


@pytest.mark.parametrize("n_files", [0, 2])
def test_anything_but_one_safetensors_file_without_a_config_is_refused(tmp_path, n_files):
    # Bug caught: a sharded config-less directory read as whichever file sorts first, or an empty one crashing.
    root, layout = _tiny(tmp_path)
    if n_files == 2:
        (root / "other.safetensors").write_bytes((root / "model.safetensors").read_bytes())
    else:
        (root / "model.safetensors").rename(root / "model.bak")
    with pytest.raises(
        DFloatFormatError,
        match=rf"^ckpt: no config\.json and {n_files} safetensors files.*single file",
    ):
        open_checkpoint(root, layouts=(layout,))


def test_a_config_less_directory_named_like_a_shard_is_refused(tmp_path):
    # Bug caught: the config-less path skipping the regular-file check the config.json path makes.
    root = tmp_path / "ckpt"
    (root / "model.safetensors").mkdir(parents=True)
    with pytest.raises(DFloatFormatError, match="not a regular file"):
        open_checkpoint(root, layouts=())


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_path_that_is_not_a_directory_is_refused_by_name(tmp_path, kind):
    # Bug caught: a typo'd --df11 path reported as a config-less file problem ("0 safetensors files"), or the
    # message printing the caller's full path.
    target = tmp_path / "typo"
    if kind == "file":
        target.write_bytes(b"x")
    with pytest.raises(DFloatFormatError, match=r"^typo: not a directory$"):
        open_checkpoint(target, layouts=())


@pytest.mark.parametrize("kind", ["broken symlink", "directory"])
def test_a_broken_config_json_is_refused_not_read_as_config_less(tmp_path, kind):
    # Bug caught: a config.json that exists but cannot be read (a dangling HF cache link, a directory) treated as
    # absent, so the file beside it is matched against the layout table instead of its own config.
    root, layout = _tiny(tmp_path)
    if kind == "directory":
        (root / "config.json").mkdir()
    else:
        (root / "config.json").symlink_to(tmp_path / "gone.json")
    with pytest.raises(
        DFloatFormatError, match=r"^ckpt/config\.json: exists but is not a readable regular file$"
    ):
        open_checkpoint(root, layouts=(layout,))


def test_the_default_table_is_read_at_call_time(tmp_path, monkeypatch):
    # Bug caught: KNOWN_LAYOUTS bound at import (tests and scripts could not supply their own).
    root, layout = _tiny(tmp_path)
    monkeypatch.setattr(_layouts, "KNOWN_LAYOUTS", (layout,))
    assert open_checkpoint(root).config_source.startswith(
        "header and spot checks match layout tiny"
    )


def _sparse_qwen(tmp_path) -> Path:
    """The published header at full file size; every data byte reads as zero (a sparse file, ~28 KB on disk).

    This relies on a filesystem with sparse files (APFS, ext4, the CI runners' disks). On one without
    them, ``truncate`` writes the full 9.7 GB of zeros.
    """
    root = tmp_path / "qwen"
    root.mkdir()
    with (root / QWEN_FILE).open("wb") as fh:
        fh.write(FIXTURE.read_bytes())
        fh.truncate(QWEN_FILE_SIZE)
    return root


def test_the_published_header_opens_with_seven_names_per_block_when_the_content_matches(tmp_path):
    # Bug caught (end to end through open_checkpoint): the real layout's counts, names or source wrong. The probes
    # are swapped for the digest of a zero page because the sparse file's data reads as zeros.
    root = _sparse_qwen(tmp_path)
    zeros = tuple(
        dataclasses.replace(p, sha256=ZERO_PAGE_SHA256) for p in QWEN_IMAGE_21_COMFYUI.probes
    )
    ckpt = open_checkpoint(
        root, layouts=(dataclasses.replace(QWEN_IMAGE_21_COMFYUI, probes=zeros),)
    )
    assert len(ckpt.groups) == 33
    assert len(ckpt.extras) == 72
    assert ckpt.groups["transformer_blocks.0"].matrix_names == tuple(
        f"transformer_blocks.0.{s}.weight" for s in BLOCK_SUBS
    )
    assert ckpt.config_source == (
        "header and spot checks match layout qwen-image-2.1-comfyui "
        f"(mingyi456/Qwen-Image-2.1-DF11-ComfyUI@{QWEN_REVISION})"
    )


def test_the_published_header_with_other_content_is_refused(tmp_path):
    # Bug caught: the default table's Qwen layout read without its content probes (the real refusal path).
    with pytest.raises(
        DFloatFormatError,
        match=rf"differs .*\(sha256 {ZERO_PAGE_SHA256} at byte 0 of transformer_blocks\.0\.sign_mantissa\)",
    ):
        open_checkpoint(_sparse_qwen(tmp_path))


@pytest.mark.network
def test_the_pinned_header_and_probes_match_the_published_file():
    # Bug caught: a pin that was never the published file's (a typo in a digest refuses every real download).
    # Reads the header, the file size and LFS sha256 and the nine 4 KiB spot checks; nothing else.
    from scripts.verify_remote_group import HfRangeSource

    src = HfRangeSource(QWEN_IMAGE_21_COMFYUI.repo_id, QWEN_IMAGE_21_COMFYUI.revision)
    assert src.size(QWEN_FILE) == 9_724_491_411
    from huggingface_hub import HfApi

    (entry,) = HfApi().get_paths_info(
        QWEN_IMAGE_21_COMFYUI.repo_id, [QWEN_FILE], revision=QWEN_IMAGE_21_COMFYUI.revision
    )
    assert entry.lfs is not None
    assert entry.lfs.sha256 == QWEN_FILE_SHA256 == QWEN_IMAGE_21_COMFYUI.file_sha256
    assert src.read(QWEN_FILE, 0, 28_416) == FIXTURE.read_bytes()
    infos = parse_header(
        _fixture_header(), data_start=28_416, file_size=QWEN_FILE_SIZE, source="qwen"
    )
    for probe in QWEN_IMAGE_21_COMFYUI.probes:
        start = infos[probe.tensor].offset + probe.offset
        assert hashlib.sha256(src.read(QWEN_FILE, start, probe.length)).hexdigest() == probe.sha256
