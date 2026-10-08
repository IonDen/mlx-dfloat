"""Krea 2 Raw and Turbo as exported for ComfyUI: config-less single files read through pinned layouts.

The fixtures are the first 35,976 bytes of each published file, exactly as on disk: the u64
little-endian header length (35,968) followed by the JSON header. Both were range-read on
2026-10-08 (file bytes 0..35975):

- ``krea-2-raw-df11-header.bin``: ``krea2_raw_bf16-DF11.safetensors`` in
  ``mingyi456/Krea-2-Raw-DF11-ComfyUI`` at revision ``8320616b25ac9340a830a7fb21f1b0237e160e66``
  (fixture sha256 ``8474548f32edddf08ae977bc0d8cb473a78154933b8d0fd5a3b29faeeefd99bf``).
- ``krea-2-turbo-df11-header.bin``: ``krea2_turbo_bf16-DF11.safetensors`` in
  ``mingyi456/Krea-2-Turbo-DF11-ComfyUI`` at revision ``978da5fb7647bd222d33125993abd8fdc2840cfc``
  (fixture sha256 ``0a45f2ab4d8c1d1415f85091024ba844265076748a3f0cbe63be5bcc50569e48``).

The spot-check digests were range-read at the same revisions on 2026-10-08: the sha256 of 4096
bytes of ``sign_mantissa`` (one byte per element), ``offset`` bytes into the tensor. Data starts at
file byte 35,976; ``blocks.0.sign_mantissa`` sits at data offset 172,776,230 (Raw) and 172,729,116
(Turbo). The offsets are the stored split points: every matrix start of ``blocks.0`` (``attn.wq``,
``wk``, ``wv``, ``gate``, ``wo``, ``mlp.gate``, ``up``, ``down``), ``blocks.27``'s first matrix,
every matrix start of ``txtfusion.layerwise_blocks.0``, the first matrix of
``txtfusion.layerwise_blocks.1``, ``txtfusion.refiner_blocks.0`` and ``txtfusion.refiner_blocks.1``,
``tmlp.0``/``tmlp.2``, ``tproj.1``, ``txtmlp.1``/``txtmlp.3``. The two first matrices of
``layerwise_blocks.1`` and ``refiner_blocks.0`` were range-read later the same day, at the same revisions. A byte
comparison of the same date matched 4096
sign/mantissa bytes at every matrix start of blocks 0, 13 and 27 and of all four text-fusion
blocks against the originals (``krea/Krea-2-Raw`` @ ``6b0ece7f…``, ``krea/Krea-2-Turbo`` @
``98e0fe11…``): 4096 of 4096 for the expected tensor, at most 32 for any other, in mflux's module
registration order.
"""

import dataclasses
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from tests._df11_fixtures import pin_layout, random_bf16, write_checkpoint

from mlx_dfloat._layouts import (
    KNOWN_LAYOUTS,
    KREA_2_RAW_COMFYUI,
    KREA_2_TURBO_COMFYUI,
    identify_layout,
)
from mlx_dfloat._safetensors import parse_header, read_header_bytes
from mlx_dfloat.errors import DFloatFormatError
from mlx_dfloat.format import config_for_layout, group_headers, open_checkpoint

FIX = Path(__file__).parent / "fixtures"
RAW = FIX / "krea-2-raw-df11-header.bin"
TURBO = FIX / "krea-2-turbo-df11-header.bin"
RAW_HEADER_SHA256 = "a0eb86513ccf0ba8baca0625147b62109707650b480046cd66043ea9a8700578"
TURBO_HEADER_SHA256 = "1de7b6f3f0dc4c04a05c3bb1e7972f0d3761902ea2ddd42916f14183aecbcafd"
RAW_FILE_SHA256 = (
    "f62cf31921ad0377f9d6e5bd2e3191a1ac9950d92f487b26fa7c052f503d9b97"  # the Hub's LFS oid
)
TURBO_FILE_SHA256 = "ccfd85551ba0f37e7f6706d86d1c00276457d2965ac72fba6fef046653ad6498"
RAW_FILE_SIZE = 17_545_490_584
TURBO_FILE_SIZE = 17_544_635_322
BLOCK = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)  # the order the byte comparison confirmed (module docstring)
RAW_PROBES = (  # (tensor, byte offset into its data, sha256 of 4096 bytes), see the module docstring
    (
        "blocks.0.sign_mantissa",
        0,
        "363d5ac87fa64889de9ef810e04d0b1e905ceb89fee078fa0e589c254298e35f",
    ),
    (
        "blocks.0.sign_mantissa",
        37_748_736,
        "ca6dd074e2d1e37891e94e867dfacc6ddb5c19f878f78a60f56e1e2f6f5e06bd",
    ),
    (
        "blocks.0.sign_mantissa",
        47_185_920,
        "6480a299d38e684bdb3a8d498752ed62afda615de694aab4b69a16a974ba518c",
    ),
    (
        "blocks.0.sign_mantissa",
        56_623_104,
        "3962f4172d9e4007c709729c5d262c1105748e472c17d714ef41ced9fd2aa40f",
    ),
    (
        "blocks.0.sign_mantissa",
        94_371_840,
        "585411a597165de162951bdd1d68cef4fb03a434caf433d33d3f60e7625d1ffe",
    ),
    (
        "blocks.0.sign_mantissa",
        132_120_576,
        "2d41ddb871e747add117545678a7097e19e9901744434de7c54b885cfa049b10",
    ),
    (
        "blocks.0.sign_mantissa",
        232_783_872,
        "01bfce386ced98d20b1bceec8ccf51ec586036c730125c9c3ef4ea76764d19a0",
    ),
    (
        "blocks.0.sign_mantissa",
        333_447_168,
        "9489cb4060011bc2edca69148499606df538751f961e35dc7ba3c2d90954b3b5",
    ),
    (
        "blocks.27.sign_mantissa",
        0,
        "d011e6504676d98bde3696da028b03d8f6cbc7d4ad5ece7e67afda38c2cc22ad",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        0,
        "edcc578d390f9a375a43660de265524fba96f8cdfb7422c2260fe7ae3c81bf95",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        6_553_600,
        "c2f20aaf9f3e23b66e4d74e26a4239e75b2409fb4baffb875344084ae8cd1920",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        13_107_200,
        "cf78e88a47707eab0acab1561a31b3c7fc75174d7ee1c872bb1a5558cf2c3d6a",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        19_660_800,
        "92fedf37baba3c13617ec9bacb6017aafacf5e61996791edb92720ebd1223cb8",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        26_214_400,
        "c74085fa9516ac66243f038846454e9f4911b5ee54891e3d6e634b27cbdacb49",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        32_768_000,
        "9b765f28f96af426c2c3ca7d48d08c11b38aab7b02c010dde4935a45fd421d55",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        50_462_720,
        "6c12c5521469efccaea92c86cf1232bf69cd7097033c23c77ea36d6899141a28",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        68_157_440,
        "5998851cd52398972b745d340839d779e5f45ba4189c08b2cd3d0493fa36e393",
    ),
    (
        "txtfusion.layerwise_blocks.1.sign_mantissa",
        0,
        "4b0fd5b9e0ea3356bc8d71b3ef3008bd4cdee28e485c89c67188d533604885db",
    ),
    (
        "txtfusion.refiner_blocks.0.sign_mantissa",
        0,
        "7470ae9e5215bb3bd6f2dc75620cc6935e4d7a957ac05358838f6b851a01dff8",
    ),
    (
        "txtfusion.refiner_blocks.1.sign_mantissa",
        0,
        "a7dd2de120a3a713fe6eb782ad926c36d2718498c739ee3baac4cd5e26577214",
    ),
    ("tmlp.sign_mantissa", 0, "356d0f90c117d207bdef0a26d602b4ca0a36a52f1e4d2588849e2dff3df4f6aa"),
    (
        "tmlp.sign_mantissa",
        1_572_864,
        "305bc2e99d6161e4674a8f2a222c25e958f9b2b2d9efb36f5db799a3a0a3e4d3",
    ),
    ("tproj.sign_mantissa", 0, "9854423ef01b650b06113ec959298735bcf4cc98d79a5b628873ed2b20a39b9d"),
    ("txtmlp.sign_mantissa", 0, "86beee24cdec56afaf7d367735d53153aed1a00956955a5aff8d1abc2b9d9182"),
    (
        "txtmlp.sign_mantissa",
        15_728_640,
        "ca3c3a6ee9e11db6c5b419f351187e46505403fdd230c3fdade6f3377527a48c",
    ),
)
TURBO_PROBES = (  # (tensor, byte offset into its data, sha256 of 4096 bytes), see the module docstring
    (
        "blocks.0.sign_mantissa",
        0,
        "11f6150d52c2100d4e46f8a8962a8b1a6acf2c38f477cbe9e113643b095caef8",
    ),
    (
        "blocks.0.sign_mantissa",
        37_748_736,
        "7dc9c9a7f947dd621467761eba193174a35913e35c09ea273109ce1c7d1995e0",
    ),
    (
        "blocks.0.sign_mantissa",
        47_185_920,
        "c8fd1bc47699725e3a1cfabaddd2cdea7e563a01bf1b0ffa10617f8456ba6f0a",
    ),
    (
        "blocks.0.sign_mantissa",
        56_623_104,
        "35e55744d6156352bc36f3b5e2d716679d4d2f8c6dc7c6ee5795d4deec7432d0",
    ),
    (
        "blocks.0.sign_mantissa",
        94_371_840,
        "f905a0181ece0d857a95cbe554bcf98968b847ccebacb332b89755f6545d0dd3",
    ),
    (
        "blocks.0.sign_mantissa",
        132_120_576,
        "171a5c675fb5bd9aff19ec75f77eedfef7dc17e6861568e2df3144dec58caaf3",
    ),
    (
        "blocks.0.sign_mantissa",
        232_783_872,
        "356076a9cd44ddcd624da9336dd56b0dc233ce3bea51e473380d739605292747",
    ),
    (
        "blocks.0.sign_mantissa",
        333_447_168,
        "7ccae2ff710cc0dbf818ca8fe768066e79b2a64881f8fb514dedcb2307eafc50",
    ),
    (
        "blocks.27.sign_mantissa",
        0,
        "9a865f96cbb3415c0d2fa6cbf04aac8e72cd914a4da7b30dfa346f48ae2f68b5",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        0,
        "8f836605424820e44b68ccd56399e183aa990ff0d3da3cec22250db6145cb3c5",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        6_553_600,
        "9a074d7d85a38cf157419326fe09f517982269097591597ffd5cf4b25438dc04",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        13_107_200,
        "2975a4ffd78330058c286a9c07d59db4f30fb30da0449cd19f770c96ca018029",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        19_660_800,
        "fe5976eb985efce2624bdfee10e3362074953a0197c4450dda7799314c905586",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        26_214_400,
        "6978c0ed44c394128072eb7671cc902112181f2f18a70a84e80d0b5a5aeed80a",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        32_768_000,
        "6a8af83a863507b9de8683470115894dea0dd302fe507e4f7f976b3034204bb0",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        50_462_720,
        "3d7417959035994ad92f9036b3636b1b35e821f438468388dc4306a6221cee4e",
    ),
    (
        "txtfusion.layerwise_blocks.0.sign_mantissa",
        68_157_440,
        "ff857b14597bfcf543d7938d71992f696177b52ee3a7574f63719a9a42dac94b",
    ),
    (
        "txtfusion.layerwise_blocks.1.sign_mantissa",
        0,
        "06d9ba6294dbc0509ab38954d11052a3d0cdc1df21aff4f5fdc1a0d30269f02f",
    ),
    (
        "txtfusion.refiner_blocks.0.sign_mantissa",
        0,
        "200e8501fc64ada5d37bf10f2d2653f273367019075f3607f58b0cb54f7c2da2",
    ),
    (
        "txtfusion.refiner_blocks.1.sign_mantissa",
        0,
        "241ff415ea140b6838ea709c1317c9d5da6129231f23ca8dbdef946b7bf056c9",
    ),
    ("tmlp.sign_mantissa", 0, "ee752a8f3b403449d3a1e95666369eba78aec410bf7879663215b58a54efac98"),
    (
        "tmlp.sign_mantissa",
        1_572_864,
        "1b2aa81eca1c9b20b9f6b96018f53833acc5f9e9617bd37432bcd4920f6b6963",
    ),
    ("tproj.sign_mantissa", 0, "52dd92bbc434a9120d8038e8637bf5b350d69aba1d508aebaeb20da2d9b57dfe"),
    ("txtmlp.sign_mantissa", 0, "72862c0bad66d0664fda6c8f8db121ceca7aafcac721268e6208d6a8fbf01f69"),
    (
        "txtmlp.sign_mantissa",
        15_728_640,
        "04724bf4d4567d5f72bb6e0e8e95a9e0560d9bd405ec19399f9b3aa84adfc2ba",
    ),
)
ZERO_PAGE_SHA256 = hashlib.sha256(bytes(4096)).hexdigest()
LAYOUTS = [
    (KREA_2_RAW_COMFYUI, RAW, RAW_FILE_SIZE, RAW_PROBES),
    (KREA_2_TURBO_COMFYUI, TURBO, TURBO_FILE_SIZE, TURBO_PROBES),
]


def _groups(path, size):
    data = path.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    infos = parse_header(data[8 : 8 + n], data_start=8 + n, file_size=size, source=path.name)
    layout = identify_layout(data[8 : 8 + n], known=KNOWN_LAYOUTS)
    assert layout is not None
    return group_headers({path: infos}, config_for_layout(layout))


@pytest.mark.parametrize(
    ("path", "digest"), [(RAW, RAW_HEADER_SHA256), (TURBO, TURBO_HEADER_SHA256)]
)
def test_the_fixtures_are_the_published_headers_as_on_disk(path, digest):
    # Bug caught: a fixture re-serialised or truncated, so the pins below test another file.
    data = path.read_bytes()
    assert len(data) == 35_976
    assert struct.unpack("<Q", data[:8]) == (35_968,)
    assert hashlib.sha256(data[8:]).hexdigest() == digest


def test_each_published_header_selects_its_own_layout():
    # Bug caught: the two pins swapped (the Turbo file read as Raw), or a pin that is not the published header.
    assert identify_layout(RAW.read_bytes()[8:], known=KNOWN_LAYOUTS) is KREA_2_RAW_COMFYUI
    assert identify_layout(TURBO.read_bytes()[8:], known=KNOWN_LAYOUTS) is KREA_2_TURBO_COMFYUI


@pytest.mark.parametrize("path", [RAW, TURBO])
def test_one_changed_header_byte_matches_no_krea_layout(path):
    # Bug caught: matching on the file name or size (a re-upload with another matrix order read with this order).
    raw = bytearray(path.read_bytes()[8:])
    raw[200] ^= 0x01
    assert identify_layout(bytes(raw), known=KNOWN_LAYOUTS) is None


@pytest.mark.parametrize(("path", "size"), [(RAW, RAW_FILE_SIZE), (TURBO, TURBO_FILE_SIZE)])
def test_the_header_groups_into_35_groups_with_the_confirmed_matrix_names(path, size):
    # Bug caught: a pattern that misses the text-fusion groups (they would become extras), `tmlp` matching `txtmlp`,
    # or the block order not the one the byte comparison confirmed.
    groups, extras = _groups(path, size)
    assert sorted(groups) == sorted(
        [f"blocks.{i}" for i in range(28)]
        + [f"txtfusion.{k}.{i}" for k in ("layerwise_blocks", "refiner_blocks") for i in (0, 1)]
        + ["tmlp", "tproj", "txtmlp"]
    )
    assert len(extras) == 169
    assert {info.dtype for _p, info in extras.values()} == {"BF16"}
    assert sum(info.nbytes for _p, info in extras.values()) == 4_560_024
    assert groups["blocks.7"].matrix_names == tuple(f"blocks.7.{s}.weight" for s in BLOCK)
    assert groups["txtfusion.layerwise_blocks.0"].matrix_names == tuple(
        f"txtfusion.layerwise_blocks.0.{s}.weight" for s in BLOCK
    )
    assert groups["txtfusion.refiner_blocks.1"].matrix_names == tuple(
        f"txtfusion.refiner_blocks.1.{s}.weight" for s in BLOCK
    )
    assert groups["tmlp"].matrix_names == ("tmlp.0.weight", "tmlp.2.weight")
    assert groups["tproj"].matrix_names == ("tproj.1.weight",)
    assert groups["txtmlp"].matrix_names == ("txtmlp.1.weight", "txtmlp.3.weight")
    assert all(g.row_plan == () for g in groups.values())  # no fused matrices in Krea 2


@pytest.mark.parametrize(
    ("layout", "repo", "revision", "file_name", "header", "lfs"),
    [
        (
            KREA_2_RAW_COMFYUI,
            "mingyi456/Krea-2-Raw-DF11-ComfyUI",
            "8320616b25ac9340a830a7fb21f1b0237e160e66",
            "krea2_raw_bf16-DF11.safetensors",
            RAW_HEADER_SHA256,
            RAW_FILE_SHA256,
        ),
        (
            KREA_2_TURBO_COMFYUI,
            "mingyi456/Krea-2-Turbo-DF11-ComfyUI",
            "978da5fb7647bd222d33125993abd8fdc2840cfc",
            "krea2_turbo_bf16-DF11.safetensors",
            TURBO_HEADER_SHA256,
            TURBO_FILE_SHA256,
        ),
    ],
)
def test_the_krea_layouts_pin_the_published_files_and_config(
    layout, repo, revision, file_name, header, lfs
):
    # Bug caught: a revision, file name or LFS digest copied wrong (a classic-cache blob of the right file refused),
    # or a typo in the pattern table (one matrix decoded under the wrong name).
    assert (layout.repo_id, layout.revision, layout.file_name) == (repo, revision, file_name)
    assert (layout.header_sha256, layout.file_sha256, layout.groups, layout.extras) == (
        header,
        lfs,
        35,
        169,
    )
    assert dict(layout.row_splits) == {}
    block = list(BLOCK)
    assert json.loads(json.dumps(layout.raw_config)) == {
        "version": "0.3.1",
        "threads_per_block": [512],
        "bytes_per_thread": 8,
        "pattern_dict": {
            r"blocks\.\d+": block,
            r"txtfusion\.layerwise_blocks\.\d+": block,
            r"txtfusion\.refiner_blocks\.\d+": block,
            "tmlp": ["0", "2"],
            "tproj": ["1"],
            "txtmlp": ["1", "3"],
        },
    }


@pytest.mark.parametrize(("layout", "_path", "_size", "probes"), LAYOUTS)
def test_the_krea_layouts_spot_check_every_matrix_start_of_block_0_and_text_fusion_0(
    layout, _path, _size, probes
):
    # Bug caught: a start left unprobed (or a digest pinned at another offset), so a re-export permuting two
    # equal-shaped matrices keeps the header and passes.
    got = tuple((p.tensor, p.offset, p.length, p.sha256) for p in layout.probes)
    assert got == tuple((t, off, 4096, digest) for t, off, digest in probes)


@pytest.mark.parametrize("layout", [KREA_2_RAW_COMFYUI, KREA_2_TURBO_COMFYUI])
def test_every_probe_digest_is_distinct(layout):
    # Bug caught: two probes pinned to one copy-pasted digest (one matrix start then checks nothing new).
    assert len({p.sha256 for p in layout.probes}) == len(layout.probes) == 25


# Two tiny blocks under the Krea pattern table: wq 4x4, wk/wv 2x4, gate/wo 4x4, mlp.gate/up 8x4, down 4x8 (elements
# 16 + 8 + 8 = 32 before attn.gate, so its start is byte 32 of the block's sign_mantissa).
TINY_SHAPES = [(4, 4), (2, 4), (2, 4), (4, 4), (4, 4), (8, 4), (8, 4), (4, 8)]
TINY_GATE_START = 32


def _tiny_krea(tmp_path):
    rng = np.random.default_rng(11)
    pattern_dict = KREA_2_RAW_COMFYUI.raw_config["pattern_dict"]
    root = write_checkpoint(
        tmp_path / "ckpt",
        groups={f"blocks.{b}": [random_bf16(rng, s) for s in TINY_SHAPES] for b in (0, 1)},
        patterns=pattern_dict,
        extras={"first.bias": random_bf16(rng, (4,))},
        single_file=True,
        write_config=False,
    )
    layout = pin_layout(
        root / "model.safetensors",
        pattern_dict=pattern_dict,
        row_splits={},
        groups=2,
        extras=1,
        key="tiny-krea",
        probe_tensor="blocks.0.sign_mantissa",
        probe_offset=TINY_GATE_START,
    )
    return root, layout


def test_a_tiny_file_pinned_like_the_krea_layout_opens_and_a_flipped_probe_byte_is_refused(
    tmp_path,
):
    # Bug caught: the Krea pattern table not usable by the reader (a grammar the config parser refuses, or names out
    # of order), or the content probes not read for a config-less file (a changed matrix start read as verified).
    root, layout = _tiny_krea(tmp_path)
    ckpt = open_checkpoint(root, layouts=(layout,))
    assert ckpt.groups["blocks.1"].matrix_names == tuple(f"blocks.1.{s}.weight" for s in BLOCK)
    file = root / "model.safetensors"
    raw, start, size = read_header_bytes(file)
    at = parse_header(raw, data_start=start, file_size=size, source="t")[
        "blocks.0.sign_mantissa"
    ].offset
    data = bytearray(file.read_bytes())
    data[at + TINY_GATE_START] ^= 0x01
    file.write_bytes(bytes(data))
    with pytest.raises(
        DFloatFormatError, match=r"differs .* at byte 32 of blocks\.0\.sign_mantissa\)"
    ):
        open_checkpoint(root, layouts=(layout,))


def _sparse(tmp_path, path, size) -> Path:
    """The published header at full file size; every data byte reads as zero (a sparse file, ~36 KB on disk)."""
    root = tmp_path / path.stem
    root.mkdir()
    with (root / "model.safetensors").open("wb") as fh:
        fh.write(path.read_bytes())
        fh.truncate(size)
    return root


@pytest.mark.parametrize(("layout", "path", "size", "_probes"), LAYOUTS)
def test_the_published_header_opens_from_the_default_table_when_the_content_matches(
    tmp_path, monkeypatch, layout, path, size, _probes
):
    # Bug caught (end to end through open_checkpoint and the default table): a Krea layout missing from
    # KNOWN_LAYOUTS, or its counts, names or source wrong. In the table, the layout's probes are swapped for the
    # digest of a zero page because the sparse file reads as zeros; a layout absent from the table swaps nothing, and
    # the open then fails on the real probes.
    from mlx_dfloat import _layouts

    zeros = tuple(dataclasses.replace(p, sha256=ZERO_PAGE_SHA256) for p in layout.probes)
    table = tuple(
        dataclasses.replace(known, probes=zeros) if known.key == layout.key else known
        for known in _layouts.KNOWN_LAYOUTS
    )
    monkeypatch.setattr(_layouts, "KNOWN_LAYOUTS", table)
    ckpt = open_checkpoint(_sparse(tmp_path, path, size))
    assert (len(ckpt.groups), len(ckpt.extras)) == (35, 169)
    assert ckpt.groups["blocks.27"].matrix_names == tuple(f"blocks.27.{s}.weight" for s in BLOCK)
    assert ckpt.config_source == (
        f"header and spot checks match layout {layout.key} ({layout.repo_id}@{layout.revision})"
    )


@pytest.mark.parametrize(("_layout", "path", "size", "_probes"), LAYOUTS)
def test_the_published_header_with_other_content_is_refused_by_the_default_table(
    tmp_path, _layout, path, size, _probes
):
    # Bug caught: the default table's Krea layout read without its content probes (the real refusal path).
    with pytest.raises(
        DFloatFormatError,
        match=rf"differs .*\(sha256 {ZERO_PAGE_SHA256} at byte 0 of blocks\.0\.sign_mantissa\)",
    ):
        open_checkpoint(_sparse(tmp_path, path, size))


@pytest.mark.network
@pytest.mark.parametrize(("layout", "path", "size", "_probes"), LAYOUTS)
def test_the_pinned_header_size_lfs_and_probes_match_the_published_file(
    layout, path, size, _probes
):
    # Bug caught: a digest copied wrong from the range reads, or a pin of a revision other than the one recorded.
    # Reads the header, the 25 probe ranges (100 KiB) and the Hub's size + LFS oid; no download.
    from huggingface_hub import HfApi
    from scripts.verify_remote_group import HfRangeSource

    src = HfRangeSource(layout.repo_id, layout.revision)
    assert src.size(layout.file_name) == size
    (entry,) = HfApi().get_paths_info(layout.repo_id, [layout.file_name], revision=layout.revision)
    assert entry.lfs is not None
    assert entry.lfs.sha256 == layout.file_sha256
    assert src.read(layout.file_name, 0, 35_976) == path.read_bytes()
    infos = parse_header(path.read_bytes()[8:], data_start=35_976, file_size=size, source=path.name)
    for probe in layout.probes:
        start = infos[probe.tensor].offset + probe.offset
        assert (
            hashlib.sha256(src.read(layout.file_name, start, probe.length)).hexdigest()
            == probe.sha256
        )
