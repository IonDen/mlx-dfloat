"""Pinned layouts for DFloat11 checkpoints published as one file without a ``config.json``.

Some DFloat11 files (ComfyUI single-file exports) carry no ``dfloat11_config``. Their matrix order
cannot be read from tensor shapes alone, because several matrices share a shape. A layout here
supplies the missing config for one published file, after its order was checked against the BF16
base model's bytes. It is matched by the sha256 of the raw safetensors header bytes and confirmed
by a few sha256 spot checks of stored bytes, at offsets chosen from the BF16 base. The header hash
pins the layout; the spot checks pin the stored data at each matrix start of the first block (and
a few more offsets), so a re-export that keeps the header but stores the matrices in another order
is refused rather than read with this one.

Standard library only: this table is read before any MLX or NumPy work starts.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentProbe:
    """A spot check: the sha256 of ``length`` bytes of one stored tensor, ``offset`` bytes into its data."""

    tensor: str
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SynthesizedLayout:
    """A ``dfloat11_config`` for one config-less file, pinned to that file's header and content.

    Attributes:
        key: Short identifier, used in messages and ``DF11Checkpoint.config_source``.
        label: Human-readable model name.
        repo_id: The Hugging Face repository the file was verified from.
        revision: The full commit of that repository.
        file_name: The file's name in the repository.
        header_sha256: sha256 of the raw JSON header bytes (after the 8-byte length).
        file_sha256: sha256 of the whole file (its Git LFS object id on the Hub). The reader
            compares it with the file's name only when that is a blob in the classic per-repo
            cache (``models--*/blobs/<sha256>``, huggingface_hub before 2.0), which names blobs by
            this sha256; the 2.0 shared blob store does not. The reader does not hash the file.
        groups: Number of compressed groups the file holds.
        extras: Number of uncompressed tensors the file holds.
        raw_config: The ``dfloat11_config`` block the file lacks, as a JSON-shaped mapping.
        row_splits: Fused stored matrices and the names of their equal row blocks, in row order.
        probes: Spot checks of stored bytes; at least one is required.
    """

    key: str
    label: str
    repo_id: str
    revision: str
    file_name: str
    header_sha256: str
    file_sha256: str
    groups: int
    extras: int
    raw_config: Mapping[str, Any]
    row_splits: Mapping[str, tuple[str, ...]]
    probes: tuple[ContentProbe, ...]


# Qwen-Image 2.1 as exported for ComfyUI: one 9,724,491,411-byte file, 33 groups (modulation.1 and
# 32 transformer blocks) and 72 BF16 extras. Each block stores six matrices; the fifth,
# img_mlp.gate_up, is gate_layer and proj stacked by rows ([gate; up], ComfyUI's SwiGLU
# convention), 24576 x 4096. The order was confirmed by comparing 4096 sign/mantissa bytes at every
# matrix start and at the gate/up seam with the BF16 base model (Qwen/Qwen-Image-2.1) in blocks 0,
# 15 and 31: 4096 of 4096 equal for the expected tensor, at most 29 for any other. The spot checks
# below cover every matrix start of block 0 and the seam, block 31's first matrix and
# modulation.1. The file records no format version. The layout uses 0.3.1, the value other
# Qwen-Image DF11 checkpoints carry; the decoder does not read it for this file.
QWEN_IMAGE_21_COMFYUI = SynthesizedLayout(
    key="qwen-image-2.1-comfyui",
    label="Qwen-Image 2.1 (ComfyUI single-file DF11)",
    repo_id="mingyi456/Qwen-Image-2.1-DF11-ComfyUI",
    revision="1b22a3a1f96293f3b328d03abe22ab2e51cbd9cc",
    file_name="qwen_image_2.1_bf16-DF11.safetensors",
    header_sha256="fa5b70707d9d5e08d1c92e40dcdeedd64f72508f88ca942217c6fd419ff5ba5e",
    file_sha256="fc07ef38e3609b1a38dc8f6ec585ef9921a4d4b5e6f74b744899e4dea91b4e2e",
    groups=33,
    extras=72,
    raw_config={
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
    },
    row_splits={"img_mlp.gate_up": ("img_mlp.gate_layer", "img_mlp.proj")},
    probes=(
        ContentProbe(  # attn.to_q
            tensor="transformer_blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="f960e614c0abeee3fc6b4d345887288841b97f7e07e6e14ca2f0bcb3d4ec9a09",
        ),
        ContentProbe(  # attn.to_k
            tensor="transformer_blocks.0.sign_mantissa",
            offset=16_777_216,
            length=4096,
            sha256="9674d29df78034e5b53974310e5d72545004db3116e97d4f97c883a78b6d21cf",
        ),
        ContentProbe(  # attn.to_v
            tensor="transformer_blocks.0.sign_mantissa",
            offset=33_554_432,
            length=4096,
            sha256="7c9eb5ca6ba99dfd5cbe65a44d332492cc2b461a8be9ca1145253896eac28db6",
        ),
        ContentProbe(  # attn.to_out.0
            tensor="transformer_blocks.0.sign_mantissa",
            offset=50_331_648,
            length=4096,
            sha256="1a117be0dca8deb6ebedab5835f1612d91f02f9881697d8ee21e046ee34b94e8",
        ),
        ContentProbe(  # gate_up row 0: gate_layer
            tensor="transformer_blocks.0.sign_mantissa",
            offset=67_108_864,
            length=4096,
            sha256="89b64f185893b34b7dc1256194c5335789b4f8540d04cfc7fac60f91222d2e08",
        ),
        ContentProbe(  # gate_up row 12288: proj
            tensor="transformer_blocks.0.sign_mantissa",
            offset=117_440_512,
            length=4096,
            sha256="022e33a91b869d7570a3115e3d2a2414ae16deb595b7858cde4071cfbb81ea19",
        ),
        ContentProbe(  # img_mlp.out
            tensor="transformer_blocks.0.sign_mantissa",
            offset=167_772_160,
            length=4096,
            sha256="732127bdf5ddadce246331844f1312b61f2a335d6634779e1ee594e42d7ea437",
        ),
        ContentProbe(  # block 31 attn.to_q
            tensor="transformer_blocks.31.sign_mantissa",
            offset=0,
            length=4096,
            sha256="c4fb1affbfaa3f620b087719a00f0d46d25a40e7857eee26abe150c116f99c8f",
        ),
        ContentProbe(  # modulation.1
            tensor="modulation.1.sign_mantissa",
            offset=0,
            length=4096,
            sha256="83a4184a43758b55bf98e4457aa2e77f4a09e85517c2ab50f894f088818adace",
        ),
    ),
)


# Krea 2 Raw and Krea 2 Turbo as exported for ComfyUI: one ~17.5 GB file each, 35 groups (the 28
# transformer blocks, the four text-fusion blocks, tmlp, tproj and txtmlp) and 169 BF16 extras. Each
# block and text-fusion block stores eight matrices. Several share a shape (wq, gate and wo, and wk
# with wv, in a transformer block; all five attention matrices in a text-fusion block) and the three
# MLP matrices hold equal element counts, so the order cannot be read from shapes. It was confirmed
# by comparing 4096
# sign/mantissa bytes at every matrix start of blocks 0, 13 and 27 and of all four text-fusion blocks
# with the originals (krea/Krea-2-Raw and krea/Krea-2-Turbo, BF16 matrices; tmlp, tproj and txtmlp
# stored there as FP32 holding BF16-exact values): 4096 of 4096 equal for the expected tensor, at most
# 32 for any other, in both files. The order is the module registration order of the transformer.
# The 25 spot checks below cover every matrix start of blocks.0 and txtfusion.layerwise_blocks.0, the
# first matrix of blocks.27 and of the other three text-fusion blocks, and each matrix of tmlp, tproj
# and txtmlp (the two first matrices of layerwise_blocks.1 and refiner_blocks.0 range-read later the
# same day, at the same revisions). They do not cover any start past the first in blocks 1 to 27 or
# in the three other text-fusion blocks: those rest on the byte comparison above and on a full
# element-by-element comparison with the originals, and a re-upload that keeps the header but
# permutes their equal-shaped matrices would pass the header and the spot checks. Both files share
# every tensor name, dtype and shape; only the codec tensors' lengths differ. The files record no
# format version; the layouts use 0.3.1 as Qwen-Image 2.1's does, and the decoder does not read it.
_KREA2_BLOCK = [
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
]
_KREA2_CONFIG: dict[str, Any] = {
    "version": "0.3.1",
    "threads_per_block": [512],
    "bytes_per_thread": 8,
    "pattern_dict": {
        r"blocks\.\d+": _KREA2_BLOCK,
        r"txtfusion\.layerwise_blocks\.\d+": _KREA2_BLOCK,
        r"txtfusion\.refiner_blocks\.\d+": _KREA2_BLOCK,
        "tmlp": ["0", "2"],
        "tproj": ["1"],
        "txtmlp": ["1", "3"],
    },
}


def _krea2_config() -> dict[str, Any]:
    """A private copy of the Krea 2 config, so the two layouts share no mutable state."""
    copy: dict[str, Any] = json.loads(json.dumps(_KREA2_CONFIG))
    return copy


KREA_2_RAW_COMFYUI = SynthesizedLayout(
    key="krea-2-raw-comfyui",
    label="Krea 2 Raw (ComfyUI single-file DF11)",
    repo_id="mingyi456/Krea-2-Raw-DF11-ComfyUI",
    revision="8320616b25ac9340a830a7fb21f1b0237e160e66",
    file_name="krea2_raw_bf16-DF11.safetensors",
    header_sha256="a0eb86513ccf0ba8baca0625147b62109707650b480046cd66043ea9a8700578",
    file_sha256="f62cf31921ad0377f9d6e5bd2e3191a1ac9950d92f487b26fa7c052f503d9b97",
    groups=35,
    extras=169,
    raw_config=_krea2_config(),
    row_splits={},
    probes=(
        ContentProbe(  # blocks.0.attn.wq
            tensor="blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="363d5ac87fa64889de9ef810e04d0b1e905ceb89fee078fa0e589c254298e35f",
        ),
        ContentProbe(  # blocks.0.attn.wk
            tensor="blocks.0.sign_mantissa",
            offset=37_748_736,
            length=4096,
            sha256="ca6dd074e2d1e37891e94e867dfacc6ddb5c19f878f78a60f56e1e2f6f5e06bd",
        ),
        ContentProbe(  # blocks.0.attn.wv
            tensor="blocks.0.sign_mantissa",
            offset=47_185_920,
            length=4096,
            sha256="6480a299d38e684bdb3a8d498752ed62afda615de694aab4b69a16a974ba518c",
        ),
        ContentProbe(  # blocks.0.attn.gate
            tensor="blocks.0.sign_mantissa",
            offset=56_623_104,
            length=4096,
            sha256="3962f4172d9e4007c709729c5d262c1105748e472c17d714ef41ced9fd2aa40f",
        ),
        ContentProbe(  # blocks.0.attn.wo
            tensor="blocks.0.sign_mantissa",
            offset=94_371_840,
            length=4096,
            sha256="585411a597165de162951bdd1d68cef4fb03a434caf433d33d3f60e7625d1ffe",
        ),
        ContentProbe(  # blocks.0.mlp.gate
            tensor="blocks.0.sign_mantissa",
            offset=132_120_576,
            length=4096,
            sha256="2d41ddb871e747add117545678a7097e19e9901744434de7c54b885cfa049b10",
        ),
        ContentProbe(  # blocks.0.mlp.up
            tensor="blocks.0.sign_mantissa",
            offset=232_783_872,
            length=4096,
            sha256="01bfce386ced98d20b1bceec8ccf51ec586036c730125c9c3ef4ea76764d19a0",
        ),
        ContentProbe(  # blocks.0.mlp.down
            tensor="blocks.0.sign_mantissa",
            offset=333_447_168,
            length=4096,
            sha256="9489cb4060011bc2edca69148499606df538751f961e35dc7ba3c2d90954b3b5",
        ),
        ContentProbe(  # blocks.27.attn.wq
            tensor="blocks.27.sign_mantissa",
            offset=0,
            length=4096,
            sha256="d011e6504676d98bde3696da028b03d8f6cbc7d4ad5ece7e67afda38c2cc22ad",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wq
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="edcc578d390f9a375a43660de265524fba96f8cdfb7422c2260fe7ae3c81bf95",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wk
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=6_553_600,
            length=4096,
            sha256="c2f20aaf9f3e23b66e4d74e26a4239e75b2409fb4baffb875344084ae8cd1920",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wv
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=13_107_200,
            length=4096,
            sha256="cf78e88a47707eab0acab1561a31b3c7fc75174d7ee1c872bb1a5558cf2c3d6a",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.gate
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=19_660_800,
            length=4096,
            sha256="92fedf37baba3c13617ec9bacb6017aafacf5e61996791edb92720ebd1223cb8",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wo
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=26_214_400,
            length=4096,
            sha256="c74085fa9516ac66243f038846454e9f4911b5ee54891e3d6e634b27cbdacb49",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.gate
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=32_768_000,
            length=4096,
            sha256="9b765f28f96af426c2c3ca7d48d08c11b38aab7b02c010dde4935a45fd421d55",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.up
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=50_462_720,
            length=4096,
            sha256="6c12c5521469efccaea92c86cf1232bf69cd7097033c23c77ea36d6899141a28",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.down
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=68_157_440,
            length=4096,
            sha256="5998851cd52398972b745d340839d779e5f45ba4189c08b2cd3d0493fa36e393",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.1.attn.wq
            tensor="txtfusion.layerwise_blocks.1.sign_mantissa",
            offset=0,
            length=4096,
            sha256="4b0fd5b9e0ea3356bc8d71b3ef3008bd4cdee28e485c89c67188d533604885db",
        ),
        ContentProbe(  # txtfusion.refiner_blocks.0.attn.wq
            tensor="txtfusion.refiner_blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="7470ae9e5215bb3bd6f2dc75620cc6935e4d7a957ac05358838f6b851a01dff8",
        ),
        ContentProbe(  # txtfusion.refiner_blocks.1.attn.wq
            tensor="txtfusion.refiner_blocks.1.sign_mantissa",
            offset=0,
            length=4096,
            sha256="a7dd2de120a3a713fe6eb782ad926c36d2718498c739ee3baac4cd5e26577214",
        ),
        ContentProbe(  # tmlp.0
            tensor="tmlp.sign_mantissa",
            offset=0,
            length=4096,
            sha256="356d0f90c117d207bdef0a26d602b4ca0a36a52f1e4d2588849e2dff3df4f6aa",
        ),
        ContentProbe(  # tmlp.2
            tensor="tmlp.sign_mantissa",
            offset=1_572_864,
            length=4096,
            sha256="305bc2e99d6161e4674a8f2a222c25e958f9b2b2d9efb36f5db799a3a0a3e4d3",
        ),
        ContentProbe(  # tproj.1
            tensor="tproj.sign_mantissa",
            offset=0,
            length=4096,
            sha256="9854423ef01b650b06113ec959298735bcf4cc98d79a5b628873ed2b20a39b9d",
        ),
        ContentProbe(  # txtmlp.1
            tensor="txtmlp.sign_mantissa",
            offset=0,
            length=4096,
            sha256="86beee24cdec56afaf7d367735d53153aed1a00956955a5aff8d1abc2b9d9182",
        ),
        ContentProbe(  # txtmlp.3
            tensor="txtmlp.sign_mantissa",
            offset=15_728_640,
            length=4096,
            sha256="ca3c3a6ee9e11db6c5b419f351187e46505403fdd230c3fdade6f3377527a48c",
        ),
    ),
)

KREA_2_TURBO_COMFYUI = SynthesizedLayout(
    key="krea-2-turbo-comfyui",
    label="Krea 2 Turbo (ComfyUI single-file DF11)",
    repo_id="mingyi456/Krea-2-Turbo-DF11-ComfyUI",
    revision="978da5fb7647bd222d33125993abd8fdc2840cfc",
    file_name="krea2_turbo_bf16-DF11.safetensors",
    header_sha256="1de7b6f3f0dc4c04a05c3bb1e7972f0d3761902ea2ddd42916f14183aecbcafd",
    file_sha256="ccfd85551ba0f37e7f6706d86d1c00276457d2965ac72fba6fef046653ad6498",
    groups=35,
    extras=169,
    raw_config=_krea2_config(),
    row_splits={},
    probes=(
        ContentProbe(  # blocks.0.attn.wq
            tensor="blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="11f6150d52c2100d4e46f8a8962a8b1a6acf2c38f477cbe9e113643b095caef8",
        ),
        ContentProbe(  # blocks.0.attn.wk
            tensor="blocks.0.sign_mantissa",
            offset=37_748_736,
            length=4096,
            sha256="7dc9c9a7f947dd621467761eba193174a35913e35c09ea273109ce1c7d1995e0",
        ),
        ContentProbe(  # blocks.0.attn.wv
            tensor="blocks.0.sign_mantissa",
            offset=47_185_920,
            length=4096,
            sha256="c8fd1bc47699725e3a1cfabaddd2cdea7e563a01bf1b0ffa10617f8456ba6f0a",
        ),
        ContentProbe(  # blocks.0.attn.gate
            tensor="blocks.0.sign_mantissa",
            offset=56_623_104,
            length=4096,
            sha256="35e55744d6156352bc36f3b5e2d716679d4d2f8c6dc7c6ee5795d4deec7432d0",
        ),
        ContentProbe(  # blocks.0.attn.wo
            tensor="blocks.0.sign_mantissa",
            offset=94_371_840,
            length=4096,
            sha256="f905a0181ece0d857a95cbe554bcf98968b847ccebacb332b89755f6545d0dd3",
        ),
        ContentProbe(  # blocks.0.mlp.gate
            tensor="blocks.0.sign_mantissa",
            offset=132_120_576,
            length=4096,
            sha256="171a5c675fb5bd9aff19ec75f77eedfef7dc17e6861568e2df3144dec58caaf3",
        ),
        ContentProbe(  # blocks.0.mlp.up
            tensor="blocks.0.sign_mantissa",
            offset=232_783_872,
            length=4096,
            sha256="356076a9cd44ddcd624da9336dd56b0dc233ce3bea51e473380d739605292747",
        ),
        ContentProbe(  # blocks.0.mlp.down
            tensor="blocks.0.sign_mantissa",
            offset=333_447_168,
            length=4096,
            sha256="7ccae2ff710cc0dbf818ca8fe768066e79b2a64881f8fb514dedcb2307eafc50",
        ),
        ContentProbe(  # blocks.27.attn.wq
            tensor="blocks.27.sign_mantissa",
            offset=0,
            length=4096,
            sha256="9a865f96cbb3415c0d2fa6cbf04aac8e72cd914a4da7b30dfa346f48ae2f68b5",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wq
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="8f836605424820e44b68ccd56399e183aa990ff0d3da3cec22250db6145cb3c5",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wk
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=6_553_600,
            length=4096,
            sha256="9a074d7d85a38cf157419326fe09f517982269097591597ffd5cf4b25438dc04",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wv
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=13_107_200,
            length=4096,
            sha256="2975a4ffd78330058c286a9c07d59db4f30fb30da0449cd19f770c96ca018029",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.gate
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=19_660_800,
            length=4096,
            sha256="fe5976eb985efce2624bdfee10e3362074953a0197c4450dda7799314c905586",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.attn.wo
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=26_214_400,
            length=4096,
            sha256="6978c0ed44c394128072eb7671cc902112181f2f18a70a84e80d0b5a5aeed80a",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.gate
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=32_768_000,
            length=4096,
            sha256="6a8af83a863507b9de8683470115894dea0dd302fe507e4f7f976b3034204bb0",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.up
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=50_462_720,
            length=4096,
            sha256="3d7417959035994ad92f9036b3636b1b35e821f438468388dc4306a6221cee4e",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.0.mlp.down
            tensor="txtfusion.layerwise_blocks.0.sign_mantissa",
            offset=68_157_440,
            length=4096,
            sha256="ff857b14597bfcf543d7938d71992f696177b52ee3a7574f63719a9a42dac94b",
        ),
        ContentProbe(  # txtfusion.layerwise_blocks.1.attn.wq
            tensor="txtfusion.layerwise_blocks.1.sign_mantissa",
            offset=0,
            length=4096,
            sha256="06d9ba6294dbc0509ab38954d11052a3d0cdc1df21aff4f5fdc1a0d30269f02f",
        ),
        ContentProbe(  # txtfusion.refiner_blocks.0.attn.wq
            tensor="txtfusion.refiner_blocks.0.sign_mantissa",
            offset=0,
            length=4096,
            sha256="200e8501fc64ada5d37bf10f2d2653f273367019075f3607f58b0cb54f7c2da2",
        ),
        ContentProbe(  # txtfusion.refiner_blocks.1.attn.wq
            tensor="txtfusion.refiner_blocks.1.sign_mantissa",
            offset=0,
            length=4096,
            sha256="241ff415ea140b6838ea709c1317c9d5da6129231f23ca8dbdef946b7bf056c9",
        ),
        ContentProbe(  # tmlp.0
            tensor="tmlp.sign_mantissa",
            offset=0,
            length=4096,
            sha256="ee752a8f3b403449d3a1e95666369eba78aec410bf7879663215b58a54efac98",
        ),
        ContentProbe(  # tmlp.2
            tensor="tmlp.sign_mantissa",
            offset=1_572_864,
            length=4096,
            sha256="1b2aa81eca1c9b20b9f6b96018f53833acc5f9e9617bd37432bcd4920f6b6963",
        ),
        ContentProbe(  # tproj.1
            tensor="tproj.sign_mantissa",
            offset=0,
            length=4096,
            sha256="52dd92bbc434a9120d8038e8637bf5b350d69aba1d508aebaeb20da2d9b57dfe",
        ),
        ContentProbe(  # txtmlp.1
            tensor="txtmlp.sign_mantissa",
            offset=0,
            length=4096,
            sha256="72862c0bad66d0664fda6c8f8db121ceca7aafcac721268e6208d6a8fbf01f69",
        ),
        ContentProbe(  # txtmlp.3
            tensor="txtmlp.sign_mantissa",
            offset=15_728_640,
            length=4096,
            sha256="04724bf4d4567d5f72bb6e0e8e95a9e0560d9bd405ec19399f9b3aa84adfc2ba",
        ),
    ),
)

KNOWN_LAYOUTS: tuple[SynthesizedLayout, ...] = (
    QWEN_IMAGE_21_COMFYUI,
    KREA_2_RAW_COMFYUI,
    KREA_2_TURBO_COMFYUI,
)


def identify_layout(
    raw_header: bytes, *, known: Sequence[SynthesizedLayout]
) -> SynthesizedLayout | None:
    """The layout whose header digest equals the sha256 of ``raw_header``, or None."""
    digest = hashlib.sha256(raw_header).hexdigest()
    return next((layout for layout in known if layout.header_sha256 == digest), None)


__all__ = [
    "KNOWN_LAYOUTS",
    "KREA_2_RAW_COMFYUI",
    "KREA_2_TURBO_COMFYUI",
    "QWEN_IMAGE_21_COMFYUI",
    "ContentProbe",
    "SynthesizedLayout",
    "identify_layout",
]
