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

KNOWN_LAYOUTS: tuple[SynthesizedLayout, ...] = (QWEN_IMAGE_21_COMFYUI,)


def identify_layout(
    raw_header: bytes, *, known: Sequence[SynthesizedLayout]
) -> SynthesizedLayout | None:
    """The layout whose header digest equals the sha256 of ``raw_header``, or None."""
    digest = hashlib.sha256(raw_header).hexdigest()
    return next((layout for layout in known if layout.header_sha256 == digest), None)


__all__ = [
    "KNOWN_LAYOUTS",
    "QWEN_IMAGE_21_COMFYUI",
    "ContentProbe",
    "SynthesizedLayout",
    "identify_layout",
]
