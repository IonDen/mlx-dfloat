"""Extract a small, real DF11 slice for the offline test suite (network; run once, commit output).

HTTP range reads only the needed prefixes of the ``model.layers.0`` group of
``DFloat11/Qwen3-4B-DF11`` (first 4 thread-blocks + 8 lookahead bytes) and the matching first
elements of ``model.layers.0.self_attn.q_proj.weight`` from ``Qwen/Qwen3-4B``. Parses safetensors
headers with its own code, independent of mlx_dfloat.

Usage:
    uv run python scripts/extract_upstream_slice.py --out src/mlx_dfloat/_canary_data
"""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, HfFileSystem

from mlx_dfloat.format import _check_pattern

DF11_REPO = "DFloat11/Qwen3-4B-DF11"
BF16_REPO = "Qwen/Qwen3-4B"
GROUP = "model.layers.0"
MATRIX = "model.layers.0.self_attn.q_proj.weight"
N_BLOCKS = 4
LOOKAHEAD = 8  # bytes of block 4: the last block-3 codes may spill up to 31 bits past it


def _header(fs: HfFileSystem, path: str) -> tuple[dict, int]:
    # Default block_size (a seekable HfFileSystemFile). block_size=0 returns a streaming file
    # whose seek() raises "Cannot seek streaming HF file" on huggingface_hub 2.0.
    with fs.open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > 100_000_000:
            raise SystemExit(f"{path}: header too large")
        return json.loads(handle.read(length)), 8 + length


def _read(fs: HfFileSystem, path: str, start: int, n: int) -> bytes:
    with fs.open(path, "rb") as handle:  # seekable; see _header
        handle.seek(start)
        data = handle.read(n)
    if len(data) != n:
        raise SystemExit(f"{path}: short read ({len(data)} of {n})")
    return data


def subpaths_for(pattern_dict: dict[str, list[str]], group: str) -> list[str]:
    """Sub-paths of the (screened) pattern that fully matches ``group``.

    The config is downloaded, so every pattern goes through the package's pattern guard before
    ``re.fullmatch`` sees it.
    """
    for pattern in pattern_dict:
        _check_pattern(pattern, source="config.json")
    return next(s for p, s in pattern_dict.items() if re.fullmatch(p, group))


def main() -> None:
    """Extract the slice and write it with provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    api, fs = HfApi(), HfFileSystem()
    df11_rev = api.model_info(DF11_REPO).sha
    bf16_rev = api.model_info(BF16_REPO).sha
    config = json.loads(fs.read_text(f"{DF11_REPO}@{df11_rev}/config.json"))["dfloat11_config"]
    subs = subpaths_for(config["pattern_dict"], GROUP)
    if not subs or subs[0] != "self_attn.q_proj":
        raise SystemExit(f"unexpected first matrix in {GROUP}: {subs[:1]}")
    df11_file = header = base = None
    for name in api.list_repo_files(DF11_REPO, revision=df11_rev):
        if name.endswith(".safetensors"):
            h, b = _header(fs, f"{DF11_REPO}@{df11_rev}/{name}")
            if f"{GROUP}.encoded_exponent" in h:
                df11_file, header, base = name, h, b
                break
    if df11_file is None:
        raise SystemExit(f"{GROUP} not found in {DF11_REPO}")
    path = f"{DF11_REPO}@{df11_rev}/{df11_file}"

    def prefix(key: str, n_bytes: int | None = None) -> bytes:
        start, end = header[f"{GROUP}.{key}"]["data_offsets"]
        size = end - start if n_bytes is None else min(n_bytes, end - start)
        return _read(fs, path, base + start, size)

    positions = np.frombuffer(prefix("output_positions", 4 * (N_BLOCKS + 1)), "<u4").copy()
    n_slice = int(positions[N_BLOCKS])
    n_bytes = N_BLOCKS * 4096 + LOOKAHEAD
    encoded = np.frombuffer(prefix("encoded_exponent", n_bytes), np.uint8)
    gaps = np.frombuffer(prefix("gaps", (N_BLOCKS + 1) * 320), np.uint8)
    sign_mantissa = np.frombuffer(prefix("sign_mantissa", n_slice), np.uint8)
    luts = np.frombuffer(prefix("luts"), np.uint8).reshape(-1, 256)
    assert encoded.size % 4096 == LOOKAHEAD, "the 8 lookahead bytes are load-bearing"

    index = json.loads(fs.read_text(f"{BF16_REPO}@{bf16_rev}/model.safetensors.index.json"))
    bf16_file = index["weight_map"][MATRIX]
    bf16_header, bf16_base = _header(fs, f"{BF16_REPO}@{bf16_rev}/{bf16_file}")
    meta = bf16_header[MATRIX]
    if meta["dtype"] != "BF16" or int(np.prod(meta["shape"])) < n_slice:
        raise SystemExit(f"unexpected original tensor {meta}")
    expected = np.frombuffer(
        _read(
            fs,
            f"{BF16_REPO}@{bf16_rev}/{bf16_file}",
            bf16_base + meta["data_offsets"][0],
            2 * n_slice,
        ),
        "<u2",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    npz = args.out / "qwen3_4b_layer0_4blocks.npz"
    np.savez_compressed(
        npz,
        encoded_exponent=encoded,
        sign_mantissa=sign_mantissa,
        luts=luts,
        gaps=gaps,
        output_positions=positions,
        split_positions=np.zeros(0, np.int64),
        expected_bf16=expected,
    )
    gap_values = np.unpackbits(gaps)[: 5 * N_BLOCKS * 512].reshape(-1, 5) @ np.array(
        [16, 8, 4, 2, 1]
    )
    provenance = {
        "df11_repo": DF11_REPO,
        "df11_revision": df11_rev,
        "df11_file": df11_file,
        "bf16_repo": BF16_REPO,
        "bf16_revision": bf16_rev,
        "bf16_file": bf16_file,
        "group": GROUP,
        "matrix": MATRIX,
        "n_blocks": N_BLOCKS,
        "n_elements": n_slice,
        "n_bytes": int(encoded.size),
        "max_code_length": int(luts[-1].max()),
        "lut_rows": int(luts.shape[0]),
        "max_gap": int(gap_values.max()),
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "license": "Qwen3-4B is Apache-2.0; the DFloat11 checkpoint is derived from it",
        "note": "4 thread-blocks + 8 lookahead bytes (load-bearing); output_positions[4] = n_elements",
    }
    (args.out / "qwen3_4b_layer0_4blocks.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
