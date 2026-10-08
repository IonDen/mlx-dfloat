"""Build small DF11 checkpoints with upstream's own (vendored) encoder, plus BF16 originals."""

import json

import mlx.core as mx
import numpy as np
from tests._upstream.dfloat11_encoder import (
    encode_weights,
    exponent_counter,
    get_32bit_codec,
    get_luts,
)


def random_bf16(rng, shape, *, exponent_low=110, exponent_high=130):
    """Random BF16 bit patterns: exponents in [low, high), random sign and mantissa."""
    exponent = rng.integers(exponent_low, exponent_high, size=shape, dtype=np.uint16)
    mantissa = rng.integers(0, 128, size=shape, dtype=np.uint16)
    sign = rng.integers(0, 2, size=shape, dtype=np.uint16)
    return ((sign << 15) | (exponent << 7) | mantissa).astype(np.uint16)


def codec_for(counter):
    """Upstream's codec, code table and LUTs for an exponent counter."""
    codec, _, table = get_32bit_codec(counter)
    return codec, table, get_luts(table)


def compress_group(matrices):
    """Compress BF16 matrices (uint16 bits) into the six stored DF11 group arrays, as upstream does."""
    combined = np.concatenate([m.reshape(-1) for m in matrices])
    codec, _, luts = codec_for(exponent_counter(combined))
    encoded, other, positions, gaps, split = encode_weights(list(matrices), codec, 8, 512)
    return {
        "encoded_exponent": encoded,
        "sign_mantissa": other,
        "luts": luts,
        "gaps": gaps,
        "output_positions": positions.view(np.uint8),  # stored as uint32 LE viewed as bytes
        "split_positions": split,
    }


def _mx(v):
    return v if isinstance(v, mx.array) else mx.array(np.ascontiguousarray(v))


def write_checkpoint(
    root,
    *,
    groups,
    pattern=None,
    sub_paths=(),
    patterns=None,
    version="0.5.0",
    extras=None,
    single_file=False,
    write_config=True,
):
    """Write a DF11 checkpoint: a shard per group (or one file), config.json, optional BF16 extras.

    ``patterns`` (pattern -> sub-paths) replaces the single ``pattern``/``sub_paths`` pair when a
    checkpoint holds more than one group family. ``write_config=False`` leaves out config.json
    (a config-less single-file checkpoint, read through a pinned layout).
    """
    root.mkdir(parents=True, exist_ok=True)
    pattern_dict = (
        {pattern: list(sub_paths)}
        if patterns is None
        else {k: list(v) for k, v in patterns.items()}
    )
    config = {
        "dfloat11_config": {
            "version": version,
            "threads_per_block": [512],
            "bytes_per_thread": 8,
            "pattern_dict": pattern_dict,
        }
    }
    if write_config:
        (root / "config.json").write_text(json.dumps(config))
    everything = {}
    for group_name, matrices in groups.items():
        tensors = {f"{group_name}.{k}": v for k, v in compress_group(matrices).items()}
        if single_file:
            everything.update(tensors)
        else:
            mx.save_safetensors(
                str(root / (group_name.replace(".", "_") + ".safetensors")),
                {k: _mx(v) for k, v in tensors.items()},
            )
    extras_mx = {k: mx.array(v).view(mx.bfloat16) for k, v in (extras or {}).items()}
    if single_file:
        everything.update(extras_mx)
        mx.save_safetensors(
            str(root / "model.safetensors"), {k: _mx(v) for k, v in everything.items()}
        )
    elif extras_mx:
        mx.save_safetensors(str(root / "model.safetensors"), extras_mx)
    return root


def write_bf16_original(root, tensors, *, with_index=True, dtype=mx.bfloat16):
    """Write 'original' tensors (uint16 bits) as up to two shards plus an HF-style index."""
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = max(1, len(names) // 2)
    shards = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for shard, shard_names in shards.items():
        if not shard_names:
            continue
        arrays = {n: mx.array(tensors[n]).view(mx.bfloat16).astype(dtype) for n in shard_names}
        mx.save_safetensors(str(root / shard), arrays)
        weight_map.update(dict.fromkeys(shard_names, shard))
    if with_index:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root
