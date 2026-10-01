"""The step bench's q8 condition: mflux's FLUX.1 transformer quantized to 8 bits at load from a pinned base.

The build order is the whole point of this module. mflux's loader returns a ``LoadedWeights`` whose
components dict holds every BF16 array of the transformer (lazy ``mx.load`` views); the applier
puts them into the model and ``nn.quantize`` swaps each Linear for a ``QuantizedLinear`` whose
weight is a lazy ``mx.quantize`` of that BF16 array. Evaluating while the dict is still referenced
keeps every BF16 array resident next to the q8 copy (about 22 + 12 GiB for FLUX.1, over a 32 GB
Mac), so the builder drops the dict and collects before the first eval, then evaluates block by
block, so at most one block's BF16 sources are materialised at a time.

mflux is imported only when a default piece (loader, applier, transformer factory, component) is
used, so the release order is unit-tested with fakes in a venv without mflux.
"""

import gc
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from mlx_dfloat.errors import DFloatIntegrationError
from mlx_dfloat.mflux import require_mflux

BITS = 8
GROUP_SIZE = 64  # mlx's affine default, which mflux's single-component applier leaves in place
MODE = "affine"
MODELS: tuple[str, ...] = ("schnell", "dev")

Loader = Callable[[Any, Path], Any]
Applier = Callable[[Any, Any, Any], None]
Evaluate = Callable[[Any], None]
MakeTransformer = Callable[[str, int, int], Any]


WEIGHT_DIRS = ("transformer", "text_encoder", "text_encoder_2", "vae")


def missing_weight_files(root: Path, patterns: Sequence[str]) -> list[str]:
    """Weight files the ``<dir>/*`` patterns promise that are absent under ``root`` (relative paths).

    A directory with a ``*.safetensors.index.json`` must hold every shard its ``weight_map`` names
    (a dangling link counts as absent); a FLUX weight directory without an index must hold at
    least one ``*.safetensors``. Other directories and the ``*`` pattern are not checked.
    """
    missing: list[str] = []
    for pattern in patterns:
        directory, sep, tail = pattern.partition("/")
        if not sep or tail != "*" or "*" in directory:
            continue
        folder = root / directory
        indexes = sorted(folder.glob("*.safetensors.index.json")) if folder.is_dir() else []
        if indexes:
            for index in indexes:
                try:
                    weight_map = json.loads(index.read_text())["weight_map"]
                    shards = sorted({str(v) for v in weight_map.values()})
                except (OSError, ValueError, KeyError, AttributeError):
                    missing.append(f"{directory}/{index.name} (unreadable)")
                    continue
                missing.extend(f"{directory}/{n}" for n in shards if not (folder / n).exists())
        elif directory in WEIGHT_DIRS and not (
            folder.is_dir() and any(p.exists() for p in folder.glob("*.safetensors"))
        ):
            missing.append(f"{directory}/*.safetensors")
    return missing


def pinned_snapshot(repo_id: str, revision: str, *, allow_patterns: Sequence[str]) -> Path:
    """The local snapshot directory of ``repo_id`` at commit ``revision``; never downloads.

    Raises:
        DFloatIntegrationError: The files are not in the local Hub cache (the message carries the
            ``hf download`` command that fetches them), or the snapshot found is not the one named
            ``revision`` (a branch name, or another commit).
    """
    patterns = list(allow_patterns)
    try:
        found = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=patterns,
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        includes = " ".join(f'--include "{p}"' for p in patterns)
        raise DFloatIntegrationError(
            f"{repo_id}@{revision} is not in the local Hub cache; download it first: "
            f"hf download {repo_id} --revision {revision} {includes}"
        ) from exc
    root = Path(str(found))
    if root.name != revision:
        raise DFloatIntegrationError(
            f"{repo_id}: the local snapshot is {root.name!r}, not the pinned revision {revision!r}"
        )
    absent = missing_weight_files(root, patterns)
    if absent:
        shown = ", ".join(absent[:5])
        more = f" (and {len(absent) - 5} more)" if len(absent) > 5 else ""
        includes = " ".join(f'--include "{p}"' for p in patterns)
        raise DFloatIntegrationError(
            f"{repo_id}@{revision}: the local snapshot is missing weight files: {shown}{more}; "
            f"download them: hf download {repo_id} --revision {revision} {includes}"
        )
    return root


def _mflux_component() -> Any:
    require_mflux()
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    return next(c for c in FluxWeightDefinition.get_components() if c.name == "transformer")


def _mflux_loader(component: Any, root: Path) -> Any:
    require_mflux()
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    return WeightLoader.load_single_local(component=component, root_path=root)


def _mflux_applier(weights: Any, transformer: Any, component: Any) -> None:
    require_mflux()
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    bits = WeightApplier.apply_and_quantize_single(
        weights,
        transformer,
        component,
        quantize_arg=BITS,
        quantization_predicate=FluxWeightDefinition.quantization_predicate,
    )
    if bits != BITS:
        raise DFloatIntegrationError(f"mflux quantized the transformer to {bits} bits, not {BITS}")


def _mflux_transformer(model: str, n_double: int, n_single: int) -> Any:
    require_mflux()
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.model.flux_transformer.transformer import Transformer

    config = ModelConfig.schnell() if model == "schnell" else ModelConfig.dev()
    return Transformer(
        config, num_transformer_blocks=n_double, num_single_transformer_blocks=n_single
    )


def q8_quantization(transformer: Any) -> dict[str, Any]:
    """The quantization the built transformer really carries, read off its first block's first QuantizedLinear.

    Raises:
        DFloatIntegrationError: The block holds no quantized Linear, or it is not 8-bit, group
            size 64, affine (``BITS``, ``GROUP_SIZE``, ``MODE``).
    """
    block = transformer.transformer_blocks[0]
    layer = next((m for _, m in block.named_modules() if isinstance(m, nn.QuantizedLinear)), None)
    if layer is None:
        raise DFloatIntegrationError("the q8 transformer's first block has no quantized Linear")
    found = {"bits": int(layer.bits), "group_size": int(layer.group_size), "mode": str(layer.mode)}
    if (found["bits"], found["group_size"], found["mode"]) != (BITS, GROUP_SIZE, MODE):
        raise DFloatIntegrationError(
            f"the q8 transformer was built as {found}, not {BITS}-bit, group size {GROUP_SIZE}, "
            f"{MODE}"
        )
    return found


def evaluate_by_block(transformer: Any) -> None:
    """Evaluate the quantized transformer one block at a time, then whatever sits outside the blocks."""
    for block in [*transformer.transformer_blocks, *transformer.single_transformer_blocks]:
        mx.eval(block.parameters())
    mx.eval(transformer.parameters())


def build_q8_transformer(
    model: str,
    root: Path,
    *,
    n_double: int = 19,
    n_single: int = 38,
    loader: Loader | None = None,
    applier: Applier | None = None,
    evaluate: Evaluate | None = None,
    make_transformer: MakeTransformer | None = None,
    component: Any = None,
) -> Any:
    """Build mflux's FLUX.1 transformer from the BF16 base snapshot ``root``, quantized to 8 bits.

    ``root`` is a pinned snapshot (``pinned_snapshot``); the loader reads ``root/transformer``
    directly, since mflux's own ``load_single`` takes no revision and would resolve the newest
    cached snapshot. Order: load, refuse a stored quantization, build the model, apply and quantize,
    drop every reference to the loaded dict and collect, evaluate (block by block by default), then
    return the freed BF16 buffers from MLX's cache. Each piece is injectable (tests pass fakes);
    a default one imports mflux.

    Raises:
        DFloatIntegrationError: ``model`` is not ``"schnell"`` or ``"dev"``, the base is already
            quantized, or mflux resolved another bit width.
        DFloatDependencyError: A default piece is used and the optional ``mflux`` extra is missing.
    """
    if model not in MODELS:
        raise DFloatIntegrationError(f"unknown model {model!r}; choose from {list(MODELS)}")
    component = _mflux_component() if component is None else component
    load = _mflux_loader if loader is None else loader
    apply = _mflux_applier if applier is None else applier
    make = _mflux_transformer if make_transformer is None else make_transformer
    run_eval = evaluate_by_block if evaluate is None else evaluate

    weights = load(component, root)
    stored = weights.meta_data.quantization_level
    if stored is not None:
        raise DFloatIntegrationError(
            f"{root}: the base is already quantized ({stored} bits); q8 needs the BF16 base"
        )
    transformer = make(model, n_double, n_single)
    apply(weights, transformer, component)
    # Release the BF16 dict before anything is evaluated (see the module docstring).
    del weights
    gc.collect()
    run_eval(transformer)
    mx.clear_cache()
    return transformer
