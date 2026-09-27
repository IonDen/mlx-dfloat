"""Encode one prompt with FLUX's T5 and CLIP text encoders into a safetensors file for the step bench.

The step bench (``scripts/bench_flux_step.py``) times the transformer alone, so the prompt is
encoded once here, never in a timed process, and read back from ``--out``. Only the two encoders
are loaded, from a local Hugging Face snapshot that holds ``text_encoder/``, ``text_encoder_2/``,
``tokenizer/`` and ``tokenizer_2/`` (no transformer or VAE is needed), through mflux 0.20's own
component loader and weight applier. ``--model dev`` encodes at dev's 512 T5 tokens with the same
encoders (schnell and dev share them). ``--synthetic`` writes seeded ``mx.random.normal`` arrays of
the same shapes instead and records that in the metadata, for a run without the encoder weights.

The file holds ``prompt_embeds`` (``(1, 256, 4096)``, dev ``(1, 512, 4096)``) and
``pooled_prompt_embeds`` (``(1, 768)``) plus string metadata: model, seed, prompt, root, token
length, whether it is synthetic, and the mflux and mlx versions.

Usage (from the repository root of a synced checkout, ``--group bench``):
    uv run python -m scripts.encode_prompt --model schnell --root SNAPSHOT_DIR --out embeds.safetensors \
        [--prompt "..."] [--seed 42]
    uv run python -m scripts.encode_prompt --model dev --synthetic --out embeds.safetensors [--seed 42]
Exit codes: 0 written, 2 an input or tool error.
"""

import argparse
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

# Run as a file, Python puts scripts/ (not the repository root) first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mlx.core as mx

    from mlx_dfloat._memory_caps import install_memory_caps
except Exception as exc:  # a broken environment is a tool error
    print(
        f"error: cannot import the project modules ({exc}); run from a synced checkout",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

EXIT_OK, EXIT_ERROR = 0, 2
T5_DIM = 4096
POOLED_DIM = 768
T5_LENGTHS = {"schnell": 256, "dev": 512}  # mflux 0.20 ModelConfig.max_sequence_length per model
DEFAULT_PROMPT = (
    "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing "
    "boat far out on the water"
)


def t5_max_length(model: str) -> int:
    """The T5 token length the bench expects for ``model`` (schnell 256, dev 512).

    Raises:
        ValueError: Unknown model.
    """
    try:
        return T5_LENGTHS[model]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(T5_LENGTHS)}") from None


def embed_shapes(model: str) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """The ``(prompt_embeds, pooled_prompt_embeds)`` shapes for ``model``."""
    return (1, t5_max_length(model), T5_DIM), (1, POOLED_DIM)


def synthetic_embeds(model: str, seed: int) -> tuple[mx.array, mx.array]:
    """Seeded standard-normal stand-ins of the real embeddings' shapes (two keys split from ``seed``)."""
    prompt_shape, pooled_shape = embed_shapes(model)
    key_prompt, key_pooled = mx.random.split(mx.random.key(seed))
    return (
        mx.random.normal(prompt_shape, key=key_prompt),
        mx.random.normal(pooled_shape, key=key_pooled),
    )


def build_metadata(
    *, model: str, seed: int, prompt: str, root: str, token_length: int, synthetic: bool
) -> dict[str, str]:
    """The safetensors metadata of an embeddings file: every value a string, as the format requires."""
    try:
        mflux_version = metadata.version("mflux")
    except metadata.PackageNotFoundError:
        mflux_version = "absent"
    return {
        "model": model,
        "seed": str(seed),
        "prompt": prompt,
        "root": root,
        "token_length": str(token_length),
        "synthetic": "true" if synthetic else "false",
        "mflux": mflux_version,
        "mlx": mx.__version__,
    }


def encode_real(model: str, prompt: str, root: Path) -> tuple[mx.array, mx.array]:
    """Encode ``prompt`` with the T5 and CLIP encoders loaded from the local snapshot ``root``.

    Loads the two encoder components through mflux's ``WeightLoader._load_component`` (the standard
    ``WeightLoader.load`` wants the transformer and VAE too), applies them with
    ``WeightApplier.apply_and_quantize(quantize_arg=None)``, builds the tokenizers with the T5
    length of ``model`` and calls ``PromptEncoder.encode_prompt``. Imports mflux here so the
    module (and ``--help``) needs no mflux.

    Raises:
        FileNotFoundError: ``root`` is not a directory.
    """
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
    from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
    from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
    from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

    if not root.is_dir():
        raise FileNotFoundError(
            f"{root}: not a directory (a local FLUX snapshot with the encoders)"
        )
    components = {c.name: c for c in FluxWeightDefinition.get_components()}
    save_subdirs = ComponentDefinition.save_subdirs(FluxWeightDefinition.get_components())
    raw_cache: dict[tuple, dict] = {}  # mflux's own annotation of the cache
    weights: dict[str, dict] = {}
    for name in ("t5_encoder", "clip_encoder"):
        loaded, _quantization, _version = WeightLoader._load_component(
            root, components[name], raw_cache, save_subdirs[name]
        )
        weights[name] = loaded
    t5, clip = T5Encoder(), CLIPEncoder()
    WeightApplier.apply_and_quantize(
        weights=LoadedWeights(
            components=weights, meta_data=MetaData(quantization_level=None, mflux_version=None)
        ),
        models={"t5_encoder": t5, "clip_encoder": clip},
        quantize_arg=None,
        weight_definition=FluxWeightDefinition,
    )
    tokenizers = TokenizerLoader.load_all(
        definitions=FluxWeightDefinition.get_tokenizers(),
        model_path=str(root),
        max_length_overrides={"t5": t5_max_length(model)},
    )
    return PromptEncoder.encode_prompt(
        prompt,
        prompt_cache={},
        t5_tokenizer=tokenizers["t5"],
        clip_tokenizer=tokenizers["clip"],
        t5_text_encoder=t5,
        clip_text_encoder=clip,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``--root`` is required unless ``--synthetic``."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=tuple(T5_LENGTHS), default="schnell")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--root", type=Path, help="local snapshot with the encoders and tokenizers")
    p.add_argument("--out", type=Path, required=True, help="safetensors file to write")
    p.add_argument("--synthetic", action="store_true", help="seeded random arrays, no encoders")
    args = p.parse_args(argv)
    if not args.synthetic and args.root is None:
        p.error("--root is required unless --synthetic")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point: encode (or synthesise), evaluate, and save with metadata."""
    args = parse_args(argv)
    start = time.perf_counter()
    try:
        if args.synthetic:
            prompt_embeds, pooled = synthetic_embeds(args.model, args.seed)
        else:
            caps = install_memory_caps()  # before the encoders (T5 alone is about 9.5 GB) load
            print(f"memory caps (wired, memory) GB: {caps}")
            prompt_embeds, pooled = encode_real(args.model, args.prompt, args.root)
        mx.eval(prompt_embeds, pooled)
        want_prompt, want_pooled = embed_shapes(args.model)
        if tuple(prompt_embeds.shape) != want_prompt or tuple(pooled.shape) != want_pooled:
            raise RuntimeError(
                f"encoded shapes {tuple(prompt_embeds.shape)} / {tuple(pooled.shape)}, "
                f"expected {want_prompt} / {want_pooled}"
            )
        md = build_metadata(
            model=args.model,
            seed=args.seed,
            prompt=args.prompt,
            root="synthetic" if args.synthetic else str(args.root.resolve()),
            token_length=want_prompt[1],
            synthetic=args.synthetic,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(args.out),
            {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled},
            metadata=md,
        )
    except Exception as exc:  # any failure is a tool error
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(
        f"ok: {args.out} prompt_embeds {tuple(prompt_embeds.shape)} {prompt_embeds.dtype}, "
        f"pooled_prompt_embeds {tuple(pooled.shape)} {pooled.dtype}, "
        f"{'synthetic' if args.synthetic else 'encoded'} in {time.perf_counter() - start:.1f} s"
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
