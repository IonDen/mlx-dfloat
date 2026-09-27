# mlx-dfloat

Run [DFloat11](https://github.com/LeanModels/DFloat11) checkpoints on a Mac with [MLX](https://github.com/ml-explore/mlx).

DFloat11 is lossless compression for BF16 model weights. Each weight is 16 bits. DFloat11 stores the 8 exponent bits
as a short variable-length code, the same idea a zip file uses, and keeps the other 8 bits (the sign and the
fraction) exactly as they are. The DFloat11 authors report models at about 70% of their BF16 size with output that is
bit for bit the same as the original.[^size] Their decoder runs on NVIDIA GPUs only. This project is an independent
reader and decoder for Apple Silicon. Once decoding is implemented, the weights will stay compressed in memory and be
decoded on the GPU right before they are used.

```
one BF16 weight, 16 bits:    s   eeeeeeee   mmmmmmm
                             |   |          |
                             |   exponent: replaced by a variable-length code   (compressed)
                             sign + fraction: kept as one raw byte              (stored as is)
```

Why that matters on a Mac: unified memory is the limit. The BF16 FLUX.1-dev transformer is about 24 GB,[^flux] more
than the GPU on a 32 GB machine can comfortably hold. At 70% it should come in under that line, and unlike 4-bit or
8-bit quantization it changes nothing in the output. Whether it really fits on a given Mac is something this project
has to measure, and the first measured numbers are in the status below.

## Status

Pre-alpha. Nothing to install yet, and not on PyPI. Two milestones decide whether the project goes ahead:

1. A bit-exact reference decoder for published DFloat11 checkpoints. **Done.** Every compressed tensor of Qwen3-4B,
   and sampled blocks of FLUX.1-schnell, FLUX.1-Krea-dev, Qwen-Image-Edit and Qwen-Image-Edit-2509, decode to
   exactly the BF16 originals. That covers all four published versions of the checkpoint format.
2. A Metal decode kernel fast enough to run inside an image-generation step. **Done.** The kernel decodes the
   published checkpoints bit-exactly at 50 GB/s on an M1 Max, and decoding every block just in time adds 5 %
   (FLUX.1-dev) to 6 % (FLUX.1-schnell) to a 1024² denoise step, measured against a control that agrees with a real
   BF16 run within 0.13 %. The recipe and numbers are in the changelog.

Both milestones passed; the next step is the mflux integration.

## Try it

There is nothing to install, but the parity check runs from a checkout. It compares a published DFloat11 repo with
its BF16 original over HTTP range reads, so it fetches only the blocks it checks instead of the whole model:

```
uv sync --group dev
uv run python -m scripts.verify_remote_group \
  --df11-repo DFloat11/FLUX.1-schnell-DF11 \
  --bf16-repo black-forest-labs/FLUX.1-schnell --bf16-subdir transformer \
  --groups max-code --out result.json
```

A few gigabytes move over the network, so expect a few minutes. The script exits 0 when every compared value
matches, 1 on a real mismatch, and 2 on any other problem such as a repo it cannot read. It runs under a memory
watchdog and stops itself with exit 70 or 71 if it gets close to the machine's memory or to its time limit. For a
checkpoint that is already on disk, `scripts/verify_checkpoint.py` (run the same way, with `--df11` and `--bf16`
directories) checks every tensor and writes one result file per group, so an interrupted run resumes where it
stopped.

## Relationship to DFloat11

This is not a fork and not affiliated with the DFloat11 authors. It reads the checkpoint format they publish. Credit
for the method goes to them: *70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via
Dynamic-Length Float* ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)).

## License

Apache-2.0. See [LICENSE](LICENSE).

[^size]: Reported in the DFloat11 paper ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)). The exponent bits of
    trained weights are far from uniformly distributed, which is what makes them compressible; the exact ratio
    depends on the model and is slightly different for each one.
[^flux]: The FLUX.1-dev transformer has about 12 billion parameters. In BF16 each takes 2 bytes, so about 24 GB for
    the weights alone, before activations and before the text encoders.
