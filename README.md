# mlx-dfloat

Run [DFloat11](https://github.com/LeanModels/DFloat11) checkpoints on a Mac with [MLX](https://github.com/ml-explore/mlx).

DFloat11 is lossless compression for BF16 model weights. It entropy-codes the 8 exponent bits of every weight and
keeps the sign and mantissa as they are. The DFloat11 authors report models at about 70% of their BF16 size, with
output that is bit-for-bit the same as the original ([paper](https://arxiv.org/abs/2504.11651)). Their decoder runs
on NVIDIA GPUs only. This project is an independent reader and decoder for Apple Silicon: the weights stay
compressed in memory and are decoded on the GPU right before they are used.

Why that matters on a Mac: unified memory is the limit. A BF16 FLUX.1-dev transformer is about 24 GB. On a 32 GB
machine that is more than the GPU can comfortably hold. Lossless compression gets it under the line without the
quality trade-off of 4-bit or 8-bit quantization.

## Status

Pre-alpha. Nothing to install yet, and not on PyPI. The package is a skeleton. The first two milestones decide
whether the project goes ahead:

1. A bit-exact reference decoder for published DFloat11 checkpoints.
2. A Metal decode kernel that is fast enough to run inside an image-generation step.

If either one fails, the repository will say so and why.

## Relationship to DFloat11

This is not a fork and not affiliated with the DFloat11 authors. It reads the checkpoint format they publish. Credit
for the method goes to them: *70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via
Dynamic-Length Float* ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)).

## License

Apache-2.0. See [LICENSE](LICENSE).
