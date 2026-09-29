# mlx-dfloat

Run [DFloat11](https://github.com/LeanModels/DFloat11) checkpoints on a Mac with [MLX](https://github.com/ml-explore/mlx).

DFloat11 is lossless compression for BF16 model weights. Each weight is 16 bits. DFloat11 stores the 8 exponent bits
as a short variable-length code, the same idea a zip file uses, and keeps the other 8 bits (the sign and the
fraction) exactly as they are. The DFloat11 authors report models at about 70% of their BF16 size with output that is
bit for bit the same as the original.[^size] Their decoder runs on NVIDIA GPUs only. This project is an independent
reader and decoder for Apple Silicon. The weights stay compressed in memory and a Metal kernel decodes them on the
GPU right before they are used; wiring that into mflux is the next step.

```
one BF16 weight, 16 bits:    s   eeeeeeee   mmmmmmm
                             |   |          |
                             |   exponent: replaced by a variable-length code   (compressed)
                             sign + fraction: kept as one raw byte              (stored as is)
```

Why that matters on a Mac: unified memory is the limit. The BF16 FLUX.1-dev transformer is about 24 GB,[^flux] more
than the GPU on a 32 GB machine can comfortably hold. At 70% it should come in under that line, and unlike 4-bit or
8-bit quantisation it changes nothing in the output. Whether it really fits on a given Mac is something this project
has to measure, and the first measured numbers are in the status below.

## Status

Pre-alpha. Nothing to install yet, and not on PyPI. Two milestones decide whether the project goes ahead:

1. A bit-exact reference decoder for published DFloat11 checkpoints. **Done.** Every compressed tensor of Qwen3-4B,
   and sampled blocks of FLUX.1-schnell, FLUX.1-Krea-dev, Qwen-Image-Edit and Qwen-Image-Edit-2509, decode to
   exactly the BF16 originals. That covers all four published versions of the checkpoint format.
2. A Metal decode kernel fast enough to run inside an image-generation step. **Done.** The kernel decodes the
   published checkpoints bit-exactly at 50 GB/s on an M1 Max. Decoding every block just in time makes a 1024²
   denoise step 5.0 % slower on FLUX.1-dev, 6.1 % slower on FLUX.1-schnell with a one-block evaluation run-ahead,
   and 8.5 % slower on FLUX.1-schnell with per-block evaluation, all well under the project's ~25 % threshold.
   That is measured against a control which,
   on a reduced-depth transformer, agrees within 0.13 % with a run whose BF16 weights are all resident, through
   the same per-block path. The recipe and numbers are in the changelog.
3. FLUX.1 image generation through mflux, straight from the compressed checkpoint to a saved image. **Done.**
   FLUX.1-schnell, FLUX.1-dev and FLUX.1-Krea-dev each generate a 1024² image on a 32 GB M1 Max within the
   machine's measured memory budget, and FLUX.1-schnell's output matches, bit for bit, the same transformer
   streaming its BF16 weights block by block instead of the compressed ones. The command and the measured numbers
   are under "Try it" below.

All three milestones are done.

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

### Generate a FLUX.1 image

Generation needs the optional `mlx-dfloat[mflux]` extra, which installs mflux 0.20.0 alongside it: `uv sync --extra
mflux` from a checkout (not on PyPI yet). The default repositories need no extra flags:

```
uv run mlx-dfloat generate --model schnell \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --steps 4 --height 1024 --width 1024 --report report.json
```

`--model` also takes `dev` and `krea-dev`. The command downloads that model's DFloat11 transformer and, from its
base repository, the text encoders, the VAE and the tokenizers; the base's own BF16 transformer is never fetched.
All three base repositories are gated on the Hub, so every model needs a Hub login (`hf auth login`, once).
FLUX.1-schnell's opens as soon as you accept its Apache-2.0 license on its Hub page. FLUX.1-dev's and
FLUX.1-Krea-dev's need a manual approval and come under Black Forest Labs' non-commercial license. Their text
encoders and VAE are byte-identical to FLUX.1-schnell's, so if you hold only the schnell license you can still run
dev or Krea-dev with `--base black-forest-labs/FLUX.1-schnell`. The dev and Krea-dev weights stay under the
non-commercial license whichever base supplies the encoders; you accept it on the FLUX.1-dev and FLUX.1-Krea-dev
Hub pages. Once everything is cached, set `HF_HUB_OFFLINE=1` to skip the Hub round trip that would otherwise check
for a newer revision on every run.

Measured on an M1 Max (32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, 2026-09-28/29) at 1024², one image per model:
FLUX.1-schnell (4 steps) peaked at 19.94 GiB and took 1 minute 54 seconds including imports; FLUX.1-dev (20 steps,
guidance 3.5) peaked at 20.10 GiB in 7 minutes 6 seconds; FLUX.1-Krea-dev (20 steps, guidance 3.5) peaked at 20.09
GiB in 7 minutes 2 seconds. The machine's budget for this, its recommended working set minus a 2 GiB reserve, is
22.96 GiB, so all three stay under it. `scripts/verify_image.py` checked FLUX.1-schnell's output against the same
transformer streaming its BF16 shards block by block instead of the compressed set: the final latents matched bit
for bit and the two images matched pixel for pixel. FLUX.1-dev and FLUX.1-Krea-dev were not checked this way,
because their BF16 transformers are gated and were not on disk to compare against.

Generating a second image in the same process pays a reload. On a 32 GB Mac every call drops the compressed
transformer before the VAE decode, whatever the image size: at 1024² holding it next to the decode measured 23.29
GiB, over the budget, and smaller sizes have not been measured, so the estimate assumes the decode needs at least as
much there. The next call
reloads it, about 26 seconds. A Mac with a larger budget keeps the transformer loaded between calls. Even a repeated
prompt pays the reload: three consecutive 4-step FLUX.1-schnell calls with the same prompt each took about 103
seconds and peaked between 19.77 and 19.98 GiB. A different prompt adds a further
reload, this time of the text encoders instead of the transformer, since the two are never resident together.
`model.encode(*prompts)` pays that encoder reload once for several prompts, by encoding all of them while the
compressed transformer is not yet loaded.

Sizes above 1024² are refused for now, because no run above 1024² has been measured on this path. `--no-fit-check`
(`fit_check=False` in Python) runs them anyway, on a memory estimate that is then an extrapolation.

Quantisation on top of DFloat11 is refused because it would change the exact bits the format exists to preserve;
LoRA, img2img and ControlNet are refused because this path does not implement them; PiD decoding is refused
because it would need an 8 GB caption encoder resident next to the compressed transformer. `--negative-prompt` is
accepted and ignored, matching mflux's own FLUX.1 behavior. The command never overwrites an existing output file;
it picks a new name instead, and the report names the file it wrote.

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
