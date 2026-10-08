# mlx-dfloat

[![PyPI version](https://img.shields.io/pypi/v/mlx-dfloat.svg)](https://pypi.org/project/mlx-dfloat/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-dfloat.svg)](https://pypi.org/project/mlx-dfloat/)
[![License: Apache-2.0](https://img.shields.io/pypi/l/mlx-dfloat.svg)](https://github.com/IonDen/mlx-dfloat/blob/main/LICENSE)

Run [DFloat11](https://github.com/LeanModels/DFloat11) checkpoints on a Mac with
[MLX](https://github.com/ml-explore/mlx). The weights stay about 30 % smaller than BF16 in memory, and the GPU decodes
each block back to exactly the same bits just before it runs. Today that means FLUX.1 image generation on a 32 GB Mac;
Z-Image, FLUX.2 Klein, Qwen-Image-2.1 and ERNIE-Image arrive in the next release.

DFloat11 is lossless compression for BF16 model weights. Each weight is 16 bits. DFloat11 stores the 8 exponent bits
as a short variable-length code, the same idea a zip file uses, and keeps the other 8 bits (the sign and the
fraction) exactly as they are. The DFloat11 authors report models at about 70% of their BF16 size with output that is
bit for bit the same as the original.[^size] Their decoder runs on NVIDIA GPUs only. This project is an independent
reader and decoder for Apple Silicon. The weights stay compressed in memory, and a Metal kernel decodes each block
on the GPU as it runs. That is how it generates FLUX.1 images through mflux; Z-Image, FLUX.2 Klein, Qwen-Image-2.1 and
ERNIE-Image follow in the next release.

```
one BF16 weight, 16 bits:    s   eeeeeeee   mmmmmmm
                             |   |          |
                             |   exponent: replaced by a variable-length code   (compressed)
                             sign + fraction: kept as one raw byte              (stored as is)
```

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-dfloat/main/docs/images/how-it-works.svg" alt="Diagram: the
DFloat11 FLUX.1 transformer stays compressed in a Mac's memory, and the GPU unpacks each of its 57 blocks to BF16 just
before it runs." width="720">
</p>

Why that matters on a Mac: unified memory is the limit. The BF16 FLUX.1-dev transformer is about 24 GB,[^flux] more
than the GPU on a 32 GB machine can comfortably hold. At 70% it comes in under that line, and unlike 4-bit or 8-bit
quantisation it changes nothing in the weights. On a 32 GB M1 Max a 1024² generation peaked at about 20 GiB for each
of the three FLUX.1 models; no other Mac has been measured. The numbers are under "Measured numbers" below.

## Does this help me?

| Your situation | What to use |
|---|---|
| A 32 GB Mac, FLUX.1-schnell, FLUX.1-dev or FLUX.1-Krea-dev, and you want the BF16 weights unchanged | mlx-dfloat. Measured peak about 20 GiB at 1024² for all three. The final latents were checked bit for bit against BF16 on FLUX.1-schnell, not yet on the other two. Unpacking makes a denoising step 3.5 % slower on dev and 4.0 % slower on schnell than a control that runs the same graph on weights decoded in advance (Krea-dev's step was not timed). |
| Less memory or a faster step, and a different image is fine | mflux's own quantised transformer. Its 8-bit mode, the only one measured here, ran a step roughly 1.3 times faster and peaked about 5 GiB lower, under the same 2.5 GB cache limit; mflux's own generate sets no cache limit, so this is not quite its default setup (see "Measured numbers" for why the ratio is rough). It changes the weights and the output; by how much was not measured. |
| A 32 GB Mac, Z-Image-Turbo or Z-Image, with the weights unchanged | mlx-dfloat, from the next release. Peak at 1024²: 13.1 to 14.3 GiB for Z-Image-Turbo over six runs, 13.15 GiB for Z-Image. See "Generate a Z-Image image". |
| A 32 GB Mac, FLUX.2 Klein (4B or 9B, base or distilled), with the weights unchanged | mlx-dfloat, from the next release. Peak at 1024²: 11.88 to 12.78 GiB for the 4B models, 17.60 to 18.40 GiB for the 9B models. See "Generate a FLUX.2 Klein image". |
| A 24 GB Mac, FLUX.2 Klein | CAPPED: FLUX.2-klein-base-4B passed, peak 12.29 GiB; FLUX.2-klein-base-9B is at risk (the fit check refuses it, and a forced run was stopped by the watchdog). See "Generate a FLUX.2 Klein image". |
| A 16 GB Mac, FLUX.2-klein-base-4B | CAPPED: passed, peak 8.15 GiB. See "Generate a FLUX.2 Klein image". |
| A 16 GB Mac, Z-Image-Turbo | CAPPED: at risk. The fit check refuses it, and a forced run was stopped by the watchdog at 9.26 GiB before the transformer ran. See "Generate a Z-Image image". |
| A 32 GB Mac, Qwen-Image-2.1, with the weights unchanged | mlx-dfloat, from the next release. Peak at 1024²: 20.42 GiB. See "Generate a Qwen-Image-2.1 image". |
| A 16 GB or 24 GB Mac, Qwen-Image-2.1 | CAPPED: at risk. The fit check refuses it, and forced runs were stopped by the watchdog while the prompt was being encoded, before the transformer ran. See "Generate a Qwen-Image-2.1 image". |
| A 32 GB Mac, ERNIE-Image or ERNIE-Image-Turbo, with the weights unchanged | mlx-dfloat, from the next release. Peak at 1024²: 16.79 GiB for ERNIE-Image (50 steps), 17.34 GiB for ERNIE-Image-Turbo (8 steps). See "Generate an ERNIE-Image image". |
| A 24 GB Mac, ERNIE-Image-Turbo | CAPPED: at risk. The fit check refuses it, 0.39 GiB over the budget; a forced run passed with a 14.39 GiB peak, 0.11 GiB under the watchdog ceiling. See "Generate an ERNIE-Image image". |
| A 24 GB Mac, ERNIE-Image | CAPPED: at risk. The fit check refuses it, and a forced run was stopped by the watchdog. See "Generate an ERNIE-Image image". |
| A 16 GB Mac, ERNIE-Image or ERNIE-Image-Turbo | CAPPED: at risk. The fit check refuses both, and forced runs were stopped by the watchdog. See "Generate an ERNIE-Image image". |
| A Mac with room for the BF16 transformer next to everything else | Plain mflux in BF16. There is nothing to decode. |
| A Mac with less than 32 GB | Not measured. The rows marked CAPPED are runs on the 32 GB Mac under the caps mlx-dfloat installs on a 16 GB or 24 GB Mac; the tables below make no other claim for those Macs. |
| Another model: an LLM, the original Qwen-Image or Qwen-Image-Edit, FLUX.2-dev, Krea-2 | Not wired up yet. The reader and the decoder handle all four published versions of the DFloat11 format, but generation in 0.1.0 covers FLUX.1 only. The next release adds Z-Image-Turbo, Z-Image, the four FLUX.2 Klein models, Qwen-Image-2.1, ERNIE-Image and ERNIE-Image-Turbo. |

## Status

Pre-alpha: version 0.1.0 is on PyPI.

Released in 0.1.0:

- All four published versions of the DFloat11 checkpoint format decode to exactly the BF16 originals: every
  compressed tensor of Qwen3-4B, and sampled blocks of FLUX.1-schnell, FLUX.1-Krea-dev, Qwen-Image-Edit and
  Qwen-Image-Edit-2509.
- A Metal kernel decodes to the same bits as the CPU reference decoder, at about 50 GB/s on an M1 Max on its default
  path, timed on Qwen3-4B and FLUX.1-schnell groups with `uv run python -m scripts.bench_decode_kernel --df11
  <checkpoint dir> --groups <group names> --out decode.json`.
- FLUX.1-schnell, FLUX.1-dev and FLUX.1-Krea-dev make 1024² images through mflux, from the command line or from
  Python. For FLUX.1-schnell, the final latents match the same transformer run block by block from its BF16 weights
  bit for bit, and the saved images match pixel for pixel.
- One command per scenario reruns the step benchmark under "Measured numbers". A tier is a Mac memory size (16, 24
  or 32 GB); the tier table has one row per model and tier, and each row is one `mlx-dfloat generate` run.
- `mlx-dfloat generate`, the parity scripts and the benchmark run under a memory watchdog that stops them when they
  cross its ceiling.

In the next release (not yet on PyPI):

- Z-Image-Turbo and Z-Image make 1024² images through mflux, from the command line or from Python. For Z-Image, every
  one of the 271 compressed matrices and 250 other tensors equals the BF16 original
  (`bench/results/parity/z-image/full-vs-bf16.json`). Its final latents (1024², 4 steps,
  guidance 4.0) match bit for bit, and its images pixel for pixel, the same transformer streaming its BF16 weights one
  block at a time; both sides ran without mflux's step compilation (record:
  `bench/results/identity/z-image-1024/compare.json`). Z-Image-Turbo has no BF16 original, because its
  published transformer is FP32. Its 271 matrices decode on the GPU to the same bits as the CPU reference
  (`bench/results/parity/z-image-turbo/kernel-vs-reference.json`), and five sampled groups (32 of the 271 matrices)
  equal the FP32 original rounded to BF16, nearest even
  (`bench/results/parity/z-image-turbo/sampled-vs-fp32-rounded.json`). Turbo's latents have not been compared with a
  reference yet.
- FLUX.2-klein-base-4B, FLUX.2-klein-4B, FLUX.2-klein-base-9B and FLUX.2-klein-9B make 1024² images through mflux, from
  the command line or from Python. For FLUX.2-klein-base-4B, every one of the 105 compressed matrices and 64 other
  tensors equals the BF16 original (`bench/results/parity/flux2-klein-base-4b/full-vs-bf16.json`), and its final
  latents (1024², 4 steps, guidance 4.0) match bit for bit, and its images pixel for pixel, the same transformer
  streaming its BF16 weights one block at a time; both sides ran without mflux's step compilation (record:
  `bench/results/identity/flux2-klein-base-4b-1024/compare.json`). For the other
  three, the GPU decoder gives the same bits as the CPU reference on every compressed matrix (105 for FLUX.2-klein-4B,
  149 for each 9B model), and sampled groups equal the BF16 originals: 29 matrices for FLUX.2-klein-4B, 41 for
  FLUX.2-klein-base-9B and 30 for FLUX.2-klein-9B. Each model's two records are `kernel-vs-reference.json` and
  `sampled-vs-bf16.json` in `bench/results/parity/<model>/` (`flux2-klein-4b`, `flux2-klein-base-9b`, `flux2-klein-9b`).
  Their latents have not been compared with a reference.
- Qwen-Image-2.1 makes 1024² images through mflux, from the command line or from Python. Its DFloat11 checkpoint
  (`mingyi456/Qwen-Image-2.1-DF11-ComfyUI`) is a single file made for ComfyUI, with no `config.json` next to it.
  mlx-dfloat carries the missing layout for that one published file, pinned to one revision on the Hub, and refuses a
  file without a `config.json` that it does not recognise. Every one of the 225 compressed matrices and 72 other tensors
  equals the BF16 original in `Qwen/Qwen-Image-2.1` (`bench/results/parity/qwen-image-2.1/full-vs-bf16.json`), and the
  GPU decoder gives the same bits as the CPU reference on all 225
  (`bench/results/parity/qwen-image-2.1/kernel-vs-reference.json`). Its final latents match, bit for bit, those of
  the same transformer streaming its BF16 weights one block at a time, and the images match pixel for pixel. That check
  ran at 1024² for 4 steps at guidance 4.0. The negative prompt was a single space, so classifier-free guidance ran
  (record: `bench/results/identity/qwen-image-2.1-1024/compare.json`). `scripts/verify_image.py` runs both sides
  without mflux's step compilation.
- ERNIE-Image and ERNIE-Image-Turbo make 1024² images through mflux, from the command line or from Python. For
  ERNIE-Image, every one of the 256 compressed matrices and 153 other tensors equals the BF16 original in
  `baidu/ERNIE-Image` (`bench/results/parity/ernie-image/full-vs-bf16.json`), and the GPU decoder gives the same bits as
  the CPU reference on all 256 (`bench/results/parity/ernie-image/kernel-vs-reference.json`). Its final latents match,
  bit for bit, those of the same transformer streaming its BF16 weights one block at a time, and the images match pixel
  for pixel. That check ran at 1024² for 4 steps at guidance 4.0, so classifier-free guidance ran (record:
  `bench/results/identity/ernie-image-1024/compare.json`). For ERNIE-Image-Turbo, all 256 compressed matrices equal the
  BF16 original in `baidu/ERNIE-Image-Turbo` as well, read from the Hub one piece at a time
  (`bench/results/parity/ernie-image-turbo/full-vs-bf16.json`), and the GPU decoder matches the CPU reference on all
  256 (`bench/results/parity/ernie-image-turbo/kernel-vs-reference.json`). Turbo's latents were not compared with a
  reference: that check needs its 16 GB BF16 transformer on disk.
- `mlx-dfloat generate --tier 16` and `--tier 24` run under the memory caps mlx-dfloat installs on a Mac of that size.
  The 16 GB and 24 GB rows under "Measured numbers" are CAPPED runs on the 32 GB Mac under those caps; no smaller Mac
  was measured.
- `mlx-dfloat selftest` checks the GPU decoder on your Mac (see "Check the GPU decoder on your Mac").

Earlier step-time numbers, measured with a smaller MLX cache limit, are in the 0.1.0 entry of the changelog.

## Install

mlx-dfloat is for Macs with Apple Silicon and needs Python 3.11 or newer. Install it from PyPI:

```
pip install "mlx-dfloat[mflux]"
```

Without the extra you get the checkpoint reader and the decoder. The `mflux` extra installs mflux 0.20, which the
`mlx-dfloat generate` command needs for FLUX.1, Z-Image, FLUX.2 Klein, Qwen-Image-2.1 and ERNIE-Image generation.

The parity scripts and the benchmark live in the repository, not in the package, so run them from a checkout with
[uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/IonDen/mlx-dfloat
cd mlx-dfloat
uv sync --extra mflux
```

A plain `uv sync` installs the reader, the decoder and the parity scripts, and `--extra mflux` adds generation. The
benchmark runs with `uv run --group bench`, which installs mflux as well.

Every FLUX.1 base repository on the Hugging Face Hub is gated, so log in once with `hf auth login` and accept the
model's licence on its Hub page (details under "Generate a FLUX.1 image"). On disk, a DFloat11 FLUX.1 transformer
takes about 16 GB, and the base's text encoders, VAE and tokenizers about 10 GB more. The benchmark's `q8`
condition also needs the base's BF16 transformer, about 24 GB, which generation itself never downloads.

## Try it

The parity check compares a published DFloat11 repo with its BF16 original over HTTP range reads, so it fetches
only the blocks it checks instead of the whole model:

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

### Check the GPU decoder on your Mac

```
mlx-dfloat selftest
```

From a checkout, prefix it with `uv run`. The command decodes two groups that ship inside the package: one cut from a
real Qwen3-4B checkpoint, and one built to use the longest bit codes the format allows, spread over more than seven
blocks. Each is decoded two ways on the GPU (the staged and direct write paths) and once by the CPU reference, and every
result is compared with bits known in advance. It exits 0 when every check passes, 1 when a check fails, and 2 when the
GPU decoder cannot run at all, for example because there is no Metal device. `--json` prints the same report as JSON.

The same check runs automatically the first time a process decodes on the GPU, once for each write path, and when a
FLUX.1, Z-Image, FLUX.2 Klein, Qwen-Image-2.1 or ERNIE-Image model or the `generate` command sets up its decoder, so a
broken GPU decoder refuses before any model loads. On an M1 Max the check itself took about 13 ms, measured after the
kernel pipelines were compiled; the one-time compile took about 0.4 s with a cold shader cache and 0.03 s with a warm
one. It does not check whether a real model fits in your memory or how fast it runs; the numbers below cover that.

### Generate a FLUX.1 image

Generation needs the `mflux` extra (`pip install "mlx-dfloat[mflux]"`, see "Install"). The default repositories need
no extra flags:

```
mlx-dfloat generate --model schnell \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --steps 4 --height 1024 --width 1024 --report report.json
```

From a checkout, prefix the command with `uv run`. `--model` also takes `dev` and `krea-dev`. The command downloads that
model's DFloat11 transformer and, from its base repository, the text encoders, the VAE and the tokenizers; the base's
own BF16 transformer is never fetched. All three base repositories are gated on the Hub, so every model needs a Hub
login (`hf auth login`, once). FLUX.1-schnell's opens as soon as you accept its Apache-2.0 license on its Hub page.
FLUX.1-dev's and FLUX.1-Krea-dev's need a manual approval and come under Black Forest Labs' non-commercial license.
Their text encoders and VAE are byte-identical to FLUX.1-schnell's, so if you hold only the schnell license you can
still run dev or Krea-dev with `--base black-forest-labs/FLUX.1-schnell`. The dev and Krea-dev weights stay under the
non-commercial license whichever base supplies the encoders; you accept it on the FLUX.1-dev and FLUX.1-Krea-dev Hub
pages. Once everything is cached, set `HF_HUB_OFFLINE=1` to skip the Hub round trip that would otherwise check for a
newer revision on every run.

The same from Python:

```python
from mlx_dfloat.mflux import DFloatFlux1

model = DFloatFlux1("schnell")  # or "dev", "krea-dev"
image = model.generate_image(
    seed=42,
    prompt="A stone lighthouse on a rocky shore at dawn",
    num_inference_steps=4,
    height=1024,
    width=1024,
)
image.save("lighthouse.png")
```

FLUX.1-dev and FLUX.1-Krea-dev want more steps; the measured runs below used 20 steps and `guidance=3.5`.
`base_path=` takes the `--base` value, and `fit_check=False` replaces `--no-fit-check` (see below). From Python, each
`generate_image` call runs under the same memory caps as `mlx-dfloat generate` and puts MLX's limits back when it
returns; a wired limit your process already set is left alone. Only the command adds a memory watchdog, so prefer it
on a Mac near its memory limit.

Measured on an M1 Max (32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0, 2026-09-28/29) at 1024², one image per model:
FLUX.1-schnell (4 steps) peaked at 19.94 GiB and took 1 minute 54 seconds including imports; FLUX.1-dev (20 steps,
guidance 3.5) peaked at 20.10 GiB in 7 minutes 6 seconds; FLUX.1-Krea-dev (20 steps, guidance 3.5) peaked at 20.09
GiB in 7 minutes 2 seconds. The machine's budget for this, its recommended working set minus a 2 GiB reserve, is
22.96 GiB, so all three stay under it. `scripts/verify_image.py` checked FLUX.1-schnell's output against the same
transformer streaming its BF16 shards block by block instead of the compressed set: the final latents matched bit
for bit and the two images matched pixel for pixel. FLUX.1-dev and FLUX.1-Krea-dev were not checked this way,
because their BF16 transformers are gated and were not on disk to compare against. A later run of each model, the
one recorded under "Measured numbers", peaked within 0.2 GiB of these figures.

Generating a second image in the same process pays a reload. On a 32 GB Mac every call drops the compressed transformer
before the VAE decode, whatever the image size: at 1024² holding it next to the decode measured 23.29 GiB, over the
budget, and smaller sizes have not been measured, so the estimate assumes the decode needs at least as much there. The
next call reloads it, about 26 seconds. A Mac with a larger budget keeps the transformer loaded between calls. Even a
repeated prompt pays the reload: three consecutive 4-step FLUX.1-schnell calls with the same prompt each took about 103
seconds and peaked between 19.77 and 19.98 GiB. A different prompt adds a further reload, this time of the text encoders
instead of the transformer, since the two are never resident together. `model.encode(*prompts)` pays that encoder reload
once for several prompts, by encoding all of them while the compressed transformer is not yet loaded.

Sizes above 1024² are refused for now, because no run above 1024² has been measured on this path. `--no-fit-check`
(`fit_check=False` in Python) runs them anyway, on a memory estimate that is then an extrapolation.

Quantisation on top of DFloat11 is refused because it would change the exact bits the format exists to preserve; LoRA,
img2img and ControlNet are refused because this path does not implement them. mflux's alternative image decoder
(`--pid-decode`) is refused because it would need an 8 GB caption encoder resident next to the compressed transformer.
`--negative-prompt` is accepted and ignored, matching mflux's own FLUX.1 behavior. The command never overwrites an
existing output file; it picks a new name instead, and the report names the file it wrote.

### Generate a Z-Image image

Z-Image-Turbo (9 steps) and Z-Image (the base model, 50 steps) run through the same command. Both are Apache-2.0 and
ungated on the Hub ([Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image),
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)), so no Hub login is needed:

```
mlx-dfloat generate --model z-image-turbo \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --base Tongyi-MAI/Z-Image --seed 42 --height 1024 --width 1024 --report report.json
```

For the base model use `--model z-image --steps 50 --guidance 4`. Without `--guidance` the base model runs at 0, as
mflux does; its model card suggests about 4.

`--base Tongyi-MAI/Z-Image` takes Turbo's text encoder, tokenizer and VAE from the base model's repository. They are the
same weight files (the Hub lists the same hashes for them), so one download serves both models. The VAE's `config.json`
differs only in metadata (`_diffusers_version` and `_name_or_path`). The measured Turbo runs used `--base
Tongyi-MAI/Z-Image`; generation from Turbo's own default base, `Tongyi-MAI/Z-Image-Turbo`, has not been run here.

The same from Python:

```python
from mlx_dfloat.mflux import DFloatModel

model = DFloatModel("z-image-turbo")  # or "z-image"
image = model.generate_image(
    seed=42,
    prompt="A stone lighthouse on a rocky shore at dawn",
    num_inference_steps=9,
    height=1024,
    width=1024,
)
image.save("lighthouse.png")
```

The DFloat11 transformer of each model is 7.8 GiB (8.4 GB) compressed. The BF16 text encoder is 7.5 GiB on disk, and
MLX's own peak while encoding the prompt was 8.4–8.6 GiB in the 32 GB runs; the VAE and tokenizer are small. mflux
compiles Z-Image's denoising step into one GPU program on M1 and M2 Max and Ultra chips and on M3 and later. This path
runs the step without that compilation on every chip, because it unpacks each block's weights just before the block
runs, and a compiled step cannot pause between blocks to do that. What that costs against stock mflux was not measured:
the only Mac measured here is an M1 Max, and no Z-Image step time is reported.

Measured on an M1 Max (32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0) at 1024², seed 42, one image per model:
Z-Image-Turbo (9 steps) peaked at 13.11 GiB and Z-Image (50 steps, guidance 4.0) at 13.15 GiB. In both runs MLX's own
memory peaked during the VAE decode. The peak varies between runs of the same command: six runs of the Turbo command on
2026-10-08 peaked between 13.11 and 14.26 GiB, and the difference came from the VAE decode. The five runs besides the
tier row's are in `bench/results/repeats/z-image-turbo-1024/`.

What was checked differs between the two models and is listed under "Status". For Z-Image, the final latents and the
image match the same transformer running on its BF16 weights (`bench/results/identity/z-image-1024/compare.json`).
Z-Image-Turbo has no BF16 original to compare with.

On a 16 GB Mac, Z-Image-Turbo is at risk. No 16 GB Mac was measured: the run behind that row is CAPPED, made on the 32
GB M1 Max under the caps mlx-dfloat installs on a 16 GB Mac and the tier's watchdog ceiling (`--tier 16`). Without
`--no-fit-check` the fit check refuses the run: it predicts a 12.2 GiB peak against a 9.17 GiB budget
(`bench/results/refusals/z-image-turbo-1024-tier16.json`). That run used the older, looser budget for a 16 GB Mac;
`generate` now uses 8.67 GiB there, which refuses it too. With the flag, the watchdog stopped the run after 4.7 s, at
the start of loading the transformer weights (the abort file names the phase), with a footprint of 9.26 GiB against the
9.17 GiB ceiling, while MLX held almost nothing (9 MB). The transformer never ran, so this run says nothing about
whether it would fit.

Quantisation, LoRA, img2img, mflux's alternative image decoder (`--pid-decode`) and ControlNet are refused, as for
FLUX.1. Sizes above 1024² are refused unless you pass `--no-fit-check`, because none has been measured.

### Generate a FLUX.2 Klein image

FLUX.2 Klein comes in two sizes, 4B and 9B, each as a base model and as a distilled model that needs far fewer steps:

| `--model` | Kind | Steps | Guidance | License |
|---|---|---|---|---|
| `flux2-klein-4b` | distilled | 4 | 1.0 only | Apache-2.0, ungated ([Hub page](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)) |
| `flux2-klein-base-4b` | base | 50 | pass `--guidance 4` | Apache-2.0, ungated ([Hub page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)) |
| `flux2-klein-9b` | distilled | 4 | 1.0 only | FLUX Non-Commercial License, gated ([Hub page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)) |
| `flux2-klein-base-9b` | base | 50 | pass `--guidance 4` | FLUX Non-Commercial License, gated ([Hub page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)) |

```
mlx-dfloat generate --model flux2-klein-4b \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report report.json
```

For a base model, add `--steps 50 --guidance 4`, as the Black Forest Labs model cards do. Without `--guidance` a
base model runs at 1.0, as mflux does, which skips classifier-free guidance. A distilled model runs at guidance 1.0
only, and the command refuses any other value, as mflux's own command does. `--scheduler` is refused for every
FLUX.2 Klein model, because mflux's FLUX.2 Klein command has no scheduler option.

The 9B weights come under Black Forest Labs' FLUX Non-Commercial License, and their Black Forest Labs repositories
are gated: a 9B model needs a Hub login (`hf auth login`, once) and the license accepted on the Hub page of the
repository that supplies its text encoder and VAE. The 4B models need neither.

Each model takes its text encoder, VAE and tokenizer from its own BF16 repository on the Hub, without downloading that
repository's transformer. A base model and the distilled model of the same size have the same text-encoder and VAE
files, and the 4B pair the same tokenizer files too (the Hub lists the same hashes for them), so `--base
black-forest-labs/FLUX.2-klein-base-4B` works for `flux2-klein-4b`, and `--base black-forest-labs/FLUX.2-klein-base-9B`
for `flux2-klein-9b`. The measured distilled runs used the base model's files that way; generation from a distilled
model's own default repository has not been run here. On disk, the DFloat11 transformer is 5.26 GB (4B) or 12.31 GB
(9B), and the text encoder about 8 GB (4B) or 16.4 GB (9B).

The same from Python:

```python
from mlx_dfloat.mflux import DFloatModel

model = DFloatModel("flux2-klein-4b")  # or another --model name from the table above
image = model.generate_image(
    seed=42,
    prompt="A stone lighthouse on a rocky shore at dawn",
    num_inference_steps=4,
    height=1024,
    width=1024,
)
image.save("lighthouse.png")
```

For a base model pass `num_inference_steps=50, guidance=4.0`. Python does not refuse another guidance on a distilled
model, just as mflux's Python API does not.

As with Z-Image, this path runs the denoising step without mflux's step compilation (see "Generate a Z-Image image"
for why), and what that costs against stock mflux was not measured.

Measured on an M1 Max (32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0) at 1024², seed 42, one image per model, with
the commands under "Measured numbers": FLUX.2-klein-4B (4 steps) peaked at 11.88 GiB in 35 s, FLUX.2-klein-base-4B
(50 steps, guidance 4.0) at 12.78 GiB in 8.9 minutes, FLUX.2-klein-9B (4 steps) at 18.40 GiB in 1.3 minutes, and
FLUX.2-klein-base-9B (50 steps, guidance 4.0) at 17.60 GiB in 19.9 minutes. The times are the `elapsed_seconds` of
each run's report, from `bench/results/tiers/flux2-klein-4b-1024.json` to `flux2-klein-base-9b-1024.json`. In all
four runs MLX's own memory peaked during the VAE decode, and no run had to drop the compressed transformer before it.

What was checked differs per model and is listed under "Status": only FLUX.2-klein-base-4B had its final latents
compared with the same transformer running on its BF16 weights
(`bench/results/identity/flux2-klein-base-4b-1024/compare.json`).

No 16 GB or 24 GB Mac was measured. The smaller-Mac rows are CAPPED runs on the 32 GB Mac under the caps mlx-dfloat
installs on those Macs. FLUX.2-klein-base-4B (50 steps, guidance 4.0) passed under both: it peaked at 8.15 GiB under a
16 GB Mac's caps, where it dropped the compressed transformer before the VAE decode to stay under the budget, and at
12.29 GiB under a 24 GB Mac's. FLUX.2-klein-base-9B is at risk on a 24 GB Mac. The fit check refuses it, predicting
16.3 GiB while denoising against a 14.50 GiB budget (`bench/results/refusals/flux2-klein-base-9b-1024-tier24.json`).
That run used the older, looser budget for a 24 GB Mac; `generate` now uses 14.00 GiB there, which refuses it too. A
forced run (`--no-fit-check`) was stopped by the watchdog while denoising, at 14.64 GiB against the 14.50 GiB
ceiling. Under a 16 GB Mac's caps only FLUX.2-klein-base-4B was run, and under a 24 GB Mac's only the two base models.

Quantisation, LoRA, img2img, mflux's alternative image decoder (`--pid-decode`) and ControlNet are refused, as for
FLUX.1. FLUX.2 Klein's image-editing and KV-cache variants are not on this path. `--negative-prompt` is ignored with
a warning: a base model above guidance 1.0 runs its negative branch on mflux's blank prompt and takes no custom one.
Sizes above 1024² are refused unless you pass `--no-fit-check`, because none has been measured.

### Generate a Qwen-Image-2.1 image

Qwen-Image-2.1 runs through the same command. Its BF16 repository and the DFloat11 checkpoint are both ungated on the
Hub, so no Hub login is needed. Both come under the Qwen Research License Agreement
([license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)).

```
mlx-dfloat generate --model qwen-image-2.1 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report report.json
```

The defaults are mflux's own for this model: 40 steps, guidance 1.0 and the `linear` scheduler. At guidance 1.0 each
step calls the transformer once, with no classifier-free guidance. As in mflux, classifier-free guidance needs both a
`--guidance` above 1 and a `--negative-prompt`. Pass a guidance above 1 without a negative prompt and the command warns
that no classifier-free guidance runs; pass a negative prompt at guidance 1 or below and it warns that the prompt has no
effect.

The command takes the DFloat11 transformer from that checkpoint (10.9 GB for either model, as listed on the Hub) and the
text encoder, VAE and tokenizer from the model's BF16 repository. It does not download that repository's BF16
transformer (16.1 GB) or its prompt enhancer (`pe/`, 7.7 GB); mflux uses neither. The text-encoder file is 7.7 GB and
also holds a vision model that is never loaded. Both BF16 repositories serve the same text-encoder file, so it is stored
once. The VAE is 0.17 GB. These sizes are the files as listed on the Hub. The default checkpoints and BF16 repositories
are pinned to the revisions the measured runs used; a `--df11` or `--base` you pass is used as given.

The same from Python:

```python
from mlx_dfloat.mflux import DFloatModel

model = DFloatModel("qwen-image-2.1")
image = model.generate_image(
    seed=42,
    prompt="A stone lighthouse on a rocky shore at dawn",
    height=1024,
    width=1024,
)
image.save("lighthouse.png")
```

`num_inference_steps` defaults to 40 here too. For classifier-free guidance pass a `guidance` above 1.0 and a
`negative_prompt`.

mflux compiles Qwen-Image-2.1's denoising step. As with Z-Image and FLUX.2 Klein, this path runs the step
without that compilation (see "Generate a Z-Image image" for why), and what that costs against stock mflux was not
measured.

Measured on an M1 Max (32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0) at 1024², seed 42, 40 steps at guidance 1.0,
with the command under "Measured numbers": the run peaked at 20.42 GiB and took 6.9 minutes (the `elapsed_seconds` of
`bench/results/tiers/qwen-image-2.1-1024.json`). MLX's own memory peaked during the VAE decode. On this 32 GB Mac the
compressed transformer stays loaded through the VAE decode: the fit estimate predicts 21.2 GiB for that step (the
report's `fit.phases.vae`), under the 22.96 GiB budget.

On a 16 GB or 24 GB Mac, Qwen-Image-2.1 is at risk. No such Mac was measured: the runs behind those rows are CAPPED,
made on the 32 GB M1 Max under the caps mlx-dfloat installs on a Mac of that size (`--tier 16` and `--tier 24`). The fit
check refuses both. It predicts a 14.6 GiB peak while the prompt is encoded, against a budget of 8.67 GiB at 16 GB and
14.00 GiB at 24 GB; the text encoder alone holds 14.1 GiB. Forced runs (`--no-fit-check`) were stopped by the
watchdog while the prompt was being encoded, before the transformer ran. At 16 GB that came after 4.4 s, when MLX's
active plus cached memory reached 9.28 GiB against the 9.17 GiB ceiling. At 24 GB it came after 4.8 s, with a
footprint of 14.59 GiB against the 14.50 GiB ceiling.

Quantisation, LoRA, img2img, mflux's alternative image decoder (`--pid-decode`) and ControlNet are refused, as for
FLUX.1. Only text-to-image is on this path; Qwen-Image-2.1's image editing is not. Sizes above 1024² are refused unless
you pass `--no-fit-check`, because none has been measured.

### Generate an ERNIE-Image image

ERNIE-Image and ERNIE-Image-Turbo run through the same command. Their BF16 repositories (`baidu/ERNIE-Image` and
`baidu/ERNIE-Image-Turbo`) and the DFloat11 checkpoints (`mingyi456/ERNIE-Image-DF11` and
`mingyi456/ERNIE-Image-Turbo-DF11`) are ungated on the Hub, so no Hub login is needed. All four list the Apache-2.0
licence on the Hub ([ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image),
[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo)).

```
mlx-dfloat generate --model ernie-image-turbo \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report report.json
mlx-dfloat generate --model ernie-image \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report report.json
```

The defaults are mflux's own for each model, and they match the steps and guidance each model card recommends at
1024x1024 ([ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image): 50 steps at guidance 4.0;
[ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo): 8 steps at guidance 1.0). ERNIE-Image-Turbo refuses
any other `--guidance`, as mflux's Turbo command does, and ignores a `--negative-prompt` with a warning. Above guidance
1, ERNIE-Image runs classifier-free guidance whether or not you pass a negative prompt; without one, mflux uses a single
space. Each step runs the prompt and the negative prompt together, as one batched transformer call. Both models use the
`linear` scheduler. An empty or blank `--prompt` is refused for both, as a user error.

The command takes the DFloat11 transformer from that checkpoint, 10.9 GB for either model, and the text encoder, VAE and
tokenizer from the model's BF16 repository. It does not download that repository's BF16 transformer (16.1 GB) or its
prompt enhancer (`pe/`, 7.7 GB), which mflux does not use. The text-encoder file is 7.7 GB and also holds a vision model
that is never loaded. Both BF16 repositories publish the same text-encoder file, and huggingface_hub 2.0's shared cache
keeps one copy of it for both models. The VAE is 0.17 GB. The default checkpoints and BF16 repositories are pinned to
the revisions the measured runs used; a `--df11` or `--base` you pass is used as given.

The same from Python:

```python
from mlx_dfloat.mflux import DFloatModel

model = DFloatModel("ernie-image")
image = model.generate_image(
    seed=42,
    prompt="A stone lighthouse on a rocky shore at dawn",
    height=1024,
    width=1024,
)
image.save("lighthouse.png")
```

Left out, `guidance`, `num_inference_steps` and `scheduler` take the same per-model defaults as the command: 4.0 and
50 steps for ERNIE-Image, 1.0 and 8 steps for ERNIE-Image-Turbo. mflux's own `ErnieImage` class defaults to 1.0 and 8
steps for both. Unlike the command, the Python class lets ERNIE-Image-Turbo run at another guidance, as mflux's class
does. An empty or blank prompt is refused here too. Each call runs under the memory caps the command installs, unless
the process has already set a wired limit.

mflux compiles ERNIE-Image's denoising step on M1 and M2 Max and Ultra chips and on M3 and later. This path runs it
without that compilation on every chip (see "Generate a Z-Image image" for why). mflux also computes ERNIE-Image's
timestep conditioning in float32, so the hidden state passed from block to block is float32, not BF16; this path runs
mflux's own block code, so it does the same. No ERNIE-Image step time is reported, against stock mflux or otherwise.

Measured on an M1 Max (32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0) at 1024², seed 42, with the commands under
"Measured numbers": ERNIE-Image-Turbo (8 steps at guidance 1.0) peaked at 17.34 GiB and took 2.9 minutes, and
ERNIE-Image (50 steps at guidance 4.0) peaked at 16.79 GiB and took 29.9 minutes (the `elapsed_seconds` of
`bench/results/tiers/ernie-image-turbo-1024.json` and `ernie-image-1024.json`). In both runs MLX's own memory peaked
during the VAE decode, and the compressed transformer stayed loaded through it. The fit estimate, mlx-dfloat's
prediction of each step's peak memory that it checks before a call, puts the VAE decode at 17.85 GiB, under the 22.96
GiB budget.

Under a 24 GB Mac's caps (`--tier 24`, run on the 32 GB M1 Max), the fit check refuses ERNIE-Image-Turbo. It predicts
14.39 GiB for the denoising step against the 14.00 GiB fit budget. A forced run (`--no-fit-check`) passed with a 14.39
GiB peak in 2.8 minutes, the compressed transformer dropped before the VAE decode
(`bench/results/tiers/ernie-image-turbo-1024-tier24.json`). That peak is only 0.11 GiB under the 14.50 GiB watchdog
ceiling. The run's report still shows the earlier 14.03 GiB estimate in its `fit` block, because the estimate was raised
from this run's peak. Two identical runs under these caps peaked 0.35 GiB apart, at 14.03 and 14.39 GiB, which is why
the estimate takes the higher one. ERNIE-Image-Turbo on a 24 GB Mac is at risk. No 24 GB Mac was measured.

ERNIE-Image under a 24 GB Mac's caps, and both models under a 16 GB Mac's, are at risk too. The fit check refuses them:
it predicts 15.2 GiB for ERNIE-Image's denoising step and 14.39 GiB for Turbo's, against budgets of 14.00 GiB at 24 GB
and 8.67 GiB at 16 GB. Forced runs (`--no-fit-check`) were stopped by the watchdog. ERNIE-Image at 24 GB was stopped in
the denoising step after 20.6 s, at a 14.80 GiB footprint against the 14.50 GiB ceiling. At 16 GB both were stopped
while the compressed transformer was loading, against the 9.17 GiB ceiling: Turbo after 18.6 s at 9.22 GiB, ERNIE-Image
after 23.3 s at 9.44 GiB.

Quantisation, LoRA, img2img, mflux's alternative image decoder (`--pid-decode`) and ControlNet are refused, as for
FLUX.1. Sizes above 1024² are refused unless you pass `--no-fit-check`, because none has been measured.

## Measured numbers

The blocks below are generated from the files under `bench/results/` by `scripts/bench_table.py`, and a test fails when
the README and those files disagree. Every row carries one of three labels. MEASURED means the run used the host's own
memory limits on a Mac with that much memory. CAPPED means a larger Mac ran under a smaller Mac's MLX memory limits and
watchdog ceiling, which shows how much memory the run needs but not how that smaller Mac performs. A CAPPED row runs
under the wired and memory caps mlx-dfloat installs on a Mac of that size (16 GB: 8 GiB wired, 10 GiB memory; 24 GB: 14
and 16 GiB), so it sees the limits a user of that Mac runs under. PROOF marks a run under a deliberately low watchdog
ceiling, made only to show that the watchdog works; it never appears as a tier row.

The 32 GB rows are MEASURED, on one M1 Max. The 16 GB and 24 GB rows are CAPPED, run on that same Mac:
FLUX.2-klein-base-4B passed under both and ERNIE-Image-Turbo under a 24 GB Mac's caps (a forced run: the fit check
refuses it), while FLUX.2-klein-base-9B under a 24 GB Mac's caps, Z-Image-Turbo under a 16 GB Mac's, ERNIE-Image-Turbo
under a 16 GB Mac's, and Qwen-Image-2.1 and ERNIE-Image under both were stopped by the watchdog. The table makes no
claim about any Mac it does not list. Each row is one `mlx-dfloat generate` run at 1024² with seed 42 (4 steps for
FLUX.1-schnell, FLUX.2-klein-4B and FLUX.2-klein-9B, 20 for FLUX.1-dev and FLUX.1-Krea-dev, 9 for Z-Image-Turbo, 40 at
guidance 1.0 for Qwen-Image-2.1, 8 at guidance 1.0 for ERNIE-Image-Turbo, 50 with guidance 4.0 for Z-Image, ERNIE-Image
and the two FLUX.2 Klein base models), with `--tier` and `--report` writing the file in the last column.

The second column is the Mac's recommended GPU working set minus a reserve. On the 16 GB and 24 GB rows the working
set is taken as two thirds of that Mac's memory. Two limits come off that working set, with different reserves. The
fit budget, which `generate` checks before a run, is the working set minus 2 GiB on every Mac: 22.96 GiB on this one,
8.67 GiB for a 16 GB Mac and 14.00 GiB for a 24 GB Mac. The watchdog ceiling, where a CAPPED run is stopped, is the
working set minus 1.5 GiB at 16 GB and 24 GB: 9.17 and 14.50 GiB. So the second column is the fit budget on the 32 GB
rows and the watchdog ceiling on the CAPPED rows. The FLUX.2 Klein and Z-Image CAPPED rows were recorded while the
fit check still used the 1.5 GiB reserve, and the stricter budget changes none of their outcomes. FLUX.2-klein-base-4B still passes at 16 GB
(a predicted 8.54 GiB against 8.67), and at 24 GB it still keeps the compressed transformer loaded for the VAE decode
(12.69 GiB predicted against 14.00). FLUX.2-klein-base-9B at 24 GB and Z-Image-Turbo at 16 GB are still refused.
The Qwen-Image-2.1 rows were recorded under the current budget, which refuses both (a predicted 14.6 GiB while it
encodes the prompt).

"Peak (watched)" is the larger of the process footprint the OS reports and MLX's active plus cached memory, and a
row's status is "target" when it stayed under the second column. "Peak MLX" is the larger of two
readings: the watchdog's sample of MLX's active plus cached memory, taken every 0.05 s, and MLX's own exact peak of
active memory in each phase of the run, which also catches a brief high point between two samples. A row the watchdog
stopped shows its peaks as lower bounds, since the run never got to its own peak. On the host's own tier the watchdog
stops a run at physical memory minus 4 GiB (28 GiB on this Mac), well above the fit budget. Each file records the
watchdog ceiling as `watchdog_ceiling_bytes`.

The three Z-Image rows come from these commands. The reports do not store the prompt, so these commands reuse the
lighthouse prompt from "Generate a Z-Image image".

```
mlx-dfloat generate --model z-image-turbo --base Tongyi-MAI/Z-Image --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/z-image-turbo-1024.json
mlx-dfloat generate --model z-image --steps 50 --guidance 4 --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/z-image-1024.json
mlx-dfloat generate --model z-image-turbo --base Tongyi-MAI/Z-Image --tier 16 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
```

The last one ends with exit 70 and writes the watchdog's `abort.json` next to the output image. Without
`--no-fit-check` it is refused with exit 2 (`bench/results/refusals/z-image-turbo-1024-tier16.json`).

The FLUX.2 Klein rows come from these commands, with the same prompt:

```
mlx-dfloat generate --model flux2-klein-4b --base black-forest-labs/FLUX.2-klein-base-4B --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-4b-1024.json
mlx-dfloat generate --model flux2-klein-base-4b --steps 50 --guidance 4 --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-base-4b-1024.json
mlx-dfloat generate --model flux2-klein-9b --base black-forest-labs/FLUX.2-klein-base-9B --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-9b-1024.json
mlx-dfloat generate --model flux2-klein-base-9b --steps 50 --guidance 4 --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-base-9b-1024.json
mlx-dfloat generate --model flux2-klein-base-4b --steps 50 --guidance 4 --tier 16 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-base-4b-1024-tier16.json
mlx-dfloat generate --model flux2-klein-base-4b --steps 50 --guidance 4 --tier 24 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/flux2-klein-base-4b-1024-tier24.json
mlx-dfloat generate --model flux2-klein-base-9b --steps 50 --guidance 4 --tier 24 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
```

The last one runs with `--no-fit-check` because without it the fit check refuses the run: it predicts 16.3 GiB while
denoising against the 14.50 GiB budget of that run, and 14.00 GiB today
(`bench/results/refusals/flux2-klein-base-9b-1024-tier24.json`). It ends with
exit 70, like the Z-Image-Turbo run above.

The Qwen-Image-2.1 rows come from these commands, with the same prompt:

```
mlx-dfloat generate --model qwen-image-2.1 --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/qwen-image-2.1-1024.json
mlx-dfloat generate --model qwen-image-2.1 --tier 16 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
mlx-dfloat generate --model qwen-image-2.1 --tier 24 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
```

In the last two the watchdog stops the run (exit 70) and writes its `abort.json` next to the output image; the
table's copies are in `bench/results/tiers/aborts/`. Without `--no-fit-check` both are refused with exit 2
(`bench/results/refusals/qwen-image-2.1-1024-tier16.json` and `qwen-image-2.1-1024-tier24.json`).

The ERNIE-Image rows come from these commands, with the same prompt:

```
mlx-dfloat generate --model ernie-image-turbo --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/ernie-image-turbo-1024.json
mlx-dfloat generate --model ernie-image --tier 32 \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/ernie-image-1024.json
mlx-dfloat generate --model ernie-image-turbo --tier 24 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024 --report bench/results/tiers/ernie-image-turbo-1024-tier24.json
mlx-dfloat generate --model ernie-image-turbo --tier 16 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
mlx-dfloat generate --model ernie-image --tier 16 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
mlx-dfloat generate --model ernie-image --tier 24 --no-fit-check \
  --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
  --seed 42 --height 1024 --width 1024
```

In the last three the watchdog stops the run (exit 70) and writes its `abort.json` next to the output image; the
table's copies are in `bench/results/tiers/aborts/`. Without `--no-fit-check` the last four commands are refused with
exit 2 (`bench/results/refusals/ernie-image-turbo-1024-tier24.json`, `ernie-image-turbo-1024-tier16.json`,
`ernie-image-1024-tier16.json` and `ernie-image-1024-tier24.json`).

Measured on an Apple M1 Max, 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0.

<!-- bench:tier-table -->
| Mac | Working set − reserve | Model | DF11 size | Peak (watched) | Peak footprint | Peak MLX (sampled active + cache, or exact phase peak) | Label | Status | Limits | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 GB | 22.96 GiB | FLUX.1-dev | 15.21 GiB | 19.96 GiB | 19.96 GiB | 19.19 GiB | MEASURED | target | host caps | `bench/results/tiers/dev-1024.json` |
| 32 GB | 22.96 GiB | ERNIE-Image | 10.17 GiB | 16.79 GiB | 16.79 GiB | 15.21 GiB | MEASURED | target | host caps | `bench/results/tiers/ernie-image-1024.json` |
| 24 GB | 14.50 GiB | ERNIE-Image-Turbo | 10.17 GiB | 14.39 GiB | 14.39 GiB | 13.35 GiB | CAPPED | target | mlx-dfloat caps for the tier | `bench/results/tiers/ernie-image-turbo-1024-tier24.json` |
| 32 GB | 22.96 GiB | ERNIE-Image-Turbo | 10.17 GiB | 17.34 GiB | 17.34 GiB | 16.16 GiB | MEASURED | target | host caps | `bench/results/tiers/ernie-image-turbo-1024.json` |
| 32 GB | 22.96 GiB | FLUX.2-klein-4B | 4.90 GiB | 11.88 GiB | 11.88 GiB | 11.90 GiB | MEASURED | target | host caps | `bench/results/tiers/flux2-klein-4b-1024.json` |
| 32 GB | 22.96 GiB | FLUX.2-klein-9B | 11.47 GiB | 18.40 GiB | 18.40 GiB | 17.57 GiB | MEASURED | target | host caps | `bench/results/tiers/flux2-klein-9b-1024.json` |
| 16 GB | 9.17 GiB | FLUX.2-klein-base-4B | 4.90 GiB | 8.15 GiB | 8.15 GiB | 7.61 GiB | CAPPED | target | mlx-dfloat caps for the tier | `bench/results/tiers/flux2-klein-base-4b-1024-tier16.json` |
| 24 GB | 14.50 GiB | FLUX.2-klein-base-4B | 4.90 GiB | 12.29 GiB | 12.29 GiB | 10.91 GiB | CAPPED | target | mlx-dfloat caps for the tier | `bench/results/tiers/flux2-klein-base-4b-1024-tier24.json` |
| 32 GB | 22.96 GiB | FLUX.2-klein-base-4B | 4.90 GiB | 12.78 GiB | 12.78 GiB | 11.91 GiB | MEASURED | target | host caps | `bench/results/tiers/flux2-klein-base-4b-1024.json` |
| 32 GB | 22.96 GiB | FLUX.2-klein-base-9B | 11.47 GiB | 17.60 GiB | 17.60 GiB | 16.58 GiB | MEASURED | target | host caps | `bench/results/tiers/flux2-klein-base-9b-1024.json` |
| 32 GB | 22.96 GiB | FLUX.1-Krea-dev | 15.21 GiB | 20.04 GiB | 20.04 GiB | 19.19 GiB | MEASURED | target | host caps | `bench/results/tiers/krea-dev-1024.json` |
| 32 GB | 22.96 GiB | Qwen-Image-2.1 | 9.06 GiB | 20.42 GiB | 20.42 GiB | 19.72 GiB | MEASURED | target | host caps | `bench/results/tiers/qwen-image-2.1-1024.json` |
| 32 GB | 22.96 GiB | FLUX.1-schnell | 15.19 GiB | 19.78 GiB | 19.78 GiB | 19.03 GiB | MEASURED | target | host caps | `bench/results/tiers/schnell-1024.json` |
| 32 GB | 22.96 GiB | Z-Image | 7.80 GiB | 13.15 GiB | 13.15 GiB | 12.65 GiB | MEASURED | target | host caps | `bench/results/tiers/z-image-1024.json` |
| 32 GB | 22.96 GiB | Z-Image-Turbo | 7.80 GiB | 13.11 GiB | 13.11 GiB | 12.20 GiB | MEASURED | target | host caps | `bench/results/tiers/z-image-turbo-1024.json` |
| 16 GB | 9.17 GiB | ERNIE-Image | not recorded | at least 9.44 GiB (stopped after 23.3 s) | at least 9.44 GiB | at least 9.03 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/ernie-image-1024-tier16.json` |
| 24 GB | 14.50 GiB | ERNIE-Image | not recorded | at least 14.80 GiB (stopped after 20.6 s) | at least 14.80 GiB | at least 14.25 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/ernie-image-1024-tier24.json` |
| 16 GB | 9.17 GiB | ERNIE-Image-Turbo | not recorded | at least 9.22 GiB (stopped after 18.6 s) | at least 9.22 GiB | at least 8.75 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/ernie-image-turbo-1024-tier16.json` |
| 24 GB | 14.50 GiB | FLUX.2-klein-base-9B | not recorded | at least 14.64 GiB (stopped after 28.5 s) | at least 14.64 GiB | at least 14.26 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/flux2-klein-base-9b-1024-tier24.json` |
| 16 GB | 9.17 GiB | Qwen-Image-2.1 | not recorded | at least 9.28 GiB (stopped after 4.4 s) | at least 8.90 GiB | at least 9.28 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/qwen-image-2.1-1024-tier16.json` |
| 24 GB | 14.50 GiB | Qwen-Image-2.1 | not recorded | at least 14.59 GiB (stopped after 4.8 s) | at least 14.59 GiB | at least 14.31 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/qwen-image-2.1-1024-tier24.json` |
| 16 GB | 9.17 GiB | Z-Image-Turbo | not recorded | at least 9.26 GiB (stopped after 4.7 s) | at least 9.26 GiB | at least 8.81 GiB | CAPPED | stopped by the watchdog | mlx-dfloat caps for the tier | `bench/results/tiers/aborts/z-image-turbo-1024-tier16.json` |
<!-- /bench:tier-table -->

The overhead block times one 1024² denoise step, five timed steps after two warm-up steps in each of three rounds, with
every condition in its own process. `df11` decodes each transformer block's weights from the compressed set just before
the block runs and evaluates after every block. `control` runs the same graph with the same per-block evaluation, but
hands every block the weights of one double and one single block, decoded once before timing started, so the gap between
the two is the cost of decoding just in time. At the earlier 1.4 GB cache limit most of that gap was allocating a fresh
buffer for each block's decoded weights; at 2.5 GB the share was not measured. The depth-2 pair evaluates one block
behind instead, so the CPU can queue the next block, decode included, while the GPU runs the current one. The eval
policy cost is what evaluating after every block adds by itself: `control` minus a control that evaluates once per step,
with no decode involved. A negative value means the per-block control was the faster of the two. The q8 ratio is DF11
with per-block evaluation over mflux's own transformer quantised to 8 bits, with one eval per step. The q8 step runs
under the scenario's 2.5 GB cache limit, like every other condition here; mflux's own generate sets no cache limit, so
this is not quite mflux's q8 as you would run it. A q8 step changes the weights and the output; a DF11 step does not.

On FLUX.1-dev a step costs 3.52 % more with per-block evaluation (19.24 s against the control's 18.59 s) and 4.61 % more
with the depth-2 run-ahead. On FLUX.1-schnell the two figures are 4.05 % (19.28 s against 18.53 s) and 4.24 %. Round 1
of the schnell run saw GPU load from outside the bench; the same pooled medians over rounds 2 and 3 alone give +4.2 %
for per-block evaluation. The dev depth-2 figure pools three rounds that disagree. Taken one round at a time (each
round's DF11 median over its own control's), they read −1.6 %, +5.9 % and +6.5 %, because round 1's depth-2 control ran
slow at 19.50 s against 18.77 s and 18.69 s later; rounds 2 and 3 alone give +6.2 %. Evaluating one block behind did not
help on either model. Evaluating after every block cost 0.41 s per step on schnell and nothing measurable on dev, where
the per-block control came out 0.07 s faster, less than either condition moved between rounds. Both sides of the
overhead comparison pay that cost, so it is not part of the decode overhead. A DF11 step takes 1.26 times as long as
mflux's q8 step on dev (15.26 s) and 1.35 times on schnell (14.25 s). Treat both ratios as rough: the q8 step's median
moved by about 2 s from one round to the next in both scenarios, while the DF11 step's moved by 0.7 s at most. The q8
step also peaked lower, at 14.8–14.9 GiB against DF11's 19.7–20.0 GiB. What DF11 keeps and q8 gives up is the exact BF16
output.

The control was validated once with `scripts/bench_control_validation.py`, at the earlier 1.4 GB cache limit and on
a reduced-depth transformer (4 double and 8 single blocks instead of 19 and 38): it agreed within 0.13 % with a run
whose BF16 weights were all resident.

<!-- bench:overhead -->
FLUX.1-dev, 1024², per-block evaluation: +3.5 % (depth-2: +4.6 %); eval policy cost -0.07 s/step; DF11 (per-block) over the mflux q8 step (one eval per step, same cache limit): 1.26×

Command: `uv run --group bench python -m scripts.bench_flux1 bench/scenarios/flux1-dev-1024.toml` (preflight skipped: not_charging)

FLUX.1-schnell, 1024², per-block evaluation: +4.0 % (depth-2: +4.2 %); eval policy cost 0.41 s/step; DF11 (per-block) over the mflux q8 step (one eval per step, same cache limit): 1.35×

Command: `uv run --group bench python -m scripts.bench_flux1 bench/scenarios/flux1-schnell-1024.toml`

Apple M1 Max, 32 GB, macOS 27.0.1, mlx 0.32.2, mflux 0.20.0, git d164fc5, 2026-10-01

DF11 and the mflux q8 step both run under a 2.5 GB MLX buffer-cache limit (decimal GB).
<!-- /bench:overhead -->

The scenario result files name the commit their runs used as it was on its pull-request branch; the `generate`
reports behind the tier rows and the harness proof record no commit. Merging onto `main` gave those commits new IDs
without changing their content: `d164fc5` (the scenario and tier runs) is `c29714d` on `main`, and `9c51869` (the
harness proof) is `3d142a7`.

The harness proof runs `mlx-dfloat generate` twice under the same lowered watchdog ceiling, once at a size that
stays under it and once at a size that does not. `bench/results/harness-proof/README.md` gives both commands and
the arithmetic behind the ceiling.

<!-- bench:harness-proof -->
Harness proof: under one 19.25 GiB cap, a 512² run passed with a watched peak of 18.64 GiB, and a 1024² run was stopped by the watchdog (memory, counter footprint) at 19.32 GiB. Records: `bench/results/harness-proof`.
<!-- /bench:harness-proof -->

### Reproduce the numbers

The overhead numbers come from one command per scenario, run from a checkout:

```
uv run --group bench python -m scripts.bench_flux1 bench/scenarios/flux1-schnell-1024.toml --results-root <a fresh directory>
```

`bench/results/` holds the published run. Without `--results-root` the command works in that directory, and it
refuses the committed files as soon as the code differs from the version that wrote them.

`bench/scenarios/` holds one scenario for FLUX.1-schnell and one for FLUX.1-dev. A scenario pins the DFloat11 and
base snapshot revisions, the prompt, the seed, the size, the step counts, the rounds, the cache limit and the
conditions. The command reads both checkpoints from the local Hugging Face cache and never downloads; when a pinned
snapshot is missing, it exits 2 and prints the `hf download` command that fetches it. Each condition and round runs
as its own process with its own memory watchdog, and writes its own JSON under `<results root>/<scenario>/`. An
interrupted run picks up where it stopped, and results from a different scenario or different code are refused
rather than mixed in.

Before anything runs, a launch check samples the machine and refuses to start (exit 2, naming what failed) when the
Mac is on battery, below 40 % battery, or on a charger that is not charging a battery below half; when
`pmset -g therm` reports a CPU speed limit below 100 or the lid is closed; when less than 20 GiB of disk or less
than 20 % of memory is free; or when another heavy process (a bench, a test run, mflux or mlx-lm) holds 1 GiB or
more. Timings taken in those states are not comparable. When macOS has recorded no CPU speed limit, as on the Mac
that produced these numbers, the speed-limit check passes. A refused launch writes nothing; once a run is accepted,
the sample is saved as `preflight.json` next to the results. The check runs again before each condition's process
starts, because a small charger can fall behind over a 40-minute run. When a check fails mid-run, the command stops
before that process (exit 2), `report.json` records which checks failed, and running the same command again picks
up from there. `--skip-preflight` skips every check, and the report records that it did. The busy check lists
a process by its executable name and pid only; set `MLX_DFLOAT_PREFLIGHT_EXCLUDE` to comma-separated substrings of
command lines that should not count as busy.

`mlx-dfloat generate --tier GB` runs under the wired and memory caps mlx-dfloat installs on a smaller Mac and a cache
limit derived from them, with the tier's watchdog ceiling and that Mac's fit budget, and labels its report CAPPED;
`--tier` set to the host's own size keeps the host's limits and gives the MEASURED rows above. `--memory-ceiling BYTES`
lowers the watchdog ceiling alone, under the host's limits, and labels the report PROOF. After a new run,
`uv run python -m scripts.bench_table` rewrites the blocks above, and `--check` exits 1 when they are out of date.

## Relationship to DFloat11

This is not a fork and not affiliated with the DFloat11 authors. It reads the checkpoint format they publish. Credit
for the method goes to them: *70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via
Dynamic-Length Float* ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)).

## License

Apache-2.0. See [LICENSE](https://github.com/IonDen/mlx-dfloat/blob/main/LICENSE).
[NOTICE](https://github.com/IonDen/mlx-dfloat/blob/main/NOTICE) credits the DFloat11 work, mflux and the test-only
encoder copied from the DFloat11 repository, each with its licence.

[^size]: Reported in the DFloat11 paper ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)). The exponent bits of
    trained weights are far from uniformly distributed, which is what makes them compressible; the exact ratio
    depends on the model and is slightly different for each one.
[^flux]: The FLUX.1-dev transformer has about 12 billion parameters. In BF16 each takes 2 bytes, so about 24 GB for
    the weights alone, before activations and before the text encoders.
