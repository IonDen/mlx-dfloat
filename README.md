# mlx-dfloat

Run [DFloat11](https://github.com/LeanModels/DFloat11) checkpoints on a Mac with [MLX](https://github.com/ml-explore/mlx).

DFloat11 is lossless compression for BF16 model weights. Each weight is 16 bits. DFloat11 stores the 8 exponent bits
as a short variable-length code, the same idea a zip file uses, and keeps the other 8 bits (the sign and the
fraction) exactly as they are. The DFloat11 authors report models at about 70% of their BF16 size with output that is
bit for bit the same as the original.[^size] Their decoder runs on NVIDIA GPUs only. This project is an independent
reader and decoder for Apple Silicon. The weights stay compressed in memory, and a Metal kernel decodes each block
on the GPU as it runs. That is how it generates FLUX.1 images through mflux.

```
one BF16 weight, 16 bits:    s   eeeeeeee   mmmmmmm
                             |   |          |
                             |   exponent: replaced by a variable-length code   (compressed)
                             sign + fraction: kept as one raw byte              (stored as is)
```

Why that matters on a Mac: unified memory is the limit. The BF16 FLUX.1-dev transformer is about 24 GB,[^flux] more
than the GPU on a 32 GB machine can comfortably hold. At 70% it should come in under that line, and unlike 4-bit or
8-bit quantisation it changes nothing in the output. Whether it really fits on a given Mac is something this project
has to measure. The numbers measured so far are under "Measured numbers" below, and they cover a 32 GB Mac only.

## Status

Pre-alpha, and not on PyPI yet; it installs from a checkout (see "Install" below). The first two milestones decided
whether the project would go ahead at all. The other two make it usable and its numbers reproducible:

1. A bit-exact reference decoder for published DFloat11 checkpoints. **Done.** Every compressed tensor of Qwen3-4B,
   and sampled blocks of FLUX.1-schnell, FLUX.1-Krea-dev, Qwen-Image-Edit and Qwen-Image-Edit-2509, decode to
   exactly the BF16 originals. That covers all four published versions of the checkpoint format.
2. A Metal decode kernel fast enough to run inside an image-generation step. **Done.** The kernel decodes the
   published checkpoints bit-exactly at 50 GB/s on an M1 Max. Decoding every block just in time makes a 1024²
   denoise step 5.0 % slower on FLUX.1-dev, 6.1 % slower on FLUX.1-schnell with a one-block evaluation run-ahead,
   and 8.5 % slower on FLUX.1-schnell with per-block evaluation, all well under the project's ~25 % threshold.
   That is measured against a control which,
   on a reduced-depth transformer, agrees within 0.13 % with a run whose BF16 weights are all resident, through
   the same per-block path. Those runs used a 1.4 GB MLX buffer-cache limit. The benchmark in milestone 4 uses
   2.5 GB, and the overhead there is lower on both models. The recipe and numbers are in the changelog.
3. FLUX.1 image generation through mflux, straight from the compressed checkpoint to a saved image. **Done.**
   FLUX.1-schnell, FLUX.1-dev and FLUX.1-Krea-dev each generate a 1024² image on a 32 GB M1 Max within the
   machine's measured memory budget, and FLUX.1-schnell's output matches, bit for bit, the same transformer
   streaming its BF16 weights block by block instead of the compressed ones. The command and the measured numbers
   are under "Try it" below.
4. A reproducible benchmark and the measured 32 GB rows. **Done.** One command runs a scenario file that pins
   the checkpoints, the prompt, the seed, the size and the cache limit, and writes one result file per condition
   and round, and the tables under "Measured numbers" are generated from those files. A 32 GB M1 Max has measured
   rows for all three FLUX.1 models. A two-run proof shows the memory watchdog stopping a run that crosses its
   ceiling and leaving one under it alone.

All four milestones are done.

## Install

mlx-dfloat is for Macs with Apple Silicon and needs Python 3.11 or newer. It is not on PyPI yet, so install it from
a checkout with [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/IonDen/mlx-dfloat
cd mlx-dfloat
uv sync --extra mflux
```

A plain `uv sync` installs the reader, the decoder and the parity scripts. `--extra mflux` adds FLUX.1 generation
and installs mflux 0.20. The benchmark runs with `uv run --group bench`, which installs mflux as well.

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

### Generate a FLUX.1 image

Generation needs the `mflux` extra (`uv sync --extra mflux`, see "Install"). The default repositories need no extra
flags:

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
because their BF16 transformers are gated and were not on disk to compare against. A later run of each model, the
one recorded under "Measured numbers", peaked within 0.2 GiB of these figures.

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

## Measured numbers

The blocks below are generated from the files under `bench/results/` by `scripts/bench_table.py`, and a test fails
when the README and those files disagree. Every row carries one of three labels. MEASURED means the run used the
host's own memory limits on a Mac with that much memory. CAPPED means a larger Mac ran under a smaller Mac's MLX
memory limits and watchdog ceiling, which shows how much memory the run needs but not how that smaller Mac performs.
A CAPPED row uses MLX's default limits for a Mac of that size, which are looser than the caps `generate` would install
there, so its peak is not understated.
PROOF marks a run under a deliberately low watchdog ceiling, made only to show that the watchdog works; it never
appears as a tier row.

So far there are only 32 GB rows, and all of them are MEASURED, on one M1 Max. The table makes no claim about any
Mac it does not list. Each row is one `mlx-dfloat generate` run at 1024² with seed 42 (4 steps for FLUX.1-schnell,
20 for FLUX.1-dev and FLUX.1-Krea-dev), with `--tier 32` and `--report` writing the file in the last column. The
fit budget is the Mac's recommended GPU working set minus a 2 GiB reserve, the same budget `generate` uses to decide
whether a run fits. "Peak (watched)" is the larger of the process footprint the OS reports and MLX's active plus
cached memory, and a row's status is "target" when it stayed under the fit budget. The fit budget is not where the
watchdog stops a run on these rows. On the host's own tier the watchdog aborts at physical memory minus 4 GiB, 28 GiB
on this Mac (`watchdog_ceiling_bytes` in each file). On a CAPPED row the two are the same number.

<!-- bench:tier-table -->
| Mac | Fit budget (budget − reserve) | Model | DF11 size | Peak (watched) | Peak footprint | Peak MLX (active + cache) | Label | Status | Limits | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 GB | 22.96 GiB | FLUX.1-dev | 15.21 GiB | 19.96 GiB | 19.96 GiB | 19.19 GiB | MEASURED | target | host caps | `bench/results/tiers/dev-1024.json` |
| 32 GB | 22.96 GiB | FLUX.1-Krea-dev | 15.21 GiB | 20.04 GiB | 20.04 GiB | 19.19 GiB | MEASURED | target | host caps | `bench/results/tiers/krea-dev-1024.json` |
| 32 GB | 22.96 GiB | FLUX.1-schnell | 15.19 GiB | 19.78 GiB | 19.78 GiB | 19.03 GiB | MEASURED | target | host caps | `bench/results/tiers/schnell-1024.json` |
<!-- /bench:tier-table -->

The overhead block times one 1024² denoise step, five timed steps after two warm-up steps in each of three rounds,
with every condition in its own process. `df11` decodes each transformer block's weights from the compressed set
just before the block runs and evaluates after every block. `control` runs the same graph with the same per-block
evaluation, but its weights were decoded once before timing started, so the gap between the two is the cost of
decoding. The depth-2 pair evaluates one block behind instead, so the CPU can queue the next block, decode included,
while the GPU runs the current one. The eval policy cost is what evaluating after every block adds by itself:
`control` minus a control that evaluates once per step, with no decode involved. A negative value means the
per-block control was the faster of the two. The q8 ratio is DF11 with per-block evaluation over mflux's own
transformer quantised to 8 bits, with one eval per step. The q8 step runs under the scenario's 2.5 GB cache limit,
like every other condition here; mflux's own generate sets no cache limit, so this is not quite mflux's q8 as you
would run it. A q8 step changes the weights and the output; a DF11 step does not.

On FLUX.1-dev a step costs 3.52 % more with per-block evaluation (19.24 s against the control's 18.59 s) and 4.61 %
more with the depth-2 run-ahead. On FLUX.1-schnell the two figures are 4.05 % (19.28 s against 18.53 s) and 4.24 %.
Round 1 of the schnell run saw GPU load from outside the bench; the same pooled medians over rounds 2 and 3 alone give
+4.2 % for per-block evaluation. The dev depth-2 figure pools three rounds that disagree. Taken one round at a time
(each round's DF11 median over its own control's), they read −1.6 %, +5.9 % and +6.5 %, because round 1's depth-2
control ran slow at 19.50 s against 18.77 s and 18.69 s later; rounds 2 and 3 alone give +6.2 %. Evaluating one
block behind did not help on either model. Evaluating after every
block cost 0.41 s per step on schnell and nothing measurable on dev, where the per-block control came out 0.07 s
faster, less than either condition moved between rounds. Both sides of the overhead comparison pay that cost, so it is
not part of the decode overhead. A DF11 step takes 1.26 times as long as mflux's q8 step on dev (15.26 s) and 1.35
times on schnell (14.25 s). Treat both ratios as rough: the q8 step's median moved by about 2 s from one round to the
next in both scenarios, while the DF11 step's moved by 0.7 s at most. The q8 step also peaked lower, at 14.8–14.9 GiB
against DF11's 19.7–20.0 GiB. What DF11 keeps and q8 gives up is the exact BF16 output.

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

`mlx-dfloat generate --tier GB` runs under a smaller Mac's MLX memory and cache limits, with that Mac's watchdog
ceiling and fit budget, and labels its report CAPPED; `--tier` set to the host's own size keeps the host's limits
and gives the MEASURED rows above. `--memory-ceiling BYTES` lowers the watchdog ceiling alone, under the host's
limits, and labels the report PROOF. After a new run, `uv run python -m scripts.bench_table` rewrites the blocks
above, and `--check` exits 1 when they are out of date.

## Relationship to DFloat11

This is not a fork and not affiliated with the DFloat11 authors. It reads the checkpoint format they publish. Credit
for the method goes to them: *70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via
Dynamic-Length Float* ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)).

## License

Apache-2.0. See [LICENSE](LICENSE). [NOTICE](NOTICE) credits the DFloat11 work, mflux and the test-only encoder
copied from the DFloat11 repository, each with its licence.

[^size]: Reported in the DFloat11 paper ([arXiv:2504.11651](https://arxiv.org/abs/2504.11651)). The exponent bits of
    trained weights are far from uniformly distributed, which is what makes them compressible; the exact ratio
    depends on the model and is slightly different for each one.
[^flux]: The FLUX.1-dev transformer has about 12 billion parameters. In BF16 each takes 2 bytes, so about 24 GB for
    the weights alone, before activations and before the text encoders.
