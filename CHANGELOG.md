# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project skeleton: package layout, MLX memory caps derived from the device's working-set size, test gates for
  slow and networked tests, and CI.
- Reading and decoding DFloat11 (DF11) checkpoints. A safetensors header reader and a DF11 format reader check the
  `dfloat11_config` block and each compressed group, and raise `DFloatFormatError` on malformed or unsafe input.
  Legacy pickle-format repos are refused and never unpickled. A pure NumPy reference decoder turns a group back into
  its BF16 bit patterns, bit for bit, and raises `DFloatResourceError` when a decode would exceed its memory budget.
  Two scripts in the repository check a conversion against the original BF16 weights: one decodes a whole local
  checkpoint, the other samples a few groups of a Hugging Face repo over range requests. They exit 0 when every
  compared value matches, 1 on a real mismatch, 2 on any other error, and 70 or 71 when the memory or wall-clock
  watchdog aborts the run.
- A Metal decode kernel for DFloat11 groups, behind `mlx_dfloat.decode`. `decode_group(group, backend="metal")`
  returns the BF16 bit patterns and a per-block status word, `check` refuses a group whose status reports an invalid
  code, a count mismatch or a broken thread chain, and `available_backends` says whether the kernel compiles and runs
  on this machine; nothing falls back silently. Every decode path is tested bit for bit against the NumPy reference:
  hand-built codebooks, encoder round-trips, a real Qwen3-4B slice, and corrupt inputs that must end with an error
  instead of a hang. On an M1 Max the kernel decodes 50 to 54 GB/s of BF16 output when a block's output is staged
  through threadgroup memory and 19 to 24 GB/s when it writes straight to device memory
  (`scripts/bench_decode_kernel.py`, median of 5 repetitions after a warm-up); staged is the default, and the direct
  path is the fallback for blocks larger than the staging buffer. `verify_checkpoint.py --decoder metal` checks a
  whole checkpoint through the kernel, against the BF16 original when one is given and against the reference
  otherwise: Qwen3-4B decodes bit-identically in 12.8 s where the reference took 223 s, and all 57 groups of
  FLUX.1-schnell match the reference.
- A measurement rig for the question the project hinges on: how much a FLUX.1 denoise step slows down when every
  transformer block's weights are decoded just in time. `scripts/bench_flux_step.py` runs one mode per process, either
  decoding per block or a control that swaps pre-decoded weights into the same graph, and reports the paired overhead;
  `scripts/bench_control_validation.py` checks that control against a run whose BF16 weights are all resident, on a
  reduced-depth transformer and through the same per-block path; `scripts/encode_prompt.py` encodes the prompt once,
  so the timed process holds no text encoder. Measured on an M1 Max (32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0) at
  1024², five timed steps in each of three rounds: FLUX.1-schnell costs 6.1 % more per step with a one-block
  evaluation run-ahead and 8.5 % with a per-block evaluation; FLUX.1-dev costs 5.0 % either way (the dev prompt was
  encoded with the schnell text encoders at 512 tokens, the same T5 and CLIP architecture, because the dev base
  repository is gated). The control agrees with that run within 0.13 %; the per-block evaluation itself costs 0.21 s
  (schnell) to 0.26 s (dev) per step, measured against a control that evaluates only at the end of the step. In the
  step, decoding costs 2.1 to 3.4 times what the isolated kernel benchmark predicts from its throughput. A follow-up
  measurement on schnell (one round of five timed steps per mode, with the bench's new `--trace` and `--cache-limit`
  options) attributes most of that difference to memory allocation: at the bench's 1.4 GB cache limit the activation
  buffers each block frees already fill MLX's buffer cache, so every block's decoded output is released instead of
  kept and allocated fresh for the next block: about 30 ms for a double block and 14 ms for a single one on the
  per-block stamps, and 1.29 s of a step as the difference between two cache limits. With a 2.5 GB limit, large enough
  that a decoded buffer survives in the cache next to those activations, the
  per-block overhead on schnell is 4.0 % for one more GiB of memory, and the in-step decode time is 1.5 times the
  isolated prediction. Decoding the next block on a second GPU stream (`df11-prefetch`, submitted after the previous
  block's evaluation returns) hides about a fifth of the kernel's time and, at that submission point, none of the
  allocation; since the cache limit removes that cost anyway, the integration will evaluate per block and size the
  cache limit instead. A timed process peaks at about 19 GiB of memory, 20.3 GiB with the look-ahead.
- The scripts' memory watchdog now enforces its ceiling on the process footprint the OS reports rather than on RSS
  plus MLX memory, which counted loaded arrays twice.
- A block-boundary integration layer, `mlx_dfloat.integrate`: zero-size placeholders for a module's matrices, weight
  providers that decode a DFloat11 group just in time, reuse one already decoded, or hand back a resident BF16
  weight, and a seam that assigns one block's weights, runs the block, evaluates by policy and restores the
  placeholders afterward. The seam and the providers raise when a matrix name falls outside the map or a weight
  comes back the wrong shape, instead of leaving the layer at its placeholder value; a separate coverage check makes
  sure every other parameter (biases, norm scales, embedders) gets exactly one tensor from the checkpoint. The FLUX.1
  adapter, `mlx_dfloat.mflux.flux1`, reads mflux's own weight mapping to name each block's matrices and builds mflux's
  transformer directly from a DFloat11 checkpoint; it needs the optional `mlx-dfloat[mflux]` extra and is tested with
  mflux 0.20.0. Three new errors mark
  this boundary: `DFloatIntegrationError` for a seam or name-map invariant that failed, `DFloatUnsupportedError` for
  an option this path does not implement, and `DFloatDependencyError` for a missing optional dependency. The step
  bench and the control validation now run through this integration code instead of a separate rig.

- `DFloatFlux1` and the `mlx-dfloat generate` command: FLUX.1 schnell, dev and Krea-dev images from a DFloat11
  transformer through mflux, decoded one block at a time. The VAE and the text encoders come from the base
  repository; its BF16 transformer is never downloaded. The text encoders and the compressed transformer are never
  in memory together: a prompt is encoded first, then the encoders are dropped and the compressed weights loaded;
  a new prompt after that reloads both (`encode()` pre-encodes several prompts at once). Each call derives an MLX
  buffer-cache limit from the checkpoint and the resolution, estimates the peak memory per phase against the
  device's budget, and restores the process's cache limit afterwards. A call that would not fit is refused, and so
  is any size above 1024², the largest measured so far (`fit_check=False` overrides both; above 1024² the estimate
  is an extrapolation). On a 32 GB Mac every call drops the compressed set before the VAE decode, because at 1024²
  the decode next to the resident set measured 23.29 GiB, over the budget, and smaller sizes have not been measured;
  the next call reloads the set in about 26 s. A Mac with a larger budget keeps it. All three base repositories are
  gated on the Hub and need `hf auth login`: FLUX.1-schnell's with automatic approval once its Apache-2.0 license is
  accepted, FLUX.1-dev's and FLUX.1-Krea-dev's with manual approval under Black Forest Labs' non-commercial license.
  Their encoders and VAE are byte-identical, so the schnell base also serves dev and Krea-dev. The command installs the memory caps and the footprint watchdog; the Python API does
  neither. Measured on an M1 Max (32 GB, macOS 27.0, mlx 0.32.2, mflux 0.20.0) at 1024²: FLUX.1-schnell (4 steps)
  peaked at 19.94 GiB (19.95 GiB with depth-2 evaluation) in 1 minute 54 seconds including imports, with per-phase
  MLX peaks of 10.09 GiB encoding the prompt, 15.20 GiB loading the compressed set, 17.44 GiB denoising and 9.55 GiB
  decoding the VAE; FLUX.1-dev (20 steps, guidance 3.5) peaked at 20.10 GiB in 7 minutes 6 seconds; FLUX.1-Krea-dev
  (20 steps, guidance 3.5) peaked at 20.09 GiB in 7 minutes 2 seconds. All three stay under the machine's measured
  22.96 GiB budget. `scripts/verify_image.py` compares the final latents with those of the same transformer
  streaming the BF16 shards block by block: FLUX.1-schnell's latents match bit for bit and are not degenerate, and
  the two images match pixel for pixel; FLUX.1-dev and FLUX.1-Krea-dev were not checked this way, because their BF16
  transformers are gated and were not on disk to compare against. Quantisation, LoRA, img2img, ControlNet and the
  PiD decoder are refused with a reason; `negative_prompt` is accepted and ignored, as mflux does for FLUX.1.

### Changed

- CI runs the whole test suite, Metal tests included, on the macOS runner, which reports a Metal device; a probe step
  prints the device it found.
