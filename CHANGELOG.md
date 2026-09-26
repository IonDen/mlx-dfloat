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
