# Harness proof

This directory shows that the memory watchdog in `mlx-dfloat generate` stops a run that goes over a set
ceiling and leaves a run under it alone. Both runs use the same ceiling, the same model and the same seed; only
the image size differs.

## The ceiling

The watched peak of FLUX.1-schnell (4 steps, seed 42) on the host, measured just before the proof runs:

| Size | Watched peak | Source |
|---|---|---|
| 512² | 18.67 GiB (20 044 378 976 bytes) | a plain `generate` run |
| 1024² | 19.78 GiB (21 233 332 920 bytes) | `../tiers/schnell-1024.json` |

The two peaks are only 1.11 GiB apart, because most of the peak is the resident compressed transformer rather than
the activations. The ceiling sits between them:

    C = 19.25 GiB = 20 669 530 112 bytes
    512² headroom:  19.25 - 18.67 = 0.58 GiB under C
    1024² overshoot: 19.78 - 19.25 = 0.53 GiB over C

Both margins are more than three times the run-to-run spread seen for schnell at 1024² (19.94, 19.81 and
19.78 GiB in three runs).

## The runs

Pass (exit 0, report `pass-512.json`, label `PROOF`, watched peak 18.25 GiB):

    uv run --group bench mlx-dfloat generate --model schnell \
      --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
      --seed 42 --steps 4 --height 512 --width 512 --memory-ceiling 20669530112 \
      --report bench/results/harness-proof/pass-512.json --output /tmp/proof-pass.png

Abort (exit 70 after 50 s, artifact `abort.json`: `reason: memory`, `verdict_counter: footprint`, peak 19.32 GiB
against the 19.25 GiB ceiling):

    uv run --group bench mlx-dfloat generate --model schnell \
      --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
      --seed 42 --steps 4 --height 1024 --width 1024 --memory-ceiling 20669530112 --no-fit-check \
      --output /tmp/proof-abort/abort.png

`--no-fit-check` turns off the up-front fit estimate, which would otherwise refuse the 1024² run before it starts,
so the watchdog is what stops it. The watchdog wrote `abort.json` next to the output path; it is copied here
unchanged.

## Setup

MacBook Pro, Apple M1 Max, 32 GB, macOS 27.0.1, on AC. mlx 0.32.2, mflux 0.20.0, mlx-dfloat at git `d164fc5`.
Recorded 2026-10-01.
