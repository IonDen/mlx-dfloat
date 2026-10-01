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

Runs of the same recipe do not land on the same peak. Schnell at 1024² measured 19.94, 19.81 and 19.78 GiB in three
runs, and three 512² runs landed between 18.25 and 18.67 GiB (18.64 GiB in the pass run below), a 0.42 GiB spread
close to the 0.53–0.58 GiB margins. Both runs still came out as predicted: the 512² run passed and the 1024² run
was stopped.

## The runs

Pass (exit 0, report `pass-512.json`, label `PROOF`, watched peak 18.64 GiB):

    uv run --group bench mlx-dfloat generate --model schnell \
      --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
      --seed 42 --steps 4 --height 512 --width 512 --memory-ceiling 20669530112 \
      --report bench/results/harness-proof/pass-512.json --output /tmp/proof-pass.png

Abort (exit 70 after 48 s, artifact `abort.json`: `reason: memory`, `verdict_counter: footprint`, peak 19.32 GiB
against the 19.25 GiB ceiling):

    uv run --group bench mlx-dfloat generate --model schnell \
      --prompt "A stone lighthouse on a rocky shore at dawn, waves breaking below it and a small fishing boat far out on the water" \
      --seed 42 --steps 4 --height 1024 --width 1024 --memory-ceiling 20669530112 --no-fit-check \
      --output /tmp/proof-abort/abort.png

`--no-fit-check` is not needed here: `--memory-ceiling` leaves the fit budget at the host's 22.96 GiB, and the 1024²
estimate fits under it. It is passed so that nothing but the watchdog can stop the run. The watchdog wrote
`abort.json` next to the output path; it is copied here unchanged.

## Setup

MacBook Pro, Apple M1 Max, 32 GB, macOS 27.0.1, on AC. mlx 0.32.2, mflux 0.20.0. The ceiling was chosen from
runs at mlx-dfloat git `d164fc5`; the two proof runs are at git `9c51869`, whose abort artifact also records the run's
model, size, seed and step count. Recorded 2026-10-01.
