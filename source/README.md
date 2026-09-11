# Dalang — self-hosted MiniMax-H3 on consumer GPUs

Running a **33B video diffusion model** (MiniMax-H3, video + audio) on three
RTX 3060 Ti with **8 GB of VRAM each**, without ComfyUI or any inference server.

This directory is the working source. The narrative — why each piece is shaped the
way it is, with measured numbers — lives in [`../knowledge-base/`](../knowledge-base/),
48 chapters of it. Start with **ch.48 · The Pipeline, Without ComfyUI** for the
architecture and **ch.47 · The Scorecard** for what the hardware actually delivers.

## What this achieves

Every row below is a render that exists, measured on 3× RTX 3060 Ti 8 GB:

| Resolution | Frames | Duration | Steps | Denoise time |
|---|---|---|---|---|
| 512×288 | 345 | 15 s | 49 | 66 min |
| 832×480 | 124 | 5 s | 4–50 | minutes–1 h |
| 1280×704 | 124 | 5 s | 23 | 55 min |
| **1280×704** | **345** | **15 s** | 3 | 5.5 h |

Native 720p at 15 seconds runs on 24 GB of consumer VRAM. It is expensive
(≈90 h at a production step count) — see ch.47 for why 480p + a 1.54× upscale is
usually the better trade.

## Layout

```
runtime/h3-consumer-bench/     the engine
  h3_cenas.py                  scenes: prompt + frames/height/width, a plain dict
  h3_encode_local.py           phase A — encode, writes h3-lab/embeds/<scene>.pt
  h3_denoise_local.py          phase B — denoise + decode, writes mp4 + wav
  h3_meminstr.py               memory instrumentation (behaviour-neutral)

dalang/                        drivers and patches
  00–04*.py                    reference capture, baselines, ring-attention proofs
  05_render_ring.py            driver: ring / context parallelism, 2 cards
  08–10_render_tiled.py        driver: single-card tiling
  12–15_shard_*.py             sequence-parallel build, validated stage by stage
  16_render_3gpu.py            driver: 3-GPU sequence parallelism
  h3_decode_fix.py             safety: device fix + latent backup
  h3_vae_cpu_accum.py          safety: keep finished pixels off the GPU in decode
  17_decode_from_latents.py    recovery: rebuild mp4/wav from a latent dump
```

## Architecture in one paragraph

H3 is two large models — a 32B encoder and a 33B DiT — and they never share a
process. **Phase A** builds a *partial* pipeline containing only the encoder blocks
and saves its full intermediate state to disk. **Phase B** starts fresh, loads only
the DiT, and reads that file. The mechanism is `diffusers`'
`SequentialPipelineBlocks.from_blocks_dict`, which lets you assemble a pipeline from
a subset of the graph. (Asking a full pipeline for one output still *runs* the whole
graph — that lesson cost a rewrite.)

The practical payoff is **encode once, render many**: embeddings are fixed per scene,
so steps, resolution, engine and seed can all change for free.

Drivers wrap phase B. Each monkeypatches the transformer's forward and then calls
`h3_denoise_local.main()` — the engine itself never changes, which is why each
parallelism strategy could be validated against a golden reference before use.

## Running it

```bash
cd runtime/h3-consumer-bench

# Phase A — once per scene set (no arguments; walks every scene in h3_cenas.py)
../../.venv/bin/python h3_encode_local.py

# Phase B — pick the driver matching the parallelism you need
H3_LATENT_BACKUP=docs/myrun \
../../.venv/bin/python ../../dalang/16_render_3gpu.py <scene> \
    --steps 23 --n 3 --saida h3-lab/<scene>.mp4

# Recovery — decode never has to cost you the denoise
../../.venv/bin/python ../../dalang/17_decode_from_latents.py \
    --video docs/myrun.video.pt --saida h3-lab/<scene>.mp4
```

**Always set `H3_LATENT_BACKUP`.** It writes the latents the instant denoise
finishes, before anything downstream can fail. A device bug in the decode stage once
destroyed six hours of completed denoise; with the backup, the same failure the
following night cost only decode time.

Add a shot by adding a dict entry to `h3_cenas.py`. That is the whole interface.

## Caveats for anyone cloning this

**Paths are hardcoded.** Roughly 25 files contain `/root/Desktop/selfhosted-minimaxi-h3`.
This is research code that grew on one machine; it was never parameterised. Search
and replace before running elsewhere.

**Versions are load-bearing.** The safety-layer patches (`h3_decode_fix.py`,
`h3_vae_cpu_accum.py`) contain faithful copies of upstream `diffusers` function
bodies with small deliberate changes. **Re-diff them against upstream after any
`diffusers` upgrade** — they are marked with `G4K` / `G4L` comments. They are
installed as monkeypatches from the driver rather than edits to `site-packages`,
so a `pip install` cannot silently revert them, but it can silently *outdate* them.

**Not in this repo:** model weights (tens of GB, fetched from Hugging Face),
`h3-lab/embeds/*.pt` (regenerate with phase A), and rendered output. All gitignored.

**Known-good settings.** `use_stream=False` on group offloading — streaming pins tens
of GB and the kernel OOM-kills the process with no traceback. SageAttention produces
pure noise on H3; leave it off. `VAEDecodeTiled` is incompatible — the latent is a
NestedTensor.

## Step counts

The most misread number in this project. Step count is a property of the **scene**,
not the model:

| Content | Steps |
|---|---|
| Tight close-up portrait | 4 |
| Character, mid-shot | 23 |
| Wide vista with a small subject | 49 |

A 15-second epic vista rendered at 3 steps is mud. The same scene at 49 steps is
finished work. Budget by content.
