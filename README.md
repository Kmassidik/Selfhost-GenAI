# Self-Hosted GenAI on 8 GB — Master Index & Session Log

_One box (`dalang@<box>`, RTX 3060 Ti 8 GB), one day: from "SSH in" to a 2-minute animated film + an exact systems optimization. This is the front door — what we did, what we tuned, and where every detail lives._

Last updated: **2026-08-09**

---

## TL;DR — what happened

We self-hosted **MiniMax-H3** (a 33-billion-parameter text/image→video+audio model) on an **8 GB gaming GPU** that officially needs an 80 GB datacenter card. Then we **profiled the real bottleneck** (the int8 quantization kernels), **patched it exactly** for a bit-identical ~2 GB VRAM win, and used the headroom to raise the 15-second native resolution ceiling **1.87×**. Along the way we produced promos, showcase clips, and a continuous **2-minute animated short film** ("Pip and the Lost Light").

---

## The machine

```
GPU     : NVIDIA RTX 3060 Ti, 8 GB VRAM, Ampere sm_86   Driver 595.84 / CUDA 13.2
CPU/RAM : 24 cores / 60 GB      Disk: 148 GB NVMe (/tmp = 31 GB tmpfs)
OS      : Ubuntu 26.04          SSH key installed (no more password races)
Runtime : Python 3.12 (uv) + PyTorch 2.11 cu128 + ComfyUI 0.30 (systemd service)
Model   : MiniMax-H3 33B, flow-matching, FLOW_AV, int8 — streamed from RAM, not resident
```

---

## What we TUNED (every lever, with results)

| # | Tune | Result | Kind |
|---|---|---|---|
| 1 | **NVIDIA driver + CUDA + PyTorch** install, GPU verified | H3 runs at all | setup |
| 2 | **Python 3.12 via uv** (system was 3.14, unsupported) | isolated env | setup |
| 3 | **Native int8 safetensors** (not GGUF — GGUF was wrong arch) | DiT loads | correctness |
| 4 | **`curl --http1.1`** for downloads | beat HF throttling | ops |
| 5 | **Structured prompt format** (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`) | biggest **quality** lever | quality |
| 6 | **Turbo-LoRA** (20 → 4-8 steps) | **~5× faster** sampling | speed |
| 7 | **Generate-small → 4x-UltraSharp upscale** | true 720p/1080p output | quality |
| 8 | **systemd service** (`comfyui.service`) | auto-start, survives reboot | ops |
| 9 | **Removed `--reserve-vram`** | reclaimed ~0.5 GB (it was costing us) | memory |
| 10 | **`--disable-cuda-malloc`** (native allocator) | enabled memory profiling + real `expandable_segments` | memory |
| 11 | **⭐ EXACT int8-matmul chunk patch** | **−2 GB sampling VRAM, bit-identical (ΔQ=0)** | memory |
| 12 | **Result of #11: raised 15 s native ceiling 576×320 → 768×448** | **1.87× more pixels** | quality |
| 13 | **Frame-chaining** (last frame → next first frame) | continuous, character-consistent film | technique |

### What we PROVED DOESN'T help (equally valuable — saved us time)
- **FFN-chunk** → flat, no VRAM benefit (bottleneck isn't the FFN)
- **Attention-split** → moot (attention isn't in the top allocations)
- **Encoder evict** → skipped (encoder is only ~3 GB, not the floor)
- **empty_cache trim** → no effect (overhead is fixed CUDA context)
- **SageAttention** → BROKEN for H3 (outputs pure noise — a known bug)
- **CPU-offloaded KV attention** → doesn't exist for DiTs, would be 3-10× slower
- **Raw diffusers instead of ComfyUI** → no VRAM win, needs ~75 GB RAM we don't have

### The headline finding (the whole experiment in one line)
> The 8 GB wall was the **int8 quantization/GEMM working buffers** — a single 2.24 GB `fast_int8_mm` temporary — NOT the FFN, attention, or model weights. Chunking that one matmul (bit-identical) freed ~2 GB and pushed the resolution ceiling up. **Found by profiling, not guessing.**

---

## Video-production techniques we built

- **Structured prompt recipe** — the reusable template (see `PROJECT-SUMMARY.md`).
- **Turbo + upscale pipeline** — 576×320 native → 1080p, ~3-6 min/clip.
- **Face fix** — realistic faces glitch (eyes) at low res + Turbo; fix = tight close-up + 20 real steps + higher res + calmer motion. Or use **animation** (sidesteps faces entirely — looks best on this hardware).
- **Frame-chaining for continuity** — extract each clip's last frame, feed as next clip's `first_frame` (LoadImage → MiniMaxH3ImageToVideo). Keeps a character consistent across a multi-clip film. NOTE: hard scene changes *morph* rather than hard-cut (reads dreamy).
- **Robust batch driver** — render clips one-at-a-time with crash recovery (ComfyUI OOM-crashes on 15 s+1080p; systemd auto-restarts and the driver resubmits). **Wait-and-retry on frame reads** (SaveVideo finalizes the mp4 asynchronously — reading too early corrupts).
- **Stitch** — ffmpeg normalize each clip to uniform 1280×720/24fps + aac, then concat.

---

## Deliverables (on the Desktop)

**Videos:** `dalang_promo_00001` (server room), `dalang_promo_v2_00001` (spokesperson, eyes fixed), `h3_duet1080`, `h3_war720_15s`, `h3_dragon720_15s`, `h3_ceiling_704` (first clip impossible pre-patch), **`pip_story_full.mp4`** (2-min animated short) + 8 scene clips.

**Docs (this folder):**
| File | What |
|---|---|
| `README.md` | ← you are here: master index + all tunes |
| `EXPERIMENT-LOG.md` | lab notebook — every experiment E-0…E-5 with data |
| `answer.md` | the master experiment plan (26 sections) |
| `8gb-experiment-answer.txt` | spec + answer, plain text for sharing |
| `SAMPLING-MATH.md` | the flow-matching / Amdahl math |
| `PROJECT-SUMMARY.md` | specs, render times, prompt recipe |
| `runpod-plan.md` | H100 recipe for scaling up (bf16, no int8 needed) |
| `knowledge-base/` | 10-chapter learning site (open `index.html`) |

---

## How to reproduce the key win (int8 patch)

The exact patch lives in `site-packages` (lost on pip reinstall). Re-apply script: `patch_q.py` (in the session scratchpad). It edits `comfy_kitchen/backends/eager/quantization.py` → `int8_linear` to chunk the matmul into the existing scaling loop. Backup at `quantization.py.orig_backup`. Requires `--disable-cuda-malloc`. Verified bit-identical with `torch.equal`.

---

## Next / open threads

- **RunPod H100** (next week) — `runpod-plan.md`. bf16 model fits fully → the int8 bottleneck vanishes → native 768p, fast, no upscale crutch.
- **W4A8 quant** (Kijai) — the approximate lever, quality-gated, needs ComfyUI upgrade + comfy-kitchen.
- **Re-render any single Pip scene** and re-stitch (pipeline is built).
- **dalang.io promo** — post the ads; the "we ran a 33B model on 8 GB" story is the pitch.
