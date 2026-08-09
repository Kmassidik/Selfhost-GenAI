# RunPod H100 Plan — MiniMax-H3 "Full Quality" Deployment

> **Status:** parked for later (next week). This is the recipe to run H3 *properly* on a rented H100, where our entire 8 GB bottleneck disappears. Written now so it's ready.

---

## Why the H100 changes everything (our experiment proves it)

Our 8 GB deep-dive found the wall was the **int8 quantization/GEMM kernels** (the `fast_int8_mm` 2.24 GB temporary — see `EXPERIMENT-LOG.md` E-3b/c). That entire problem **only exists because bf16 doesn't fit in 8 GB**, forcing int8.

On an **H100 (80 GB)**, the **40 GB bf16 model fits fully resident** → **no int8 path → the bottleneck vanishes.** Plus native 768p (no upscale crutch), and CUDA 13 enables the native kernels that cu128 disabled for us. So on the H100 you get **fast AND high-quality** with zero of our workarounds.

**Community-verified benchmark (the target):** H100 80 GB · ComfyUI latest · `minimax_h3_fl2va_pruned_bf16` (40 GB) · PyTorch 2.13 + CUDA 13.0 · Turbo LoRA → **10 s clip in under 3 minutes, native 768p.**

---

## The stack to install

```
GPU      : RunPod H100 80 GB (SXM or PCIe)
Template : PyTorch 2.13 + CUDA 13.0  (or install into a fresh venv)
Engine   : ComfyUI latest (git clone, NOT pinned to 0.30)
Model    : Kijai/MiniMax-H3_comfy  →  minimax_h3_fl2va_pruned_bf16  (~40 GB)
           + Qwen3-VL encoder (bf16)  + video/audio VAEs (fp16/fp32)
Speed    : MiniMax-H3 Turbo-LoRA (larryvrh) — same 5× lever we used
```

## Setup steps

1. **Pod:** launch H100 80 GB, PyTorch 2.13 / CUDA 13 image. Attach a volume with ≥ 80 GB for weights.
2. **ComfyUI:** `git clone` latest + `uv venv` + `uv pip install -r requirements.txt` + install PyTorch 2.13 cu13.
3. **Weights** → `models/`: bf16 DiT (40 GB), Qwen3-VL bf16 encoder, video VAE fp16, audio VAE fp32. Use an `HF_TOKEN` (`hf download`, Xet). Repo: `Kijai/MiniMax-H3_comfy` (or `Comfy-Org/MiniMax-H3` bf16 files).
4. **Turbo:** `git clone` `Larryvrh/ComfyUI-MiniMax-H3-Turbo` → node; download `minimax_h3_turbo_v4_step600_ema.safetensors` → `models/loras/`.
5. **Workflow (the key difference from 8 GB):**
   - `UNETLoader` bf16 DiT → `MiniMaxH3TurboLoRA` → sampler
   - **Native 768p** (short side 768, e.g. **1344×768**) — NO upscale node needed
   - `MiniMaxH3TurboSampler`, steps **4–8**
   - **NO int8, NO FFN-chunk, NO upscale, NO int8 patch** — all of that was 8 GB-only

## What transfers from the 8 GB work / what doesn't

**Transfers:**
- The **Turbo-LoRA** speed lever (identical 5× win)
- The **structured prompt format** (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`) — biggest quality lever, hardware-independent
- The finding that **audio needs post-processing** (normalize/limit — universal to H3)
- ComfyUI graph structure via the `/prompt` API

**Does NOT apply (8 GB-specific):** the int8 quant path + our exact int8 patch, the generate-small-then-upscale pipeline, the FFN-chunk / attention-split / memory-snapshot experiments, `--disable-cuda-malloc`.

## Expected result & cost

- **10–15 s at native 768p in ~3–6 min**, high quality, no upscale artifacts.
- Audio: run a post pass (`ffmpeg` loudnorm/limiter, or a DAW touch-up).
- Cost: H100 ≈ $2–4/hr on RunPod — spin up, render a batch, spin down.

## For 2K / longer

- **2K** needs `H3-Regenerate-2K`, which MiniMax has **NOT open-sourced** (API only) — not available in any local stack yet.
- Batching becomes possible with 80 GB (multiple clips per forward) — unlike 8 GB.

---
_This is the "graduation": 8 GB was hard mode that taught us the machine; the H100 is where that understanding runs unconstrained._
