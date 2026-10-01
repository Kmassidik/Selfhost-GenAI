# 🗂️ Build Queue — living task list

> The running list of what we're building and in what order. Update status as we go.
> Last updated: 2026-09-02.

> **🔀 REPRIORITIZED 2026-09-02:** user called it — stop the 1-GPU treadmill. **Skipping ahead to Task 3 (the 3-GPU engine).** Tasks 1 (cleanup) + 2 (f16-at-length) are **paused** until the engine exists. An fp16 canta render is still finishing on GPU 0 in the background (it needs no attention and gives the quality A/B for free; the engine's design phase needs no GPU).

---

## 🔨 ACTIVE BUILD → Task 3: the 3-GPU block-cache engine
See task 3 below. Approach: R&D — **research the concrete sharding mechanism first** (don't blind-code a huge pipeline), then build, then test on the freed box.

## ⏳ Background (no attention needed)
- **fp16 canta render** — full-precision bf16, 124 frames, reusing canta's encode. Output `exemplo-canta-f16.mp4` for the int8-vs-fp16 A/B. ~2 hrs.

## 🎯 The 3 queued builds (in order — they chain)

### 1. 🧹 Box folder cleanup
- **What:** tidy `/root/Desktop/selfhosted-minimaxi-h3/` — see [[BOX-CLEANUP-PLAN]].
- **Why first:** clean layout before we add f16 / engine work.
- **Blocker:** must run only when the box is idle (no active render).
- **Status:** ⏸️ PAUSED (do after the engine).

### 2. 🎯 f16 (full-precision) render
- **What:** run H3 at native bf16 — no int8 quant, no quality loss (the fal.ai ceiling).
- **How:** drop the `TorchAoConfig` int8 quant from `h3_denoise_local.py`; load `dtype=torch.bfloat16`; keep block-streaming.
- **Cost:** ~2× slower + tighter frames (f16 block 1.29 GB vs int8 0.65 GB → ~7.52 GB + 0.64 ≈ OOM at 209 frames → must drop to ~150 frames / ~6 s).
- **Goal:** prove full-precision quality on a short clip, side-by-side vs the int8 canta (same seed 2047).
- **Status:** 🟡 short-clip A/B RUNNING (background); f16-at-length PAUSED until the engine.

### 3. 🔌 3-GPU block-cache engine
- **What:** pipeline the 50 blocks across the 3 cards; the 2 idle GPUs hold ~20 blocks pre-loaded (resident cache) so we stream ~30 instead of 50 → ~40% less streaming-wait. Only the thin hidden state (~0.27 GB) crosses the no-NVLink link at stage boundaries.
- **Why:** ~1.3–1.5× faster, still int8-quality — **and it's what makes f16 practical at length** (f16 doubles streaming; the cache claws it back).
- **Cost:** real build (pipeline + cache orchestration), not a config flag. Sequential (no compute speedup — the win is less waiting).
- **Status:** 🔨 **ACTIVE — building now.** Step 1: research the concrete sharding mechanism (accelerate device_map + per-GPU offload vs custom torch pipeline) before writing the pipeline.

## The chain
```
4-women lands → 🧹 cleanup → 🎯 f16 (short, prove quality)
             → 🔌 engine (cache across 3 GPUs) → f16 at LENGTH + speed
```
The engine + f16 aren't separate goals — **the engine is what lets f16 run long and fast.**

---

## 📋 Backlog (not forgotten, just later)
- **v3 eyes tweak** — "eyes open + occasional natural blink" (v2 overcorrected to an unblinking stare).
- **Control pipeline (no training)** — deterministic fixes around the good model: audio-driven **lip-sync** post-stage (kills the silent-2nd-singer), **face restoration** (GFPGAN/CodeFormer), **i2v** reference conditioning for identity/eyes, **best-of-N** seed selection. Fetch current-best tools before picking (post-cutoff, moves fast).
- **Pre-made H3 LoRA search** — Civitai/HF: does a good one exist for faces/realism? (The fal LoRA we tried *softened* the output — bad fit.)
- **15 s via chunk-and-chain** — now that single-shot reaches 8.7 s, 15 s = **2 × 8.7 s takes (one seam)** instead of 3 × 5 s. Buildable without kernel surgery.
- **Kernel surgery** — fused rotary + chunked attention/FFN → true 15 s **single-shot** (zero seams). The big one.
- **Step distillation (49 → ~4 steps)** — the algorithmic-core move (needs a rented run). ~12× fewer steps. See [Ch.35](../knowledge-base/35-is-it-really-ours.html).

## Golden rules
- Reach the box over **Tailscale** (`100.122.45.32`), never the flaky public IP. Never rapid-retry SSH.
- **Never reorganize / move files a running render depends on.** Tidy only when idle.
