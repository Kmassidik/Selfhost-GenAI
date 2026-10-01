# MiniMax-H3 8 GB — Experiment Log (Lab Notebook)

> **Purpose:** a durable, chronological record of every measurement, decision, and result, so this project continues even if nobody remembers the context. Append-only. Newest experiments at the bottom of the log section.
>
> **How to use:** read "Current State" first, then the numbered experiments. Each entry has Question / Method / Result / Conclusion / Next. Companion docs: `answer.md` (the master plan/spec), `8gb-experiment-answer.txt` (spec + answer for sharing), `SAMPLING-MATH.md`, `knowledge-base/` (concepts).

Last updated: **2026-08-09**

> **See `README.md` for the master index of the whole session (all tunes + deliverables).**

---

## Quality Findings (2026-08-10) — the "why does it look broken" investigation

**RAM correction:** the box is **64 GB DDR3-1600 ECC LRDIMM** (2×32 GB Hynix), NOT generic "60 GB". ~60 GB usable. It's an older server/workstation board (many empty DIMM slots report DDR2 default). DDR3-1600 ≈ ~25 GB/s dual-channel — 2-3.5× slower than DDR4/DDR5 — which is a secondary bottleneck on the RAM→VRAM weight streaming. Roomy but slow; well-suited to the offload strategy, but caps streaming speed.

**RunPod: ARCHIVED.** Decision — optimize this 8 GB box, don't rent. `runpod-plan.md` banner-archived.

**The "broken video" root cause (measured, not guessed):**
1. File not corrupt (ffmpeg decodes clean).
2. Upscaler innocent — adds only 1.02× temporal change (measured).
3. Turbo steps innocent for *motion* (6-step ≈ 20-step frame-diff).
4. **Motion jitter** → fixed by **RIFE** (24→48 fps, 1.8× smoother, measured).
5. **Melty faces** → it's **pixels-per-face**, not the model. Root cause = 8 GB forces low native res → small faces get ~20 px → melt.

**The pixels-per-face rule (the key insight):**
- 1 person, face fills frame → flawless even at 480p.
- 10 people framed as a group photo, 1080p → all foreground faces good.
- 30-person wide crowd, 576×320 → tiny faces (~20 px) → melt.
- It's face SIZE in pixels that matters, not people count.

**A single still fits FAR higher res than video** (no 362-frame temporal cost). On 8 GB: **1920×1080 single frame fits** and looks photoreal. Ladder (single person): 480p flawless, 720p magazine-grade, 1080p indistinguishable from a photo.

**Teeth fix (scored the 10-person image down to 6/10):**
- ⭐ **Turbo was the main culprit** — its 6-step distillation under-renders fine detail; teeth are the finest. **Full 20-step euler (no Turbo) forms teeth properly.**
- Fewer/bigger faces (5 not 10) → each tooth gets ~4× pixels.
- **CodeFormer** restore (low fidelity) rebuilds teeth better than GFPGAN.
- Combined → crisp teeth, approved.

**Quality-first hero recipe (PROVEN for stills/portraits):**
> single 1080p still · full 20-step euler (NO Turbo) · ≤5-6 framed faces · CodeFormer restore

**Post-processing nodes installed:** ComfyUI-Frame-Interpolation (RIFE VFI), facerestore_cf (GFPGAN + CodeFormer + facexlib + lpips), ComfyUI-ClipProj (4B encoder swap — see docs/clipproj-benchmark.md).

**Turbo vs quality tradeoff (the meta-lesson):** Turbo-LoRA is ~5× faster but trades away fine detail (teeth, small features). Use Turbo for drafts/motion tests; use **full 20-step for hero shots**. Speed and quality were trading off and Turbo was on the wrong side for quality work.

---

## Video Production Log (2026-08-09)

Techniques proven while producing the deliverables (promos + the 2-min Pip film):

- **Face artifact fix (eyes):** realistic faces destabilize (asymmetric/glassy eyes) at low native res + Turbo + high motion. Fix that worked: **tight close-up** (more pixels on face) + **20 real steps** (not distilled Turbo) + **768×448** + **calmer expression**. Verified by frame-checking frames 60–80. Alternative: use **animation** — sidesteps realistic faces entirely and looks best on this hardware.
- **Frame-chaining for a continuous multi-clip film:** H3 caps at 15 s/clip. To make a 2-min film, chain clips: extract each clip's LAST frame → feed as next clip's `first_frame` (LoadImage → MiniMaxH3ImageToVideo `first_frame` input). Keeps the character (Pip) consistent AND flows scene-to-scene. Verified: pip_02's opening frame ≈ pip_01's ending frame. CAVEAT: hard scene changes morph rather than hard-cut (reads dreamy — fine for storybook).
- **Batch crash recovery:** ComfyUI OOM-crashes on 15 s + 1080p-upscale loads (5 s clips are fine); systemd auto-restarts but that WIPES the queue. Fix: a driver that renders **one clip at a time**, polls the prompt_id for completion, and on crash waits for restart + resubmits that one clip. So a crash costs 1 clip, not the batch.
- **Async-write race:** `SaveVideo` finalizes the mp4 asynchronously (moov atom written after the file appears). Reading the last frame too early → `InvalidDataError` and the driver dies. Fix: **wait-and-retry** (up to ~25× / 3 s) on any frame extraction.
- **Stitch:** no ffmpeg by default → `apt-get install ffmpeg`. Normalize each clip to uniform `1280×720, 24fps, aac 32kHz` (scale+pad+fps), then `concat` demuxer with `-c copy`. Output: `pip_story_full.mp4`, 120.86 s, 2896 frames.

Deliverables: 2 promos, duet/war/dragon showcase clips, `h3_ceiling_704` (first clip impossible before the int8 patch), and **`pip_story_full.mp4`** (2-min animated short).

---

## Current State (read this first)

- **Goal:** hold a fixed baseline quality while pushing peak VRAM LOWER (free headroom), then spend it on higher res/frames. Quality first; runtime is free. Partition the computation, never shrink the problem.
- **Baseline B:** the woman+man duet, native **576×320 × 362 frames** (15 s, 24 fps), upscaled to 1080p. Peak ~7.57–7.83 GB. Stresses identity+face+mouth+motion+temporal+audio+long context.
- **Key finding so far:** the peak VRAM is **NOT** the encoder and **NOT** frame-scaling attention. It is the **DiT sampling forward pass (~7.8 GB)**, with the **VAE decode a close second (~7.6 GB)**. There are TWO bottlenecks.
- **Next experiment:** FFN-chunk sweep (running) → then attention-split sweep.

---

## Machine & stack (the box)

```
GPU     : NVIDIA RTX 3060 Ti, 8 GB VRAM, Ampere sm_86
Driver  : 595.84 / CUDA 13.2
CPU/RAM : 24 cores / 60 GB      Disk: 148 GB NVMe (/tmp = 31 GB tmpfs)
OS      : Ubuntu 26.04          Host: dalang@<box> (SSH key installed)
Runtime : Python 3.12 (uv) + PyTorch 2.11 cu128 + ComfyUI 0.30
ComfyUI : systemd service `comfyui.service`
          ExecStart: main.py --port 8188 --disable-smart-memory
          Env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
          (NOTE: --reserve-vram was REMOVED — it cost usable VRAM; see D-4)
          Manage: sudo systemctl {status,restart} comfyui ; logs: sudo journalctl -u comfyui -f
Model   : MiniMax-H3 (33B, flow-matching, FLOW_AV). Weights streamed from RAM, not resident.
```

**Files (`~/ComfyUI/models/`):** DiT `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`; encoder `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors`; VAEs `vae/minimax_h3_{video_vae_fp16,audio_vae_fp32}.safetensors`; Turbo LoRA `loras/minimax_h3_turbo_v4_step600_ema.safetensors`; upscaler `upscale_models/4x-UltraSharp.pth`.

**Custom nodes:** ComfyUI-GGUF, KJNodes, rgthree, ComfyMath, **ComfyUI-MiniMax-H3-Turbo** (Turbo LoRA + Turbo Sampler), **ComfyUI-sol-attn** (`MiniMaxH3ChunkFeedForward` = exact FFN partition; skip its approximate `...ScheduledSolAttentionPatch`).

---

## Levers: exact vs approximate (quality gate)

**Exact (execution-preserving, ΔQ ≈ 0 — bit-identity NOT yet proven, only numerical closeness):**
- FFN-chunk (`MiniMaxH3ChunkFeedForward`, params: chunks 1–64, min_tokens threshold default 8192, min 256)
- attention_split (`--use-split-cross-attention`)
- weight offload (already on), layer streaming, activation offload/recompute, async prefetch

**Approximate (ΔQ ≠ 0 — investigate LAST, quality-gate vs B):**
- W4A8 quant DiT (Kijai — see BACKLOG), sparse/tau attention, lower quant, resolution/frame reduction
- SageAttention — **BROKEN for H3** (outputs pure noise, ComfyUI bug). Do not use.

---

## EXPERIMENTS

### E-0 — Differential VRAM profiling (peak vs problem size)
- **Q:** does peak VRAM scale with resolution / frames, or is there a fixed floor?
- **Method:** run H3 at several sizes, steps=2 (peak is per-forward, independent of step count), sample nvidia-smi peak.
- **Result (peak MB):**
  | Config | Peak |
  |---|---|
  | 512×288 × 5f | 7412 (warmup-inflated) |
  | 512×288 × 73f | 6570 |
  | 512×288 × 181f | 6570 (identical → FLAT) |
  | 512×288 × 362f | 7834 |
  | 576×320 × 362f | 7832 (~same as 512×288 → resolution nearly free) |
- **Conclusion:** fixed floor ≈ **6.5 GB**; frame-scaling adds ~1.3 GB only at 362f; **resolution is nearly free** within tested range (but NOT at the top — see D-2).
- **Next:** decompose the floor (E-1).

### E-1 — Phase attribution of the peak  *(your plan Experiment #1)*
- **Q:** what phase owns the ~7.8 GB peak — encoder, DiT sampling, or VAE decode?
- **Method:** one 576×320×362 run (steps=2), sample nvidia-smi at 5 Hz with timestamps, align to ComfyUI journal phase markers (TEModel load = encoder; MiniMaxH3 load = sampling; VideoVAE load = decode).
- **Result:**
  ```
  overall peak      : 7832 MB
  ENCODER phase peak:  2996 MB
  SAMPLING phase    : 7832 MB   <- the peak
  VAE_DECODE phase  : 7578 MB   <- second peak
  ```
- **Conclusion:** **Hypothesis A (encoder residency) REJECTED** — encoder is only ~3 GB. The peak is the **DiT sampling forward**; the **VAE decode is a second near-peak.** Two bottlenecks, not one.
- **Next:** skip E-2; run FFN sweep (E-3); attention sweep (E-4).

### E-2 — Encode → evict encoder  *(your plan Experiment #2)*
- **Status:** **SKIPPED.** E-1 proved the encoder is not the floor, so eviction cannot help. (Measuring saved the detour — the whole point of profiling first.)

### E-3 — FFN-chunk sweep  *(your plan Experiment #3)* — IN PROGRESS
- **Q:** how much does X_FFN reduce the 7.8 GB sampling peak, and where does it asymptote?
- **Method:** 576×320×362, steps=2, MiniMaxH3ChunkFeedForward(chunks ∈ {off,2,4,8,16}, min_tokens=256), measure peak + time.
- **Result (576×320×362, steps=2, min_tokens=256):**
  | X_FFN | peak MB | time |
  |---|---|---|
  | off | 7832 | 202 s |
  | 2 | 7832 | 172 s |
  | 4 | 7832 | 175 s |
  | 8 | **7162** | 180 s |
  | 16 | 7832 | 191 s |
  - (First attempt used min_tokens=0 → HTTP 400; min_tokens min is 256. Re-ran with 256.)
- **Conclusion:** **FFN-chunk gives NO reliable VRAM reduction.** The curve is flat at 7832 except a single non-monotonic dip at X=8 (7162) that does NOT hold at X=16 → almost certainly allocator/reserved-pool measurement noise, not a real FFN effect (a true effect would be monotonic). **The FFN intermediate is NOT the dominant term in the 7.8 GB sampling peak** — contradicts the general "FFN is the largest transient" claim for this stack. Consistent with the 640×384 OOM-with-FFN-chunk observation.
- **Caveat:** nvidia-smi measures RESERVED (high-water mark); FFN-chunk could lower FFN's own allocation while a different allocation still sets the 7832 ceiling. Only the pickle snapshot (allocation-site) can separate these.
- **Next:** the 7.8 GB is set by something else → (a) attention-split sweep E-4, and (b) **pickle memory snapshot** to finally attribute the sampling peak to weights vs attention vs workspace vs allocator. Snapshot is now the highest-value unknown.

### E-3b — Pickle memory snapshot: allocation-site breakdown of the peak  *(the big one)*
- **Q:** what allocations actually hold the ~7.8 GB (or the 5.9 GB of real tensors) during sampling?
- **Method:** custom nodes `MemRecordStart`/`MemDump` (in `custom_nodes/mem_probe/`) wrap the sampler: reset peak + `_record_memory_history` before, `_dump_snapshot('/tmp/h3_mem.pickle')` + `memory_stats` after. Parse the pickle's `device_traces`, replay alloc/free to the max-total moment, group live allocations by stack site (`analyze_snap.py`).
- **Gotcha:** `_record_memory_history` is **incompatible with `cudaMallocAsync`** (ComfyUI's default allocator). Had to relaunch with **`--disable-cuda-malloc`** (native allocator). SIDE FINDING: our `expandable_segments:True` was being **ignored under cudaMallocAsync** the whole time; it only applies to the native allocator (saved ~200 MB slack once actually active — small).
- **Result — decomposition of the 7.84 GB nvidia-smi peak:**
  | Component | Size |
  |---|---|
  | Real live tensors (allocated peak) | ~5.9 GB |
  | Allocator slack (reserved − allocated) | ~0.5–0.7 GB (tunable) |
  | CUDA context + driver + non-torch + record overhead | ~1.2–1.4 GB (mostly fixed) |
- **Result — top allocation SITES within the 5.9 GB peak (272 live allocs):**
  ```
  2240 MB  fast_int8_mm        (quantization.py:754)   <- one 2.24 GB temporary
  1119 MB  int8_linear x9      (quantization.py:1048)
  1119 MB  int8_linear         (quantization.py:1051)
   ~600 MB int8_linear / _rotate_activation / _round_int8 (convrot int8 path)
   ~500 MB model forward (rope_rotation_table, patchify_video, _forward)
   (91 MB torch::unwind = recording artifact, ignore)
  ```
- **Conclusion (THE finding):** **~88% of the real peak memory is the int8 quantization / GEMM working buffers** (`fast_int8_mm`, `int8_linear`, activation rotation, rounding) — the temporaries the **convrot-int8 kernels** allocate to run the quantized matmuls. It is **NOT** the FFN intermediate, **NOT** attention, **NOT** resident weights. This explains ALL prior negatives: FFN-chunk and attention-split can't help because the memory isn't in the activations they partition — it's in the quant math.
- **The real lever:** a more memory-efficient int8 matmul (chunk the `fast_int8_mm` 2.24 GB temporary), OR a different quant kernel path. **Kijai's W4A8 + comfy-kitchen** ("chunked fused int4→int8 dequant + strided INT8 GEMM") is the prime candidate — smaller GEMM working buffers — BUT it is 4-bit weights = APPROXIMATE (quality-gate). Also worth checking: does the convrot int8 path have a chunked/low-mem mode (would be an EXACT win)?
- **Next:** (1) investigate whether `fast_int8_mm`/`int8_linear` can chunk their temporaries exactly (comfy quant_ops); (2) evaluate W4A8 as the approximate lever (needs ComfyUI upgrade + comfy-kitchen, quality-gate vs B).

### E-3c — EXACT int8-matmul chunk patch (the payoff of E-3b) — ✅ SUCCESS
- **Q:** can `int8_linear`'s 2.24 GB int32 matmul temporary be chunked exactly (no quality cost)?
- **Finding:** YES. `int8_linear` (comfy_kitchen/backends/eager/quantization.py) already chunks the *scaling* loop but materializes the FULL int32 [M,N] matmul result first (`_int8_matmul_accumulate(x_8, weight.T)`) = the 2.24 GB. Moving the matmul *inside* the row-chunk loop + writing into a pre-allocated output kills both the 2.24 GB int32 AND the ~1.1 GB cat-doubling.
- **Exactness:** PROVEN **bit-identical** — `torch.equal(orig, chunked) = True`, max_abs_diff = 0.0 (convrot on & off). Rows are independent (row i = x_8[i]@wt, per-row scale) so blocking changes nothing → **ΔQ = 0 exactly** (not just ≈0).
- **Patch:** applied to `.../comfy_kitchen/backends/eager/quantization.py` (backup at `...quantization.py.orig_backup`). Requires `--disable-cuda-malloc` (native allocator) which we now run. NOTE: this file is in site-packages → lost on `pip reinstall`; keep the patch script `patch_q.py` to re-apply.
- **Result (576×320×362, 2 steps):**
  | Metric | before | after |
  |---|---|---|
  | Sampling allocated peak | 5925 MB | **3887 MB (−2038, −34%)** |
  | Sampling reserved peak | 6426 MB | 4066 MB |
  | Whole-run nvidia-smi peak | 7840 MB | 7570 MB (only −270) |
- **Conclusion:** the exact patch frees **~2 GB during the DiT sampling phase, bit-identical.** BUT the whole-run peak barely moved because the **VAE decode (~7.6 GB) is now the binding bottleneck** (the E-1 "second peak"). Classic whack-a-mole: killed the sampling wall, VAE-decode wall now caps the total.
- **Next:** attack the VAE decode exactly — can it be chunked along frames/time without changing the result? (D-3: `VAEDecodeTiled` broken for H3 nested latent; a frame-wise exact chunk may work.) THEN the sampling 2 GB headroom becomes usable for higher res/frames.
- **TODO validation:** run a FULL render post-patch and eyeball vs baseline B (bit-identity proven on random data; confirm on the real pipeline).

### E-3d — VAE decode snapshot + RESOLUTION CEILING (the payoff) — ✅
- **VAE decode is CHEAP, not a bottleneck:** wrapped the probe around VAEDecode (reset peak before) → **allocated_peak = 1162 MB**. The E-1 "VAE phase 7.6 GB" was leftover RESERVED pool from sampling, NOT VAE consumption. (Correction: VAE is not the wall; earlier claim retracted.)
- **New 15 s resolution ceiling WITH the int8 patch (362 frames, 2 steps):**
  | Config | peak | result |
  |---|---|---|
  | 640×384 | 7618 MB | ✅ (OOM'd pre-patch) |
  | 704×384 | 7594 MB | ✅ |
  | **768×448** | **7828 MB** | ✅ (OOM'd HARD pre-patch) |
  | 832×480 | — | ❌ OOM |
- **Conclusion:** the exact int8 patch raised the 15 s native ceiling from **576×320 → 768×448 = 1.87× the pixels**, bit-identical, zero quality cost, no upgrade, no W4A8. Systems optimization → real resolution/quality gain. **This is the headline result of the whole 8 GB experiment.**
- **Practical rec for renders:** 704×384 for reliable margin (~600 MB), 768×448 is the hard ceiling (~360 MB margin — Turbo LoRA may tip it, fall back to 704 if OOM).

### E-3e — CUDA/reserved-overhead trim (empty_cache) — ❌ NO EFFECT
- **Q:** can the ~1.9 GB gap between nvidia-smi peak (~7.6 GB) and torch tensors (~5.9 GB, ~3.9 GB post-patch) be reclaimed by `torch.cuda.empty_cache()` between encode and sampling?
- **Method:** A/B at 576×320×362 (2 steps), `EmptyCacheLatent` node inserted after the encoder, before the sampler.
- **Result:** A (no trim) = 7568 MB; B (empty_cache) = 7578 MB → **no difference (noise).**
- **Conclusion:** the overhead is **fixed CUDA context + driver + genuinely-needed reserved**, not stranded cache. Not reclaimable. This is the honest **floor of exact optimizations on 8 GB**: the E-3c int8 patch (768×448 15 s ceiling) is the win; there's no more free VRAM to squeeze exactly. Further gains need APPROXIMATE levers (W4A8) or bigger hardware.

### E-4 — Attention-split sweep — MOOT
- E-3b shows attention is not in the top allocations → skip.
- **Method:** `--use-split-cross-attention` (exact query-tiling, full K/V), measure peak vs baseline. Confirm exactness.

### E-5+ — layer streaming / activation offload / recompute / combine — TODO (your §12–15, §24)

---

## DECISIONS & RULES (D-log)

- **D-1:** Partition the computation, NOT the video. Never slice 362 frames into independent clips — breaks the shared latent, destroys temporal coherence. The 362-frame gen stays one logical generation.
- **D-2:** "Resolution is free" holds only at LOW frame counts. At 15 s (362f), 768×448 and 640×384 both OOM'd during sampling → ~576×320 is the native 15 s ceiling on this box.
- **D-3:** `VAEDecodeTiled` is INCOMPATIBLE with H3 (latent is a NestedTensor → TypeError). Use plain `VAEDecode`. The VAE decode second-peak (E-1) therefore has no easy tiling fix yet.
- **D-4:** `--reserve-vram` was removed — it subtracted usable VRAM and caused OOMs that otherwise fit. Keep only `--disable-smart-memory` + `expandable_segments`.
- **D-5:** ΔQ = 0 requires proven bit-identity; `torch.testing.assert_close` only proves closeness. Report exact levers as **ΔQ ≈ 0** until a true equality test is run.
- **D-6:** ComfyUI caches node outputs — resubmitting an identical graph with only decode/output nodes changed reuses cached sampling (skips the re-render). Used this to recover a failed 720p decode for free.
- **D-7:** Target M_peak ≤ 7 GB (leave ~1 GB headroom), not 8 GB.

---

## RENDER MEASUREMENTS (delivered clips — timing reference)

| Clip | Native | Steps | Output | Render time |
|---|---|---|---|---|
| smoke | 512×320×5 | 20 | 512×320 | ~57 s |
| ocean demo | 768×448×73 | 20 | 768×448 | 348 s |
| 15 s plain | 512×288×362 | 20 | 512×288 | 14m36 |
| dragon (edge) | 576×320×362 | 20 | 1280×720 | 27m59 |
| turbo neon | 512×288×73 | **4 (turbo)** | 1280×720 | **180 s** |
| duet 1080p | 576×320×362 | 6 (turbo) | 1920×1080 | 18m05 |

- **Turbo-LoRA** cuts sampling ~5× (20→4-8 steps). But at 1080p the **4× upscale of 362 frames became the new bottleneck** (Amdahl: killed sampling cost → upscale tail dominates). 720p keeps the tail manageable.

---

## BACKLOG / PARKED (with reasons)

- **Kijai `MiniMax-H3-experimental` W4A8 DiT** (12.5 GB vs our 20.9 GB int8, ~40% smaller, **runs on Ampere** — INT8/W4A8 has no hw floor unlike fp8/nvfp4). Could lower the resident-weight part of the sampling peak. **BUT it's 4-bit weights = APPROXIMATE (ΔQ≠0)** → quality-gate, do LAST. Needs ComfyUI > 0.30 (PRs #15308 + #15334) + `pip install comfy-kitchen`. WIP per Kijai.
- **CPU-offloaded exact K/V attention** — does NOT exist off-the-shelf for a DiT (no reusable KV cache; LLM offload machinery doesn't transfer). Custom build, ~3–10× slowdown for a term SDPA already keeps small. Low priority.
- **VAE decode second-peak (~7.6 GB)** — no easy fix (tiled decode broken, D-3). Needed to push frames/res higher even after sampling is optimized.
- **Turbo-LoRA quality check** — 4-step distilled may be softer than 20-step; eyeball vs baseline before adopting for final renders.

---

## OPEN QUESTIONS

1. Within the 7.8 GB sampling peak, what is the split (DiT resident weights vs attention workspace vs FFN vs allocator)? → needs the pickle memory snapshot (`torch.cuda.memory._record_memory_history` → `_dump_snapshot` → pytorch.org/memory_viz). Phase attribution (E-1) got us to "sampling phase"; the snapshot gets us to the allocation site.
2. Does attention-split (E-4) move the peak where FFN-chunk (E-3) does not?
3. Can the VAE decode second-peak be chunked exactly (given D-3)?
