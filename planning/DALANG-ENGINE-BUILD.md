# Dalang Engine — the full custom build

*Our own inference engine for MiniMax-H3, built to beat the int8-buffer wall that
diffusers' context-parallelism cannot. One CPU "dalang" (puppet-master) holds one
shared model and directs the GPU "wayang" (puppets). Started 2026-09-05.*

---

## Why we build it (the mandate)

Tonight's 3-experiment proof (see KB ch.43) established, with hard data, that:

1. **The wall is the ~2.2 GB int8 `convrot` GEMM buffer + fixed CUDA context (~1.2 GB)** — *not* pixels, *not* frames. Confirmed three ways:
   - ring-2 @ 832×480 × 345f → OOM by ~36 MB
   - ring-2 @ 768×448 × 345f → OOM ~7.5 GB (−14% res freed nothing)
   - ring-2 @ 832×480 × 243f → OOM ~7.28 GB (−30% frames freed ~120 MB)
   - Only a **−63% pixel nuke** (512×288) fit 15 s (Option 1, PASS).
2. **The free dodge is dead in our engine.** Q4 GGUF is absent on the box *and* won't load into `MiniMaxH3Transformer3DModel` (no `from_single_file`; Q4 only ever ran under ComfyUI, a different engine).
3. **diffusers' CP replicates weights per rank** (ring-2 = 66 GB RAM, ring-3 = 99 GB → SIGKILL) and its group-offload **re-copies per process** during the forward (issue #12533) — so we can't share weights through it either.

**Therefore the only doors to full-res 15 s / native 720p are ours to build:**
- **(A)** a single-process ÷N context-parallel engine that *truly* splits the activation **and** the int8 buffer per card, sharing ONE model in RAM, and
- **(B)** eventually, a custom memory-efficient int8/int4 CUDA kernel that shrinks the 2.2 GB buffer directly.

Dalang is (A). It also makes (B) plug-in-able later.

## Architecture (recap)

- **One process** (the dalang / CPU) = one address space → the model loads **once** (~33 GB int8, shared by default; the OS-isolation replication of ch.42 vanishes because there is only one program).
- **The video sequence is split ÷N** across GPU 0/1/2. Each card holds only its 1/N slice of activations — and because each card runs its slice's matmuls, its int8 buffer is 1/N too (the thing diffusers' CP failed to do).
- **Attention across the split = Ring Attention** (arXiv 2310.01889): K/V slices rotate around the ring; after N passes every token has attended to the whole sequence; the online-softmax running max + denominator keeps it mathematically correct; the next K/V transfer overlaps the current compute to hide the no-NVLink PCIe cost.
- **GPUs are pure workers**; the CPU is the only scheduler.

## The build — staged, each stage a PROVABLE milestone

> **Iron rule: correctness before speed.** Every stage verifies its output against a
> diffusers reference *before* we move on. We never scale a stage we haven't proven correct.

### Stage 0 — the correctness harness  *(do first, no engine code yet)*
- On the box, run the existing diffusers denoise on a **tiny** input (few frames, few steps, fixed seed) and **dump**: the input latent/state, and the transformer's output tensor, to `.pt` files.
- This is our **golden reference**. Every Dalang stage must reproduce it bit-for-bit (or within fp tolerance).
- Deliverable: `ref_tiny.pt` (inputs) + `ref_tiny_out.pt` (expected output).

### Stage 1 — single-process, single-GPU baseline  *(load once, run, match)*
- One script: load `MiniMaxH3Transformer3DModel` **once**, run the full forward on **one GPU** for the Stage-0 tiny input, compare to `ref_tiny_out.pt`.
- No parallelism yet — this proves our **loader + forward plumbing** is correct and that we can drive the model outside the diffusers pipeline wrapper.
- **Proof:** max abs diff vs reference < 1e-3 (or matches diffusers' own run-to-run noise).

### Stage 2 — 2-GPU context-parallel, hand-written ring attention  *(THE milestone)*
- Same single process, now split the sequence across **GPU 0 + GPU 1**. Implement the attention as **ring attention with online softmax** (running max `m` + denominator `l`, corrected each K/V pass). Everything else (norms, FFN, rotary) runs per-slice.
- Weights: acceptable to still hold one copy per device *for now* (correctness first); sharing comes in Stage 3.
- **Proof:** output on the reassembled sequence matches `ref_tiny_out.pt`. This is the make-or-break — if our ring output matches diffusers, the engine is real.

### Stage 3 — share ONE model across the GPU streams  *(the RAM win)*
- Load the int8 weights **once** into CPU RAM (mmap / pinned) and stream each block to whichever GPU needs it, **without per-process re-copy** (the thing #12533 broke — but we control the forward now, so no group-offload hook).
- **Proof:** `free -g` shows ~33 GB resident (not 66), output still matches reference. RAM wall gone → ring-3 becomes RAM-feasible.

### Stage 4 — scale to 3 GPUs (ring-3) + confirm the buffer splits  *(the VRAM win)*
- ÷3 the sequence across all three cards. Verify (via `nvidia-smi` peak) that per-card VRAM ≈ fixed context + (buffer + activation)/3 — i.e. the int8 buffer **actually shrinks per card** (the diffusers-CP failure we're fixing).
- **Proof:** a real render that diffusers' ring-2/3 could not fit.

### Stage 5 — the payoff runs
- **Full-res 15 s:** 832×480 × 345f across 3 cards — the clip that OOM'd all night.
- **Native 720p short:** 1280×704 × ~124f — native high-res the model actually computes.
- **Proof:** clips exist, VRAM logged, quality eyeballed. Then upscale as usual.

### Stage 6 (later) — the custom kernel
- Once the engine is correct and scaling, attack the 2.2 GB buffer at the source: a memory-efficient int8/int4 matmul (CUDA C++), or true int8 tensor-core path (the `convrot` int8 currently dequants to bf16, wasting the tensor cores — ch.13/41). This is the deep frontier; the engine makes it a drop-in.

## Risks & how each is tested (no hand-waving)
- **Ring online-softmax correctness** → Stage 2's bit-for-bit check against the golden reference. If it diverges, the running max/denominator math is wrong; debug on the tiny input, not a full render.
- **No-NVLink PCIe comms cost** → measure wall-clock per stage; if the ring transfer dominates, add compute/comms overlap (the paper's key trick) and consider fewer, larger K/V blocks.
- **Single-process multi-GPU + Python GIL** → GPU kernels run async, so the GIL rarely bottlenecks; if it does, per-GPU CUDA streams + minimal Python in the hot loop. Measure before optimizing.
- **int8 buffer not shrinking per card** → Stage 4 VRAM peak proves it; if it doesn't shrink, the matmul isn't seeing the ÷N slice — fix the slicing, not the kernel.

## Ground rules (carried from the whole project)
- **Measure, don't argue.** Every claim = a number from the box.
- **Box access:** Tailscale `root@dalang-z9pe-d8-ws` ONLY. One connection per check. No polling loops. Verify GPU VRAM free before every launch. (See memory: box-connection-tailscale.)
- **Correctness before speed, always.** A fast wrong engine is worthless.

## Status
- **Designed** (KB ch.43) + this roadmap. **Stage 0 is the next action.**
- Tangible wins already banked tonight: a real 15 s single-card clip (soft), and the wall named with data.
