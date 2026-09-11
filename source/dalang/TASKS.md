# Dalang Engine — task tracker (small → large, one at a time)

*No rush. Each task is tiny, concrete, and PROVEN before we tick it. We build the
foundation first and only go bigger once the small thing is measured and works.
Full plan: `../../planning/DALANG-ENGINE-BUILD.md`. Rule: correctness before speed.*

Status key:  `[ ]` todo · `[~]` in progress · `[x]` done (with the number/proof)

---

## Phase A — Foundations (tiny probes, minutes each)

- [x] **A1 · Can ONE CPU process hold the int8 model? → YES, ~31 GB, one copy.**
  Ran `00_probe_load.py`. **System RAM used: 2.6 → 33.5 GB → one-copy footprint = 30.8 GB.**
  33.1 B params, 50 blocks, loaded on CPU in 139 s. So Dalang (one shared copy across N ranks)
  needs ~31 GB regardless of N — vs torchrun's replication (2×≈66, 3×≈99 GB = the ring-3 RAM wall).
  *Note:* process RSS read 90 GB — that's inflated by the **mmap'd fp16 source safetensors** (file-backed,
  reclaimable, not real added RAM); the true footprint is the **30.8 GB system-used delta**. Premise confirmed.

- [x] **A2 · Can the CPU dispatch that model to a GPU? → YES** (covered by A3's forward — the
  transformer ran on the GPU via block-offload and produced [1,14430,96]). Clean per-stage VRAM
  logging folds into Stage 1/4. Dispatch premise confirmed.

- [x] **A3 · Golden reference → CAPTURED.**
  Ran `00_dump_reference.py ns-face --steps 2`. Hook fired on the first transformer forward.
  **Saved `dalang_ref/ref_in.pt` + `ref_out.pt`** on the box; reference output tensor = **[1, 14430, 96] float32.**
  **H3 transformer forward signature (args=none, kwargs):**
  `hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
  attention_kwargs, return_dict, position_ids, token_tags, video_indices, audio_indices, text_indices`
  ← this is the exact call Stage 1 (`01_baseline_single_gpu.py`) must reproduce.
  *Note:* a VAE-decode dtype error ("Half vs float bias") fired AFTER capture — downstream of the
  transformer, harness-only (didn't replicate the full denoise's VAE offload/dtype setup). Reference unaffected.
  *(A2 GPU-dispatch is effectively covered — this forward ran on the GPU via block-offload.)*

## Phase B — Single-GPU baseline (our own forward)

- [x] **B1 · Wrote `01_baseline_single_gpu.py`** — loads int8 once, feeds `ref_in.pt` through the
  transformer ourselves (outside diffusers' pipeline wrapper) via the 12 captured kwargs. Ran first try.
- [x] **B2 · PROVEN — bit-for-bit.** Our forward output [1,14430,96] vs `ref_out.pt`:
  **max diff = 0.000e+00, mean = 0.000e+00.** Not "close" — *identical.* ← **Stage 1 DONE.**
  (Load 118 s, forward 86 s, inputs-as-captured/CPU path worked; noise floor is 0, so any future
  stage that diverges from 0 has a real bug — a strict, clean correctness anchor.)

## Phase C — Ring attention on 2 GPUs (the make-or-break)

- [x] **C1 · H3 attention read & understood** → `NOTES-attention.md`.
  Flow: QKV proj → split heads → RMSNorm(q,k) → rotary(q,k) → `dispatch_attention_fn` (softmax(QKᵀ/√d)V,
  **non-causal, no mask, full**) → out-proj. **Key: only the softmax step needs cross-GPU comms; everything
  else is per-`seq`-slice local.** So the ring = replace `dispatch_attention_fn` with our online-softmax ring.
  diffusers' CP rings via multi-PROCESS torch.distributed (→ weight replication / RAM wall); Dalang is
  single-process, so we write our own device→device ring. Split axis = `seq` (dim 1).
- [x] **C2 · Online softmax PROVEN exact** → `02_online_softmax.py`.
  Blockwise running-max + denominator attention vs PyTorch SDPA on random tensors:
  **fp32 max 8.9e-7** (exact; ×3 block sizes incl. non-divisible S), **bf16 max 7.8e-3 = 1 ULP**
  (bf16's physical floor — as close as the format allows). Learned + applied: **accumulate m/l/O
  in fp32 internally** (like FlashAttention), cast to input dtype at the end. The core ring math is correct.
- [x] **C3 · 2-GPU device→device ring PROVEN** → `03_ring_2gpu.py`.
  Seq split cuda:0/cuda:1; K/V rotated device→device (PCIe); online-softmax accumulate over 2 passes;
  reassembled vs full SDPA: **fp32 max 2.4e-6, bf16 max 4.9e-3 (1 ULP). PASS.** Context parallelism in
  ONE process (no torchrun, no weight replication) — two cards computing one attention. The Dalang mechanism works.
- [x] **C4 · Ring wired into the real H3 transformer** → `04_ring_in_model.py` (+ diagnostics `04b`, `04c`).
  Journey: OOM (naive ring materialized full 7215² score matrix → *rediscovered why FlashAttention exists*) →
  fixed with K-tiling → end-to-end diff 0.34 (CLOSE, not bit-for-bit) → **C4b** isolated a per-block bug (max 0.625)
  → **C4c** pinned it: bf16 `QKᵀ` matmul (fp32 fixed it 18×; orig == plain SDPA exactly; residual 0.25 = 1 bf16 ULP
  at magnitude). Can't be bit-for-bit vs flash — it's a *different* valid bf16 attention, so 1-ULP/block compounds
  over 50 layers to ~0.017 mean. **Correct, not identical.**
- [x] **C5 · RENDER validation (the real Stage-2 sign-off)** → `05_render_ring.py`.
  Rendered `ns-face` WITH the ring vs normal (same seed/steps). Needed proper **Q+K tiling** (OOM'd in the full-denoise
  context first — GPU0 more loaded; scores now (1024,512) tiles → **peak 5.7 GB, fits**). **GPU 1 held 1.4 GB = the ring's
  far slice, live.** Result: frames **perceptually identical**, **PSNR 37.6 dB**, differences confined to background bokeh.
  ✅ **STAGE 2 DONE — our context-parallel engine renders correct video across 2 GPUs, in one process, our code.**

## Phase D — Share ONE model (the RAM win) — **already free by being single-process**

- [x] **D1/D2 · One model copy = achieved by construction.** Our engine is SINGLE-PROCESS, so the model
  loads ONCE (31 GB, per A1) — no per-process replication. The torchrun ring-3 RAM wall (3×31≈93 GB) simply
  **does not exist for us.** Phase D was won by the architecture, not extra work.

## Phase D-test — does the ring beat the frame wall? → **NO (important finding)**

- [x] **Rendered cp10 (243f @ 832×480) through the ring → OOM at 7.30 GB** (single-card OOM'd at ~7.28 GB —
  so the ring gave **~0 frame headroom**). ROOT CAUSE: our ring splits only the **attention**; GPU 0 still runs
  **QKV projection + FFN on the FULL sequence**, so its activations + int8 buffers still scale with frames.
  **Stage 2 proved correctness, NOT the memory win.** The attention-only ring is a stepping stone, not the summit.

## Phase E — the REAL memory win (revised)

**Insight (the "different way"):** the block is **per-token EXCEPT attention** — QKV/FFN/norms/rotary are all
independent per token (chunkable); only attention is cross-token, and **we already tile that** (C4/C5). So:

- [x] **E1 · Per-linear chunking → FAILED, but corrected a wrong premise** → `06_chunk_linear.py`.
  Wrapped 367 big linears to chunk the token dim. Result: **peak 5.09 GB UNCHANGED** (baseline 5.09), diff 0.018.
  **Two corrections the box taught us:** (1) our torchao engine is `Int8WeightOnlyConfig` (weight-only, dequant→bf16,
  **NO 2.2 GB activation scratch buffer** — that was ComfyUI's convrot int8, ch.13; I'd carried the old model in).
  (2) Per-linear chunking leaves the **FFN 14336-dim intermediate** materialized full-sequence between the two
  chunked linears. So the real lever is the **activation**, chiefly the FFN intermediate — chunk the FFN *as a unit*.
- [x] **E1b · Chunk the FFN as a unit → WORKS** → `06b_chunk_ffn.py`.
  Wrapped 52 `.ff` modules to run up→act→down over token-chunks (intermediate is chunk-sized). @124f:
  **NOCHUNK peak 5.09 GB (diff 0) → FFN-CHUNKED peak 4.34 GB (−0.75 GB), diff mean 0.015** (bf16 matmul-size
  rounding, imperceptible — same class as the ring's 0.017). **Proves the activation wall is chunkable and it SCALES
  with frames** (the FFN intermediate is ~2.8× bigger at 345f → ~2 GB saved where we OOM). The correct lever.
- [x] **E1c · 345f single-card, FFN-chunked → OOM, but MOVED the wall** → `07_render_chunked.py`.
  Did NOT die in the FFN this time — the FFN chunk held and carried us into the **attention front-end**, where it
  OOM'd inside `_apply_rotary_emb` (7.60 GB used / 7.66 GB cap, over by ~60 MB, alloc 422 MB). **New peak located:**
  full-S query + rotary's ~3 full transient copies (`hidden_states_rotated` cat, the cos/sin result, the final
  contiguous cat). FFN wall cleared ✅; attention rotary transient is now the ~60 MB straw.
- [x] **E1d · FFN + rotary chunked → past rotary, OOM at int8 linear dequant** → `08_render_tiled.py`.
  Rotary chunk freed ~300 MB (free 18→318 MB). Next 422 MB straw: torchao weight-only int8 makes **two full-S
  bf16 transients per linear** (`m = X@Wᵀ`, `y = m*scale`, ~430 MB each @345f for the attention projections).
- [x] **E1e · all three chunkers → OOM on my own torch.cat** → `09_render_tiled.py`.
  Chunks computed fine; died reassembling them: `torch.cat` holds the chunk list (full-S) AND allocates a 2nd
  full-S contiguous to cat into = ~2× the output (the 422 MB straw). The chunk math is right; the *reassembly* was wasteful.
- [x] **E1f · preallocate outputs → transient 422 MB → 84 MB, hit the RESIDENT FLOOR** → `10_render_tiled.py`.
  Preallocation killed the cat doubling; the tipping alloc shrank 422→**84 MB**. But now **7.52 GB is resident**
  (block weights + full-S residual stream + CUDA context) with only ~0.14 GB headroom, and it OOM'd on an 84 MB alloc
  with *98 MB free* = **fragmentation at the edge**. Chunking has done its job — transients are tiny; what's left is
  the irreducible full-S resident floor. **0.14 GB short.**
- [x] **E1g · 🎉 THE DENOISE FIT ON ONE CARD AT 345f.** `10_render_tiled.py --achunk 2048 --fchunk 1024`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:128`.
  **seq = 41,153 tokens (15s @ 832×480), all 50 blocks × 3 steps completed in 9m31s, alloc steady 4.36 GB, peak 7.71 GB,
  NO OOM.** The full-15s DiT denoise — flatly impossible before — ran on a single 8 GB RTX 3060 Ti. Every wall we
  knocked down (FFN→rotary→int8 dequant→cat-doubling→fragmentation) stacked; smaller chunks + anti-frag allocator
  closed the last 0.14 GB. **The frame wall is broken at the transformer.**
  *Next micro-wall:* VAE decode of 345 frames is a separate memory regime — E1h below.

- [x] **E1h · 🎬 FULL END-TO-END 15s ON ONE CARD → `e1h-15s.mp4` (832×480, 345f, 14.375s, +audio), rc=0.**
  `11_render.py`. Found the first blind decode wasn't OOM'ing — it was **crawling under `leaf_level` VAE offload**
  (streams every layer CPU↔GPU across thousands of tile/clip forwards). The VAE already chunks temporally
  (`_decode_clip`) + tiles spatially (256 px), so it's small. Fix (NOT "chunk the VAE" — already chunked):
  **defer moving the VAE onto cuda until the FIRST decode clip** (by then the transformer's weights are offloaded, so
  there's room; moving at setup OOM'd) **+ log each clip**. Result: 20 clips decoded visibly (gpu ~6.4 GB), then
  cat+export. **Total pipeline `gerado em 3432s` (~57 min).** Two setup bugs fixed en route: (i) the offload-patch
  signature (`module=` kw, not positional) — broke the transformer's own offload; (ii) SSH sessions dropping on
  in-command `sleep` + a launcher left ATTACHED (streaming) — memory rule written (never stream; launch detached & return).

**🏁 PHASE E COMPLETE — the frame wall is broken end-to-end.** A full 15s video+audio clip renders on a single 8 GB
  RTX 3060 Ti. It is SLOW (~57 min, block-offload is sequential) — fitting ≠ fast. **We do NOT chase speed (user's
  principle, 2026-09-06: "we not charge the speed… its not our style").** Capability, not throughput.

- [x] **E3 · 🎬 NATIVE 720p END-TO-END ON ONE CARD → `e720-5s.mp4` (1280×704, 124f, 5.17s, +audio), rc=0.**
  Scene `cp720` (re-encoded), rendered with the same `11_render.py` tiling engine. **Constraint learned:** H3 needs
  height & width **multiples of 32** (720 is invalid → use 704; error `must be multiples of 32, got 720x1280`).
  **seq = 33,193 tokens, alloc steady 4.10 GB, peak 6.81 GB, NO OOM** — LOWER than the 480p×345f run (7.71 GB),
  because 720p×124f is fewer total tokens (tokens ∝ H×W×frames). ~41 min (`gerado em 2476s`). **The recipe carried
  the resolution jump with HEADROOM → a LONGER 720p clip is possible.** HD capability on a $200 8 GB card: confirmed.

**THE RECIPE (single-card native 15s denoise):** int8 weight-only + block-offload (1 block/group)
  + FFN chunked as a unit (1024) + rotary chunked (4096) + attention-linear chunked (2048), **all preallocating
  outputs (never torch.cat)** + anti-frag allocator. Correct by construction (every chunked op is per-token; only
  attention is cross-token and stays full/flash). This is Dalang's single-GPU DiT-tiling engine.

  **PATTERN (the honest read):** each fix frees ~0.3–0.75 GB and we hit the *next* identically-sized ~422 MB
  full-S transient. We are **structurally ~0.5 GB short on ONE card for 345f** — single-card chunking is
  whack-a-mole against a fixed floor. If E1e fits, great; if it hits yet another transient, that is the *proof*
  the real win is **sharding the sequence across GPUs through the whole block** (true DiT sequence parallelism —
  every transient becomes S/N), reusing the ring (C5) only for the attention gather. That is the summit of Phase E.
- [ ] **E2 · Render full-res 15 s** (832×480 × 345f) once tiling fits it — single card, sequential (trades speed for fitting).
- [ ] **E3 · Native 720p short** (1280×704).
- [ ] **E4 (later) · Layer the multi-GPU ring on top for speed** once the single-GPU tiling fits.

## Phase G — 3-GPU sequence parallelism (the summit: native 720p × 15s straight)

**Goal (user-chosen 2026-09-06):** a **15-second straight** continuous take at **native 720p** (1280×704 × 345f ≈
**93k tokens** — 2.3× the single-card ceiling of ~41k). Single-card tiling can't fit it; the answer is to **shard the
sequence S across all 3 GPUs** (≈31k tokens/card, which we KNOW fits — 720p×124f was 33k @ 6.81 GB). Not for speed
(we don't chase speed) — for **capability**: fitting a job no single card can hold. Correctness-first, 1-by-1, like A–E.

**What we already have:** the ring (C5, `05_render_ring.py`) = the cross-GPU **attention gather**, proven. The NEW work
is sharding the **per-token** ops (norm/QKV/rotary/FFN) so each card only ever holds its S/N slice — then the block's
attention gathers across cards via the ring. Single process, ONE model copy in RAM (per A1), block weights replicated
to the N active cards per block.

- [ ] **G0 · Design + notes** — read how `11_render.py`/block-offload places the current block; decide the shard
  architecture (replicate current block's int8 weights to cuda:0/1/2, split hidden S/3, per-token ops local, ring
  attention across 3). Write `NOTES-seqparallel.md`. Confirm per-card budget (~31k tokens ≈ 6.8 GB + block wts).
- [x] **G0/G1a · recon + capture** — block is per-token EXCEPT `self.attn` (the clean SP case). Captured real block-0
  I/O → `dalang_ref/blk0_io.pt` (`13`/`12_capture_block.py`): hidden (1,15030,5376), temb (1,2688) SHARED,
  adaln_indices (15030,) sliced by S, rotary (cos,sin 15030×96) sliced by S, heads=56 head_dim=96.
- [x] **G1 · Sharded block across 2 GPUs = correct** → `13_shard_block.py`. Baseline (single-card) vs captured = 0.000
  exact. **Seq-parallel (S/2, per-token local + C5 ring for attn) vs captured: mean 4.0e-03, REL mean 0.51% / p99 2.86%
  / max 5.63%** — pure bf16 compounding (worst abs 256 sits on a |35k|-magnitude token = 0.74% rel; 0 NaN). Same class
  as C5 (validated by render). **Mechanism proven.**
- [x] **G2 · Sharded block across 3 GPUs = correct** → `14_shard_block_ngpu.py --n 3`. Result **identical** to G1
  (mean 4.0e-03, REL mean 0.51%) — proving the online-softmax ring is associative: split count doesn't change the
  answer. N-GPU sequence-parallel block is correct.
- [x] **G3 · Full 50-block 3-GPU sharded forward = CORRECT** → `15_shard_forward.py`. Build packed hidden on cuda:0 →
  shard S/3 → stream each block to all 3 cards (deepcopy the cuda:0 original, G1/G2 pattern) + run sharded, keeping
  hidden sharded across all 50 blocks → gather → proj_out. **vs ref_out: mean 1.78e-02, REL mean 0.95% / p99 1.9%** —
  *identical* to C5's 0.017 (the bf16 level already render-validated at PSNR 37.6). **Memory held FLAT across all 50
  blocks** (g0 2.29 / g1 0.07 / g2 0.95 GB @ 15k tokens). Two bugs fixed: (i) deepcopy CPU-int8 block → illegal access;
  fix = move original to cuda:0, deepcopy THAT (G1/G2 pattern); (ii) multi-GPU async race + uneven replica free →
  OOM/illegal; fix = `torch.cuda.synchronize(i)` barriers after cross-device moves + explicit replica `.to("cpu")`.
  **The 3-GPU sequence-parallel engine is proven.**
- [~] **G4 · 🎯 Render native 720p × 345f (15s straight) across 3 GPUs — RENDERING (all 3 cards lit).**
  `16_render_3gpu.py cp720long` (1280×704×345f, scene `cp720long`, ~93k tokens). Patches the transformer forward with
  the G3 sharded forward + neuters the transformer's group-offload + keeps the E1h deferred-VAE. **The key realization:
  each card's S/3 shard STILL needs the Phase-E tiling** — sequence-parallel (across cards) + tiling (within each card)
  must COMBINE. Integration fixes, in order: (1) free the full-S `hid`/`ve`/`ae`/`te` off cuda:0 after sharding;
  (2) shrink attention score tiles 2048→512 (940 MB → 59 MB); (3) chunk the FFN + attn-linears per card — **the chunk
  name-match had to change from full-model paths (`transformer_blocks.N.ff`) to block-relative (`ff`, `attn.*`)**;
  used a rotating ring (hold one remote K/V shard at a time). **Result: block 0/50 cleared, GPUs g0 7.1 / g1 5.0 / g2 6.3 GB,
  no OOM — the combined engine renders 93k tokens no single card can hold.** Slow (streaming + chunk + 3-GPU, by choice).
  *In flight:* grind 50 blocks × 4 steps → deferred VAE → `g4-720p15s.mp4`. Validate by watching. Then quality re-render (more steps).

## Phase F — Custom kernel (later, the deep frontier)

- [ ] **F1 · Profile the 2.2 GB int8 buffer**; design a memory-efficient int8/int4 matmul (CUDA C++).

---

## Log (newest first)
- 2026-09-06 — **🎉🎉 FRAME WALL BROKEN (E1g).** Full 15s (345f, seq 41,153) DiT denoise ran on ONE 8 GB card —
  50 blocks × 3 steps, 9m31s, peak 7.71 GB, no OOM. Path: E1 (per-linear, no effect → corrected the int8-buffer
  myth) → E1b (FFN-unit chunk, −0.75 GB, works) → E1c (345f OOM moved to rotary) → E1d (rotary chunk, OOM moved to
  int8 dequant) → E1e (attn-linear chunk, OOM on my own torch.cat) → E1f (preallocate, 422→84 MB, hit resident floor,
  0.14 GB short) → E1g (smaller chunks + anti-frag allocator = FIT). Six walls, each measured & logged. The RECIPE
  is documented above. Next: VAE decode (separate regime) → then the multi-GPU ring on top for SPEED (E4).
- 2026-09-05 — **🎉 STAGE 2 DONE (C4+C5).** Ring wired into real H3; per-block bug (bf16 QKᵀ) found via C4b/C4c and fixed (fp32 scores); render validation `ns-face` ring-vs-normal = **perceptually identical, PSNR 37.6 dB**, GPU1 live-held the far slice. **We built a working single-process 2-GPU context-parallel engine, proven to render correct video.** Next: **Phase D** (share ONE model in RAM) → **Phase E** (ring-3 + full-res 15s / native 720p).
- 2026-09-05 — **C3 DONE:** 2-GPU device→device ring == full SDPA (fp32 2.4e-6, bf16 1 ULP). Context parallelism, single process, ours.
- 2026-09-05 — **C2 DONE:** online-softmax attention proven exact (fp32 8.9e-7; bf16 1 ULP). fp32 internal accumulation. Ring math is correct.
- 2026-09-05 — **C1 DONE:** H3 attention read (`NOTES-attention.md`) — only the softmax needs cross-GPU comms; full/non-causal → clean ring case.
- 2026-09-05 — **STAGE 1 (PHASE B) DONE.** Our own forward reproduces the golden reference **bit-for-bit (max diff 0.000e+00).** We now own the forward path.
- 2026-09-05 — **PHASE A DONE.** A3: golden reference `ref_out.pt` = [1,14430,96] f32; forward signature captured (12 kwargs). A2: GPU dispatch confirmed (covered by A3).
- 2026-09-05 — **A1 DONE:** one CPU process holds int8 H3 in **30.8 GB** (one copy), 33.1 B params / 50 blocks, 139 s load. Premise confirmed.
- 2026-09-05 — folder created; roadmap + `00_probe_load.py` + `00_dump_reference.py` written.
