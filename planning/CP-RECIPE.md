# Context / Sequence Parallelism for MiniMax-H3 denoise across 3× RTX 3060 Ti

**Status:** research + code-level recipe. Verified against current diffusers source (the `minimax-h3`
PR branch, head `f37ab93`, and `main`), September 2026. Nothing here has been run on the box yet.
Every API claim is cited to source below.

---

## 0. The goal (restated, and why)

We are **NOT chasing speed.** We want to **distribute the activation memory** of the video denoise
across the 3 cards so each GPU holds ~1/3 of the packed video/audio/text token sequence. The weights
already stream from RAM (int8 = 33 GB, fp16 = 66 GB, both ≫ 24 GB total VRAM, so they can never go
resident). The target is the **~5.7 GB activation**, split 3 ways → **~1.9 GB/GPU**, freeing headroom
to run **full-precision fp16 at longer length and/or higher resolution (720p)**.

Context parallelism (CP) is exactly the right tool for this: the diffusers docs classify it as the
strategy that "splits the input sequence" and "reduces **activation memory**" — as opposed to tensor
parallelism (splits weights) or `device_map` (splits components). See the strategy table in the
distributed-inference guide.
Source: <https://huggingface.co/docs/diffusers/main/en/training/distributed_inference> ("Choosing a strategy").

**Bottom line up front:** the *GPU-side* memory math works and fits 8 GB with room to grow. But two
hard blockers stand between us and the fp16 dream, and one of them is fatal on current RAM:

1. **H3 ships no `_cp_plan`.** The transformer does not declare how to shard its inputs, so
   `enable_parallelism()` raises unless we **author a `cp_plan` by hand** and pass it. H3's
   packed-sequence + `index_copy`/`index_select` design makes that plan non-trivial (§1, §4).
2. **CP replicates the full weights per rank.** CP shards *activations*, not weights. With
   `torchrun --nproc_per_node=3`, each of the 3 processes holds its **own** full copy of the model in
   CPU RAM to stream from. int8 = 33 GB × 3 = **99 GB** (fits 125 GB, tight). fp16 = 66 GB × 3 =
   **198 GB ≫ 125 GB RAM — does not fit.** fp16-at-length via CP is blocked by **RAM**, not VRAM,
   unless the weights are memory-mapped and shared across processes via the OS page cache (§3, §7).

So: CP realistically lets us **grow resolution / frame count at int8** by freeing activation headroom.
Getting to **fp16** additionally requires solving the 3×-weight-replication RAM problem (mmap-shared
weights) — that is the single biggest risk and it is unverified.

---

## 1. Does diffusers have context parallelism today, and how is it invoked?

**Yes.** It is a first-class, if "experimental", feature on `main` and present on the `minimax-h3`
branch. The public surface:

- `diffusers.ContextParallelConfig(ring_degree=..., ulysses_degree=..., convert_to_fp32=True,
  rotate_method="allgather", mesh=None, ulysses_anything=False, ring_anything=False, ...)`
  Source: <https://huggingface.co/docs/diffusers/main/en/api/parallel> (`ContextParallelConfig`).
- `model.enable_parallelism(*, config, cp_plan=None)` — a `ModelMixin` method (call it on the
  **transformer**, not the pipeline).
  Source (branch, sha f37ab93): `src/diffusers/models/modeling_utils.py` L1601–1686.

Canonical invocation (from the docs, Ring flavor):

```python
import torch
from torch import distributed as dist
from diffusers import DiffusionPipeline, ContextParallelConfig

dist.init_process_group(backend="nccl")
rank = dist.get_rank(); world = dist.get_world_size()
device = torch.device(f"cuda:{rank}"); torch.cuda.set_device(device)

pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", dtype=torch.bfloat16).to(device)
pipeline.transformer.set_attention_backend("_native_cudnn")        # CP-compatible backend
pipeline.transformer.enable_parallelism(config=ContextParallelConfig(ring_degree=world))
```

Launched with `torchrun --nproc-per-node <N> script.py`.
Source: <https://huggingface.co/docs/diffusers/main/en/training/distributed_inference> ("Context parallelism").

**The mechanism** (this is what matters for H3): `enable_parallelism()` does **not** magically shard.
It reads a **`cp_plan`** — a dict mapping module names → which forward inputs to split and which
outputs to gather — and installs pre-/post-forward hooks that call an `EquipartitionSharder` on those
tensors. The attention itself is made distributed because H3's attention processor passes
`parallel_config=self._parallel_config` into `dispatch_attention_fn`, which runs the ring/ulysses
collective on K/V.
Sources (sha f37ab93): `src/diffusers/hooks/context_parallel.py` L80–252 (`apply_context_parallel`,
`ContextParallelSplitHook`, `ContextParallelGatherHook`, `EquipartitionSharder`);
`src/diffusers/models/_modeling_parallel.py` L41+ (`ContextParallelInput`/`ContextParallelOutput`).

The plan is expressed with two dataclasses:

```python
@dataclass(frozen=True)
class ContextParallelInput:
    split_dim: int
    expected_dims: int | None = None
    split_output: bool = False

@dataclass(frozen=True)
class ContextParallelOutput:
    gather_dim: int
    expected_dims: int | None = None
```

Example the framework ships for QwenImage (input hidden_states IS the sequence — the easy case):

```python
_cp_plan = {
    "": {  # "" = the model root; split these named forward args
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
    },
    "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),  # gather back at the end
}
```

Officially covered models are the plain DiTs that ship a `_cp_plan`: **QwenImage, Flux, Flux.2,
Wan, LTX, CogView4, HunyuanVideo, Cosmos** and similar (each declares `_cp_plan`/`_tp_plan` in its
transformer file). **A plain DiT is exactly the supported shape — the problem is not the architecture,
it is that H3 simply hasn't declared its plan yet.**

---

## 2. Does it work with a MODULAR pipeline (`SequentialPipelineBlocks` / `ModularPipeline`)?

**Effectively yes — because CP is a property of the transformer *model*, not the pipeline.**
`enable_parallelism()` is a `ModelMixin` method. Our modular setup already loads a concrete
`MiniMaxH3Transformer3DModel` and hands it to the denoise blocks; we call `enable_parallelism()`
on that model object directly. The modular `ModularPipeline` / `SequentialPipelineBlocks` never needs
to know CP exists — the split/gather hooks and the distributed attention live entirely inside the
transformer's forward.

Caveats that are on **us**, not on diffusers:
- Every rank must run the **same** denoise loop over the **same** latents. Seed/generator must be
  synced across ranks (the docs stress this: "Must specify generator so all ranks start with same
  latents"). Build the initial noise on rank 0 and broadcast, or use a fixed `manual_seed` identically
  on all ranks.
- The modular block code must not branch on rank or do rank-divergent RNG. Our `["denoise","decode"]`
  partial pipeline is deterministic given latents, so this is fine — but **decode should run on rank 0
  only** (the VAE is small, keep it off the CP path; gather the final latent to rank 0 first).

There is **no diffusers guarantee or test** for modular + CP specifically (the CP tests target the
monolithic Flux/Qwen pipelines). Treat "modular works" as *very likely correct but unverified* — the
first feasibility test (§6) exists precisely to confirm it.

---

## 3. Does CP compose with `enable_group_offload` block-streaming?

**Two separate questions — keep them apart.**

**(a) Do the hooks conflict?** CP installs diffusers `HookRegistry` hooks (split on entry modules,
gather on exit) via `apply_context_parallel`. `enable_group_offload(offload_type="block_level")`
installs its own `HookRegistry` hooks that move each block CPU→GPU before its forward and back after.
They operate at *different* module granularities and are, in principle, composable: the block is on the
GPU when its forward runs, and CP's split/gather act on tensors already on that GPU.

**BUT** there is a known, open, unresolved hook-ordering bug between CP and offload:
Issue **#12533 "Hooks conflicts: Context Parallelism and CPU Offload"** — enabling offload *before* CP
produces `RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048)` on the
**second** pipeline call, inside rotary embedding. Root cause is hook-application order; assigned to
@yiyixuxu, **no fix merged**. The reporter's workaround: **enable CP first, then offload.**
Source: <https://github.com/huggingface/diffusers/issues/12533>.
(That issue is about `enable_model_cpu_offload` (accelerate hooks); `enable_group_offload` is the
native-diffusers hook path, so the exact failure may differ — but the ordering lesson stands and this
is the single most likely thing to break for us. See §7.)

**Order to use:** `set_attention_backend()` → `enable_parallelism(...)` → `enable_group_offload(...)`.

**(b) Does CP assume weights are resident per rank?** CP shards **activations only**; it never touches
weights (that is tensor parallelism's job). So CP does **not** require resident weights — it is happy
for group-offload to stream each block in and out. The conflict is **not** conceptual, it is the
per-rank **replication**: each `torchrun` process is a full, independent Python process with its **own**
model instance and therefore its **own** full weight set to stream from CPU RAM. That is the RAM
blocker in §0 and §7, and it is the real reason fp16 is hard — not a CP/offload incompatibility.

---

## 4. The `cp_plan` H3 needs (we must write it — this is the crux)

**Confirmed: `MiniMaxH3Transformer3DModel` defines NO `_cp_plan`.** Grep of the transformer file at
sha f37ab93 (`src/diffusers/models/transformers/transformer_minimax_h3.py`, 631 lines) finds no
`_cp_plan`, no `ContextParallelInput`, no `_tp_plan`. The base `ModelMixin` default is `_cp_plan = None`
(`modeling_utils.py` L256), and `enable_parallelism()` raises if both the arg and the attribute are
None:

```python
# modeling_utils.py L1681-1685 (sha f37ab93)
if cp_plan is None and self._cp_plan is None:
    raise ValueError("`cp_plan` must be provided either as an argument or set in the model's `_cp_plan` attribute.")
cp_plan = cp_plan if cp_plan is not None else self._cp_plan
apply_context_parallel(self, config.context_parallel_config, cp_plan)
```

So the API path is: **pass our own `cp_plan=` to `enable_parallelism()`.** Good news — that arg exists
and is public.

**Why H3's plan is harder than QwenImage's.** In QwenImage the input `hidden_states` *is* the sequence
that flows through every block, so you split at the root `""` and gather at `proj_out`. In H3 the
forward receives **three separate modality tensors** (`hidden_states` = video rows,
`audio_hidden_states`, `encoder_hidden_states` = text) plus a fistful of **full-length index tensors**
(`position_ids (seq,3)`, `token_tags (seq,)`, `timestep_indices (seq,)`, `video_indices`,
`audio_indices`, `text_indices`). The actual **packed sequence** of length `seq_len` is built *inside*
forward via `index_copy`, and the output is pulled back out via `index_select` on **absolute global
positions**. Those `*_indices` tensors would be **wrong if sharded** (they name positions in the full
sequence). So we **cannot** split at the root `""`.

The correct split point is **after packing, at the transformer-block stack**, where the tensor is a
clean `(B, seq_len, hidden)` and attention is full self-attention over `seq_len`. The block forward is
`block(hidden_states, temb, adaln_indices, rotary_emb)`:
- `hidden_states` `(B, seq, 5376)` → split `dim=1`
- `temb` `(num_timesteps, ...)` → **not** split (shared)
- `adaln_indices` `(seq,)` → split `dim=0` (per-row AdaLN table index — shards correctly with the rows)
- `rotary_emb` = `(cos, sin)`, each `(seq, head_dim)` → split each `dim=0`

Because the loop threads `hidden_states = block(...)`, we split `hidden_states` **once** at block 0 and
it stays sharded down the residual stream (attention's ring collective handles cross-token K/V). But
`adaln_indices` and `rotary_emb` are re-supplied full to **every** block, so they must be split at every
block. Then gather `hidden_states` back to full length at the **last** block's output, *before*
`norm_out`/`proj_out` (which use full-length `timestep_indices`/`video_indices`).

**Candidate hand-authored plan (UNVERIFIED — must be numerically validated, see §6/§7):**

```python
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput

NUM_LAYERS = 50  # transformer.config.num_layers
h3_cp_plan = {
    # split the residual stream once, on entry to block 0
    "transformer_blocks.0": {
        "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
    },
    # every block gets the full-length positional/AdaLN tensors, so shard them per block
    "transformer_blocks.*": {
        "adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1),
        "rotary_emb": [
            ContextParallelInput(split_dim=0, expected_dims=2),   # cos
            ContextParallelInput(split_dim=0, expected_dims=2),   # sin
        ],
    },
    # gather the sequence back to full length at the last block's output, before norm_out/proj_out
    f"transformer_blocks.{NUM_LAYERS - 1}": ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

Notes / risks baked into this plan:
- The wildcard `transformer_blocks.*` requires the prefix to resolve to an `nn.ModuleList` (it does) and
  allows exactly one `*` (`context_parallel.py` L357–369). It also fires on block 0 — harmless, it only
  shards `adaln_indices`/`rotary_emb` there, while the separate `transformer_blocks.0` entry shards
  `hidden_states`; different arg names, different hook names, both register cleanly
  (`apply_context_parallel` keys hooks by module id).
- The split arg **names must match** the block's forward parameter names exactly
  (`hidden_states, temb, adaln_indices, rotary_emb`) — the hook looks inputs up by name/position
  (`ModuleForwardMetadata._get_parameter_from_args_kwargs`).
- `rotary_emb` is a **tuple**; the hook supports a per-element list of `ContextParallelInput` (see
  `pre_forward` handling of `list/tuple` inputs, `context_parallel.py` L166–174).
- **This plan is my best reading of the source, not a tested artifact.** A wrong split silently produces
  a plausible-but-wrong video. Validate numerically (§6).

---

## 5. Ulysses vs Ring — and which one our hardware forces

**Diffusers implements both, plus a unified 2D combo and "anything" variants:**
- **Ulysses** — all-to-all, splits **heads** across `ulysses_degree`. Requires `ulysses_degree` to
  divide the head count. Low latency, **high bandwidth** (all-to-all every layer). Wants NVLink.
- **Ring** — splits the **sequence**; passes K/V around the ring. Lower per-step bandwidth, tolerant of
  slow links. Best for long sequences with limited bandwidth.
- `ulysses_anything` / `ring_anything` — relax the divisibility constraints on head count / sequence
  length.
Source: <https://huggingface.co/docs/diffusers/main/en/training/distributed_inference> (Ulysses/Ring/Unified sections).

**For our 3× 3060 Ti (56 heads, no NVLink, ~4.9 GB/s PCIe): use RING, degree 3.**

- **Ulysses is disqualified by head count.** H3 has **56 attention heads**. `ulysses_degree` must
  divide the head count; **3 does not divide 56** (56 = 8·7). Plain Ulysses-3 is invalid. `ulysses_anything`
  could bypass that, but Ulysses' all-to-all is the *worst* pattern for our slow PCIe. Skip it.
- **Ring tolerates slow links, and this is verified on hardware like ours.** The diffusers docs publish
  a Ring benchmark on **a node of 4× RTX 4090 (48 GB), consumer cards without NVLink**, for FLUX.1-dev —
  ring ran successfully at ~2.9 steps/s. That is the "CP on 4×4090 over PCIe" data point, confirmed in
  the official docs.
  Source: <https://huggingface.co/docs/diffusers/main/en/training/distributed_inference> ("Ring Anything Attention" benchmark table, "4 RTX 4090 (48GB)").
- **Sequence divisibility:** `EquipartitionSharder` splits `seq_len` evenly across ring_degree. If our
  packed `seq_len` is not divisible by 3, either pad to a multiple of 3 **or** set
  `ring_anything=True` (pads/gathers arbitrary lengths). `ring_anything` is inference-only and requires
  `attn_mask=None` — which is exactly H3's case (packed single document, no mask). So `ring_anything=True`
  is the safe default for us.

Config to use:

```python
ContextParallelConfig(ring_degree=3, ring_anything=True)
# and add the gloo backend to init_process_group to avoid CUDA syncs from the anything-padding:
#   dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```

Attention backend: `set_attention_backend("_native_cudnn")` (CP-supported, works on Ampere sm_86) or
`"_native_flash"`. Confirmed CP-supporting backends include `_native_cudnn`, `_native_flash`, the flash
variants, and sage — registered with `supports_context_parallel=True` in
`src/diffusers/models/attention_dispatch.py`. `enable_parallelism()` will **hard-error** if the active
backend does not support CP, so set it first.

---

## 6. The honest per-GPU memory math

Per-GPU budget with **block-level streaming** (only one block resident at a time) **+ ring CP degree 3**:

| Item | int8 | fp16 | Notes |
|---|---|---|---|
| 1 resident transformer block | ~0.66 GB | ~1.29 GB | block streamed in for its forward |
| Activation (was ~5.7 GB) ÷ 3 | ~1.9 GB | ~1.9 GB | **the whole point** — sequence split 3 ways |
| Ring K/V comm buffers + context | ~1.2 GB | ~1.2 GB | ring holds ~1/N K/V at a time + working set |
| **Per-GPU total** | **~3.8 GB** | **~4.4 GB** | **fits 8 GB** with ~3.6–4.2 GB headroom |

So **on the GPU side the math works**: splitting the sequence 3 ways genuinely cuts per-GPU activation
to ~1/3, and even fp16 leaves ~4 GB of headroom on each 8 GB card to grow frames or push to 720p. This
is the result we wanted. (Activation scales ~linearly with token count, so at 720p / more frames the
5.7 GB grows, but it also stays ÷3 — the headroom is what lets us push it.)

**The catch is not VRAM, it is host RAM (see §3/§7):** with 3 independent `torchrun` processes each
holding a full CPU copy of the weights to stream from:
- **int8: 33 GB × 3 = 99 GB** → fits in 125 GB RAM (tight; watch pinned-memory + OS overhead).
- **fp16: 66 GB × 3 = 198 GB** → **does NOT fit 125 GB RAM.**

fp16 is therefore **not reachable by naive CP** on this box. It becomes reachable only if the 66 GB of
weights are **memory-mapped once and shared across the 3 processes via the OS page cache**
(`safetensors` mmap / `low_cpu_mem_usage=True`, read-only), so RAM holds ~one copy (~66 GB) instead of
three. That is the key unlock for the fp16 goal and it is **unverified** with group-offload (which may
copy block weights into per-process pinned buffers on the way to the GPU, defeating the sharing). This
is the experiment that decides whether "fp16-at-length" is real for us.

---

## 7. Minimal FEASIBILITY TEST (run this FIRST, before any full render)

Goal: answer "does CP even initialize on H3 across 3 GPUs without erroring, with our hand-written
plan?" — cheaply, before chasing a full video. Run in stages, stop at the first failure.

**Stage 0 — NCCL sanity on our no-NVLink topology (30 s).** Confirm the collective works over PCIe/SYS
before involving H3 at all:

```bash
# nccl_smoke.py: each rank all-reduces a tensor
cat > nccl_smoke.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(1, device=f"cuda:{r}") * r
dist.all_reduce(t)
print(f"rank {r}: all_reduce -> {t.item()} (expect 3.0)")
dist.destroy_process_group()
PY
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 torchrun --nproc_per_node=3 nccl_smoke.py
```
- **`NCCL_P2P_DISABLE=1` is expected/required** on our mixed PHB/SYS topology where GPU0 crosses the
  QPI link and P2P is not available on the SYS path. If NCCL hangs or throws a P2P/`cudaErrorPeerAccess`
  error, this flag is the fix. `NCCL_IB_DISABLE=1` too (no InfiniBand). Consider `NCCL_DEBUG=INFO` to
  see the chosen transport.

**Stage 1 — does `enable_parallelism()` accept our plan on H3 (no full forward)?**
In each rank: load the transformer only, set backend, call `enable_parallelism` with the plan. This
alone catches: missing `_cp_plan` handling, backend-not-CP-compatible errors, wildcard/module-name
mistakes in the plan.

```python
transformer.set_attention_backend("_native_cudnn")
transformer.enable_parallelism(
    config=ContextParallelConfig(ring_degree=dist.get_world_size(), ring_anything=True),
    cp_plan=h3_cp_plan,   # from §4
)
# success = no exception; hooks are now registered
```

**Stage 2 — one denoise step, tiny.** 1–2 denoise steps at the SMALLEST resolution/frame count that
still packs a real multi-modality sequence, group-offload ON, order = backend → CP → group_offload.
Watch for the `#12533`-style shape mismatch on the **second** step (run at least 2 steps).

**Stage 3 — numerical correctness.** Generate a short clip on a **single** GPU (no CP) with a fixed
seed, then the **same** seed under ring-CP-3, and diff the latents. A wrong `cp_plan` runs fine but
produces a different (wrong) result. They should match to within fp tolerance. **Do not trust the plan
until this passes.**

**Stage 4 — the RAM question (decides fp16).** Before attempting fp16, measure host RAM with 3 ranks
loaded: `int8` first (expect ~99 GB). Then try fp16 **only** with mmap-shared weights
(`low_cpu_mem_usage=True`, safetensors mmap, no pinned copy) and watch RSS across the 3 processes — if
total RAM approaches ~66 GB (shared) rather than ~198 GB, fp16-at-length is on; if it hits 198 GB / OOMs,
it is off and we stay int8-and-grow-resolution.

---

## 8. Blunt risk assessment — what breaks, and the fallback

**Most likely to break (ranked):**

1. **fp16 weight replication OOMs host RAM (198 GB > 125 GB).** *This is the single biggest risk and it
   directly threatens the stated goal.* CP does nothing to weights; 3 processes = 3 weight copies.
   *Mitigation:* mmap-shared read-only weights via OS page cache (Stage 4). *Fallback:* **stay int8**
   (99 GB fits) and spend the freed ÷3 activation headroom on **higher resolution / more frames**, which
   is itself a real quality win even without fp16.
2. **The hand-written `cp_plan` is subtly wrong.** H3's packed-sequence/index design is unusual; my plan
   (§4) is reasoned from source, not tested. A wrong split gives a plausible-but-wrong video, silently.
   *Mitigation:* Stage 3 numerical diff against single-GPU. *Fallback:* file the plan upstream / open a
   diffusers issue asking the H3 authors (apolinario / yiyixuxu) to bless a `_cp_plan` — a plain DiT like
   this is squarely in scope and they've added plans for every other DiT.
3. **CP × group_offload hook-ordering bug (#12533).** Real, open, unresolved; shows up on the *second*
   step. *Mitigation:* apply CP **before** offload; run ≥2 steps in Stage 2 to catch it. *Fallback:* if
   group_offload and CP genuinely can't coexist, drop group_offload and use plain per-rank CPU offload of
   whole-model weights with mmap (slower, but CP still splits activations).
4. **NCCL over no-NVLink hangs.** *Mitigation:* `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`, `NCCL_DEBUG=INFO`
   (Stage 0). Ring's low per-step bandwidth is what makes ~4.9 GB/s tolerable.
5. **Ring latency at our bandwidth.** Ring passes K/V every attention layer over PCIe; at ~4.9 GB/s this
   adds real wall-clock. But we are **not chasing speed** — as long as it fits and is correct, slow is
   acceptable. The 4×4090-over-PCIe ring benchmark in the docs shows it runs.
6. **Modular pipeline + CP untested.** Low risk (CP lives in the model), but confirm in Stage 1/2. Keep
   `decode` on rank 0 only.

**One-line verdict:** CP is genuinely the right mechanism and the GPU-side memory math fits 8 GB with
room to grow — but H3 needs a hand-authored `cp_plan` (not shipped) and fp16 additionally needs the
weights mmap-shared across the 3 processes or it OOMs 125 GB RAM. **Realistically: CP + int8 to push
resolution/frames is achievable; CP + fp16 is contingent on the shared-weights experiment.**

---

## Sources (all current as of Sep 2026)

- Parallelism API reference (ContextParallelConfig, ParallelConfig, apply_context_parallel):
  <https://huggingface.co/docs/diffusers/main/en/api/parallel>
- Distributed inference guide (CP invocation, Ring/Ulysses/Unified/anything, torchrun, gloo tip,
  strategy table, **4×RTX 4090 ring benchmark**):
  <https://huggingface.co/docs/diffusers/main/en/training/distributed_inference>
- CP × CPU-offload hook conflict (open issue, ordering workaround):
  <https://github.com/huggingface/diffusers/issues/12533>
- MiniMax-H3 integration PR (Modular-Diffusers-only):
  <https://github.com/huggingface/diffusers/pull/14355>
- Source read at PR head sha `f37ab93e621d5ce206c9662e8291ca8b67d9c555`:
  - `src/diffusers/models/transformers/transformer_minimax_h3.py` — **no `_cp_plan`**; processor uses
    `dispatch_attention_fn(..., parallel_config=self._parallel_config)`; 56 heads × 128, hidden 5376, 50 layers.
  - `src/diffusers/models/modeling_utils.py` L256 (`_cp_plan = None`), L1601–1686 (`enable_parallelism`,
    the `cp_plan` requirement/raise).
  - `src/diffusers/hooks/context_parallel.py` L80–369 (`apply_context_parallel`, split/gather hooks,
    `EquipartitionSharder`, wildcard rules).
  - `src/diffusers/models/_modeling_parallel.py` (ContextParallelInput/Output dataclasses).
  - `src/diffusers/models/attention_dispatch.py` (`supports_context_parallel=True` backends:
    `_native_cudnn`, `_native_flash`, flash variants, sage).
