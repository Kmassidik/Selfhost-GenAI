# ENGINE-DESIGN — Sharding the MiniMax‑H3 video DiT across 3× RTX 3060 Ti

Status: research + design (no code run, no box touched). Written 2026‑09‑02.
APIs verified against current docs (diffusers **v0.40.0**, accelerate **v1.14.0 / v1.0.0**); links inline. Where I say "unverified" I mean I could not confirm it from primary sources and it must be checked on the box.

---

## 1. The goal (restated) and the one‑paragraph verdict

**Goal:** make **one** render of the H3 DiT (`MiniMaxH3Transformer3DModel`: 50 transformer blocks, hidden 5376, ~33 B params, int8 ≈ 33 GB) *lower‑latency* by using all 3 GPUs, via a "block‑cache pipeline": split the 50 blocks into 3 stages, one GPU computes at a time, and the 2 idle GPUs hold pre‑loaded blocks as a resident cache so fewer blocks stream from RAM each step.

**Verdict up front (honest):** the block‑cache pipeline **as briefed will not meaningfully speed up a single render**, for four independent reasons that each alone is close to fatal:

1. **Pipeline/block parallelism does not reduce single‑sample latency.** It raises *throughput* by keeping every stage busy with *different* samples. With one render there is one latent and one active step at a time, so at any instant exactly one GPU computes and two idle — the wall‑clock is the same sequential chain a single GPU would run, minus overheads. (This is the textbook property of pipeline parallelism; the diffusion loop is also strictly sequential across steps — step *t+1* needs step *t*'s full output — so there is no cross‑step pipeline to fill either.)
2. **The 5.7 GB activation forces the active GPU to evict its own cache.** A GPU can hold a big block cache *only while idle*. The instant it becomes the compute stage it must make room for the 5.7 GB activation scratch, leaving ~0.76 GB ≈ **1 resident block**. So the cache built while idle does not survive into the compute phase — it buys ~0 streaming reduction. (Worked out in §3.)
3. **No‑NVLink P2P (4.9 GB/s) is *slower* than RAM→GPU (12 GB/s).** So the fallback idea — "active GPU pulls a cached block from an idle peer instead of from RAM" — is strictly worse than just streaming from RAM. Peer‑as‑cache is a net loss.
4. **The render is compute‑bound, not transfer‑bound.** Measured on our box: **SM 100%, ~8 s/step**; streaming 33 GB from RAM is ≈1–1.5 s of that (DDR3‑1600 ~25 GB/s on the old box; DDR4 faster on the new one). Even *eliminating all streaming* caps the win at ~1.1–1.15×, and the block‑cache scheme can't eliminate it (reason 2). (Source: our own `dalang-box-h3-setup` measurements.)

**What actually could make one render faster on 3 GPUs:** only **intra‑step compute splitting** — tensor parallelism (TP) or **sequence/context parallelism (CP)**. CP is the interesting one because it *also* shrinks the 5.7 GB activation (the real 8 GB wall is ~88% activations, not weights — our own finding). That is the higher‑ceiling, higher‑risk v2. See §4/§6.

**What is guaranteed to use 3 GPUs well:** throughput parallelism — 3 independent renders / seeds / shots at once, linear 3× on total work, zero new engine code. If the deliverable is "a minute of video" (many shots/seeds) rather than "this one clip, faster," this is the correct answer and should not be overlooked. See §6.

---

## 2. Per‑GPU memory budget (worked out)

Constants (given, measured):

| Item | int8 | bf16 |
|---|---|---|
| Usable VRAM / card (3060 Ti, 8 GB) | 7.66 GB | 7.66 GB |
| CUDA context / driver | 1.20 GB | 1.20 GB |
| One transformer block (of 50) | 0.65 GB | 1.29 GB |
| Activation scratch @ ~25k tokens (209 f) | 5.70 GB | 5.70 GB* |
| Hidden state crossing a stage boundary | 0.27 GB | 0.27 GB |
| Full model resident | 33 GB | 66 GB |

\* activation scratch is dtype‑of‑compute; H3's int8 path dequants to bf16 for the GEMM, so the scratch is bf16‑sized regardless. (Our profiling: ~88% of the 5.9 GB real‑tensor peak is int8 GEMM/convrot working buffers — `fast_int8_mm` alone is a 2.24 GB temp — not resident weights.)

**Two states a GPU can be in:**

- **ACTIVE (computing a block):** `1.20 (ctx) + 5.70 (activation) = 6.90 GB` fixed → **0.76 GB free → 1 int8 block** (or 0 bf16 blocks — bf16 does not fit the active GPU at all with this activation; bf16 is off the table for this box, consistent with prior findings). So *while computing*, every GPU is a 1‑resident‑block machine, identical to the single‑GPU case.
- **IDLE (holding cache only):** `7.66 − 1.20 = 6.46 GB` → **~9–10 int8 blocks cached**.

The whole scheme lives or dies on whether the IDLE‑state cache can survive the transition to ACTIVE. It cannot (§3), unless the activation is shrunk (§4/CP).

---

## 3. The catch about activations (answering Q3 directly)

Take the natural pipeline‑parallel partition: GPU0 = blocks 0–16, GPU1 = 17–33, GPU2 = 34–49. Within one denoise step:

- **Phase 0:** GPU0 active. Needs each of blocks 0–16 resident *as it reaches them*. It has 0.76 GB free → holds 1 block, streams the other 16 from its pinned RAM region. GPU1/GPU2 idle; they pre‑load their ranges into cache (~10 blocks each).
- **Phase 1:** hidden state (0.27 GB) crosses GPU0→GPU1 over P2P. GPU1 becomes active → must host the 5.7 GB activation → must **evict its ~10‑block cache down to 1 block**. The 6.5 GB of blocks it pre‑loaded while idle are thrown away *before it computes them*. It then streams blocks 17–33 from RAM one at a time, exactly as if it had never cached.
- **Phase 2:** same for GPU2.

**Net streaming:** ~49 of 50 blocks still stream from RAM per step — **no reduction over single‑GPU.** The briefed arithmetic ("~20 cached across 2 idle GPUs → ~30 streamed") silently assumes the cache survives into compute; it does not, because *the GPU that cached the blocks is the same GPU that must evict to compute them*. The two idle GPUs' caches only ever help a *future* phase on *that same GPU*, and that future phase is exactly when the eviction happens.

The only escape without changing the activation: cache block *N* on a **peer** and pull it over P2P when the active GPU needs it. But P2P = 4.9 GB/s < RAM = 12 GB/s, so that is slower than streaming from RAM. Dead end (reason 3 in §1).

**The escape that works: shrink the activation.** Split the ~25k‑token sequence across the 3 GPUs (context/sequence parallelism) → each GPU holds ~8.3k tokens → activation scratch ≈ **5.7/3 ≈ 1.9 GB**. Now the ACTIVE budget is `1.20 + 1.90 = 3.10 GB` → **~7 resident int8 blocks survive while computing**. *And* the three GPUs now compute the block simultaneously on their token shards (real latency win), and the true 8 GB wall (activations) is relieved. This is why the honest recommendation pivots away from "cache weights" toward "split the sequence." See §4‑C.

---

## 4. Candidate mechanisms

### A — `accelerate` big‑model‑inference `device_map` (naive PP + CPU overflow)
**What it is.** Load the transformer with a fixed layer→device map; whatever fits on GPUs stays resident, the overflow is offloaded to CPU (or disk) and streamed per forward.

Current, verified API (accelerate v1.0.0 / v1.14.0):
```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers import MiniMaxH3Transformer3DModel   # ModelMixin subclass

with init_empty_weights():
    model = MiniMaxH3Transformer3DModel.from_config(cfg)

model = load_checkpoint_and_dispatch(
    model, checkpoint="…/transformer",
    device_map="auto",                       # or an explicit {"transformer_blocks.0": 0, …}
    max_memory={0: "6GiB", 1: "6GiB", 2: "6GiB", "cpu": "110GiB"},
    no_split_module_classes=["MiniMaxH3TransformerBlock"],   # keep a block whole (residual)
    offload_folder=None, dtype=torch.bfloat16,
)
```
diffusers exposes the same through `AutoModel.from_pretrained(subfolder="transformer", device_map="auto", max_memory=…)` (distributed‑inference doc, v0.40.0).

**How its offload actually behaves (verified).** `AlignDevicesHook` runs just before each module's forward: it moves CPU‑offloaded weights **onto GPU 0**, runs, then moves them back — *serially, no prefetch, and always through GPU 0*. GPU‑assigned layers stay resident (a fixed partition). So it gives you exactly the "resident where it fits + serial stream for the rest" behavior — **but it does *not* do the idle‑caches‑active‑computes optimization**, and its stream target is GPU 0 only.

**Pros:** ~15 lines; current, documented, correct; honors `max_memory` per GPU; the fastest way to get a *running* 3‑GPU baseline and, more importantly, a **measurement harness** for the compute:transfer ratio that decides everything.
**Cons:** it is pipeline‑parallel → **no single‑render latency win** (§1.1); CPU overflow streams serially through GPU 0 (not per‑GPU pinned regions); no activation relief. Expect it to be *equal to or slightly slower than* today's single‑GPU streamed setup for one render.
**Caveat:** H3 in diffusers is **Modular‑Diffusers‑only** (`MiniMaxH3ModularPipeline`; there is no `DiffusionPipeline` half). The transformer is still a `ModelMixin`, so `device_map` on the *model* should apply, but the pipeline‑level `device_map="balanced"` examples (which only split *pipeline components* — text‑encoder/VAE/transformer — not a single model's blocks) do **not** transfer. Verify `load_checkpoint_and_dispatch` on the H3 transformer on the box.

Sources: [accelerate Big Model Inference](https://huggingface.co/docs/accelerate/v1.0.0/en/usage_guides/big_modeling), [accelerate concept guide — how offload/AlignDevicesHook works](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference), [HF blog: How Accelerate runs very large models](https://huggingface.co/blog/accelerate-large-models), [diffusers distributed inference (device_map), v0.40.0](https://huggingface.co/docs/diffusers/en/training/distributed_inference).

### B — Custom torch block‑cache pipeline (the briefed design, hand‑rolled)
**What it is.** Manually assign block ranges to `cuda:0/1/2`; keep a per‑GPU resident cache; stream overflow from *per‑GPU pinned* CPU RAM; move the 0.27 GB hidden state between GPUs with `.to(device)` at stage boundaries; use per‑GPU CUDA streams to prefetch block *N+1* while computing *N*.

**Why it still fails for single‑render latency:** it is Candidate A done by hand — same pipeline‑parallel structure (§1.1), same eviction problem (§3). Hand‑rolling *does* let you (a) use per‑GPU pinned buffers instead of A's route‑through‑GPU‑0, and (b) add true double‑buffered prefetch. But (a)+(b) are **single‑GPU** wins (recover the serial‑stream cost on the one active GPU); they don't need 3 GPUs. The 2 idle GPUs still contribute nothing a single GPU with a prefetch double‑buffer wouldn't.

**Real gotchas if built anyway:**
- **P2P availability:** call `torch.cuda.can_device_access_peer(i, j)` — on this board (PCIe, no NVLink) it is likely **False**, so `.to()` between GPUs bounces through host pinned memory at ~4.9 GB/s. Budget the 0.27 GB hop at ~55 ms each, ×2 boundaries ×steps.
- **`torch.cuda.set_device` / streams:** each GPU needs its own stream and its own pinned staging buffer; you must `set_device` before allocating pinned host buffers you intend to DMA to a given GPU, and synchronize the prefetch stream before the compute kernel reads the weight.
- **Pinned‑RAM blow‑up (known landmine):** the existing `use_stream=True` path in diffusers pins host RAM and **kills the process** — because unbounded pinning of a 33 GB model exhausts pinnable memory. A hand‑rolled version must use a **small, fixed ring of pinned buffers** (e.g. 2–3 block‑sized, ~2 GB total), not pin the whole model. This is the single most important implementation constraint.
- **Group‑offload hook conflict:** you cannot mix this with diffusers `enable_group_offload` / `apply_group_offloading` — both install pre‑forward hooks that move weights; they will fight. Pick one owner of weight movement.

Sources: [PyTorch `can_device_access_peer`](https://pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html), [diffusers group offloading (`use_stream`, `num_blocks_per_group`), v0.35.1](https://huggingface.co/docs/diffusers/en/optimization/memory), [diffusers issue #12319 — broken block_level group offload](https://github.com/huggingface/diffusers/issues/12319).

### C — Sequence / Context parallelism (the only real single‑render win) — v2
**What it is.** Split the token sequence across the 3 GPUs. Each GPU computes every block on its ~8.3k‑token shard; attention exchanges K/V across GPUs (Ring) or does an all‑to‑all on heads (Ulysses). This **parallelizes compute** (true latency reduction) **and shrinks the activation** to ~1.9 GB/GPU (§3), which is the actual 8 GB wall.

Current, verified API (diffusers v0.40.0 — brand new, native):
```python
from diffusers import ContextParallelConfig
transformer.set_attention_backend("_native_cudnn")           # CP-compatible backend
transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=3))
# launched with:  torchrun --nproc-per-node 3 render.py
```
diffusers ships Ulysses, Ring, Unified, and the "Anything" variants (arbitrary seq‑len / head‑count — relevant because 3 rarely divides our token count or head count cleanly). Benchmarks in the doc were run on **4× RTX 4090 (PCIe, no NVLink)** and **4× L20 (PCIe)** — i.e. this is designed to work without NVLink. Ulysses gave the best throughput in HF's own numbers.

**Why it's v2 not v1 (honest risks):**
- **Weights must still be resident or streamed *per GPU*.** CP does not shard weights (that's TP). Each rank needs the block it's computing → 33 GB won't fit in 8 GB, so you must *combine CP with per‑GPU streaming/offload* — and **CP + group‑offload is not a validated combination** (diffusers explicitly marks even TP+CP as experimental). This is the crux risk. The activation relief buys ~7 resident blocks/GPU while active (§3), so per‑step you'd stream ~43/50 — better than today but still streaming, and now with per‑layer all‑to‑all comms on top.
- **Attention‑backend compatibility on H3 is unverified.** H3's latent is a **NestedTensor**; `VAEDecodeTiled` already breaks on it and **SageAttention outputs pure noise on H3**. Whether the CP backends (`_native_cudnn` / ring / ulysses) accept H3's packed multimodal NestedTensor sequence is **unknown and must be tested**. H3 is Modular‑Diffusers‑only and very new (PR #14355); `enable_parallelism()` also requires the model to cooperate (and TP additionally needs a `_tp_plan`, which a brand‑new model almost certainly lacks).
- **Per‑layer comms over PCIe:** Ulysses all‑to‑all every attention layer × 50 blocks × ~35 steps is a lot of 4.9 GB/s traffic; the HF 4090 benchmarks are on datacenter‑ish PCIe and *fit weights resident*. Our added streaming makes the balance worse. Measure before believing.

Sources: [diffusers Context Parallelism (Ulysses/Ring/Unified + Anything variants) + benchmarks on 4×4090/4×L20, v0.40.0](https://huggingface.co/docs/diffusers/en/training/distributed_inference), [diffusers `ContextParallelConfig` API](https://huggingface.co/docs/diffusers/v0.40.0/en/api/parallel), [MiniMax‑H3 diffusers pipeline (Modular‑only)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3), [Add MiniMax‑H3 PR #14355](https://github.com/huggingface/diffusers/pull/14355).

### Prior art we should reuse instead of hand‑rolling
- **xDiT / PipeFusion** ([arXiv 2411.01738](https://arxiv.org/abs/2411.01738), [github PipeFusion/PipeFusion](https://github.com/PipeFusion/PipeFusion)) — explicitly built for **PCIe/Ethernet without NVLink** ("PCIe and Ethernet are enough"), benchmarked on 4×T4‑16GB PCIe. **But:** (1) it is *patch‑level pipeline* parallel → same throughput‑not‑latency caveat; (2) supported models are PixArt/Hunyuan/SD3/Flux/DiT‑XL — **not H3**; (3) **no CPU‑offload** — each stage's weights must be resident. 33 GB / 3 = 11 GB > 8 GB, so H3 **does not fit** PipeFusion's stages on 3060 Ti even split 3 ways. Its reusable idea is its *sequence‑parallel* component, which diffusers now ships natively (Candidate C). Verdict: study, don't adopt.
- **DeepSpeed‑Inference pipeline / FSDP:** FSDP is training‑oriented and our own roadmap already ruled it out on this box (33 B video activations + 4.9 GB/s link). Not applicable.
- **accelerate big‑model‑inference:** = Candidate A; the sanctioned "doesn't fit + slow interconnect" path per diffusers' own strategy table ("`device_map` … the model doesn't fit and the interconnect is slow").

---

## 5. RECOMMENDED v1 — measure first, with Candidate A as the harness

The whole design hinges on one unmeasured number: **what fraction of the 8 s/step is weight streaming vs compute?** Everything the block‑cache pipeline attacks is inside that streaming fraction, and our existing evidence (SM 100%) says it's small. So v1 is deliberately a **measurement + honest‑baseline** step, not the pipeline itself:

**v1 = load the H3 DiT with Candidate A (`load_checkpoint_and_dispatch`, `max_memory` = {0:6GiB,1:6GiB,2:6GiB, cpu:110GiB}, `no_split_module_classes` on the block class), run ONE proven‑safe render (e.g. 576×320, our known‑good config), and instrument per‑step compute vs H2D‑transfer time.**

Why this is the right v1:
- It is ~15 lines of *current, verified* API and produces a **running 3‑GPU render** (correctness first).
- It yields the decision number. If transfer is, say, >30% of step time → a streaming‑reduction scheme is worth building, and we go to a **single‑GPU double‑buffered prefetch** first (recovers most of it with *no* multi‑GPU complexity, and sidesteps the `use_stream=True` pinned‑RAM crash via a bounded pinned ring — Candidate B's only genuinely useful pieces). If transfer is <~15% (likely, given SM 100%) → **stop building weight‑streaming tricks**; the block‑cache pipeline is dead, and 3‑GPU value must come from Candidate C (latency) or throughput parallelism (§6).
- It costs nothing we throw away: the `max_memory`/offload plumbing and the instrumentation are reused by every later option.

**Concrete first implementation step (do exactly this):**
1. On the box, in the ComfyUI venv (Python 3.12), write `scratchpad/shard_probe.py` that: (a) `load_checkpoint_and_dispatch` the `transformer/` folder with the `max_memory` above; (b) print `model.hf_device_map` to confirm the block split; (c) run one 576×320 render through the Modular pipeline with the transformer swapped in; (d) wrap the transformer forward with `torch.cuda.Event` timers separating kernel time from `AlignDevicesHook` H2D copies (or simplest proxy: total step time vs `nvidia-smi dmon` SM% duty cycle), log per‑step compute‑ms and transfer‑ms.
2. Read off the compute:transfer ratio. Record it in `dalang-box-h3-setup` memory.
3. Branch on the result per the decision rule above.

Do **not** build the custom block‑cache pipeline (Candidate B full form) before step 2. It is the most code for a benefit the measurement is likely to show is near zero.

---

## 6. Honest expected payoff, risks, showstoppers

**Block‑cache pipeline (as briefed), single‑render latency:** expected **~1.0×** (range 0.9–1.1×). Not the hoped 1.3–1.5×. It is defeated by, in order of severity: (1) pipeline parallelism is a throughput tool, not a latency tool — one render never has two stages busy at once; (2) the 5.7 GB activation evicts any cache the moment a GPU computes; (3) P2P (4.9) < RAM (12) kills peer‑caching; (4) the render is compute‑bound (SM 100%) so the streaming it targets is only ~10–15% of the step even before (2)/(3) neutralize the caching. Add pipeline‑bubble + two P2P hidden‑state hops/step + cross‑process sync and it likely lands slightly **negative**.

**Where real single‑render speedup can come from (ranked):**
- **Candidate C (sequence/context parallelism):** the only mechanism that both parallelizes compute *and* attacks the true activation wall. Optimistic ceiling maybe ~2× *if* it runs on H3 and the added per‑GPU streaming + PCIe all‑to‑all don't eat it. **High uncertainty** — gated on unverified H3 attention‑backend compatibility (NestedTensor; SageAttention already noise on H3) and on the unvalidated CP‑plus‑offload combo. Treat as a research spike, not a delivery.
- **Single‑GPU prefetch double‑buffer:** recovers whatever the transfer fraction is (measurement‑bounded, ~1.1× if transfer‑bound), no multi‑GPU, no P2P, low risk. Cheapest real win if v1 shows transfer matters.

**The pragmatic alternative that *guarantees* 3‑GPU value — throughput, not latency.** If the real objective is "more finished video per hour" (our 60 s films are 6× 10 s shots; our hero workflow is draft‑then‑final; seeds are cheap), run **3 independent renders in parallel**, one per GPU, via accelerate's `PartialState.split_between_processes` (verified, v1.14.0) or simply 3 ComfyUI instances (the new box already runs GPU0→:8189, GPU1→:8190). This is a **linear 3× on total work**, zero engine code, zero P2P. For everything except "this exact clip must finish sooner," this beats every sharding option here and should be the default. (Source: [diffusers distributed inference — Accelerate `split_between_processes`](https://huggingface.co/docs/diffusers/en/training/distributed_inference).)

**Showstoppers to state plainly:**
- bf16 is off the table on this box (active‑GPU budget can't hold the activation + even one bf16 block cheaply; matches prior findings). int8/Q4 only.
- H3 being Modular‑Diffusers‑only + brand new means every diffusers parallel API (device_map on the model, `enable_parallelism`, `_tp_plan`) is **compat‑unverified on H3** and must be smoke‑tested before design weight is put on it.
- No NVLink is not a tuning problem, it's structural: any scheme with per‑layer cross‑GPU comms pays 4.9 GB/s. That is why diffusers' own guidance says `device_map` (not TP) for slow interconnects, and why the latency‑winning option (CP) is simultaneously the riskiest here.

---

## 7. Sources (verified this session)
- diffusers **v0.40.0** — Distributed inference (device_map, Accelerate `split_between_processes`, Context Parallelism Ulysses/Ring/Unified + Anything, Tensor Parallelism, strategy table): https://huggingface.co/docs/diffusers/en/training/distributed_inference
- diffusers `ContextParallelConfig` / `TensorParallelConfig` API: https://huggingface.co/docs/diffusers/v0.40.0/en/api/parallel
- diffusers Reduce‑memory / group offloading (`enable_group_offload`, `block_level`, `use_stream`, `num_blocks_per_group`) **v0.35.1**: https://huggingface.co/docs/diffusers/en/optimization/memory
- diffusers issue #12319 — broken `block_level` group offloading: https://github.com/huggingface/diffusers/issues/12319
- accelerate **v1.0.0** Big Model Inference (`load_checkpoint_and_dispatch`, `no_split_module_classes`, `max_memory`, offload): https://huggingface.co/docs/accelerate/v1.0.0/en/usage_guides/big_modeling
- accelerate concept guide — offload mechanics / `AlignDevicesHook` (moves offloaded weights to GPU 0 just‑in‑time, back after): https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference
- HF blog — How Accelerate runs very large models (device_map, hooks): https://huggingface.co/blog/accelerate-large-models
- MiniMax‑H3 in diffusers (Modular‑only pipeline): https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3 · PR #14355: https://github.com/huggingface/diffusers/pull/14355
- xDiT / PipeFusion (PCIe/Ethernet without NVLink; patch‑level pipeline parallel; model list; no CPU offload): https://arxiv.org/abs/2411.01738 · https://github.com/PipeFusion/PipeFusion
- PyTorch `torch.cuda.can_device_access_peer`: https://pytorch.org/docs/stable/generated/torch.cuda.can_device_access_peer.html
- Our own measured facts: `~/.claude/.../memory/dalang-box-h3-setup.md`, `h3-optimization-roadmap.md` (SM 100% / ~8 s/step; 8 GB wall ≈ 88% activations; DDR3‑1600 ~25 GB/s; `use_stream=True` pins host RAM and kills the process; SageAttention = noise on H3; NestedTensor latent).
