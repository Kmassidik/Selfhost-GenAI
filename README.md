# selfhostgenai — self-hosted generative AI on 8 GB cards

One box — `dalang-Z9PE-D8-WS`, **3× RTX 3060 Ti (8 GB each)** — running models that
officially need datacenter GPUs, on our **own** inference engines (no ComfyUI), every
number measured on the metal. This repo holds the engines, the operational record,
and a book-length knowledge base explaining how it all works.

Siblings on the same box: `../selfhostaudioai` (sound), `../selfhostllm` (code).
Same rule everywhere: **official weights, the reference pipeline, quality first —
and the 8 GB wall solved by memory placement, not by throwing away precision.**

## What's here

| Path | What |
|---|---|
| `dalang/` | The H3 video engine — the build, step by step: single-GPU baseline → online-softmax → ring attention → block sharding → **3-GPU sequence-parallel 720p render**. |
| `runtime/` | The H3 runtime: loader, RAM→VRAM offload, flow-matching sampling loop. |
| `scenes/` | Scene scripts for rendered films. |
| `image-engine/` | The **Qwen-Image 2.1** engine — reference-quality stills on one 8 GB card, int8 + KV cache + offload. See [its README](image-engine/README.md). |
| `knowledge-base/` | A 49-chapter learning site (`index.html`) — concepts, deep-dives, and the measured findings behind every decision. |
| `docs/` | Operational log — what was installed, downloaded, and run, and why (rebuildable by anyone). |
| `experiments/` | The lab notebook — int8-vs-Q4, multi-GPU research, sampling math, seed studies. |
| `planning/` | Specs, checkpoints, engine design, setup, queue. |
| `roadmap/`, `posting/`, `history/` | Forward plan, write-ups, and the project's story. |

## The two engines, in one line each

**Video — MiniMax-H3 (33B).** The 8 GB wall turned out to be the int8 GEMM working
buffers, not the weights — found by profiling, not guessing. Patched bit-identically,
then pushed further: three cards split one video's sequence to render **native 720p ×
15 s** that no single 8 GB card could hold. Story: knowledge-base ch. 43–48.

**Image — Qwen-Image 2.1 (7B).** The video playbook applied to stills: official
weights + reference `diffusers` pipeline + int8 + CPU offload → reference-quality
1024² images at ~365 s each on a single card. int8 matched bf16 again, and 2.1's
causal-conditioning **KV cache** cut memory in half for free. Story: ch. 49.

## The rule

Every spec in the knowledge base is measured on this box, not rounded or invented —
including the negative results (the LoRA that made faces worse, the kernels that
didn't help, the early multi-GPU verdict that turned out wrong). Weights, engines,
media, and secrets are **git-ignored**; they live on the box and are regenerable.
