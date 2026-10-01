# Multi-GPU Inference on our box — reference + the spatial-parallel verdict

## The interconnect (design against these numbers)
- 3x RTX 3060 Ti = **24 GB AGGREGATE but three isolated 8 GB islands**.
- **No NVLink, no P2P** -> GPU<->GPU routes through host RAM = **~4.9 GB/s measured** (two PCIe hops).
- PCIe Gen3 x16 host<->device ~12 GB/s (idles at Gen1/2, ramps under load).
- Topology: **GPU1<->GPU2 = PHB** (same socket/host bridge). **GPU0 = SYS** (crosses QPI to the other Xeon socket) -> slower + NUMA-remote. Pin GPU0's work to minimize cross-socket chatter.
- Design rule: **treat every inter-GPU byte as ~100x more expensive than on NVLink**.

## Parallelism taxonomy — does it survive our 4.9 GB/s link?
| Scheme | splits | comm/step | survives our link? |
|---|---|---|---|
| Data parallel (independent renders) | whole jobs | 0 | YES — best fit, free |
| Pipeline (layer stages) | layers | seams only | YES |
| PipeFusion (patch pipeline, shards weights 1/N) | layers+patches | lowest | BEST sharded option |
| Tensor parallel | every matmul | all-reduce/layer | NO |
| Sequence — Ulysses | tokens | all-to-all/layer | NO (surges on QPI) |
| Sequence — Ring | tokens | ring KV/layer | marginal |
| Spatial/patch tiling | latent tiles | depends on attn | only if attention is LOCAL |

## Spatial-parallel-720p thesis — VERDICT: ~5-10%, DON'T build it
Two independent, each-fatal reasons:
1. **THE WALL IS WEIGHTS, NOT ACTIVATIONS.** 33B: bf16 66 GB, int8 33 GB — both > 24 GB aggregate. Model fits our 3 cards ONLY at Q4 (~16.5 GB) AND weight-sharded (~5.5 GB/card). Splitting the *activation* misses the real wall entirely.
2. **GLOBAL ATTENTION forbids independent tiles.** Every token attends to every other, so a spatial tile is NOT independently computable — correct attention forces full KV exchange EVERY layer (all-to-all / ring), ~130 MB/layer over 4.9 GB/s ~= 27 ms/layer -> **~45-55 s of unhideable comm PER CLIP**. There is no "boundary-only" version of a global op.
- And: DistriFusion (the only slow-link-tolerant patch method) **replicates the full model on every card** -> impossible on 8 GB regardless of interconnect.

## The ONE real 3-GPU single-render path: 4-bit PipeFusion-style pipeline
- Weight-sharded pipeline: each card holds ~1/N of params (~5.5 GB at Q4) + patch pipeline with stale KV.
- Lowest comm cost; tolerates a slow fabric; the ONLY scheme that both shards weights AND survives our link.
- BUT: still per-step KV exchange; proven only on fabrics ~2.5x faster than ours; comm-bound; quality risk from Q4 + stale KV. **A bounded 1-2 day spike, not a multi-week build.**

## Honest recommendations to ship a 720p deliverable (in order)
1. **Render at a fittable res + tiled upscale to 720p** (tiled upscale is the ONE place tiling legitimately works on our link; data-parallel across all 3 cards).
2. **Rent ONE 48-80 GB GPU ~$1-2/hr** for the handful of true native-720p renders (zero inter-GPU comm).
3. **Use the 3 cards as 3 data-parallel workers** (throughput) — their highest-value use.

Sources: DistriFusion 2402.19481 · xDiT 2411.01738 · PipeFusion 2405.14430 · USP 2405.07719 · Flash Communication 2412.04964
