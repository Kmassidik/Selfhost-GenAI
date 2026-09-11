# selfhosted-minimaxi-h3

Our OWN MiniMax-H3 inference runtime — NO ComfyUI. Runs on this box
(dalang-Z9PE-D8-WS, 3x RTX 3060 Ti). Same shape as how fal serves H3
(a lean runtime around the open Diffusers model), on our own metal.

Knowledge / concepts live on the Mac (selfhostgenai/knowledge-base).
THIS repo is the running app + the operational record.

## Layout
- docs/     operational log — what was installed, downloaded, run, and WHY (rebuildable by anyone)
- runtime/  our H3 engine: loader, RAM->VRAM offload, flow-matching sampling loop, Triton kernels
- models/   H3 weights (Diffusers format) + video/audio VAEs + Qwen3-VL encoder
- outputs/  generated clips

## Status
M0 (base stack) in progress -> docs/00-setup-log.md

## Direction (decided 2026-08-31)
BUILD #1: our own **4-bit weight-sharded PIPELINE** inference across the 3 GPUs (PipeFusion-style).
Accept slow. Target quality = fal.ai H3-Max (768p, fixed faces). Rationale + physics: docs/gpu-reference/02.

Honest: the pipeline fits the WEIGHTS (~5.5GB/card at Q4) but each card holds the full stage
activation -> native 768p needs our activation kernels (fused-FFN, flash-attn, tiled-VAE) too.
So fal-quality 768p = pipeline + kernels. Order: pipeline running first, then kernels to reach 768p.

### Alternatives (documented, NOT building now)
2. Single-GPU engine + our quality kernels + tiled upscale to 720p. Fastest path to the clip; faces fixed.
3. Rent one 48-80GB GPU (~$1-2/hr) for true native 768p, zero inter-GPU comm.
