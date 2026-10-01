# Pipeline Engine — Theory & Design (our own 3-GPU H3 inference, NO ComfyUI)

## The model we're splitting (grounded from the real config.json)
- **MiniMaxH3DiTModel**: `self.transformer_blocks = ModuleList([MiniMaxH3TransformerBlock() x 50])` -> **50 blocks**.
- hidden **5376**, **56 heads x 128** head_dim, FFN **14336**, patch [1,2,2], latents_dim 24, audio_latents_dim 32,
  + 2 token-refiner layers, AdaLN, RoPE. Denoises ONE packed sequence (text cond + video + audio latents).
- Scheduler sigma-shift **12 (video) / 3 (audio)**.

## The scheme: pipeline (inter-layer) parallelism, weight-sharded, Q4
- Split 50 blocks into 3 contiguous stages: ~[0-16] GPU0, [17-33] GPU1, [34-49] GPU2.
- Each GPU holds ONLY its stage's **Q4 weights resident (~1/3 of 16.5GB = ~5.5GB)** -> NO streaming.
- Forward: latent -> GPU0(0-16) -> [PCIe handoff] -> GPU1(17-33) -> [handoff] -> GPU2(34-49) -> out.
- **Activation crosses a GPU boundary only at the 2 seams** (not per layer) -> our 4.9 GB/s link tolerates it.
- Denoise loop (35-50 steps) repeats the 3-stage flow.

## Why this scheme (from docs/gpu-reference/02)
- FITS: 33B doesn't fit 24GB at int8 (33GB); only **Q4 (16.5GB) weight-sharded (5.5GB/card)** fits.
- SURVIVES the link: pipeline crosses only at seams; tensor/sequence parallel cross EVERY layer -> die on 4.9 GB/s.
- NUMA: GPU1<->GPU2 = PHB (tight); GPU0 = SYS (crosses QPI). Prefer GPU0 as an END stage / minimize its cross-socket handoffs.

## Honest limits
- Fits WEIGHTS, not activation. Each card still holds the full stage activation. **Native 768p activation likely won't
  fit the ~2.5GB free/card WITHOUT our kernels** (fused-FFN, flash-attn, tiled-VAE). Order: pipeline running at a
  fittable res first, THEN kernels to climb to 768p.
- Single clip: stages run sequentially (GPU1 waits GPU0) -> no parallel latency win for ONE clip; the WIN is
  resident weights vs streaming 16GB through one 8GB card. GPU utilization comes from the advanced version below.

## Build stages
- **P1**: basic 3-stage resident-Q4 pipeline. Load 50 blocks Q4, place across 3 GPUs (torch.distributed.pipelining
  OR manual per-block .to(device) + handoff), run OUR sampling loop. Prove: our 3-GPU H3 clip, no ComfyUI.
- **P2**: our Triton int8/Q4 kernels inside the blocks (fused-FFN, flash-attn) -> shrink activation + speed up.
- **P3**: PipeFusion patch-pipelining (stale KV) to fill pipeline bubbles -> keep all 3 GPUs busy.

## References (real; PipeFusion & xDiT were independently web-verified by our research agent)
- PipeFusion — arXiv **2405.14430** (patch pipeline for DiTs; shards weights 1/N; PCIe/Ethernet-proven) [agent-verified]
- xDiT — arXiv **2411.01738** · github.com/xdit-project/xDiT (unified DiT parallel engine) [agent-verified]
- GPipe — arXiv 1811.06965 (micro-batch pipeline) · Megatron-LM — arXiv 1909.08053 · github.com/NVIDIA/Megatron-LM
- **torch.distributed.pipelining** (PiPPy, now in PyTorch) — the stage-splitting API we'll use
- DistriFusion — arXiv 2402.19481 (contrast: replicates full model per card -> NOT for us)
- ViDiT-Q — arXiv 2406.02540 (diffusion quant for the Q4 step)

## Open decisions (resolve at build time)
- Q4 source: quantize the fp16 diffusers weights ourselves (= Track B) vs map an existing Q4 GGUF.
- Exact stage-split blocks (balance compute + minimize GPU0/SYS traffic).
- Handoff payload: hidden_states + audio_hidden_states + index/position tensors that cross each seam.
