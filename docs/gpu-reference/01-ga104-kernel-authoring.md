# GA104 / sm_86 Kernel-Authoring Reference (RTX 3060 Ti) — distilled

Target: hand-written **Triton** kernels (JIT->PTX, no nvcc) for H3's DiT on our card.
Full sourced version compiled 2026-08-31; key facts below.

## The card (sm_86)
38 SMs - 4864 FP32 cores (128/SM) - 152 Tensor cores (4/SM, 3rd gen) - 8GB GDDR6 256-bit @ **448 GB/s**
- boost ~2100 MHz (measured; ref 1665) - L2 **4 MB** - regs **65536/SM (<=255/thread)**
- shared/L1 **128 KB unified/SM, up to 100 KB opt-in shared** - **48 warps/SM**, **16 blocks/SM**.

## Peak throughput (why INT8)
| dtype | @2100MHz | vs FP32 |
|---|---|---|
| FP32 | ~20 TFLOPS | 1x |
| FP16 tensor, **FP32 accum** | ~41 TFLOPS | 2x |
| FP16 tensor, FP16 accum | ~82 | 4x |
| **INT8 tensor (INT32 accum)** | **~163 TOPS** | **8x** |
| INT4 tensor | ~327 | 16x |

**KEY FINDING:** GeForce Ampere runs **FP16-with-FP32-accumulate at HALF rate**. Since attention/large
GEMM need FP32 accum for safety, the fp16 path tops out ~2x FP32. **INT8 IMMA (INT32 accum, no penalty)
= 4x that path, 8x FP32** -> INT8 is the right target for the DiT's big GEMMs on this card.

## Memory hierarchy + roofline
regs ~0cyc / shared ~23-30 / L1 ~33 / L2 ~200 / GDDR6 ~450-650 cyc. Ridge (ops/byte @2100):
FP32 46, FP16-fp32acc 91, **INT8 365**. Implications:
- **GEMM/linear** = compute-bound with big tiles -> feed tensor cores.
- **Attention** naive = memory-bound (S×S > 4MB L2) -> MUST flash-fuse (scores in shared/regs, never write S×S).
- **norms/RoPE/activation/add** = always memory-bound (AI~0.1-1) -> **fuse everything**, coalesce, 128-bit loads, high occupancy.
Two wins on this card: (1) fuse all memory-bound ops; (2) quantize big GEMMs to INT8.

## INT8 IMMA rules
- shape m16n8k32.s8.s8->s32; **BLOCK_K must be multiple of 32**.
- Do NOT dequant int8->bf16->fp16 matmul (pays half-rate + wastes 4x). Instead:
  `tl.dot(a_int8,b_int8,out_dtype=tl.int32)` -> IMMA, accumulate int32, **fold dequant (per-token x per-channel scale)+bias into the epilogue**.
- **Verify the PTX/SASS actually contains IMMA/mma...s8** - Triton silently picks fp16 MMA if an operand is float (known footgun).

## Occupancy / autotune (A100 configs DON'T fit - 100KB shared cap)
- Full occupancy (48 warps) needs <=42 regs/thread. GEMM at ~17% occ is FINE (MMA throughput = ILP + cp.async, not warp count).
- A100 default 128x256x64 stages3 = ~123 KB shared -> **does not fit** sm_86. Use:
  - FP16 GEMM: 128x128x32, num_warps 4, num_stages 3 (~55-74KB shared).
  - **INT8 GEMM**: BLOCK_K in {32,64,128}, M/N in {64,128,256}, warps {4,8}, stages {3,4} (int8 halves bytes -> deeper pipelines fit). Fold dequant epilogue.
  - Flash attention: seq tiles 64/128, BLOCK_D=head_dim(128), warps {4,8}, stages {2,3}, running softmax in regs, K/V in shared via cp.async.
  - elementwise/norm: 1D blocks 1024-4096, stages 1, chase occupancy not tiles.
- Grid: want >= 2-4 waves over 38 SMs (~150-300 tiles); if GEMM small, use 64x64 tiles for more blocks.

## Sources
CUDA C Programming Guide (cc 8.6 tables) - Ampere Tuning Guide - GA102/GA10x whitepaper -
TechPowerUp RTX 3060 Ti - Ampere microbenchmark arXiv 2208.11174 - Native INT8 DiT GEMM arXiv 2606.14598 -
Triton issue #7188 (verify IMMA emission).
