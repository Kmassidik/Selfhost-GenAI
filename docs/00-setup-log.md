# Setup Log

Chronological record of every install / download / operation on this box.

Box: dalang-Z9PE-D8-WS - 3x RTX 3060 Ti 8GB - dual Xeon E5-2665 (32 threads)
     128GB DDR3 ECC - 818GB free - Ubuntu 24.04.3 - driver 595.84 (CUDA 13.2) - NO nvcc.
Goal: our own H3 runtime (no ComfyUI). Reference target = fal H3-Max output
      (1344x768, 24fps, 32kHz stereo audio, ~5.18s).

## M0 - Base stack
- [ ] apt prereqs: build-essential, python3.12-dev, python3.12-venv, ffmpeg, git, curl
- [ ] uv installed
- [ ] venv + torch 2.9 + triton + diffusers + transformers + safetensors + accelerate + huggingface_hub
- [ ] verify: 3 GPUs visible, Triton compiles, int8 tensor cores work
- [ ] download H3 Diffusers weights (MiniMaxAI/MiniMax-H3)

## Log
(2026-08-31) Project scaffold created under /root/Desktop/selfhosted-minimaxi-h3.

## M0 progress (2026-08-31)
- [x] apt prereqs installed (build-essential, python3.12-dev, ffmpeg, git, curl) + Python.h present
- [x] uv 0.12.7; venv at .venv (Python 3.12.3)
- [x] torch 2.9.0+cu128 · triton 3.5.0 · cuda 12.8 · all 3 GPUs visible
- [x] libs: diffusers, transformers, accelerate, safetensors, huggingface_hub[hf_transfer], einops, sentencepiece, av, imageio
- [x] VERIFIED: Triton JIT compiles + int8 tensor-core matmul works (cc 8.6) -> Track B kernel path good
- [~] downloading H3 FL2VA (144 GB fp16) -> models/H3-FL2VA (see m0-download.log)

## Model repo notes (MiniMaxAI/MiniMax-H3, public, 498 GB total = both variants)
FL2VA variant (text/image -> video+audio), 144 GB:
  transformer/  66.3 GB (33B DiT, 13 shards)   text_encoder/ 66.7 GB (Qwen3-VL-32B)
  video_vae/    10.4 GB                          audio_vae/     0.6 GB   + configs
Encoder(67)+DiT(66)=133 GB > 125 GB RAM -> load sequentially (encode, free, then denoise).
