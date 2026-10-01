# image-engine — Qwen-Image 2.1 on one 8 GB card

A second engine beside the H3 video pipeline: the **official Qwen-Image 2.1** (7B
visual-gen DiT + Qwen3-VL text encoder) rendering reference-quality stills on a
single RTX 3060 Ti (8 GB), int8, no ComfyUI. The full story — the 2.0→2.1 class
trap, int8≈bf16 measured again, the KV-cache finding, the VAE-decode OOM — is
[knowledge-base chapter 49](../knowledge-base/49-the-image-engine.html).

Runtime (weights, venvs, outputs) lives on the box and is git-ignored; this folder
is the source of truth for the code.

## The stack

| piece | file | job |
|---|---|---|
| server | `app/server.py` | FastAPI: serves the page + gallery, enqueues jobs. Never touches the GPU. |
| worker | `app/worker_qwen.py` | Resident diffusers worker. Loads the model once, renders in-process on GPU 1. |
| queue | `app/jobs.py`, `app/db.py` | SQLite, one job at a time. |
| catalog | `app/catalog/*.json` | One JSON per model (`engine`, `defaults`, `limits`, `device`). No per-model code. |
| page | `app/web/index.html` | Prompt → image UI, password-gated generation. |
| units | `deploy/*.service` | systemd: app on :8097, resident worker on GPU 1. |

## The engine config (measured, see ch.49)

- **Model:** official `Qwen/Qwen-Image-2.1`, loaded with `QwenImage21Pipeline` /
  `QwenImage21Transformer2DModel` (needs diffusers **git-main**).
- **int8** weight-only (torchao) — identical quality to bf16, ~18% faster under
  offload because half the weight bytes cross PCIe per step. Toggle: `QWEN_INT8=1`.
- **KV cache** — `use_kv_cache=True`; 2.1's causal-conditioning cache. ~5% faster,
  peak GPU memory 1.4 GB → 0.8 GB, no quality change. Toggle: `QWEN_KV_CACHE=1`.
- **Offload** — `enable_sequential_cpu_offload()`; the 7B model streams layer-by-
  layer from system RAM, only the active layer sits on the 8 GB card.
- **OOM fixes** — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (clears the
  VAE-decode fragmentation OOM) and `torch.cuda.empty_cache()` after every job.

Benchmark (1024×1024, 30 steps, seed 509632276): bf16 470 s · int8 385 s ·
int8+KV 365 s — all visually identical.

## Isolation — three cards, three tenants

- `CUDA_VISIBLE_DEVICES=1` in the worker unit: the engine can only see GPU 1.
  **GPU 0 is the audio service and is never visible to it.**
- Its own venv (`.venv-qwen21`, diffusers git-main) — the H3 video engine's venv is
  never modified.

## Run (on the box)

```bash
# one-time: dedicated venv with diffusers git-main (see build_venv21.sh on the box)
uv venv .venv-qwen21 --python 3.12
uv pip install --python .venv-qwen21/bin/python torch==2.11.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-qwen21/bin/python \
  "git+https://github.com/huggingface/diffusers" transformers accelerate \
  torchao safetensors pillow sentencepiece protobuf

# services (OWNER_PASSWORD lives in /root/.env.imageai, never in git)
systemctl enable --now selfhostimage-app selfhostimage-worker
```

Generation is password-gated (owner creates, anyone views); viewing the gallery is
open. The page is served behind the shared Cloudflare tunnel on its own subdomain.
