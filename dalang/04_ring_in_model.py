#!/usr/bin/env python3
"""Dalang Engine — C4 (Stage 2 milestone): our ring INSIDE the real H3 transformer.

We replace H3's attention (`dispatch_attention_fn`) with our proven 2-GPU ring, run
the FULL transformer on the Stage-0 reference input, and diff the output against
`ref_out.pt`. If it matches (within bf16 noise), our ring is correctly integrated
into the model — Stage 2 achieved.

Note: this C4 validates CORRECTNESS (does our ring, in the real model, produce the
right output?). It does NOT yet save VRAM — the hidden_states are still full outside
attention; the full split-forward (the VRAM win) is Phase D/E. Correctness first.

Run (box, >=2 GPUs, dalang_ref/ present):  python 04_ring_in_model.py --ref ./dalang_ref
"""
import argparse
import math
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch
import torch.nn.functional as F

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
D1 = "cuda:1"  # second device for the ring's far slice (free; no model weights live here)


def log(m):
    print(f"[dalang-s2] {m}", flush=True)


def ring_step(q, kj, vj, m, l, O):
    scale = 1.0 / math.sqrt(q.shape[-1])
    # FP32 QKᵀ matmul (flash accumulates in fp32 anyway) — bf16 matmul here caused the C4b/C4c drift.
    Sj = torch.matmul(q.float(), kj.float().transpose(-1, -2)) * scale
    m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
    corr = torch.exp(m - m_new)
    Pj = torch.exp(Sj - m_new)
    l = l * corr + Pj.sum(dim=-1, keepdim=True)
    O = O * corr + torch.matmul(Pj, vj.float())
    return m_new, l, O


def fold_kv(q, k, v, m, l, O, block=512):
    """Fold a whole K/V slice into (m,l,O), TILED over the key dim so we never form the
    full QKᵀ (flash-style). This is what keeps memory small."""
    for j0 in range(0, k.shape[2], block):
        m, l, O = ring_step(q, k[:, :, j0:j0 + block], v[:, :, j0:j0 + block], m, l, O)
    return m, l, O


def ring_attention_2gpu(q, k, v, d0, d1=D1, block=512):
    """q,k,v: (B,H,S,D) on d0. Split S across d0/d1, 2-pass ring (each pass tiled), return (B,H,S,D) on d0."""
    B, H, S, D = q.shape
    h = S // 2
    dt = q.dtype

    def init(dev, Sq):
        return (torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, 1), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, D), dtype=torch.float32, device=dev))

    q0, k0, v0 = q[:, :, :h], k[:, :, :h], v[:, :, :h]                      # already on d0
    q1, k1, v1 = q[:, :, h:].to(d1), k[:, :, h:].to(d1), v[:, :, h:].to(d1)  # far slice -> d1

    m0, l0, O0 = init(d0, h)
    m0, l0, O0 = fold_kv(q0, k0, v0, m0, l0, O0, block)                 # local (tiled)
    m0, l0, O0 = fold_kv(q0, k1.to(d0), v1.to(d0), m0, l0, O0, block)   # ring from d1 (tiled)
    out0 = (O0 / l0).to(dt)

    m1, l1, O1 = init(d1, S - h)
    m1, l1, O1 = fold_kv(q1, k1, v1, m1, l1, O1, block)                 # local (tiled)
    m1, l1, O1 = fold_kv(q1, k0.to(d1), v0.to(d1), m1, l1, O1, block)   # ring from d0 (tiled)
    out1 = (O1 / l1).to(dt)

    return torch.cat([out0, out1.to(d0)], dim=2)


# our replacement for dispatch_attention_fn: same signature, routes through the ring.
def ring_dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  backend=None, parallel_config=None, **kw):
    # processor hands us (B, S, H, D); our ring wants (B, H, S, D)
    d0 = query.device
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = ring_attention_2gpu(q, k, v, d0=d0)          # (B, H, S, D) on d0
    return out.transpose(1, 2).contiguous()            # back to (B, S, H, D)


def extract_sample(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return getattr(out, "sample", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./dalang_ref")
    args = ap.parse_args()

    # monkeypatch H3's attention to use OUR ring
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    h3mod.dispatch_attention_fn = ring_dispatch
    log("patched dispatch_attention_fn -> our 2-GPU ring")

    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    t0 = time.time()
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out"],
        ),
    )
    tr.requires_grad_(False)
    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    log(f"loaded in {time.time()-t0:.0f}s")

    ref_in = torch.load(f"{args.ref}/ref_in.pt", weights_only=False)
    ref_out = torch.load(f"{args.ref}/ref_out.pt", weights_only=False)
    kwargs = dict(ref_in["kwargs"])

    log("running FULL transformer with the ring as its attention ...")
    t = time.time()
    with torch.no_grad():
        sample = extract_sample(tr(**kwargs))
    log(f"forward done in {time.time()-t:.0f}s; out shape={tuple(sample.shape)}")

    a, b = sample.float().cpu(), ref_out.float().cpu()
    if a.shape != b.shape:
        log(f"!! SHAPE MISMATCH ours={tuple(a.shape)} ref={tuple(b.shape)}"); return
    d = (a - b).abs()
    maxd, meand = d.max().item(), d.mean().item()
    log(f"DIFF vs golden reference:  max={maxd:.3e}  mean={meand:.3e}")
    # our ring's attention differs from diffusers' by ~1 bf16 ULP; propagated through 50 blocks
    # a small diff is EXPECTED and correct. A large diff = integration bug.
    if maxd < 5e-2:
        log("VERDICT: PASS ✅  the ring is correctly integrated into H3. STAGE 2 (correctness) achieved.")
    elif maxd < 5e-1:
        log("VERDICT: CLOSE ⚠️  functional but drifting — inspect dtype/scale/rotary handling.")
    else:
        log("VERDICT: FAIL ❌  integration bug — the ring output is wrong in the model.")
    log("STAGE2-DONE")


if __name__ == "__main__":
    main()
