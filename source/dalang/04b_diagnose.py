#!/usr/bin/env python3
"""Dalang Engine — C4b: is the drift a bug or harmless bf16 compounding?

Compare OUR ring vs H3's ORIGINAL attention (`dispatch_attention_fn`) for the FIRST
real block's q/k/v. If they agree to ~1 bf16 ULP, the per-block attention is correct
and the end-to-end C4 drift (max 0.34) is just 1-ULP differences compounding over 50
residual layers (harmless — both are valid bf16). If they diverge, it's an integration
bug (scale/shape/transpose) to fix.

Run (box, >=2 GPUs, dalang_ref/ present):  python 04b_diagnose.py --ref ./dalang_ref
"""
import argparse
import math
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
D1 = "cuda:1"


def log(m):
    print(f"[dalang-s2b] {m}", flush=True)


def ring_step(q, kj, vj, m, l, O):
    scale = 1.0 / math.sqrt(q.shape[-1])
    Sj = (torch.matmul(q, kj.transpose(-1, -2)) * scale).float()
    m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
    corr = torch.exp(m - m_new)
    Pj = torch.exp(Sj - m_new)
    l = l * corr + Pj.sum(dim=-1, keepdim=True)
    O = O * corr + torch.matmul(Pj, vj.float())
    return m_new, l, O


def fold_kv(q, k, v, m, l, O, block=512):
    for j0 in range(0, k.shape[2], block):
        m, l, O = ring_step(q, k[:, :, j0:j0 + block], v[:, :, j0:j0 + block], m, l, O)
    return m, l, O


def ring_attention_2gpu(q, k, v, d0, d1=D1, block=512):
    B, H, S, D = q.shape
    h = S // 2
    dt = q.dtype
    def init(dev, Sq):
        return (torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, 1), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, D), dtype=torch.float32, device=dev))
    q0, k0, v0 = q[:, :, :h], k[:, :, :h], v[:, :, :h]
    q1, k1, v1 = q[:, :, h:].to(d1), k[:, :, h:].to(d1), v[:, :, h:].to(d1)
    m0, l0, O0 = init(d0, h)
    m0, l0, O0 = fold_kv(q0, k0, v0, m0, l0, O0, block)
    m0, l0, O0 = fold_kv(q0, k1.to(d0), v1.to(d0), m0, l0, O0, block)
    out0 = (O0 / l0).to(dt)
    m1, l1, O1 = init(d1, S - h)
    m1, l1, O1 = fold_kv(q1, k1, v1, m1, l1, O1, block)
    m1, l1, O1 = fold_kv(q1, k0.to(d1), v0.to(d1), m1, l1, O1, block)
    out1 = (O1 / l1).to(dt)
    return torch.cat([out0, out1.to(d0)], dim=2)


def ring_dispatch(query, key, value):
    d0 = query.device
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = ring_attention_2gpu(q, k, v, d0=d0)
    return out.transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./dalang_ref")
    args = ap.parse_args()

    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    orig = h3mod.dispatch_attention_fn         # keep the real attention
    state = {}

    def diag(query, key, value, **kw):
        if "done" not in state:
            ref = orig(query, key, value, **kw)          # H3's real (flash) attention
            ours = ring_dispatch(query, key, value)      # our ring
            d = (ours.float() - ref.float()).abs()
            log(f"PER-BLOCK attention diff (ours vs original): max={d.max().item():.3e} "
                f"mean={d.mean().item():.3e}  q.shape={tuple(query.shape)} dtype={query.dtype}")
            ulp = 2 ** -7  # bf16 spacing at ~1
            log(f"bf16 1-ULP ≈ {ulp:.2e}. If max ~<= a few ULP -> ring is correct per block "
                f"(end-to-end drift is harmless compounding). If max >> that -> integration bug.")
            state["done"] = True
            raise RuntimeError("C4B_STOP")               # we have our answer; stop early
        return orig(query, key, value, **kw)

    h3mod.dispatch_attention_fn = diag

    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    t0 = time.time()
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out"]),
    )
    tr.requires_grad_(False)
    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    log(f"loaded in {time.time()-t0:.0f}s")

    ref_in = torch.load(f"{args.ref}/ref_in.pt", weights_only=False)
    kwargs = dict(ref_in["kwargs"])
    log("running forward until the first attention call ...")
    try:
        with torch.no_grad():
            tr(**kwargs)
    except RuntimeError as e:
        if "C4B_STOP" not in str(e):
            raise
    log("C4B-DONE")


if __name__ == "__main__":
    main()
