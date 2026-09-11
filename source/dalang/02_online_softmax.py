#!/usr/bin/env python3
"""Dalang Engine — C2: online-softmax attention, proven == plain softmax.

The ONE piece of real math in ring attention: instead of forming the full
softmax(QKᵀ/√d)·V at once, process the keys/values in BLOCKS, carrying a running
max `m` and running denominator `l` so the final result is exact. This is the
core we'll spread across GPUs in C3 (each ring pass = one K/V block from another
device). Here we prove it matches PyTorch's reference SDPA — no devices yet.

Run:  python 02_online_softmax.py
"""
import math
import torch
import torch.nn.functional as F


def online_softmax_attention(q, k, v, block=128):
    """q,k,v: (B, H, S, D). Returns (B, H, S, D). Exact softmax(QKᵀ/√D)·V, blockwise.
    Accumulates m/l/O in FP32 internally (like FlashAttention/SDPA) for numerical
    stability with bf16 inputs; casts back to the input dtype at the end."""
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    O = torch.zeros(B, H, Sq, D, dtype=torch.float32, device=q.device)             # fp32 accum
    m = torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=q.device)
    l = torch.zeros(B, H, Sq, 1, dtype=torch.float32, device=q.device)
    for j0 in range(0, Sk, block):
        Kj = k[:, :, j0:j0 + block, :]
        Vj = v[:, :, j0:j0 + block, :]
        Sj = (torch.matmul(q, Kj.transpose(-1, -2)) * scale).float()   # (B,H,Sq,bk) fp32
        m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m - m_new)                                    # rescale prior accum
        Pj = torch.exp(Sj - m_new)                                     # (B,H,Sq,bk) fp32
        l = l * corr + Pj.sum(dim=-1, keepdim=True)
        O = O * corr + torch.matmul(Pj, Vj.float())                    # (B,H,Sq,D) fp32
        m = m_new
    return (O / l).to(q.dtype)


def check(dtype, device, S=384, H=4, D=64, block=128):
    torch.manual_seed(0)
    q = torch.randn(1, H, S, D, dtype=dtype, device=device)
    k = torch.randn(1, H, S, D, dtype=dtype, device=device)
    v = torch.randn(1, H, S, D, dtype=dtype, device=device)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)        # ground truth
    ours = online_softmax_attention(q, k, v, block=block)
    d = (ours.float() - ref.float()).abs()
    maxd, meand = d.max().item(), d.mean().item()
    # bf16 output ULP at magnitude ~1 is 2^-7 = 7.8e-3; being within ~1-2 ULP of the
    # reference is the BEST bf16 can do (not a math error). fp32 must be near-exact.
    tol = 1e-4 if dtype == torch.float32 else 1.2e-2
    ok = maxd < tol
    print(f"[C2] {str(dtype).split('.')[-1]:>8} on {device}: max={maxd:.2e} mean={meand:.2e} "
          f"tol={tol:.0e}  -> {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[C2] proving online softmax == plain SDPA on random (S,H,D) tensors, block-processed")
    results = []
    results.append(check(torch.float32, dev))
    results.append(check(torch.float32, dev, block=64))   # different block size -> same answer
    results.append(check(torch.float32, dev, S=513, block=100))  # non-divisible S/block
    if dev == "cuda":
        results.append(check(torch.bfloat16, dev))         # the real render dtype
    print(f"[C2] {'ALL PASS ✅ — online softmax is exact. Ready for C3 (spread across GPUs).' if all(results) else 'SOME FAIL ❌ — fix the math before C3.'}")
    print("C2-DONE")


if __name__ == "__main__":
    main()
