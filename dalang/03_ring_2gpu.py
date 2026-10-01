#!/usr/bin/env python3
"""Dalang Engine — C3: context-parallel ring attention across 2 GPUs (single process).

Split the sequence in half: slice 0 on cuda:0, slice 1 on cuda:1. Each device holds
its own Q/K/V slice. Ring:
  pass 1  — each device attends its Q to its LOCAL K/V (online-softmax init).
  pass 2  — K/V slices rotate device->device (PCIe, no NVLink); each device attends
            its Q to the ARRIVING K/V and accumulates (online-softmax update).
After N=2 passes every query has seen the whole sequence, but no device ever held
more than 1/2 of it. Reassemble -> prove it equals full-sequence SDPA.

This is context parallelism, for real, in ONE process (the Dalang way — no torchrun,
no weight replication). Still random tensors; the model gets plugged in at C4.

Run (box, needs >=2 GPUs):  python 03_ring_2gpu.py
"""
import math
import torch
import torch.nn.functional as F


def ring_step(q, kj, vj, m, l, O):
    """One online-softmax update: fold K/V block (kj,vj) into running (m,l,O). All fp32 accum.
    q:(B,H,Sq,D) on q.device; kj,vj already moved to q.device."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    Sj = (torch.matmul(q, kj.transpose(-1, -2)) * scale).float()
    m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
    corr = torch.exp(m - m_new)
    Pj = torch.exp(Sj - m_new)
    l = l * corr + Pj.sum(dim=-1, keepdim=True)
    O = O * corr + torch.matmul(Pj, vj.float())
    return m_new, l, O


def ring_attention_2gpu(q, k, v, d0="cuda:0", d1="cuda:1"):
    """q,k,v: (B,H,S,D). Split S in half across d0,d1; 2-pass ring; return full output on d0."""
    B, H, S, D = q.shape
    h = S // 2
    dt = q.dtype

    def init(dev, Sq):
        return (torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, 1), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, D), dtype=torch.float32, device=dev))

    # place slices on their home devices
    q0, k0, v0 = q[:, :, :h].to(d0), k[:, :, :h].to(d0), v[:, :, :h].to(d0)
    q1, k1, v1 = q[:, :, h:].to(d1), k[:, :, h:].to(d1), v[:, :, h:].to(d1)

    # device 0: Q0 attends local(K0,V0) then remote(K1,V1 -> d0)
    m0, l0, O0 = init(d0, h)
    m0, l0, O0 = ring_step(q0, k0, v0, m0, l0, O0)                 # pass 1 (local)
    m0, l0, O0 = ring_step(q0, k1.to(d0), v1.to(d0), m0, l0, O0)   # pass 2 (ring: from d1)
    out0 = (O0 / l0).to(dt)

    # device 1: Q1 attends local(K1,V1) then remote(K0,V0 -> d1)
    m1, l1, O1 = init(d1, S - h)
    m1, l1, O1 = ring_step(q1, k1, v1, m1, l1, O1)                 # pass 1 (local)
    m1, l1, O1 = ring_step(q1, k0.to(d1), v0.to(d1), m1, l1, O1)   # pass 2 (ring: from d0)
    out1 = (O1 / l1).to(dt)

    return torch.cat([out0, out1.to(d0)], dim=2)                  # reassemble on d0


def check(dtype, S=384, H=4, D=64):
    torch.manual_seed(0)
    q = torch.randn(1, H, S, D, dtype=dtype)
    k = torch.randn(1, H, S, D, dtype=dtype)
    v = torch.randn(1, H, S, D, dtype=dtype)
    ref = F.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=False).cpu()
    ours = ring_attention_2gpu(q, k, v).cpu()
    d = (ours.float() - ref.float()).abs()
    maxd, meand = d.max().item(), d.mean().item()
    tol = 1e-4 if dtype == torch.float32 else 1.2e-2
    ok = maxd < tol
    print(f"[C3] {str(dtype).split('.')[-1]:>8}: 2-GPU ring vs full SDPA  max={maxd:.2e} mean={meand:.2e} "
          f"tol={tol:.0e} -> {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main():
    n = torch.cuda.device_count()
    print(f"[C3] GPUs visible: {n}")
    if n < 2:
        print("[C3] need >=2 GPUs — abort"); return
    res = [check(torch.float32), check(torch.bfloat16)]
    print(f"[C3] {'ALL PASS ✅ — device->device ring + online softmax works. Two cards, one attention.' if all(res) else 'FAIL ❌ — debug the ring accumulation.'}")
    print("C3-DONE")


if __name__ == "__main__":
    main()
