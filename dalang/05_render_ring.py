#!/usr/bin/env python3
"""Dalang Engine — C5: render a REAL clip with our ring, to prove visual equivalence.

Patches H3's attention with our 2-GPU ring, then runs the FULL proven denoise
(h3_denoise_local.py) — transformer + VAE + audio — to produce an actual .mp4.
Render the same scene/seed/steps once WITH the ring and once normal; if the clips
look identical, the ring is validated for real use (the end-to-end 0.017 tensor
diff is imperceptible, exactly like the int8-vs-fp16 plateau).

Run (box, h3-consumer-bench dir, >=2 GPUs):
    python 05_render_ring.py <cena> --steps 6 --saida ./h3-lab/<name>.mp4        # WITH ring
    python 05_render_ring.py <cena> --steps 6 --saida ./h3-lab/<name>.mp4 --nopatch  # normal
"""
import argparse
import math
import os
import sys

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

D1 = "cuda:1"


def ring_step(q, kj, vj, m, l, O):
    scale = 1.0 / math.sqrt(q.shape[-1])
    Sj = torch.matmul(q.float(), kj.float().transpose(-1, -2)) * scale   # fp32 scores (the C4c fix)
    m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
    corr = torch.exp(m - m_new)
    Pj = torch.exp(Sj - m_new)
    l = l * corr + Pj.sum(dim=-1, keepdim=True)
    O = O * corr + torch.matmul(Pj, vj.float())
    return m_new, l, O


def fold_kv(q, k, v, m, l, O, block_k=512):
    for j0 in range(0, k.shape[2], block_k):
        m, l, O = ring_step(q, k[:, :, j0:j0 + block_k], v[:, :, j0:j0 + block_k], m, l, O)
    return m, l, O


def attend_all(qb, k_local, v_local, k_remote, v_remote, block_k):
    """One Q-tile qb attends the whole sequence (local + remote K/V), tiled over K."""
    B, H, Qb, D = qb.shape
    m = torch.full((B, H, Qb, 1), float("-inf"), dtype=torch.float32, device=qb.device)
    l = torch.zeros((B, H, Qb, 1), dtype=torch.float32, device=qb.device)
    O = torch.zeros((B, H, Qb, D), dtype=torch.float32, device=qb.device)
    m, l, O = fold_kv(qb, k_local, v_local, m, l, O, block_k)
    m, l, O = fold_kv(qb, k_remote, v_remote, m, l, O, block_k)
    return (O / l).to(qb.dtype)


def ring_attention_2gpu(q, k, v, d0, d1=D1, block_q=1024, block_k=512):
    """Flash-style TILED over BOTH Q and K so scores are (Qtile, Ktile) — memory-frugal."""
    B, H, S, D = q.shape
    h = S // 2
    dt = q.dtype
    q0, k0, v0 = q[:, :, :h], k[:, :, :h], v[:, :, :h]
    q1, k1, v1 = q[:, :, h:].to(d1), k[:, :, h:].to(d1), v[:, :, h:].to(d1)
    k1_0, v1_0 = k1.to(d0), v1.to(d0)   # remote half -> d0 (once)
    k0_1, v0_1 = k0.to(d1), v0.to(d1)   # local half  -> d1 (once)

    out0 = torch.empty((B, H, h, D), dtype=dt, device=d0)
    for qi in range(0, h, block_q):
        out0[:, :, qi:qi + block_q] = attend_all(q0[:, :, qi:qi + block_q], k0, v0, k1_0, v1_0, block_k)
    out1 = torch.empty((B, H, S - h, D), dtype=dt, device=d1)
    for qi in range(0, S - h, block_q):
        out1[:, :, qi:qi + block_q] = attend_all(q1[:, :, qi:qi + block_q], k1, v1, k0_1, v0_1, block_k)
    return torch.cat([out0, out1.to(d0)], dim=2)


def ring_dispatch(query, key, value, **kw):
    d0 = query.device
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = ring_attention_2gpu(q, k, v, d0=d0)
    return out.transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cena")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--nopatch", action="store_true")
    a = ap.parse_args()

    if not a.nopatch:
        import diffusers.models.transformers.transformer_minimax_h3 as h3mod
        h3mod.dispatch_attention_fn = ring_dispatch
        print("[dalang-render] attention PATCHED -> Dalang 2-GPU ring", flush=True)
    else:
        print("[dalang-render] normal engine (no patch)", flush=True)

    # run the proven denoise, unchanged, with our patch already in place
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench"):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()


if __name__ == "__main__":
    main()
