#!/usr/bin/env python3
"""Dalang Engine — C4c: is the C4b bug bf16 matmul precision in the ring?

Compare, on the FIRST real attention block:
  (a) H3 original dispatch_attention_fn   (reference)
  (b) our ring, bf16 internal matmuls     (current — C4b showed max 0.625)
  (c) our ring, FP32 internal matmuls     (upcast q,k,v -> fp32 in the ring)
  (d) plain F.scaled_dot_product_attention
If (c) matches (a) closely but (b) doesn't -> the bug is bf16 QKᵀ precision;
fix = do the ring's matmuls in fp32 (flash accumulates in fp32 anyway).

Run:  python 04c_precision.py --ref ./dalang_ref
"""
import argparse, math, os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch
import torch.nn.functional as F

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
D1 = "cuda:1"


def log(m): print(f"[dalang-s2c] {m}", flush=True)


def ring_step(q, kj, vj, m, l, O, cdt):
    scale = 1.0 / math.sqrt(q.shape[-1])
    Sj = (torch.matmul(q.to(cdt), kj.to(cdt).transpose(-1, -2)) * scale).float()
    m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
    corr = torch.exp(m - m_new)
    Pj = torch.exp(Sj - m_new)
    l = l * corr + Pj.sum(dim=-1, keepdim=True)
    O = O * corr + torch.matmul(Pj, vj.float())
    return m_new, l, O


def fold_kv(q, k, v, m, l, O, cdt, block=512):
    for j0 in range(0, k.shape[2], block):
        m, l, O = ring_step(q, k[:, :, j0:j0 + block], v[:, :, j0:j0 + block], m, l, O, cdt)
    return m, l, O


def ring(q, k, v, d0, cdt, block=512):
    B, H, S, D = q.shape; h = S // 2; dt = q.dtype
    def init(dev, Sq):
        return (torch.full((B, H, Sq, 1), float("-inf"), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, 1), dtype=torch.float32, device=dev),
                torch.zeros((B, H, Sq, D), dtype=torch.float32, device=dev))
    q0, k0, v0 = q[:, :, :h], k[:, :, :h], v[:, :, :h]
    q1, k1, v1 = q[:, :, h:].to(D1), k[:, :, h:].to(D1), v[:, :, h:].to(D1)
    m0, l0, O0 = init(d0, h)
    m0, l0, O0 = fold_kv(q0, k0, v0, m0, l0, O0, cdt, block)
    m0, l0, O0 = fold_kv(q0, k1.to(d0), v1.to(d0), m0, l0, O0, cdt, block)
    out0 = (O0 / l0).to(dt)
    m1, l1, O1 = init(D1, S - h)
    m1, l1, O1 = fold_kv(q1, k1, v1, m1, l1, O1, cdt, block)
    m1, l1, O1 = fold_kv(q1, k0.to(D1), v0.to(D1), m1, l1, O1, cdt, block)
    out1 = (O1 / l1).to(dt)
    return torch.cat([out0, out1.to(d0)], dim=2)


def ring_dispatch(query, key, value, cdt):
    d0 = query.device
    q = query.transpose(1, 2).contiguous(); k = key.transpose(1, 2).contiguous(); v = value.transpose(1, 2).contiguous()
    return ring(q, k, v, d0, cdt).transpose(1, 2).contiguous()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="./dalang_ref"); args = ap.parse_args()
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    orig = h3mod.dispatch_attention_fn
    state = {}

    def diag(query, key, value, **kw):
        if "done" not in state:
            ref = orig(query, key, value, **kw).float()
            ours_bf16 = ring_dispatch(query, key, value, torch.bfloat16).float()
            ours_fp32 = ring_dispatch(query, key, value, torch.float32).float()
            qh, kh, vh = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            plain = F.scaled_dot_product_attention(qh, kh, vh, is_causal=False).transpose(1, 2).float()
            def dd(x): d = (x - ref).abs(); return f"max={d.max().item():.3e} mean={d.mean().item():.3e}"
            log(f"(b) bf16-ring vs orig : {dd(ours_bf16)}")
            log(f"(c) fp32-ring vs orig : {dd(ours_fp32)}")
            log(f"(d) plainSDPA vs orig : {dd(plain)}")
            log(f"    bf16-ring vs plainSDPA: {(ours_bf16-plain).abs().max().item():.3e}")
            log(f"    fp32-ring vs plainSDPA: {(ours_fp32-plain).abs().max().item():.3e}")
            state["done"] = True
            raise RuntimeError("C4C_STOP")
        return orig(query, key, value, **kw)

    h3mod.dispatch_attention_fn = diag
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    t0 = time.time()
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out"]))
    tr.requires_grad_(False)
    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    log(f"loaded in {time.time()-t0:.0f}s")
    ref_in = torch.load(f"{args.ref}/ref_in.pt", weights_only=False)
    try:
        with torch.no_grad():
            tr(**dict(ref_in["kwargs"]))
    except RuntimeError as e:
        if "C4C_STOP" not in str(e): raise
    log("C4C-DONE")


if __name__ == "__main__":
    main()
