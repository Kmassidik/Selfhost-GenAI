#!/usr/bin/env python3
"""Dalang G2: sequence-parallel ONE block across N GPUs (default 3), match single-card.
Generalizes G1: shard S/N across cuda:0..N-1, per-token ops local, ring attention gathers
ALL shards to each card. N=3 is the config that fits native 720p x 15s (~31k tokens/card).
"""
import os, sys, math, copy, argparse, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
REF = "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench/dalang_ref"

def log(m): print(f"[g2] {m}", flush=True)

def fold_kv(q, k, v, m, l, O, bk=1024):
    scale = 1.0 / math.sqrt(q.shape[-1])
    for j in range(0, k.shape[2], bk):
        kj, vj = k[:, :, j:j+bk], v[:, :, j:j+bk]
        Sj = torch.matmul(q.float(), kj.float().transpose(-1, -2)) * scale
        m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m - m_new); Pj = torch.exp(Sj - m_new)
        l = l * corr + Pj.sum(dim=-1, keepdim=True)
        O = O * corr + torch.matmul(Pj, vj.float()); m = m_new
    return m, l, O

def attend_shards(q, kv_shards, bk=1024, bq=1024):
    """q attends over a list of (k,v) shards already on q's device."""
    B, H, Q, Dh = q.shape
    out = torch.empty((B, H, Q, Dh), dtype=q.dtype, device=q.device)
    for i in range(0, Q, bq):
        qb = q[:, :, i:i+bq]; n = qb.shape[2]
        m = torch.full((B, H, n, 1), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((B, H, n, 1), dtype=torch.float32, device=q.device)
        O = torch.zeros((B, H, n, Dh), dtype=torch.float32, device=q.device)
        for k, v in kv_shards:
            m, l, O = fold_kv(qb, k, v, m, l, O, bk)
        out[:, :, i:i+bq] = (O / l).to(q.dtype)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3); a = ap.parse_args()
    N = a.n; devs = [f"cuda:{i}" for i in range(N)]
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    _rot = h3mod._apply_rotary_emb
    ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in","audio_proj_in","context_embedder","time_embedder",
                "time_proj","token_refiner","norm_out","proj_out","audio_proj_out"]))
    tr.requires_grad_(False)
    io = torch.load(f"{REF}/blk0_io.pt", weights_only=False)
    H = tr.transformer_blocks[0].attn.heads
    hidden = io["hidden_states"]; temb = io["temb"]; adaln = io["adaln_indices"]
    cos, sin = io["rotary_emb"]; ref = io["__out__"].float().cpu()
    S = hidden.shape[1]; log(f"N={N} devs={devs} S={S} heads={H}")

    # baseline (single-card, exact)
    b0 = tr.transformer_blocks[0].to(devs[0])
    with torch.no_grad():
        base = b0(hidden.to(devs[0]), temb.to(devs[0]), adaln.to(devs[0]),
                  (cos.to(devs[0]), sin.to(devs[0]))).float().cpu()
    log(f"baseline vs captured: max={(base-ref).abs().max():.3e}")

    # N block copies, one per device
    blocks = [b0] + [copy.deepcopy(tr.transformer_blocks[0]).to(devs[i]) for i in range(1, N)]

    # even sequence split
    bounds = [round(k*S/N) for k in range(N+1)]
    def pre(blk, dev, lo, hi):
        hid = hidden[:, lo:hi].to(dev); ad = adaln[lo:hi].to(dev)
        c = cos[lo:hi].to(dev); s = sin[lo:hi].to(dev); tb = temb.to(dev)
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = blk.adaln_proj(tb)
        residual = hid
        nh = blk.norm1(hid); nh = nh*(1+sc_msa.index_select(0, ad)) + sh_msa.index_select(0, ad)
        q = blk.attn.norm_q(blk.attn.to_q(nh).unflatten(-1, (H, -1)))
        k = blk.attn.norm_k(blk.attn.to_k(nh).unflatten(-1, (H, -1)))
        v = blk.attn.to_v(nh).unflatten(-1, (H, -1))
        q = _rot(q, c, s); k = _rot(k, c, s)
        q = q.transpose(1,2).contiguous(); k = k.transpose(1,2).contiguous(); v = v.transpose(1,2).contiguous()
        return residual, (g_msa, sh_mlp, sc_mlp, g_mlp, ad), q, k, v
    def post(blk, residual, packed, attn_bhsd):
        g_msa, sh_mlp, sc_mlp, g_mlp, ad = packed
        o = attn_bhsd.transpose(1,2).flatten(2,3); o = blk.attn.to_out[0](o)
        hid = residual + g_msa.index_select(0, ad)*o; residual = hid
        nh = blk.norm2(hid); nh = nh*(1+sc_mlp.index_select(0, ad)) + sh_mlp.index_select(0, ad)
        return residual + g_mlp.index_select(0, ad)*blk.ff(nh)

    with torch.no_grad():
        pres, qs, ks, vs = [], [], [], []
        for i in range(N):
            r, p, q, k, v = pre(blocks[i], devs[i], bounds[i], bounds[i+1])
            pres.append((r, p)); qs.append(q); ks.append(k); vs.append(v)
        outs = []
        for i in range(N):
            kv = [(ks[j].to(devs[i]), vs[j].to(devs[i])) for j in range(N)]   # gather all shards
            attn_i = attend_shards(qs[i], kv)
            outs.append(post(blocks[i], pres[i][0], pres[i][1], attn_i).float().cpu())
        sp = torch.cat(outs, dim=1)

    d = (sp - ref).abs()
    relt = d[0].max(dim=-1).values / (ref[0].abs().max(dim=-1).values + 1e-6)
    log(f"SEQ-PARALLEL-{N}GPU vs captured: abs max={d.max():.3e} mean={d.mean():.3e}")
    log(f"per-token REL err: max={relt.max():.4%} mean={relt.mean():.4%} p99={relt.float().quantile(0.99):.4%}")
    log("G2-DONE")

if __name__ == "__main__":
    main()
