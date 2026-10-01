#!/usr/bin/env python3
"""Dalang G1: sequence-parallel ONE block across 2 GPUs, match the single-card block.

Anchor: dalang_ref/blk0_io.pt (real block-0 I/O, from G1a). We shard hidden/adaln_indices/
rotary by sequence S/2 across cuda:0/1, run every PER-TOKEN op locally on each card's slice,
and the ONE cross-token op (self.attn) gathers across cards via the C5 ring. Then reassemble
and compare to the captured single-card output. Correct => the summit's core mechanism works.
"""
import os, sys, math, copy, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
REF = "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench/dalang_ref"
D0, D1 = "cuda:0", "cuda:1"

def log(m): print(f"[g1] {m}", flush=True)

# ---- C5 ring: one Q-tile attends the whole (local+remote) K/V, tiled, fp32 scores ----
def fold_kv(q, k, v, m, l, O, bk=1024):
    scale = 1.0 / math.sqrt(q.shape[-1])
    for j in range(0, k.shape[2], bk):
        kj, vj = k[:, :, j:j+bk], v[:, :, j:j+bk]
        Sj = torch.matmul(q.float(), kj.float().transpose(-1, -2)) * scale
        m_new = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m - m_new); Pj = torch.exp(Sj - m_new)
        l = l * corr + Pj.sum(dim=-1, keepdim=True)
        O = O * corr + torch.matmul(Pj, vj.float())
        m = m_new
    return m, l, O

def attend_all(q, kL, vL, kR, vR, bk=1024, bq=1024):
    B, H, Q, Dh = q.shape
    out = torch.empty((B, H, Q, Dh), dtype=q.dtype, device=q.device)
    for i in range(0, Q, bq):
        qb = q[:, :, i:i+bq]
        m = torch.full((B, H, qb.shape[2], 1), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((B, H, qb.shape[2], 1), dtype=torch.float32, device=q.device)
        O = torch.zeros((B, H, qb.shape[2], Dh), dtype=torch.float32, device=q.device)
        m, l, O = fold_kv(qb, kL, vL, m, l, O, bk)
        m, l, O = fold_kv(qb, kR, vR, m, l, O, bk)
        out[:, :, i:i+bq] = (O / l).to(q.dtype)
    return out

def main():
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
    log(f"heads={H}, head_dim={5376//H}")

    b0 = tr.transformer_blocks[0].to(D0)
    hidden = io["hidden_states"].to(D0); temb = io["temb"].to(D0)
    adaln = io["adaln_indices"].to(D0); cos, sin = [t.to(D0) for t in io["rotary_emb"]]
    ref_out = io["__out__"].float()
    S = hidden.shape[1]; h = S // 2
    log(f"S={S}, split {h}/{S-h}")

    # ---- baseline: real single-card block (sanity vs captured output) ----
    with torch.no_grad():
        base = b0(hidden, temb, adaln, (cos, sin)).float().cpu()
    log(f"baseline vs captured: max={ (base-ref_out.cpu()).abs().max():.3e}")

    # ---- second copy of the block on cuda:1 ----
    b1 = copy.deepcopy(tr.transformer_blocks[0]).to(D1)

    def half(dev, sl):
        return (hidden[:, sl].to(dev), adaln[sl].to(dev), cos[sl].to(dev), sin[sl].to(dev),
                temb.to(dev))

    def per_token_pre(blk, hid, tb, ad, c, s):
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = blk.adaln_proj(tb)
        residual = hid
        nh = blk.norm1(hid)
        nh = nh * (1 + sc_msa.index_select(0, ad)) + sh_msa.index_select(0, ad)
        # qkv + heads + norm + rotary -> (B,Hd,s,Dh)
        q = blk.attn.norm_q(blk.attn.to_q(nh).unflatten(-1, (H, -1)))
        k = blk.attn.norm_k(blk.attn.to_k(nh).unflatten(-1, (H, -1)))
        v = blk.attn.to_v(nh).unflatten(-1, (H, -1))
        q = _rot(q, c, s); k = _rot(k, c, s)
        q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous(); v = v.transpose(1, 2).contiguous()
        return residual, (g_msa, sh_mlp, sc_mlp, g_mlp, ad), q, k, v

    def per_token_post(blk, residual, packed, attn_bhsd):
        g_msa, sh_mlp, sc_mlp, g_mlp, ad = packed
        o = attn_bhsd.transpose(1, 2).flatten(2, 3)          # (B,s,5376)
        o = blk.attn.to_out[0](o)
        hid = residual + g_msa.index_select(0, ad) * o
        residual = hid
        nh = blk.norm2(hid)
        nh = nh * (1 + sc_mlp.index_select(0, ad)) + sh_mlp.index_select(0, ad)
        ff = blk.ff(nh)
        return residual + g_mlp.index_select(0, ad) * ff

    with torch.no_grad():
        h0, a0, c0, s0, t0 = half(D0, slice(0, h))
        h1, a1, c1, s1, t1 = half(D1, slice(h, S))
        r0, p0, q0, k0, v0 = per_token_pre(b0, h0, t0, a0, c0, s0)
        r1, p1, q1, k1, v1 = per_token_pre(b1, h1, t1, a1, c1, s1)
        # ring: each card attends local + remote
        o0 = attend_all(q0, k0, v0, k1.to(D0), v1.to(D0))
        o1 = attend_all(q1, k1, v1, k0.to(D1), v0.to(D1))
        out0 = per_token_post(b0, r0, p0, o0)
        out1 = per_token_post(b1, r1, p1, o1)
        sp = torch.cat([out0.float().cpu(), out1.float().cpu()], dim=1)

    ref=ref_out.cpu()
    d = (sp - ref).abs()
    log(f"SEQ-PARALLEL vs captured: max={d.max():.3e} mean={d.mean():.3e}")
    log(f"nan(sp)={torch.isnan(sp).sum().item()} inf(sp)={torch.isinf(sp).sum().item()}")
    tokmax = d[0].max(dim=-1).values           # (S,)
    big = (tokmax > 1.0).nonzero().flatten()
    log(f"tokens with err>1.0: n={big.numel()} split_at={S//2} -> first20={big[:20].tolist()}")
    # global-max position + relative error
    gi=int(d.flatten().argmax()); gp=gi//5376; gc=gi%5376
    log(f"GLOBAL max @ token {gp} ch{gc}: sp={sp.flatten()[gi]:.2f} ref={ref.flatten()[gi]:.2f} tok|max|={ref[0,gp].abs().max():.1f} REL={(d.flatten()[gi]/(ref[0,gp].abs().max()+1e-6)):.4%}")
    # relative error over ALL tokens (per-token max err / per-token max magnitude)
    relt = d[0].max(dim=-1).values / (ref[0].abs().max(dim=-1).values + 1e-6)
    log(f"per-token RELATIVE err: max={relt.max():.4%} mean={relt.mean():.4%} p99={relt.float().quantile(0.99):.4%}")
    log("G1-DONE")

if __name__ == "__main__":
    main()
