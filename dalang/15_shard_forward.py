#!/usr/bin/env python3
"""Dalang G3: FULL 50-block sequence-parallel forward across N GPUs, vs ref_out anchor.
Build packed hidden on cuda:0 -> shard S/N -> stream each block to all N cards + run sharded
(G2) keeping hidden sharded across blocks -> gather -> norm_out/proj_out. Compare video_out
to dalang_ref/ref_out.pt (the B1 zero-diff anchor). Correct(ish, bf16) => G4 render next.
"""
import os, sys, math, copy, gc, argparse, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
REF = "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench/dalang_ref"
MOD = 3  # MINIMAX_H3_MODALITY_NUM

def log(m): print(f"[g3] {m}", flush=True)

def fold_kv(q, k, v, m, l, O, bk=2048):
    scale = 1.0/math.sqrt(q.shape[-1])
    for j in range(0, k.shape[2], bk):
        kj, vj = k[:, :, j:j+bk], v[:, :, j:j+bk]
        Sj = torch.matmul(q.float(), kj.float().transpose(-1,-2))*scale
        mn = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m-mn); Pj = torch.exp(Sj-mn)
        l = l*corr + Pj.sum(dim=-1, keepdim=True); O = O*corr + torch.matmul(Pj, vj.float()); m = mn
    return m, l, O

def attend_shards(q, kv, bk=2048, bq=2048):
    B,H,Q,Dh = q.shape; out = torch.empty((B,H,Q,Dh), dtype=q.dtype, device=q.device)
    for i in range(0, Q, bq):
        qb = q[:,:,i:i+bq]; n = qb.shape[2]
        m = torch.full((B,H,n,1), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((B,H,n,1), dtype=torch.float32, device=q.device)
        O = torch.zeros((B,H,n,Dh), dtype=torch.float32, device=q.device)
        for k,v in kv: m,l,O = fold_kv(qb,k,v,m,l,O,bk)
        out[:,:,i:i+bq] = (O/l).to(q.dtype)
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
    d0 = devs[0]
    for name in ["proj_in","audio_proj_in","context_embedder","token_refiner","time_proj",
                 "time_embedder","rope","norm_out","proj_out","audio_proj_out"]:
        getattr(tr, name).to(d0)
    H = tr.transformer_blocks[0].attn.heads
    ri = torch.load(f"{REF}/ref_in.pt", weights_only=False)["kwargs"]
    ref_out = torch.load(f"{REF}/ref_out.pt", weights_only=False).float().cpu()
    k = {kk: (v.to(d0) if torch.is_tensor(v) else v) for kk, v in ri.items()}

    with torch.no_grad():
        rotary = tr.rope(k["position_ids"]); cos, sin = rotary
        seqlen = k["position_ids"].shape[0]
        ve = tr.proj_in(k["hidden_states"].to(tr.proj_in.weight.dtype))
        ae = tr.audio_proj_in(k["audio_hidden_states"].to(tr.audio_proj_in.weight.dtype))
        te = tr.token_refiner(tr.context_embedder(k["encoder_hidden_states"].to(tr.context_embedder.weight.dtype)))
        hid = te.new_zeros((te.shape[0], seqlen, te.shape[-1]))
        hid = hid.index_copy(1, k["text_indices"], te)
        hid = hid.index_copy(1, k["video_indices"], ve.to(te.dtype))
        hid = hid.index_copy(1, k["audio_indices"], ae.to(te.dtype))
        temb = tr.time_embedder(tr.time_proj(k["timestep"]).to(tr.time_embedder.linear_1.weight.dtype))
        adaln = k["timestep_indices"]*MOD + k["token_tags"]
        log(f"packed hidden {tuple(hid.shape)}; seqlen={seqlen}")

        # shard across devices
        b = [round(x*seqlen/N) for x in range(N+1)]
        Hd = [hid[:, b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Ad = [adaln[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Cd = [cos[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Sd = [sin[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Td = [temb.to(devs[i]) for i in range(N)]

        for bi, blk_cpu in enumerate(tr.transformer_blocks):
            blk0 = blk_cpu.to(devs[0])                                   # move ORIGINAL to cuda:0 (G1/G2 pattern)
            blk = [blk0] + [copy.deepcopy(blk0).to(devs[i]) for i in range(1, N)]
            res, pk, qs, ks, vs = [], [], [], [], []
            for i in range(N):
                sh_msa,sc_msa,g_msa,sh_mlp,sc_mlp,g_mlp = blk[i].adaln_proj(Td[i])
                r = Hd[i]; nh = blk[i].norm1(Hd[i])
                nh = nh*(1+sc_msa.index_select(0,Ad[i])) + sh_msa.index_select(0,Ad[i])
                q = blk[i].attn.norm_q(blk[i].attn.to_q(nh).unflatten(-1,(H,-1)))
                kk = blk[i].attn.norm_k(blk[i].attn.to_k(nh).unflatten(-1,(H,-1)))
                vv = blk[i].attn.to_v(nh).unflatten(-1,(H,-1))
                q = _rot(q,Cd[i],Sd[i]); kk = _rot(kk,Cd[i],Sd[i])
                res.append(r); pk.append((g_msa,sh_mlp,sc_mlp,g_mlp,Ad[i]))
                qs.append(q.transpose(1,2).contiguous()); ks.append(kk.transpose(1,2).contiguous()); vs.append(vv.transpose(1,2).contiguous())
            for i in range(N):
                kv = [(ks[j].to(devs[i]), vs[j].to(devs[i])) for j in range(N)]
                torch.cuda.synchronize(i)                                # finish cross-device moves before use
                ao = attend_shards(qs[i], kv).transpose(1,2).flatten(2,3)
                g_msa,sh_mlp,sc_mlp,g_mlp,ad = pk[i]
                ho = res[i] + g_msa.index_select(0,ad)*blk[i].attn.to_out[0](ao); r2 = ho
                nh = blk[i].norm2(ho); nh = nh*(1+sc_mlp.index_select(0,ad)) + sh_mlp.index_select(0,ad)
                Hd[i] = r2 + g_mlp.index_select(0,ad)*blk[i].ff(nh)
            for i in range(N): torch.cuda.synchronize(i)                # ALL kernels done before we free
            Hd = [h.detach() for h in Hd]
            del qs, ks, vs, res, pk
            for i in range(1, N): blk[i].to("cpu")                       # pull replicas off their GPUs
            del blk, blk0
            blk_cpu.to("cpu")
            gc.collect()
            for i in range(N): torch.cuda.empty_cache()
            if bi < 3 or bi % 10 == 0:
                mem = " ".join(f"g{i}:{torch.cuda.memory_allocated(i)/1e9:.2f}" for i in range(N))
                log(f"block {bi} done  mem[{mem}]")

        hid = torch.cat([Hd[i].to(d0) for i in range(N)], dim=1)
        hid = tr.norm_out(hid, temb, k["timestep_indices"]).to(tr.proj_out.weight.dtype)
        video_out = tr.proj_out(hid).index_select(1, k["video_indices"]).float().cpu()

    d = (video_out - ref_out).abs()
    rel = d[0].max(dim=-1).values / (ref_out[0].abs().max(dim=-1).values + 1e-6)
    log(f"FULL SP-{N}GPU forward vs ref_out: abs max={d.max():.3e} mean={d.mean():.3e}")
    log(f"per-token REL: max={rel.max():.4%} mean={rel.mean():.4%} p99={rel.float().quantile(0.99):.4%}")
    log("G3-DONE")

if __name__ == "__main__":
    main()
