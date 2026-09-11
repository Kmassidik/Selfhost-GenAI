#!/usr/bin/env python3
"""Dalang G4: render native 720p x 15s STRAIGHT across 3 GPUs (sequence parallel).
Patches the transformer forward with the proven G3 sharded forward (blocks streamed to
3 cards, hidden sharded S/3), skips the transformer's group-offload (sharded forward
streams blocks itself), keeps the E1h deferred-VAE fix, and runs the full denoise.
"""
import os, sys, math, copy, gc, argparse, torch
import torch.nn as nn
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MOD = 3

def log(m): print(f"[g4] {m}", flush=True)

def fold_kv(q, k, v, m, l, O, bk=2048):
    scale = 1.0/math.sqrt(q.shape[-1])
    for j in range(0, k.shape[2], bk):
        kj, vj = k[:, :, j:j+bk], v[:, :, j:j+bk]
        Sj = torch.matmul(q.float(), kj.float().transpose(-1,-2))*scale
        mn = torch.maximum(m, Sj.max(dim=-1, keepdim=True).values)
        corr = torch.exp(m-mn); Pj = torch.exp(Sj-mn)
        l = l*corr + Pj.sum(dim=-1, keepdim=True); O = O*corr + torch.matmul(Pj, vj.float()); m = mn
    return m, l, O

def attend_rotating(q, local_kv, remote_kv, bk=512, bq=512):
    """q attends local shard then each remote shard IN TURN (bring one, fold, free)."""
    B,H,Q,Dh = q.shape; out = torch.empty((B,H,Q,Dh), dtype=q.dtype, device=q.device)
    for i in range(0, Q, bq):
        qb = q[:,:,i:i+bq]; n = qb.shape[2]
        m = torch.full((B,H,n,1), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((B,H,n,1), dtype=torch.float32, device=q.device)
        O = torch.zeros((B,H,n,Dh), dtype=torch.float32, device=q.device)
        m,l,O = fold_kv(qb, local_kv[0], local_kv[1], m, l, O, bk)
        for (kr, vr) in remote_kv:
            kk = kr.to(q.device); vv = vr.to(q.device)
            m,l,O = fold_kv(qb, kk, vv, m, l, O, bk); del kk, vv
        out[:,:,i:i+bq] = (O/l).to(q.dtype)
    return out

def _chunk_apply(o, x, chunk, *a, **k):
    if not torch.is_tensor(x) or x.dim() < 2 or x.shape[-2] <= chunk: return o(x, *a, **k)
    S = x.shape[-2]; first = o(x[..., :chunk, :], *a, **k)
    out = torch.empty((*x.shape[:-2], S, first.shape[-1]), dtype=first.dtype, device=first.device)
    out[..., :chunk, :] = first; del first
    for i in range(chunk, S, chunk): out[..., i:i+chunk, :] = o(x[..., i:i+chunk, :], *a, **k)
    return out

def chunk_block(blk, ffc=1024, attnc=1024):
    for name, mod in blk.named_modules():
        if getattr(mod, "_ck", False): continue
        if name == "ff" or name.endswith(".ff"):
            o = mod.forward; mod.forward = (lambda o, c: (lambda x, *a, **k: _chunk_apply(o, x, c, *a, **k)))(o, ffc); mod._ck = True
        elif isinstance(mod, nn.Linear) and (name.startswith("attn.") or ".attn." in name) and mod.in_features >= 1024:
            o = mod.forward; mod.forward = (lambda o, c: (lambda x, *a, **k: _chunk_apply(o, x, c, *a, **k)))(o, attnc); mod._ck = True

def make_sharded_forward(N=3):
    devs = [f"cuda:{i}" for i in range(N)]
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    _rot = h3mod._apply_rotary_emb; Out = h3mod.MiniMaxH3TransformerOutput
    st = {"moved": False}
    def fwd(self, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
            timestep_indices, token_tags, position_ids, video_indices, audio_indices,
            text_indices, attention_kwargs=None, return_dict=True):
        d0 = devs[0]; orig_dev = hidden_states.device
        if not st["moved"]:
            for nm in ["proj_in","audio_proj_in","context_embedder","token_refiner","time_proj",
                       "time_embedder","rope","norm_out","proj_out","audio_proj_out"]:
                getattr(self, nm).to(d0)
            st["moved"] = True; log("pre/post modules on cuda:0; blocks stream from CPU")
        g = lambda t: t.to(d0) if torch.is_tensor(t) else t
        hidden_states, audio_hidden_states, encoder_hidden_states = g(hidden_states), g(audio_hidden_states), g(encoder_hidden_states)
        timestep, timestep_indices, token_tags = g(timestep), g(timestep_indices), g(token_tags)
        position_ids = g(position_ids)
        video_indices, audio_indices, text_indices = g(video_indices), g(audio_indices), g(text_indices)
        H = self.transformer_blocks[0].attn.heads
        seqlen = position_ids.shape[0]
        cos, sin = self.rope(position_ids)
        ve = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        ae = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        te = self.token_refiner(self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype)))
        hid = te.new_zeros((te.shape[0], seqlen, te.shape[-1]))
        hid = hid.index_copy(1, text_indices, te)
        hid = hid.index_copy(1, video_indices, ve.to(te.dtype))
        hid = hid.index_copy(1, audio_indices, ae.to(te.dtype))
        del ve, ae, te; torch.cuda.empty_cache()          # free packing embeds off cuda:0 immediately
        temb = self.time_embedder(self.time_proj(timestep).to(self.time_embedder.linear_1.weight.dtype))
        adaln = timestep_indices*MOD + token_tags
        b = [round(x*seqlen/N) for x in range(N+1)]
        Hd = [hid[:, b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Ad = [adaln[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Cd = [cos[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Sd = [sin[b[i]:b[i+1]].to(devs[i]) for i in range(N)]
        Td = [temb.to(devs[i]) for i in range(N)]
        del hid, cos, sin, adaln          # free full-S intermediates off cuda:0 (only shards persist)
        torch.cuda.empty_cache()
        for bi, blk_cpu in enumerate(self.transformer_blocks):
            blk = [copy.deepcopy(blk_cpu).to(devs[i]) for i in range(N)]   # each replica direct CPU->its card
            for i in range(N): chunk_block(blk[i])
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
            for i in range(N): torch.cuda.synchronize(i)
            for i in range(N):
                remote = [(ks[j], vs[j]) for j in range(N) if j != i]
                ao = attend_rotating(qs[i], (ks[i], vs[i]), remote).transpose(1,2).flatten(2,3)
                torch.cuda.synchronize(i)
                g_msa,sh_mlp,sc_mlp,g_mlp,ad = pk[i]
                ho = res[i] + g_msa.index_select(0,ad)*blk[i].attn.to_out[0](ao); r2 = ho
                nh = blk[i].norm2(ho); nh = nh*(1+sc_mlp.index_select(0,ad)) + sh_mlp.index_select(0,ad)
                Hd[i] = r2 + g_mlp.index_select(0,ad)*blk[i].ff(nh)
            for i in range(N): torch.cuda.synchronize(i)
            Hd = [h.detach() for h in Hd]
            del qs, ks, vs, res, pk
            for i in range(N): blk[i].to("cpu")
            del blk; gc.collect()
            for i in range(N): torch.cuda.empty_cache()
            if bi % 10 == 0: log(f"  block {bi}/50")
        vo_parts, ao_parts = [], []
        for i in range(N):                                   # per-shard heads: never hold full-S hidden on cuda:0
            hi = Hd[i].to(d0)
            hi = self.norm_out(hi, temb, timestep_indices[b[i]:b[i+1]]).to(self.proj_out.weight.dtype)
            vo_parts.append(self.proj_out(hi)); ao_parts.append(self.audio_proj_out(hi)); del hi
            torch.cuda.empty_cache()
        video_out = torch.cat(vo_parts, dim=1).index_select(1, video_indices).to(orig_dev)
        audio_out = torch.cat(ao_parts, dim=1).index_select(1, audio_indices).to(orig_dev)
        if not return_dict: return (video_out, audio_out)
        return Out(sample=video_out, audio_sample=audio_out)
    return fwd

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cena"); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--saida", required=True); ap.add_argument("--n", type=int, default=3); a = ap.parse_args()
    import diffusers, diffusers.hooks
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    h3mod.MiniMaxH3Transformer3DModel.forward = make_sharded_forward(a.n)
    log(f"transformer.forward patched -> {a.n}-GPU sequence parallel")
    # after load: neuter the transformer's group-offload (we stream blocks ourselves)
    _fp = diffusers.MiniMaxH3Transformer3DModel.from_pretrained.__func__
    def patched(cls, *A, **K):
        m = _fp(cls, *A, **K); m.enable_group_offload = (lambda *x, **y: log("skip transformer group-offload")); return m
    diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)
    # E1h deferred-VAE fix
    _ago = diffusers.hooks.apply_group_offloading
    def ago(*A, **K):
        model = A[0] if A else (K.get("module") or K.get("model"))
        if model is not None and hasattr(model, "_decode_clip"):
            _dc = model._decode_clip; c={"i":0}; saved=(A,K)
            def dc(z,*aa,**kk):
                c["i"]+=1
                if c["i"]==1:
                    try: model.to("cuda:0"); log("VAE -> cuda:0 at decode")
                    except Exception as e: log(f"VAE cuda OOM {e}; leaf fallback"); _ago(*saved[0], **saved[1])
                log(f"[vae] clip {c['i']}"); return _dc(z,*aa,**kk)
            model._decode_clip = dc; return
        return _ago(*A, **K)
    diffusers.hooks.apply_group_offloading = ago
    # G4K: decode device fix + latent backup (dalang/h3_decode_fix.py).
    # g4j completed all 3 denoise steps (~6h10m) then died on the first line
    # of decode with a cpu/cuda:0 mismatch and lost the whole run.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import h3_decode_fix
    h3_decode_fix.install(os.environ.get("H3_LATENT_BACKUP"), log=log)
    # G4M: also dump latents after EVERY denoise step. At ~110 min/step a
    # 23-step native 720p run is ~42 h; a crash at step 20 would otherwise
    # throw away a day and a half.
    h3_decode_fix.install_step_checkpoint(log=log)
    # G4L: keep finished pixel chunks off the GPU during decode. Without this
    # the inline decode OOMs at ~clip 14 at 1280x704x345 (that is what ended g4k).
    import h3_vae_cpu_accum
    h3_vae_cpu_accum.install(log=log)
    sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()

if __name__ == "__main__":
    main()
