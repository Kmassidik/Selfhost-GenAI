#!/usr/bin/env python3
"""Dalang E1h: full end-to-end 15s (345f) single-card = winning denoise + FAST, VISIBLE VAE.

The E1g denoise fits (proven). The VAE decode was crawling under leaf_level offload
(streams every layer CPU<->GPU across thousands of tile/clip forwards). But the VAE
already chunks temporally (_decode_clip) + tiles spatially (256px) by design, so it's
small. Fix: put the VAE straight on cuda (transformer weights are offloaded by decode
time) to kill the thrash, and log each decode clip so we can SEE progress.
"""
import argparse, os, sys, torch
import torch.nn as nn
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def chunk_apply(o, x, chunk, *a, **k):
    if not torch.is_tensor(x) or x.dim() < 2 or x.shape[-2] <= chunk:
        return o(x, *a, **k)
    S = x.shape[-2]
    first = o(x[..., :chunk, :], *a, **k)
    out = torch.empty((*x.shape[:-2], S, first.shape[-1]), dtype=first.dtype, device=first.device)
    out[..., :chunk, :] = first; del first
    for i in range(chunk, S, chunk):
        out[..., i:i+chunk, :] = o(x[..., i:i+chunk, :], *a, **k)
    return out

def patch_ffn_chunk(model, chunk=1024):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".ff") and hasattr(mod, "forward"):
            o = mod.forward; mod.forward = (lambda o: (lambda x, *a, **k: chunk_apply(o, x, chunk, *a, **k)))(o); n += 1
    return n

def patch_attn_linear_chunk(model, chunk=2048, min_in=1024):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".attn" in name and mod.in_features >= min_in:
            o = mod.forward; mod.forward = (lambda o: (lambda x, *a, **k: chunk_apply(o, x, chunk, *a, **k)))(o); n += 1
    return n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cena"); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--saida", required=True); ap.add_argument("--rchunk", type=int, default=4096)
    a = ap.parse_args()

    import diffusers, diffusers.hooks
    import diffusers.models.transformers.transformer_minimax_h3 as h3mod

    # --- denoise: the winning three chunkers ---
    _orig_rotary = h3mod._apply_rotary_emb
    def rotary_chunked(hs, cos, sin, _c=a.rchunk):
        S = hs.shape[1]
        if S <= _c: return _orig_rotary(hs, cos, sin)
        out = torch.empty_like(hs)
        for i in range(0, S, _c):
            out[:, i:i+_c] = _orig_rotary(hs[:, i:i+_c], cos[i:i+_c], sin[i:i+_c])
        return out
    h3mod._apply_rotary_emb = rotary_chunked
    _fp = diffusers.MiniMaxH3Transformer3DModel.from_pretrained.__func__
    def patched(cls, *A, **K):
        m = _fp(cls, *A, **K)
        print(f"[e1h] denoise chunkers: rotary + {patch_ffn_chunk(m)} FFN + {patch_attn_linear_chunk(m)} attn", flush=True); return m
    diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)

    # --- VAE: kill the leaf-offload thrash + log each clip ---
    _orig_ago = diffusers.hooks.apply_group_offloading
    def ago(*A, **K):
        model = A[0] if A else (K.get("module") or K.get("model"))
        if model is not None and hasattr(model, "_decode_clip"):   # the video VAE only
            _dc = model._decode_clip; c = {"i": 0}; saved = (A, K)
            def dc(z, *aa, **kk):
                c["i"] += 1
                if c["i"] == 1:   # first clip: transformer is offloaded now, so there's room
                    try:
                        model.to("cuda"); print("[e1h-vae] VAE moved to cuda at DECODE time (no thrash)", flush=True)
                    except Exception as e:
                        print(f"[e1h-vae] cuda OOM at decode ({e}); leaf-offload fallback", flush=True)
                        _orig_ago(*saved[0], **saved[1])
                        z = z.to("cpu") if False else z
                print(f"[e1h-vae] decode clip {c['i']}  z={tuple(z.shape)}  gpu={torch.cuda.memory_allocated(0)/1e9:.2f}GB", flush=True)
                return _dc(z, *aa, **kk)
            model._decode_clip = dc
            print("[e1h-vae] VAE offload DEFERRED to decode time (+per-clip log)", flush=True)
            return   # do NOT offload at setup — the transformer still owns the card here
        return _orig_ago(*A, **K)
    diffusers.hooks.apply_group_offloading = ago

    sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()

if __name__ == "__main__":
    main()
