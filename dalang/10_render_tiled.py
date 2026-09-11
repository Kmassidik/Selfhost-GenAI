#!/usr/bin/env python3
"""Dalang E1f: single-card 345f — same three chunkers but PREALLOCATE outputs.

E1e died on the chunk wrapper's own torch.cat (holds the chunk list AND allocates a
2nd full-S tensor to cat into = ~2x the output, the 422MB we were short). Fix: write
chunks into a preallocated buffer (the trick that got rotary through). No cat doubling.
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

def patch_ffn_chunk(model, chunk=2048):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".ff") and hasattr(mod, "forward"):
            orig = mod.forward
            def make(o):
                return lambda x, *a, **k: chunk_apply(o, x, chunk, *a, **k)
            mod.forward = make(orig); n += 1
    return n

def patch_attn_linear_chunk(model, chunk=8192, min_in=1024):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".attn" in name and mod.in_features >= min_in:
            orig = mod.forward
            def make(o):
                return lambda x, *a, **k: chunk_apply(o, x, chunk, *a, **k)
            mod.forward = make(orig); n += 1
    return n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cena"); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--saida", required=True); ap.add_argument("--rchunk", type=int, default=4096)
    ap.add_argument("--achunk", type=int, default=2048); ap.add_argument("--fchunk", type=int, default=1024)
    a = ap.parse_args()

    import diffusers.models.transformers.transformer_minimax_h3 as h3mod
    _orig_rotary = h3mod._apply_rotary_emb
    def rotary_chunked(hidden_states, cos, sin, _c=a.rchunk):
        S = hidden_states.shape[1]
        if S <= _c: return _orig_rotary(hidden_states, cos, sin)
        out = torch.empty_like(hidden_states)
        for i in range(0, S, _c):
            out[:, i:i+_c] = _orig_rotary(hidden_states[:, i:i+_c], cos[i:i+_c], sin[i:i+_c])
        return out
    h3mod._apply_rotary_emb = rotary_chunked

    import diffusers
    _fp = diffusers.MiniMaxH3Transformer3DModel.from_pretrained.__func__
    def patched(cls, *A, **K):
        m = _fp(cls, *A, **K)
        nf, na = patch_ffn_chunk(m, a.fchunk), patch_attn_linear_chunk(m, a.achunk)
        print(f"[e1f] preallocated chunkers: rotary + {nf} FFN + {na} attn-linears", flush=True); return m
    diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)

    sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()

if __name__ == "__main__":
    main()
