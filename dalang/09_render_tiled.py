#!/usr/bin/env python3
"""Dalang E1e: single-card 345f = FFN-chunk + rotary-chunk + attn-linear-chunk.

Each big int8 linear makes two full-S bf16 transients (m=X@Wt, y=m*scale ~430MB@345f).
E1d cleared rotary; the next 422MB straw was the attn projection dequant. Fix: chunk
the ATTENTION linears (to_q/k/v/to_out, not .ff) over tokens so m,y are chunk-sized.
All three chunkers are per-token -> correct by construction. Does 15s fit ONE card now?
"""
import argparse, os, sys, torch
import torch.nn as nn
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def patch_ffn_chunk(model, chunk=2048):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".ff") and hasattr(mod, "forward"):
            orig = mod.forward
            def make(o):
                def fwd(x, *a, **k):
                    if x.dim() < 2 or x.shape[-2] <= chunk: return o(x, *a, **k)
                    return torch.cat([o(x[..., i:i+chunk, :], *a, **k) for i in range(0, x.shape[-2], chunk)], dim=-2)
                return fwd
            mod.forward = make(orig); n += 1
    return n

def patch_attn_linear_chunk(model, chunk=8192, min_in=1024):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".attn" in name and mod.in_features >= min_in:
            orig = mod.forward
            def make(o):
                def fwd(x, *a, **k):
                    if x.dim() < 2 or x.shape[-2] <= chunk: return o(x, *a, **k)
                    return torch.cat([o(x[..., i:i+chunk, :], *a, **k) for i in range(0, x.shape[-2], chunk)], dim=-2)
                return fwd
            mod.forward = make(orig); n += 1
    return n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cena"); ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--saida", required=True); ap.add_argument("--rchunk", type=int, default=8192)
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
        nf, na = patch_ffn_chunk(m), patch_attn_linear_chunk(m)
        print(f"[e1e] chunked: rotary + {nf} FFN + {na} attn-linears", flush=True); return m
    diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)

    sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()

if __name__ == "__main__":
    main()
