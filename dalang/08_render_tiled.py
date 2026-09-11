#!/usr/bin/env python3
"""Dalang E1d: single-card 345f = FFN-chunk + rotary-chunk.

E1c proved FFN-chunk clears the FFN wall but 345f then OOMs inside
_apply_rotary_emb (full-S query + ~3 full rotary transient copies). Fix: chunk
the rotary over the sequence so its transients are chunk-sized. Correct by
construction (rotary is per-token). Combined with FFN-chunk -> does 15s fit ONE card?
"""
import argparse, os, sys, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

def patch_ffn_chunk(model, chunk=2048):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".ff") and hasattr(mod, "forward"):
            orig = mod.forward
            def make(orig):
                def fwd(x, *a, **k):
                    if x.dim() < 2 or x.shape[-2] <= chunk: return orig(x, *a, **k)
                    return torch.cat([orig(x[..., i:i+chunk, :], *a, **k) for i in range(0, x.shape[-2], chunk)], dim=-2)
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
    print(f"[e1d] rotary chunked (chunk={a.rchunk} tokens)", flush=True)

    # apply FFN chunk right after the transformer loads
    import diffusers
    _fp = diffusers.MiniMaxH3Transformer3DModel.from_pretrained.__func__
    def patched(cls, *A, **K):
        m = _fp(cls, *A, **K); print(f"[e1d] FFN-chunked {patch_ffn_chunk(m)} modules", flush=True); return m
    diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)

    sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
    sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
    import h3_denoise_local
    h3_denoise_local.main()

if __name__ == "__main__":
    main()
