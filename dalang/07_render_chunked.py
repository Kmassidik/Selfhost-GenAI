#!/usr/bin/env python3
"""Dalang E1c: render with FFN chunking — does 15s (345f) fit ONE card now?"""
import os, sys, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/dalang")
from importlib import import_module

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

# monkeypatch MiniMaxH3Transformer3DModel.from_pretrained to apply FFN chunking after load
import diffusers
_orig = diffusers.MiniMaxH3Transformer3DModel.from_pretrained.__func__
def patched(cls, *a, **k):
    m = _orig(cls, *a, **k)
    print(f"[e1c] FFN-chunked {patch_ffn_chunk(m)} modules", flush=True)
    return m
diffusers.MiniMaxH3Transformer3DModel.from_pretrained = classmethod(patched)

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("cena"); ap.add_argument("--steps", type=int, default=4); ap.add_argument("--saida", required=True)
a = ap.parse_args()
sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
sys.argv = ["h3_denoise_local.py", a.cena, "--steps", str(a.steps), "--saida", a.saida]
import h3_denoise_local
h3_denoise_local.main()
