#!/usr/bin/env python3
"""Dalang Engine — E1b: chunk the WHOLE FFN as a unit (not per-linear).

E1 showed our torchao engine is weight-only int8 (no activation scratch buffer);
the real peak driver is resident ACTIVATIONS — chiefly the FFN's 14336-dim
intermediate. Per-linear chunking left that intermediate full-sequence. Fix: wrap
each block's FeedForward (`ff`) to run up->act->down over token-CHUNKS, so the
intermediate is chunk-sized. Correct by construction (FFN is per-token).

Run: python 06b_chunk_ffn.py --ref ./dalang_ref [--nochunk]
"""
import argparse, os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def log(m): print(f"[dalang-e1b] {m}", flush=True)


def patch_ffn_chunk(model, chunk=2048):
    n = 0
    for name, mod in model.named_modules():
        if name.endswith(".ff") and hasattr(mod, "forward"):
            orig = mod.forward
            def make(orig):
                def fwd(x, *a, **k):
                    if x.dim() < 2 or x.shape[-2] <= chunk:
                        return orig(x, *a, **k)
                    return torch.cat([orig(x[..., i:i + chunk, :], *a, **k)
                                      for i in range(0, x.shape[-2], chunk)], dim=-2)
                return fwd
            mod.forward = make(orig)
            n += 1
    return n


def extract_sample(out):
    return out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./dalang_ref")
    ap.add_argument("--nochunk", action="store_true")
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    t0 = time.time()
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out"]))
    tr.requires_grad_(False)
    if not a.nochunk:
        n = patch_ffn_chunk(tr, a.chunk); log(f"FFN-CHUNKED: wrapped {n} FFN modules (chunk={a.chunk})")
    else:
        log("NOCHUNK baseline")
    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    log(f"loaded {time.time()-t0:.0f}s")
    ref_in = torch.load(f"{a.ref}/ref_in.pt", weights_only=False)
    ref_out = torch.load(f"{a.ref}/ref_out.pt", weights_only=False)
    torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        s = extract_sample(tr(**dict(ref_in["kwargs"])))
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    d = (s.float().cpu() - ref_out.float().cpu()).abs()
    log(f"PEAK VRAM cuda:0 = {peak:.2f} GB   DIFF max={d.max().item():.3e} mean={d.mean().item():.3e}")
    log("E1B-DONE")


if __name__ == "__main__":
    main()
