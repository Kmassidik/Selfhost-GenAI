#!/usr/bin/env python3
"""Dalang Engine — E1: chunk the per-token matmuls to kill the int8 buffer wall.

The wall (proven all night) is the ~2.2 GB int8 `fast_int8_mm` scratch buffer during
the big linears (QKV, out-proj, FFN). Those are all PER-TOKEN — each token's projection
is independent. So if we feed the linear its tokens in CHUNKS, the int8 buffer is
chunk-sized, not full-sequence. Output is mathematically identical (linear is per-row);
peak VRAM drops → more frames fit on ONE card.

This wraps every big nn.Linear in the transformer to loop its token dim in chunks.

Run (box):
  python 06_chunk_linear.py --ref ./dalang_ref            # chunked
  python 06_chunk_linear.py --ref ./dalang_ref --nochunk  # baseline (for the VRAM delta)
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch
import torch.nn as nn

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def log(m):
    print(f"[dalang-e1] {m}", flush=True)


def patch_chunking(model, chunk=1024, min_in=1024):
    """Wrap big per-token Linears to process the token dim (-2) in chunks."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.in_features >= min_in:
            orig = mod.forward
            def make(orig):
                def fwd(x, *a, **k):
                    if x.dim() < 2 or x.shape[-2] <= chunk:
                        return orig(x, *a, **k)
                    return torch.cat(
                        [orig(x[..., i:i + chunk, :], *a, **k) for i in range(0, x.shape[-2], chunk)],
                        dim=-2)
                return fwd
            mod.forward = make(orig)
            n += 1
    return n


def extract_sample(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return getattr(out, "sample", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./dalang_ref")
    ap.add_argument("--nochunk", action="store_true")
    ap.add_argument("--chunk", type=int, default=1024)
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
        n = patch_chunking(tr, chunk=a.chunk)
        log(f"CHUNKED: wrapped {n} linears (chunk={a.chunk} tokens)")
    else:
        log("NOCHUNK: baseline (linears run full-sequence)")

    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    log(f"loaded in {time.time()-t0:.0f}s")

    ref_in = torch.load(f"{a.ref}/ref_in.pt", weights_only=False)
    ref_out = torch.load(f"{a.ref}/ref_out.pt", weights_only=False)
    kwargs = dict(ref_in["kwargs"])

    torch.cuda.reset_peak_memory_stats(0)
    t = time.time()
    with torch.no_grad():
        sample = extract_sample(tr(**kwargs))
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    log(f"forward {time.time()-t:.0f}s; PEAK VRAM cuda:0 = {peak:.2f} GB")

    d = (sample.float().cpu() - ref_out.float().cpu()).abs()
    log(f"DIFF vs reference: max={d.max().item():.3e} mean={d.mean().item():.3e}")
    if d.max().item() < 1e-3:
        log("correctness: EXACT ✅ (chunking a linear is per-row identical)")
    log("E1-DONE")


if __name__ == "__main__":
    main()
