#!/usr/bin/env python3
"""Dalang Engine — STAGE 1: single-GPU baseline (OUR forward, matched to reference).

We load the int8 H3 transformer ONCE and call its forward OURSELVES with the exact
inputs captured in Stage 0 (`ref_in.pt`), then check the output reproduces
`ref_out.pt`. This proves our loader + forward plumbing is correct — that we can
drive H3 outside the diffusers pipeline wrapper. No parallelism yet.

PROOF TARGET: max abs diff vs ref_out < 1e-3 (deterministic forward → should be ~0).

RUN (box, from h3-consumer-bench dir; dalang_ref/ from Stage 0 must exist):
    python 01_baseline_single_gpu.py --ref ./dalang_ref
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def log(m):
    print(f"[dalang-s1] {m}", flush=True)


def extract_sample(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, dict):
        return next(iter(out.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="./dalang_ref")
    args = ap.parse_args()

    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    t0 = time.time()
    log("loading int8 transformer (single process, single GPU) ...")
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
            ],
        ),
    )
    tr.requires_grad_(False)
    # same block-level offload the reference forward ran under
    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    log(f"loaded in {time.time()-t0:.0f}s")

    ref_in = torch.load(f"{args.ref}/ref_in.pt", weights_only=False)
    ref_out = torch.load(f"{args.ref}/ref_out.pt", weights_only=False)
    kwargs = dict(ref_in["kwargs"])
    log(f"ref_in kwargs: {list(kwargs.keys())}")
    log(f"ref_out: shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")

    def run(kw):
        with torch.no_grad():
            return extract_sample(tr(**kw))

    log("calling OUR forward (attempt 1: inputs as captured / CPU) ...")
    t = time.time()
    try:
        sample = run(kwargs)
    except RuntimeError as e:
        log(f"attempt 1 failed ({type(e).__name__}: {str(e)[:80]}); attempt 2: move tensors to cuda ...")
        kw2 = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        sample = run(kw2)
    log(f"forward done in {time.time()-t:.0f}s; out shape={tuple(sample.shape)} dtype={sample.dtype}")

    a = sample.float().cpu()
    b = ref_out.float().cpu()
    if a.shape != b.shape:
        log(f"!! SHAPE MISMATCH: ours={tuple(a.shape)} ref={tuple(b.shape)} — cannot compare"); return
    d = (a - b).abs()
    maxd, meand = d.max().item(), d.mean().item()
    log(f"DIFF vs golden reference:  max={maxd:.3e}   mean={meand:.3e}")
    if maxd < 1e-3:
        log("VERDICT: PASS ✅  our single-GPU forward reproduces the reference. Stage 1 done.")
    elif maxd < 5e-2:
        log("VERDICT: CLOSE ⚠️  small diff — likely fp/nondeterminism; run reference twice to get the noise floor.")
    else:
        log("VERDICT: FAIL ❌  our forward diverges — the call/inputs are not equivalent; debug on this tiny case.")
    log("STAGE1-DONE")


if __name__ == "__main__":
    main()
