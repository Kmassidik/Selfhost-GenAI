#!/usr/bin/env python3
"""Dalang Engine — STAGE 0: the correctness harness.

Before we build anything parallel, we need a GOLDEN REFERENCE: what does H3's
transformer output for a fixed tiny input? Every later Dalang stage (single-GPU
baseline, 2-GPU ring attention, shared weights, ring-3) must reproduce this
tensor bit-for-bit (within fp tolerance). No reference → no way to know our
engine is correct → we'd be scaling a black box. So this comes first.

WHAT IT DOES
  - Loads the H3 transformer exactly as our denoise engine does (bf16 + int8),
    on ONE GPU, via the existing box setup.
  - Registers a forward hook on the transformer to capture, for the FIRST
    denoise step: the input kwargs (hidden_states, conditioning, rotary, etc.)
    and the output tensor.
  - Runs a TINY denoise (few frames, few steps, fixed seed) off a saved state.
  - Dumps  ref_in.pt  (captured forward inputs, CPU) and  ref_out.pt  (output).

  Later stages load ref_in.pt, feed it through the Dalang forward, and diff
  against ref_out.pt. Target: max abs diff < 1e-3 (or within diffusers'
  own run-to-run nondeterminism, measured by running this twice).

RUN (on the box, dalang-z9pe-d8-ws, from the h3-consumer-bench dir):
    python 00_dump_reference.py <cena> --steps 2 --out ./dalang_ref

STATUS: FIRST DRAFT — to be run and refined ON THE BOX. The hook-capture of the
transformer's exact call signature is the part to verify interactively (H3's
forward kwargs may differ from the guess below; print them once and adjust).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
EMB = os.path.expanduser("./h3-lab/embeds")


def log(m):
    print(f"[dalang-s0] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cena")
    ap.add_argument("--steps", type=int, default=2)   # tiny: keep it fast/cheap
    ap.add_argument("--out", default="./dalang_ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- load the transformer + partial pipeline exactly like h3_denoise_local.py ---
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from torchao.quantization import Int8WeightOnlyConfig

    t0 = time.time()
    base = ModularPipeline.from_pretrained(ROOT)
    sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in ("denoise", "decode")}
    pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline(ROOT)

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
    pipe.update_components(transformer=tr, transformer_ref=tr)
    restante = [n for n in getattr(pipe, "component_names", [])
                if n not in ("transformer", "transformer_ref")]
    if restante:
        pipe.load_components(names=restante, dtype=torch.bfloat16)

    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    pipe.transformer.requires_grad_(False)
    log(f"loaded in {time.time()-t0:.0f}s")

    # --- capture the FIRST forward's inputs + output via a hook ---
    captured = {}

    def hook(module, args_, kwargs_, output):
        if "done" in captured:
            return  # only the first call
        # move everything to CPU so the reference is portable + device-agnostic
        def to_cpu(x):
            return x.detach().to("cpu") if torch.is_tensor(x) else x
        captured["args"] = tuple(to_cpu(a) for a in args_)
        captured["kwargs"] = {k: to_cpu(v) for k, v in kwargs_.items()}
        captured["output"] = to_cpu(output[0] if isinstance(output, (tuple, list)) else output)
        captured["done"] = True
        # print the exact signature ONCE so we can verify/adjust the harness
        log(f"HOOK sig: args={[type(a).__name__+str(tuple(a.shape)) if torch.is_tensor(a) else type(a).__name__ for a in args_]}")
        log(f"HOOK kwargs keys: {list(kwargs_.keys())}")

    h = pipe.transformer.register_forward_hook(hook, with_kwargs=True)

    # tiny state: reuse a saved encode but we only need a couple of steps
    estado = torch.load(f"{EMB}/{args.cena}.pt", weights_only=False)
    kwargs = {k: v for k, v in estado.items()}
    kwargs["generator"] = torch.Generator().manual_seed(2047)
    kwargs["num_inference_steps"] = args.steps
    kwargs["num_frames"] = max(22, int(kwargs.get("num_frames") or 0))  # smallest valid-ish, cheap
    kwargs["output"] = ["videos"]

    log(f"running {args.steps}-step tiny denoise to trip the hook...")
    try:
        pipe(**kwargs)
    except Exception as e:
        # we may deliberately stop early once captured; that's fine
        log(f"denoise stopped: {type(e).__name__}: {e}")
    h.remove()

    if "done" not in captured:
        log("!! hook never fired — the transformer wasn't called; inspect the pipeline path")
        sys.exit(1)

    torch.save({"args": captured["args"], "kwargs": captured["kwargs"]}, f"{args.out}/ref_in.pt")
    torch.save(captured["output"], f"{args.out}/ref_out.pt")
    o = captured["output"]
    log(f"REFERENCE SAVED: ref_in.pt + ref_out.pt  (out shape={tuple(o.shape)} dtype={o.dtype})")
    log("Stage 0 done. Next: 01_baseline_single_gpu.py must reproduce ref_out.pt.")


if __name__ == "__main__":
    main()
