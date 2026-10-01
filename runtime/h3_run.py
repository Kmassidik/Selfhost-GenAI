#!/usr/bin/env python3
"""
M1 - our own MiniMax-H3 runtime (t2va: text -> video+audio). NO ComfyUI.
Loads the FL2VA modular pipeline, streams the 61.7 GB fp16 transformer through an
8 GB card via offload, denoises the joint video+audio sequence, decodes + muxes to mp4.
v0: control + correctness first. Kernels/quant (Track B) come after it runs.
"""
import os, time, argparse, torch
from diffusers import ModularPipeline

BASE  = os.path.expanduser("~/Desktop/selfhosted-minimaxi-h3")
MODEL = f"{BASE}/models/H3-FL2VA/FL2VA"
OUT   = f"{BASE}/outputs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps",   type=int,   default=35)
    ap.add_argument("--frames",  type=int,   default=124)   # ~5.17s @ 24fps
    ap.add_argument("--height",  type=int,   default=480)
    ap.add_argument("--width",   type=int,   default=832)   # 832/16=52 even (patch-aligned)
    ap.add_argument("--guidance",type=float, default=3.0)
    ap.add_argument("--shift",   type=float, default=12.0)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--gpu",     type=int,   default=0)
    ap.add_argument("--out",     default=f"{OUT}/m1_clip.mp4")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_device(args.gpu); dev = f"cuda:{args.gpu}"
    print(f"[load] building t2va pipeline from {MODEL}"); t0 = time.time()
    pipe = ModularPipeline.from_pretrained(MODEL, workflow="t2va", torch_dtype=torch.float16)
    # CORE: 61.7GB transformer can't fit 8GB -> stream per-layer (this is our runtime's value)
    try:
        pipe.enable_sequential_cpu_offload(device=dev); print("[load] sequential offload on")
    except Exception as e:
        print(f"[load] seq offload n/a ({e}); model offload"); pipe.enable_model_cpu_offload(device=dev)
    print(f"[load] ready in {time.time()-t0:.0f}s")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"[gen] {args.width}x{args.height} {args.frames}f {args.steps}steps :: {args.prompt[:70]}")
    t0 = time.time()
    result = pipe(prompt=args.prompt, height=args.height, width=args.width,
                  num_frames=args.frames, num_inference_steps=args.steps,
                  guidance_scale=args.guidance, shift=args.shift, generator=gen)
    print(f"[gen] finished in {(time.time()-t0)/60:.1f} min")
    save_result(result, args.out); print(f"[done] -> {args.out}")

def save_result(result, path):
    import imageio
    frames = getattr(result, "frames", None) or getattr(result, "videos", None)
    audio  = getattr(result, "audio",  None)
    if frames is None:
        print(f"[save] inspect result: {type(result)} attrs={[a for a in dir(result) if not a.startswith(chr(95))][:25]}"); return
    imageio.mimsave(path, frames, fps=24)  # v0 video; audio mux added once attrs confirmed

if __name__ == "__main__":
    main()
