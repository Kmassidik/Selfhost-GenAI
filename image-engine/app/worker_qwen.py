#!/usr/bin/env python3
"""Resident diffusers worker for the OFFICIAL Qwen-Image 2.1 (int8, reference pipeline).

Runs in the H3 venv (torch + diffusers + torchao). Loads the model ONCE — int8
weight-only transformer + CPU offload so it fits the 8 GB card — then claims jobs
and renders in-process (no per-image reload). Pinned to GPU 1 by the unit's
CUDA_VISIBLE_DEVICES; GPU 0 (audio) is never visible to it.

This is the H3 recipe (official weights, reference pipeline, int8 ~= bf16 quality)
applied to a smaller, image-only model.
"""
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import db
import jobs

S = config.settings()
MODEL = str(S.root / "models" / "qwen-image-2.1-official")

_pipe = None


def load_pipe():
    import torch
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    t0 = time.time()
    int8 = os.environ.get("QWEN_INT8", "0") == "1"
    if int8:
        from diffusers import TorchAoConfig
        from torchao.quantization import Int8WeightOnlyConfig
        print(f"[qwen] loading 2.1 transformer INT8 from {MODEL}", flush=True)
        qc = TorchAoConfig(Int8WeightOnlyConfig(version=2))
        tr = QwenImage21Transformer2DModel.from_pretrained(
            MODEL, subfolder="transformer", torch_dtype=torch.bfloat16, quantization_config=qc)
        pipe = QwenImage21Pipeline.from_pretrained(MODEL, transformer=tr, torch_dtype=torch.bfloat16)
    else:
        print(f"[qwen] loading 2.1 pipeline BF16 (reference quality) from {MODEL}", flush=True)
        pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    # Fit the 8 GB card: submodule-level CPU offload (robust for any component size).
    # Quality is unaffected (offload is about *where* weights sit, not precision).
    pipe.enable_sequential_cpu_offload()
    # VAE decode of a 1024px latent is the memory *peak* on an 8 GB card — tile +
    # slice it so the decode never spikes past the budget (output is identical).
    for fn in ("enable_vae_tiling", "enable_vae_slicing"):
        try:
            getattr(pipe, fn)()
        except Exception as e:
            print(f"[qwen] {fn} unavailable: {e}", flush=True)
    print(f"[qwen] pipeline ready in {time.time()-t0:.0f}s "
          f"({'int8' if int8 else 'bf16'})", flush=True)
    return pipe


def render(job) -> str:
    import torch
    p = job["params"]
    out_file = S.out / f"{int(time.time())}-{job['id']}.png"
    gen = torch.Generator(device="cpu").manual_seed(int(p["seed"]))
    kw = dict(
        prompt=p["prompt"],
        negative_prompt=p.get("negative", "") or " ",
        width=int(p["width"]), height=int(p["height"]),
        num_inference_steps=int(p["steps"]),
        true_cfg_scale=float(p["cfg"]),
        generator=gen,
    )
    # 2.1 has a causal-conditioning KV cache; reuse the conditioning K/V across steps.
    if os.environ.get("QWEN_KV_CACHE", "0") == "1":
        kw["use_kv_cache"] = True
    img = _pipe(**kw).images[0]
    img.save(out_file)
    # sidecar params
    out_file.with_suffix(".json").write_text(
        __import__("json").dumps({**p, "file": out_file.name}, indent=2))
    return f"/img/{out_file.name}"


def tick(conn) -> bool:
    job = jobs.claim(conn)
    if not job:
        return False
    t0 = time.time()
    try:
        image = render(job)
        jobs.finish(conn, job["id"], image=image, seconds=round(time.time() - t0, 1))
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs.finish(conn, job["id"], error=f"{type(e).__name__}: {e}",
                    seconds=round(time.time() - t0, 1))
    finally:
        # A failed render can leave cached allocations on the card; free them so the
        # next job in this resident worker still fits the 8 GB budget.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return True


_stopping = False


def _stop(*_):
    global _stopping
    _stopping = True
    print("stop asked for; finishing the job in hand, then exiting", flush=True)


def main():
    global _pipe
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    conn = db.connect(S.db_path)
    db.init(conn)
    stale = jobs.reset_stale(conn)
    print(f"worker starting · {stale} interrupted job(s) cleared · loading model…", flush=True)
    _pipe = load_pipe()
    print("worker up · model resident · waiting for jobs", flush=True)
    while not _stopping:
        if not tick(conn):
            for _ in range(20):
                if _stopping:
                    break
                time.sleep(0.1)


if __name__ == "__main__":
    main()
