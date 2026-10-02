#!/usr/bin/env python3
"""Multi-model GPU worker — one model on the 8 GB card at a time, swapped on demand.

Mirrors the audio studio's single-GPU discipline: each job names a model; if it is
not the one currently loaded, the worker UNLOADS the current pipeline, frees the card,
and LOADS the requested one, then renders. Pinned to GPU 1 by the unit's
CUDA_VISIBLE_DEVICES; GPU 0 (audio) and GPU 2 (LLM) are never visible to it.

Model families (catalog "family" field) decide how to load + call the pipeline:
  qwen21 · flux · sdxl · sd3
"""
import gc
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import db
import jobs
import registry

S = config.settings()
_cur = {"id": None, "pipe": None, "render": None}


def _free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _path(spec):
    return str(S.root / spec["model"])


# ---- per-family loaders: return (pipe, render_fn(params)->PIL image) ----
def _load_qwen(spec):
    import torch
    from diffusers import QwenImage21Pipeline
    mp = _path(spec)
    if spec.get("int8"):
        from diffusers import TorchAoConfig, QwenImage21Transformer2DModel
        from torchao.quantization import Int8WeightOnlyConfig
        qc = TorchAoConfig(Int8WeightOnlyConfig(version=2))
        tr = QwenImage21Transformer2DModel.from_pretrained(
            mp, subfolder="transformer", torch_dtype=torch.bfloat16, quantization_config=qc)
        pipe = QwenImage21Pipeline.from_pretrained(mp, transformer=tr, torch_dtype=torch.bfloat16)
    else:
        pipe = QwenImage21Pipeline.from_pretrained(mp, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    kv = bool(spec.get("kv_cache"))

    def render(p):
        g = torch.Generator("cpu").manual_seed(int(p["seed"]))
        return pipe(prompt=p["prompt"], negative_prompt=p.get("negative") or " ",
                    width=int(p["width"]), height=int(p["height"]),
                    num_inference_steps=int(p["steps"]), true_cfg_scale=float(p["cfg"]),
                    generator=g, use_kv_cache=kv).images[0]
    return pipe, render


def _load_flux(spec):
    import torch
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(_path(spec), torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    def render(p):  # guidance-distilled: no negative prompt
        g = torch.Generator("cpu").manual_seed(int(p["seed"]))
        return pipe(prompt=p["prompt"], width=int(p["width"]), height=int(p["height"]),
                    num_inference_steps=int(p["steps"]), guidance_scale=float(p["cfg"]),
                    generator=g).images[0]
    return pipe, render


def _load_sdxl(spec):
    import torch
    from diffusers import StableDiffusionXLPipeline
    mp = _path(spec)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            mp, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            mp, torch_dtype=torch.float16, use_safetensors=True)
    pipe.enable_sequential_cpu_offload()

    def render(p):
        g = torch.Generator("cpu").manual_seed(int(p["seed"]))
        return pipe(prompt=p["prompt"], negative_prompt=p.get("negative") or "",
                    width=int(p["width"]), height=int(p["height"]),
                    num_inference_steps=int(p["steps"]), guidance_scale=float(p["cfg"]),
                    generator=g).images[0]
    return pipe, render


def _load_sd3(spec):
    import torch
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(_path(spec), torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()

    def render(p):
        g = torch.Generator("cpu").manual_seed(int(p["seed"]))
        return pipe(prompt=p["prompt"], negative_prompt=p.get("negative") or "",
                    width=int(p["width"]), height=int(p["height"]),
                    num_inference_steps=int(p["steps"]), guidance_scale=float(p["cfg"]),
                    generator=g).images[0]
    return pipe, render


LOADERS = {"qwen21": _load_qwen, "flux": _load_flux, "sdxl": _load_sdxl, "sd3": _load_sd3}


def switch_to(model_id):
    """Make model_id the one model on the card. Returns the render fn."""
    if _cur["id"] == model_id and _cur["render"] is not None:
        return _cur["render"]
    spec = registry.get(model_id)
    if not spec:
        raise ValueError(f"unknown model {model_id}")
    fam = spec.get("family")
    if fam not in LOADERS:
        raise ValueError(f"{model_id}: unknown family {fam}")
    # unload the current one and free the card
    if _cur["pipe"] is not None:
        print(f"[swap] unloading {_cur['id']}", flush=True)
        _cur["pipe"] = None
        _cur["render"] = None
        _cur["id"] = None
        _free()
    t0 = time.time()
    print(f"[swap] loading {model_id} (family {fam})…", flush=True)
    pipe, render = LOADERS[fam](spec)
    _cur.update(id=model_id, pipe=pipe, render=render)
    print(f"[swap] {model_id} ready in {time.time()-t0:.0f}s", flush=True)
    return render


def do_job(conn, job):
    import torch
    p = job["params"]
    out_file = S.out / f"{int(time.time())}-{job['id']}.png"
    t0 = time.time()
    try:
        render = switch_to(job["model"])
        img = render(p)
        img.save(out_file)
        out_file.with_suffix(".json").write_text(
            __import__("json").dumps({**p, "model": job["model"], "file": out_file.name}, indent=2))
        jobs.finish(conn, job["id"], image=f"/img/{out_file.name}",
                    seconds=round(time.time() - t0, 1))
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs.finish(conn, job["id"], error=f"{type(e).__name__}: {e}",
                    seconds=round(time.time() - t0, 1))
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


_stopping = False


def _stop(*_):
    global _stopping
    _stopping = True
    print("stop asked for; finishing the job in hand, then exiting", flush=True)


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    conn = db.connect(S.db_path)
    db.init(conn)
    stale = jobs.reset_stale(conn)
    print(f"multi-model worker up · {stale} interrupted job(s) cleared · "
          f"models load on demand, one at a time", flush=True)
    while not _stopping:
        job = jobs.claim(conn)
        if not job:
            for _ in range(20):
                if _stopping:
                    break
                time.sleep(0.1)
            continue
        print(f"[job {job['id']}] model={job['model']}", flush=True)
        do_job(conn, job)


if __name__ == "__main__":
    main()
