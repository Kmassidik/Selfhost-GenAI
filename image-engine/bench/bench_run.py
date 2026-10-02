#!/usr/bin/env python3
"""Single-GPU image-model benchmark (mirrors the audio app's load->run->unload->swap).

Runs on GPU 2 (CUDA_VISIBLE_DEVICES=2). For each engine: load it, render the whole
prompt suite, record wall-time + peak VRAM, save the image, then unload and free the
card before the next engine. Resumable: re-running skips (engine,prompt) pairs already
in results.json. GPU 0 (audio) and GPU 1 (image app) are never touched.
"""
import os, json, time, gc, traceback
import torch

BENCH = "/root/Desktop/selfhostimageai/models/bench"
QWEN  = "/root/Desktop/selfhostimageai/models/qwen-image-2.1-official"
OUT   = f"{BENCH}/_results"
os.makedirs(OUT, exist_ok=True)

# (id, prompt, seed) — fixed seeds for reproducibility (not comparable across archs).
PROMPTS = [
 ("portrait",  "close-up portrait of an elderly fisherman, weathered skin, kind eyes, soft golden-hour light, shot on 85mm f1.4, photographic, highly detailed", 101),
 ("landscape", "a serene japanese garden, a red wooden bridge over a koi pond, autumn maple trees, soft morning mist, photographic, highly detailed", 102),
 ("text_sign", 'a vintage coffee shop storefront with a sign that reads "MORNING BREW" in bold retro lettering, brick wall, warm evening light', 103),
 ("text_menu", 'a cafe chalkboard sign that clearly reads "TODAY: FRESH PASTA $12", neat white handwriting, wooden frame', 104),
 ("count",     "exactly three red apples and two green pears arranged in a row on a wooden table, studio lighting, photographic", 105),
 ("anime",     "a girl walking under cherry blossom trees at sunset, Studio Ghibli style anime, soft cel shading, vibrant colors, detailed background", 106),
 ("product",   "a sleek matte-black wireless headphone floating on a clean light-grey gradient studio background, soft rim light, premium product photography", 107),
 ("hands",     "a smiling young woman raising one hand with five clearly separated fingers, natural pose, photographic, detailed realistic hand", 108),
 ("city",      "a futuristic city skyline at dusk, neon signs reflecting on wet streets, flying cars, cinematic, highly detailed", 109),
 ("fantasy",   "a regal elven queen in ornate silver filigree armor standing in an enchanted glowing forest, dramatic rim light, intricate detail, digital painting", 110),
]
NEG = "blurry, low quality, deformed, bad anatomy, extra fingers, text, watermark, duplicate, jpeg artifacts"
W = H = 1024
STEPS = 30

def free():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)

def load_qwen():
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(QWEN, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    def render(p, seed):
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe(prompt=p, negative_prompt=NEG, width=W, height=H,
                    num_inference_steps=STEPS, true_cfg_scale=4.0,
                    generator=g, use_kv_cache=True).images[0]
    return pipe, render

def load_flux():
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(f"{BENCH}/flux1-dev", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    def render(p, seed):  # FLUX.1-dev is guidance-distilled: guidance_scale, no negative
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe(prompt=p, width=W, height=H, num_inference_steps=STEPS,
                    guidance_scale=3.5, generator=g).images[0]
    return pipe, render

def load_sdxl():
    from diffusers import StableDiffusionXLPipeline
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            f"{BENCH}/sdxl-base", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            f"{BENCH}/sdxl-base", torch_dtype=torch.float16, use_safetensors=True)
    pipe.enable_sequential_cpu_offload()
    def render(p, seed):
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe(prompt=p, negative_prompt=NEG, width=W, height=H,
                    num_inference_steps=STEPS, guidance_scale=7.0, generator=g).images[0]
    return pipe, render

def load_sd35():
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(f"{BENCH}/sd35-medium", torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    def render(p, seed):
        g = torch.Generator("cpu").manual_seed(seed)
        return pipe(prompt=p, negative_prompt=NEG, width=W, height=H,
                    num_inference_steps=STEPS, guidance_scale=4.5, generator=g).images[0]
    return pipe, render

# Order matches the download order (sdxl -> sd35 -> flux) so each model is ready by
# the time the benchmark reaches it. Qwen is already present. (dl_marker = what the
# downloader prints in bench_curl_dl.log when that repo finishes; None = no wait.)
ENGINES = [("qwen", load_qwen, None), ("sdxl", load_sdxl, "sdxl-base"),
           ("sd35-medium", load_sd35, "sd35-medium"), ("flux-dev", load_flux, "flux1-dev")]
DL_LOG = "/root/Desktop/selfhostimageai/bench_curl_dl.log"

def wait_for_download(marker, max_min=90):
    if not marker:
        return True
    import re
    for _ in range(max_min * 2):  # poll every 30s
        try:
            log = open(DL_LOG).read()
        except Exception:
            log = ""
        if f"DONE-REPO {marker}" in log or "ALL-CURL-DONE" in log:
            return True
        time.sleep(30)
    print(f"WAIT-TIMEOUT for {marker}", flush=True)
    return False

resfile = f"{OUT}/results.json"
results = json.load(open(resfile)) if os.path.exists(resfile) else []
done = {(r["engine"], r["prompt_id"]) for r in results}

for ename, loader, dl_marker in ENGINES:
    if all((ename, pid) in done for pid, _, _ in PROMPTS):
        print(f"##### {ename}: all prompts done, skipping", flush=True); continue
    if not wait_for_download(dl_marker):
        print(f"##### SKIP {ename}: download not ready", flush=True); continue
    print(f"##### loading {ename} …", flush=True)
    free(); t0 = time.time()
    try:
        pipe, render = loader()
    except Exception as e:
        print(f"LOAD-FAIL {ename}: {type(e).__name__}: {e}", flush=True); traceback.print_exc(); continue
    load_s = round(time.time() - t0, 1)
    print(f"##### {ename} loaded in {load_s}s", flush=True)
    for pid, prompt, seed in PROMPTS:
        if (ename, pid) in done:
            print(f"  skip {ename}/{pid}", flush=True); continue
        free(); t = time.time()
        try:
            img = render(prompt, seed)
            fn = f"{OUT}/{ename}__{pid}.png"; img.save(fn)
            rec = {"engine": ename, "prompt_id": pid, "prompt": prompt, "seed": seed,
                   "seconds": round(time.time() - t, 1), "peak_vram_gb": peak_gb(),
                   "load_s": load_s, "image": fn, "ok": True}
        except Exception as e:
            rec = {"engine": ename, "prompt_id": pid, "prompt": prompt, "seed": seed,
                   "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}", "ok": False}
            traceback.print_exc()
        results.append(rec); json.dump(results, open(resfile, "w"), indent=2)
        print(f"  {ename}/{pid}: {rec.get('seconds')}s vram={rec.get('peak_vram_gb')}GB ok={rec['ok']}", flush=True)
    del pipe, render; free()
    print(f"##### {ename} done + unloaded", flush=True)

print("BENCH-COMPLETE", flush=True)
