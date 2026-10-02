#!/usr/bin/env python3
"""Build per-prompt 4-model comparison strips + summary stats from the benchmark."""
import json, os, statistics, collections
from PIL import Image, ImageDraw, ImageFont

OUT = "/root/Desktop/selfhostimageai/models/bench/_results"
MONT = f"{OUT}/montages"; os.makedirs(MONT, exist_ok=True)
ORDER = ["qwen", "flux-dev", "sdxl", "sd35-medium"]
LABEL = {"qwen": "Qwen-Image 2.1", "flux-dev": "FLUX.1-dev", "sdxl": "SDXL 1.0", "sd35-medium": "SD3.5 Medium"}
CELL = 512; PAD = 10; TOP = 46

r = json.load(open(f"{OUT}/results.json"))
by = {(x["engine"], x["prompt_id"]): x for x in r if x.get("ok")}
prompts = []
seen = set()
for x in r:
    if x["prompt_id"] not in seen:
        seen.add(x["prompt_id"]); prompts.append((x["prompt_id"], x["prompt"]))

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except Exception:
    font = small = ImageFont.load_default()

for pid, prompt in prompts:
    n = len(ORDER)
    W = n * CELL + (n + 1) * PAD
    H = TOP + CELL + PAD + 30
    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 12), f"[{pid}]  {prompt[:110]}", fill=(235, 235, 240), font=small)
    for i, eng in enumerate(ORDER):
        x0 = PAD + i * (CELL + PAD)
        rec = by.get((eng, pid))
        if rec and os.path.exists(rec["image"]):
            im = Image.open(rec["image"]).convert("RGB").resize((CELL, CELL))
            canvas.paste(im, (x0, TOP))
            cap = f"{LABEL[eng]}   {rec['seconds']}s · {rec['peak_vram_gb']}GB"
        else:
            d.rectangle([x0, TOP, x0 + CELL, TOP + CELL], fill=(40, 40, 44))
            cap = f"{LABEL[eng]}   (missing)"
        d.text((x0, TOP + CELL + 6), cap, fill=(200, 200, 210), font=small)
    canvas.save(f"{MONT}/cmp_{pid}.png")
    print("montage", pid)

# summary stats
print("\n## Speed + VRAM (per engine)")
print("| engine | renders | median s | mean s | peak VRAM GB |")
print("|---|---|---|---|---|")
summary = {}
for eng in ORDER:
    secs = [x["seconds"] for x in r if x["engine"] == eng and x.get("ok")]
    vram = [x["peak_vram_gb"] for x in r if x["engine"] == eng and x.get("ok")]
    if not secs:
        print(f"| {LABEL[eng]} | 0 | - | - | - |"); continue
    summary[eng] = {"n": len(secs), "median_s": round(statistics.median(secs), 1),
                    "mean_s": round(statistics.mean(secs), 1), "peak_vram_gb": max(vram)}
    print(f"| {LABEL[eng]} | {len(secs)} | {statistics.median(secs):.1f} | {statistics.mean(secs):.1f} | {max(vram):.2f} |")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print("\nDONE-MONTAGE")
