# 📌 CHECKPOINT v0.1 — Self-Hosting MiniMax-H3 on 8GB

**Date frozen:** 2026-08-10
**Status:** Stable baseline. Everything below is *measured and reproducible* on the box. This is the line we build the W4A8 experiments on top of — if v0.2 breaks something, we roll back to here.

> **The one-line story:** we run a **33-billion-parameter** video+audio AI model on a **$300 8GB gaming GPU**, and after finding the real bottleneck and every quality trap, we can produce genuinely shareable clips — with a clear, honest map of exactly where the ceiling is and why.

---

## 1. The system (what "it" is)

| Layer | What |
|---|---|
| **Box** | `dalang@192.168.16.246` · Ubuntu 26.04 · 24-core CPU · 60GB RAM · RTX **3060 Ti 8GB** (Ampere sm_86) |
| **Model** | MiniMax-H3 — 33B flow-matching (FLOW_AV) text/image→video+audio |
| **Weights** | native **int8** DiT (20GB) + Qwen3-VL-32B encoder (26GB int8) + video/audio VAEs — 46GB total |
| **Engine** | ComfyUI 0.30 as a **systemd service** (auto-restart), PyTorch 2.11 cu128 |
| **How 46GB fits in 8GB** | DynamicVRAM offload — stream layers from RAM through VRAM one slice at a time |
| **Repo** | `github.com/dalang-io/gen-ai` (clean, no secrets/weights) |

---

## 2. Proven recipes (the reusable wins)

| Recipe | Settings | Best for | Score |
|---|---|---|---|
| **Quality-max single face** | 576×320 native → supersample 480p · 50 steps euler · **no Turbo** · one big close face · **no dialogue** · CodeFormer | hero portrait shots | **~85** |
| **Talking podcast (2-person)** | tight two-shot · 50 steps · **CodeFormer** (eyes+teeth) · **RIFE** 48fps (motion) | podcast / interview | **~83** |
| **Intercut solo podcast** | solo close-ups per speaker, stitched (5+5) | max face quality, edited feel | **~80** |
| **80s cel animation** | cel-style prompt · figure framed *from behind* (sidesteps faces) · 50 steps | stylized / music-video | **~85** |
| **Single sharp still** | native 1080p single frame (fits!) | photoreal stills | **~90** |

**Universal levers baked into all of them:**
- **Full 50-step euler, never Turbo** (Turbo under-renders teeth/mouths)
- **Big close faces** (pixels-per-face) — never wide crowds
- **Near-native resolution + minimal upscale** (supersample downscale for sharpness)
- **CodeFormer** for eyes/teeth on big faces · **RIFE** for motion · light deflicker for boil

---

## 3. Key deliverables (on Desktop, honest scores)

| File | What | Score / note |
|---|---|---|
| `hero480_portrait.mp4` | 480p cinematic portrait | **~85** — the quality-max proof |
| `podcast_2shot_eyesfix.mp4` | 2-person podcast, eyes fixed (CodeFormer) | **~83** — best 2-person |
| `podcast_eyesfix_RIFE48.mp4` | above + RIFE 48fps | smoothest motion |
| `podcast_5plus5.mp4` | intercut solo podcast | ~80 |
| `anime80_hero15s.mp4` | 15s 80s-anime night walk | ~85 style (seed-80 ghost panel; seed-777 clean) |
| `singer_song_SHORT.mp4` | 3s singer, big face | sharp — the short+sharp proof |
| `hero_crowd_final.mp4` | 30-person crowd | **~60 — kept as the cautionary example** |

Plus ~44 total renders on the box (tsunami, pip story, promos, experiments).

---

## 4. Optimizations achieved

- **INT8 chunked-matmul kernel patch** — chunked the 2.24GB `fast_int8_mm` int32 buffer per-row inside the existing scaling loop (`comfy_kitchen/.../quantization.py`). **Bit-identical** (`torch.equal`), freed **~2GB VRAM**, raised the 15s resolution ceiling. *This is the standout engineering.*
- **Allocator tuning** — `--disable-cuda-malloc` (enables `expandable_segments` + memory profiling), `--disable-smart-memory`.
- **Supersample-downscale** for crisp low-res output.
- **Full profiling** — CUDA memory snapshots pinned the real VRAM hog (int8 GEMM working buffers, ~88% of real tensors — not FFN/attention/weights).

---

## 5. The rules we discovered (our field laws)

1. **Pixels-per-face** — face quality = face *size in pixels*, not people count. Frame big & close.
2. **Pixels-per-eye / the degradation ladder** — as a face shrinks, features fail in order: **eyes → teeth → whole face**. Eyes are finest, fail first.
3. **Short+sharp OR long+soft** — a clip fits VRAM at high native res *or* long duration, never both.
4. **Turbo kills detail** — great for drafts, ruins teeth/mouths. Full steps for heroes.
5. **Ghost panel = seed-dependent** — a VAE decode artifact fixed by choosing a clean seed, not a setting.
6. **Boil = temporal instability** — upscalers amplify it; RIFE/deflicker reduce it.
7. **Motion judder = missing motion blur** — AI frames are too sharp; RIFE interpolation fixes it.

---

## 6. The honest ceiling (measured + research-confirmed)

- **Bottleneck (measured live):** **compute-bound** — GPU SM at 100%, power-capped at 200W (raisable to 210W ≈ +5%). RAM (30GB free) and PCIe streaming are **idle** — *not* the wall.
- **Native 720p 15s on 8GB = not achievable** by any published method. The wall is *activation memory* (attention over frames), which quantization barely touches (QuantSparse: most-aggressive W4A8 + 85% attention pruning on a 14B model still needed **28GB**).
- **RAM can't "up the DiT"** — the model is limited by VRAM (where it computes), not RAM (where it's stored). Warehouse vs. workbench.
- **The only ways up:** a **24GB GPU** (bf16 + native res), or on this box **down-the-weights/up-the-res** (W4A8 → free VRAM → higher native res).

---

## 7. Documentation produced

- **Knowledge base** — 12 HTML chapters (`knowledge-base/`), incl. new **11-glossary** and **12-pushing-past-8gb**.
- **Project docs** — EXPERIMENT-LOG, PROJECT-SUMMARY, SETUP, math models, README, (runpod-plan archived).
- **Persistent memory** — the box setup, all findings, recipes, stall/hang lessons.
- **Grounding research** — ViDiT-Q (2406.02540), QuantSparse (2509.23681), Native INT8 GEMM (2606.14598).

---

## 8. Known issues / open items

- Eyes still soft in *wide/two-shot* framing (fixed via CodeFormer + tight framing; inherent below ~60px/eye).
- Residual motion judder — RIFE gets ~90%; the rest needs native high-fps (VRAM).
- 80s-anime background text "boils" (inherent to diffusion; hidden by bokeh/deflicker).
- Anime seed-80 had a ghost panel → use seed-777 (judged clean).

---

## 9. → v0.2 roadmap

1. **W4A8 experiment** — load the Q4 H3 DiT (~11GB), measure freed VRAM, push native res 576×320 → 704×384, compare *Q4-at-higher-res* vs *int8-at-lower-res-upscaled*. Let the eyes decide.
2. **Package the win** — the activation-chunk patch + 8GB recipe as a **ComfyUI node + HF writeup** (applied contribution, under lucacadalora, every number reproducible).
3. **Optional** — 210W power limit (+5%), `res_multistep` sampler A/B.

---

*v0.1 frozen. Everything here is measured, honest, and reproducible. The box does real work; the ceiling is known; the next experiment is defined. If we die tomorrow, this project continues.*
