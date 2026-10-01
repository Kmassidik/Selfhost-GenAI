# 🔬 Multi-GPU + quality research — will the 2nd GPU actually fix the eyes?

**Question:** does adding the 2nd RTX 3060 Ti improve *quality* (the soft-eyes problem), or just speed?
**Answer from research: YES, it can improve quality — and there are TWO compounding levers.**

---

## Lever 1 — Multi-GPU (ComfyUI-MultiGPU / DisTorch2) → higher resolution → more pixels per eye

**The tool:** [`pollockjj/ComfyUI-MultiGPU`](https://github.com/pollockjj/ComfyUI-MultiGPU) — a maintained ComfyUI node with **DisTorch2 "Virtual VRAM."**

**How it fixes the eyes (the mechanism):**
- On a single 8GB card, the DiT **weights + activations share the same 8GB**. Activations (= resolution × frames) get squeezed → we're forced to low res → each eye gets ~30px → soft.
- DisTorch2 **offloads the DiT's weights onto GPU 1's VRAM** (e.g. allocation `cuda:0,4gb;cuda:1,*` = keep 4GB on card 0, park the rest on card 1). Now **GPU 0's 8GB is mostly free for activations** → we can generate at **higher native resolution** → **more pixels per eye → cleaner irises.**

**Confirmed facts:**
- ✅ **Works with our GGUF** — it ships `UnetLoaderGGUFDisTorch2MultiGPU` (drop-in for our `UnetLoaderGGUF`).
- ✅ Supports video/GGUF loaders explicitly.
- ✅ Bonus: offloading to GPU 1's **VRAM is far faster than our current system-RAM offload** → also speeds renders.

**Honest caveat:** it does NOT create a true 16GB pool. It *frees the compute card's VRAM for activations* by parking weights elsewhere. Real gain, but bounded — and PCIe traffic between cards adds some overhead. We must **measure the actual max resolution** it unlocks, not assume.

---

## Lever 2 — the HF model you flagged: `fal/MiniMax-H3-Realism-People-LoRA` → better eyes at the SAME resolution

**Yes, the HF model you gave helps — directly on faces/eyes.** From its own docs:
- *"Skin keeps its texture instead of smoothing out, **eyes and micro-expressions stay coherent**, light behaves like a film set."*
- Purpose-built for **"portraits, faces, hands, crowds, everyday characters."**
- Compatible with "any MiniMax H3 implementation" (standard H3 layout) → **should apply to our Q4 GGUF** (needs a quick compat test).
- Usage: drop in `models/loras/`, standard **Load LoRA** node, strength **0.8–1.0**, trigger word **`r34l1sm`**.

---

## ⭐ The key insight — the two levers COMPOUND

The LoRA's own training notes say the quiet part out loud:
> *"Skin texture, pores, fine hair live in **high spatial frequencies**, and at a lower [resolution] the latents barely carry them."*

Translation: **the Realism LoRA needs higher resolution to deliver its detail.** So:
- **Multi-GPU (DisTorch2)** provides the higher resolution.
- **Realism LoRA** fills that resolution with coherent eyes/skin/pores.
- **Together = the actual eye fix** — more pixels *and* a model tuned to render faces well in them. Neither alone is as strong.

---

## What the other HF models you listed are for (honest triage)
| Model | Helps eyes-per-frame? | What it's actually for |
|---|---|---|
| **fal Realism-People-LoRA** | ✅ **YES** | face/eye/skin detail — our target |
| Ref2VA (t8star / xmarre) | ❌ no | *identity consistency across clips* (the drift problem, different issue) |
| full-33B GGUF (vantagewithai) | ~ marginal | more params, but bigger/harder to fit — resolution matters more than params for eyes |
| Turbo LoRA | ❌ (hurts) | speed, but degrades detail |

---

## The plan for `multiple-gpu-gen-ai` (data-driven, isolated from the working single-GPU app)
1. **Install** `ComfyUI-MultiGPU` in the multi-GPU project.
2. **Offload the Q4 DiT to GPU 1** via `UnetLoaderGGUFDisTorch2MultiGPU` → free GPU 0 for activations.
3. **Measure the ceiling:** push native resolution up (960×540 → 1024×576 → higher) until VRAM says stop. Record the real max.
4. **Add the Realism-People-LoRA** (`r34l1sm`, strength 0.8) — test GGUF compat.
5. **The proof:** render the *same cafe-portrait prompt/seed* as our single-GPU 480p, at the new higher res + LoRA → **crop the eyes → compare.** Data, not argument.

**Expected honest outcome:** meaningfully better eyes (more pixels + face-tuned LoRA). *How much* better is the thing we measure. If DisTorch only buys a small res bump, the LoRA still helps; if it buys a big one, the combo could be a real jump.

## Sources
- ComfyUI-MultiGPU (DisTorch2): github.com/pollockjj/ComfyUI-MultiGPU
- Realism LoRA: huggingface.co/fal/MiniMax-H3-Realism-People-LoRA
- ComfyUI multi-GPU behavior: Comfy-Org/ComfyUI Discussion #836
