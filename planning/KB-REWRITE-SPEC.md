# Knowledge-base rewrite — spec and verified facts

Every chapter in `knowledge-base/` is being rewritten to one house standard.
**Read this file completely before editing any chapter.**

The credibility of this project rests on one rule: **every number was measured.**
If a number is not in this file and not in the chapter you are editing, **do not
invent it.** Write around it, or say the measurement was not taken.

---

## 1 · House format (do not deviate)

Each chapter is a single `.html` file with this exact skeleton:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NN · Chapter Title</title>
<meta name="description" content="One sentence, for link previews and search.">
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="assets/katex.min.css">
</head>
<body>
<script type="text/markdown" id="md">
<span class="kicker">Chapter NN · Section</span>

# Chapter Title

<p class="lede">One or two sentences that answer the title directly.</p>

...body...

## The vocabulary you now own

<div class="callout spec"><span class="t">Terms from this chapter</span>
<p><span class="badge">term</span><span class="badge">term</span></p>
<p>If those feel solid, ...</p></div>

Next: one sentence pointing at the following chapter.
</script>
<script src="assets/marked.min.js"></script>
<script src="assets/katex.min.js"></script>
<script src="assets/app.js"></script>
</body>
</html>
```

### Hard rules

1. **NEVER load `mermaid.min.js`.** Convert every ```mermaid block to inline SVG.
2. **NEVER put a blank line inside a raw HTML block or an `<svg>`.** The markdown
   renderer terminates the HTML block at the first blank line and the rest is
   mangled. This is the single most common way to break a page.
3. **Every chapter ends with "The vocabulary you now own"** followed by a one-line
   pointer to the next chapter.
4. **No second person about the machine.** Never "your box", "your GPU", "your
   render", "we did together", "you asked". The reader is a stranger. Use "the
   box", "this build", "the machine", or first-person plural for actions we took.
5. **Answer the title in the first sentence** of the lede, then build outward.
6. Keep every internal link valid. Chapter files are `NN-slug.html` in the same
   directory; verify the file exists before linking.

### Voice

Narrative openings, measured bodies. Open a chapter with the concrete, surprising
or human thing (a number that shouldn't be true, a failure, a question). Then be
precise and unadorned. Analogies are encouraged when they carry real structure —
countertop/pantry for VRAM/RAM works because the distances are genuinely different.
Avoid decorative metaphor.

Be honest about failures. Where an earlier prediction was later disproved, keep the
prediction, quote it, and show what actually happened. That contrast is the most
valuable content in this guide.

### Callouts

```html
<div class="callout truth"><span class="t">Title</span><p>...</p></div>
<div class="callout spec"><span class="t">Title</span><p>...</p></div>
<div class="callout warn"><span class="t">Title</span><p>...</p></div>
<div class="callout danger"><span class="t">Title</span><p>...</p></div>
```

`truth` = a grounded conclusion. `spec` = reference data or a rule.
`warn` = a trap. `danger` = a hard limit.

### SVG diagrams

Aim for **2–4 diagrams per chapter**, replacing mermaid and adding where a picture
carries real mechanism. Style: dark theme, 1px hairlines, no shadows, one accent
colour highlighting the focal element, generous whitespace, all coordinates
divisible by 2. Prefer *showing the mechanism* over decorating the text.

```
viewBox="0 0 760 H"  width="100%"  style="max-width:920px"
role="img" aria-label="A full sentence describing what the diagram shows."
```

Theme colours (use these exactly):

| Role | Hex |
|---|---|
| page background | `#0d1017` |
| panel | `#151b26` |
| deep panel | `#12161f` |
| border | `#232c3b` |
| ink | `#e7ecf3` |
| ink soft | `#aab4c5` |
| ink faint | `#6f7b8f` |
| accent (green) | `#5ee0c0` |
| accent green fill | `#14342c` |
| accent 2 (blue) | `#7aa2ff` |
| blue fill | `#1a2440` |
| warn | `#ffb454` |
| danger | `#ff6b6b` |
| danger fill | `#241a1a` |

Give every `<marker>` a **unique id per page** (`a1`, `b1`, …) or arrowheads collide.

---

## 2 · Verified facts — the only numbers you may use

### The machine, then and now

| | At the start | Now |
|---|---|---|
| GPU | 1 × RTX 3060 Ti, 8 GB | **3 × RTX 3060 Ti, 8 GB each (24 GB)** |
| Architecture | Ampere, sm_86 | same |
| Per-card bandwidth | ~448 GB/s | same |
| Power limit | 200 W per card | same |
| PCIe | 3.0 ×16 | same |
| CPU | 24 cores | **2 × Xeon E5-2665, 16 cores / 32 threads** |
| RAM | 60 GB | **125 GB DDR3-1600 (4 × 32 GB)** |
| Disk | 148 GB | **879 GB, 585 GB free** |
| Driver | 595.84 (exposes CUDA 13.2) | same |

### Software stack (measured on the box)

Python 3.12.3 · torch 2.11.0+cu128 · CUDA 12.8 · cuDNN 91900 ·
diffusers 0.40.0.dev0 · torchao 0.18.0 · transformers 5.16.1 ·
accelerate 1.14.0 · safetensors 0.8.0. ComfyUI 0.30 was the original engine and
**has been removed**; the pipeline is now our own (ch.48).

### The model — four parts

| Part | Size |
|---|---|
| Encoder (Qwen3-VL-32B), tapped at layer 50 | **26 GB** |
| Omni-Transformer / DiT, int8, pruned | **20 GB** |
| Visual VAE | 4.9 GB |
| Audio VAE | 0.6 GB |
| Original bf16 transformer | ~66 GB |

Capabilities: 4–15 s output, 24 fps, 32 kHz stereo, 11 languages, short side 768
by default. The 15 s ceiling is the **model's trained range**, not a hardware limit.

### Transformer config (from `transformer/config.json`)

`num_layers` 50 · `hidden_size` 5376 · `num_attention_heads` 56 ·
`attention_head_dim` 128 · `ffn_dim` 14336 · `in_channels` 24 ·
`audio_in_channels` 32 · `patch_size` [1,2,2] · `num_refiner_layers` 2 ·
`text_dim` 5120 · `time_embed_dim` 2688 · `rope_theta` 10000.

Derived and verified: ~395 M parameters per block × 50 ≈ **20 billion**, which at
one byte per parameter under int8 is the 20 GB file. ~0.4 GB per block streamed;
a 23-step render moves ~460 GB across PCIe.

### VAE config

Video VAE: `latent_channels` 24, `clip_length` 17, `token_drop` 3, spatial
compression **÷16**, **tiling on by default** at 256×256 with 64 px overlap.
Audio VAE: `latent_channels` 32.
Observed latent shape for 832×480×124: **`(1, 24, 37, 30, 52)`**.

### Measured renders — the complete table

| Scene | Resolution × frames | Steps | Denoise | s/step | Result |
|---|---|---|---|---|---|
| smoke | 512×320 × 5 | — | ~57 s | — | works |
| ocean demo | 768×448 × 73 (3.0 s) | 20 | 348 s | — | works |
| `cmp-ring` / `cmp-normal` | 832×480 × 124 | 4 | ~2 min | — | sharp; ring == reference |
| `cp15sd` | 512×288 × 345 (15 s) | **49** | 3965 s (66 min) | 81 s | **beautiful** |
| `e1h-15s` | 832×480 × 345 (15 s) | 3 | 578 s | 193 s | under-sampled |
| `an1` | **1280×704** × 124 (5 s) | **23** | 3295 s (55 min) | 143 s | **beautiful** |
| `e720hq` | 1280×704 × 124 | 23 | 3486 s (58 min) | 152 s | good |
| `g4k` | 1280×704 × 345 (14.4 s) | 3 | ~5.5 h | 6600 s | **mud** |
| `g4final` | 1280×704 × 345 | 23 | ~44 h | 6845 s | in progress |

**Steps needed by content:** tight close-up **4** · character mid-shot **23** ·
wide vista with small distant subject **49**. Step count is a property of the
*scene*, not the model.

Sharded-path cost: 832×480×345 → 1280×704×345 is 2.26× the tokens and **34×** the
time. Quadratic attention predicts ~5×, so the 3-GPU path carries roughly 7×
overhead beyond the maths.

### Failures — all real, all usable

- **g4j**: completed 6 h 10 m of denoise, then died on the first line of decode —
  `_execution_device` resolved to cuda:0 while the sharded forward returned latents
  on CPU. Both decode steps had the bug (`decoders.py:185` and `:245`).
- **g4k**: decode OOM at `[vae] clip 14` — `_decode` appended every finished pixel
  chunk to a GPU-resident list (~216 MB each at 1280×704). Not a capacity wall; an
  accumulation bug. Fix: move finished chunks to CPU on append.
- **The 848-pixel trap**: 848 ÷ 16 = 53, odd, so 2×2 patching fails with a shape
  error that never mentions resolution. Use widths where ÷16 is even (832 → 52).
- **ring-3 context parallelism**: OOM — offload re-copied weights per process.
- **SageAttention**: produces pure noise on H3. Keep it off.
- **`VAEDecodeTiled`**: incompatible — H3's latent is a NestedTensor.
- **LoRA at 480p**: softened the image; the base model was sharper.
- **`use_stream=True`** on offloading: pins tens of GB and the kernel OOM-kills the
  process with no traceback. Keep it `False`.
- Readable on-screen text is impossible for any current video diffusion model.
- Identity drifts across chained clips for photoreal faces; stylised characters
  hold far better.

### Costs and the honest verdict

Native 720p × 15 s **works** and costs ~44 h at 23 steps (~90 h at 49). The same
15 s at 832×480, 49 steps, is **2.6 h of denoise**, then a 1.54× upscale to 720p —
roughly 35× cheaper for a difference most viewers cannot see. Native 720p×15 s is a
proven capability, not a workflow.

---

## 2b · Other authorised sources of measured facts

The table above is not the whole record. These files contain further **real
measurements** from earlier phases of the project and are authorised sources —
cite freely from them, and prefer them over anything you would otherwise guess:

| File | What it holds |
|---|---|
| `experiments/BENCHMARK-int8-vs-q4.md` | The int8 vs Q4 head-to-head: 107 min vs 52 min, 7.5 GB vs 7.06 GB, native resolutions |
| `experiments/EXPERIMENT-LOG.md` | The 8 GB memory-profiling work, VRAM decompositions |
| `experiments/math-model.md` | The 2.24 GB `fast_int8_mm` buffer analysis and the memory model |
| `experiments/answer.md` | The "what fits in 8 GB" analysis |
| `planning/ENGINE-DESIGN.md` | Per-block sizes (0.65 GB int8, 1.29 GB bf16), engine architecture |
| `planning/PROJECT-SUMMARY.md` | Recipes and measured configurations |
| `planning/CP-RECIPE.md` | Context-parallel recipe details |
| `experiments/MULTI-GPU-RESEARCH.md` | Multi-GPU / NVLink / interconnect research |
| `experiments/SAMPLING-MATH.md` | Sampler and scheduler mathematics |
| `experiments/SEED-EXPERIMENT.md` | Seed reproducibility measurements |
| `experiments/itung-itungan.md` | Cost and throughput calculations |
| `history/JOURNEY.md` | The day-by-day build record |

**Rule unchanged:** a number must come from this spec, from one of these files, or
from the chapter you are editing. If you take a number from one of these files,
keep its context — do not restate a 480p measurement as though it were 720p.

Also note the **arc**: the project began on one 8 GB card running ComfyUI and now
runs three cards on its own pipeline. Older documents describe the earlier state.
That is not an error to correct — it is history to frame. Where an old measurement
has been superseded, show both and say what changed.

---

## 3 · Before you finish a chapter

Run these and fix anything they catch:

```bash
python3 - <<'EOF'
import re, os, sys, glob, xml.etree.ElementTree as ET
for f in sys.argv[1:] or glob.glob("*.html"):
    h = open(f, encoding="utf-8").read()
    ok = True
    for i, svg in enumerate(re.findall(r'<svg.*?</svg>', h, re.S)):
        if re.search(r'\n[ \t]*\n', svg): print(f"{f}: BLANK LINE in svg {i+1}"); ok = False
        try: ET.fromstring(svg)
        except Exception as e: print(f"{f}: svg {i+1} malformed: {e}"); ok = False
    if h.count('<div class="callout') != h.count("</div>"): print(f"{f}: callout/div mismatch")
    if "mermaid.min.js" in h: print(f"{f}: STILL LOADS MERMAID")
    if "vocabulary you now own" not in h: print(f"{f}: missing vocabulary section")
    links = set(re.findall(r'href="([^"]+\.html)"', h)) | set(re.findall(r'\]\(([^)]+\.html)\)', h))
    for l in links:
        if "assets/" in l or l.startswith(("..", "http")): continue
        if not os.path.exists(l): print(f"{f}: BROKEN LINK {l}")
    if ok: print(f"{f}: ok")
EOF
```

Do not mark a chapter done until it prints `ok` and no other warnings.
