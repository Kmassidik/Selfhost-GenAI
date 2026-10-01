# Dalang Knowledge Roadmap

*From "I followed along" to "I can rebuild it blind."*

This is the learning path for everything we built — tied to our own knowledge base
(`../knowledge-base`) and our own engine code (`../source/dalang`). It's written from
the teacher's side of the table, but the order is deliberate: **build first, read to
fill the gaps, explain to lock it in.** Reading alone teaches nothing here.

## How to use it

```bash
bash roadmap.sh            # the whole map + your progress bar
bash roadmap.sh next       # the next thing to do
bash roadmap.sh show 3     # full detail of a stage (read / build / checkpoint)
bash roadmap.sh done 3     # tick a stage off
bash roadmap.sh reset      # start over
```

Each stage has three parts:
- **READ** — our KB chapter(s), plus one external paper/resource where it matters.
- **BUILD** — a small piece of code you write yourself. This is where the learning is.
- **CHECKPOINT** — a question. If you can answer it out loud, you own that stage.

## The ratio that matters

```
Reading:    ~20%   (you're mostly there already)
Building:   ~60%   (rewrite, break, measure)
Explaining: ~20%   (the knowledge base — you're already doing it)
```

Most people invert this — read 90%, build 10%, understand nothing. Do it backwards-correctly.

## The stages

| # | Stage | The one thing it teaches |
|---|-------|--------------------------|
| 0 | Orientation | the mission + the wall, before any code |
| 1 | The math floor | matmul is the whole game |
| 2 | What a model IS | it's just numbers in a shape |
| 3 | Attention | the one mechanism everything wraps |
| 4 | FlashAttention & online softmax | why attention fits in memory (and tiles) |
| 5 | Diffusion / flow-matching | how noise becomes a video in a few steps |
| 6 | Precision & quantization | precision ≠ quality, but it IS memory |
| 7 | The GPU & its memory | what actually sits in VRAM, and the 8 GB wall |
| 8 | Multi-GPU / context parallelism | two cards, one attention, one process |
| 9 | Python as glue vs native | where the real cost lives |
| 10 | **THE EXAM** | rebuild the Dalang engine from a blank file |

## The exam (Stage 10)

The real measure of understanding isn't pages read — it's this:

> **Can you rewrite the ~500-line Dalang engine from an empty file, no reference,
> and render a clip on one 8 GB card?**

The recipe you're rebuilding (documented in KB ch.44):

```
int8 weight-only + block-offload (1 block/group)
  + FFN chunked as a unit
  + rotary chunked
  + attention-linears chunked
  + ALL preallocating outputs (never torch.cat)
  + anti-frag allocator
  + VAE deferred onto the GPU at decode time
→ every per-token op is chunk-sized; only attention stays full/flash
```

Pass it, and you don't "understand the knowledge here" — you **own** it.

## Rough time

Not a shelf of books. About **5 papers + ~1.5 books' worth of chapters**, and — the
part that counts — **rewriting the engine 2–3 times.** Months of evenings, not years.
You're likely 60–70% of the way to "rebuild it blind" *right now*, because you built
each wall with your own hands instead of reading about someone else's.

🐢🔥
