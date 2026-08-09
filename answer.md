# MiniMax-H3 on 8 GB VRAM — Memory-First Execution Experiment

## Machine

```text
GPU        : NVIDIA GeForce RTX 3060 Ti
VRAM       : 8 GB
Architecture: Ampere / sm_86
Driver     : 595.84
CUDA       : 13.2

CPU        : 24 cores
RAM        : 60 GB
Disk       : 148 GB NVMe (~90 GB free)
/tmp       : 31 GB tmpfs (RAM disk)

OS         : Ubuntu 26.04 LTS
Runtime    : Python 3.12 + uv
PyTorch    : 2.11 (cu128)
ComfyUI    : 0.30

ComfyUI:
  --disable-smart-memory
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Model

```text
MiniMax-H3
33B
Flow Matching
model_type = FLOW_AV

DiT:
  minimax_h3_fl2va_pruned_int8_convrot.safetensors
  ~19.5 GB

Encoder:
  qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  ~26 GB

Video VAE:
  minimax_h3_video_vae_fp16.safetensors
  ~4.9 GB

Audio VAE:
  minimax_h3_audio_vae_fp32.safetensors
  ~0.6 GB
```

Weights are not permanently resident in VRAM.

The current system already streams model weights through the 60 GB system RAM into the 8 GB GPU.

Therefore:

[
\boxed{
\text{VRAM is a working buffer, not the model's permanent storage}
}
]

---

# 1. The Original Question Is Already Answered

The original question was:

> Can MiniMax-H3 reproduce the current output quality while keeping VRAM <= 8 GB?

The answer is already:

[
\boxed{\text{YES}}
]

The current baseline successfully runs:

[
576\times320\times362
]

with approximately:

[
M_{\text{peak}}\approx7.57\text{ GB}
]

on an 8 GB RTX 3060 Ti.

Therefore, the interesting research question is no longer:

[
\text{"Can H3 fit into 8 GB?"}
]

It is:

[
\boxed{
\text{"How much can we reduce the H3 working-set VRAM while preserving the same computation and quality?"}
}
]

The freed VRAM can then be spent on:

* higher resolution
* more frames
* larger latent dimensions
* additional context
* larger generation targets

The ultimate objective is:

[
\boxed{
M_{\text{VRAM}}\downarrow
\quad+\quad
Q\approx Q_{\text{baseline}}
}
]

while accepting:

[
T_{\text{runtime}}\uparrow
]

---

# 2. Core Philosophy

Do not make the generation problem smaller.

Instead:

[
\boxed{
\textbf{Make the memory working set smaller.}
}
]

Do not immediately reduce:

[
W,\ H,\ F
]

For example, do not assume:

[
576\times320\times362
\rightarrow
384\times216\times121
]

is necessary.

Instead investigate:

[
576\times320\times362
\rightarrow
\text{same logical computation}
\rightarrow
\text{smaller VRAM working set}
]

The fundamental distinction is:

[
\boxed{
\text{partition the computation}
\neq
\text{reduce the problem}
}
]

---

# 3. Baseline Video

Use the existing duet video as the baseline benchmark.

The video is approximately:

```text
Resolution : 1920x1080 output
Duration   : ~15 seconds
FPS        : 24
Frames     : 362
Scene      : two people singing into a microphone
```

This is a strong benchmark because it stresses:

* identity consistency
* facial consistency
* mouth/lip motion
* body motion
* temporal consistency
* lighting consistency
* composition
* long temporal context
* audio/video relationship

Do not use a simple static scene as the primary benchmark.

Define:

[
B=\text{current baseline generation}
]

Every optimized implementation must be compared against:

[
B
]

---

# 4. Exact vs Approximate Optimizations

Separate optimization techniques into two categories.

## 4.1 Execution-preserving optimizations

These change **how** computation is scheduled rather than intentionally changing **what** is computed.

Candidate techniques:

* FFN chunking
* attention splitting
* exact tiled attention
* layer streaming
* weight offloading
* activation offloading
* activation recomputation
* CPU/RAM staging
* asynchronous prefetching

For these methods, the target is:

[
Q_{\text{optimized}}\approx Q_{\text{baseline}}
]

Numerical differences may exist because of floating-point execution order.

Do not automatically claim:

[
\Delta Q=0
]

unless exact bitwise equality has actually been demonstrated.

`torch.testing.assert_close()` demonstrates numerical closeness within tolerance; it does not by itself prove bit-identical tensors.

Only claim:

[
\boxed{\Delta Q=0}
]

if an appropriate exact equality test has been performed.

Otherwise use:

[
\boxed{\Delta Q\approx0}
]

---

## 4.2 Approximate optimizations

These intentionally change the computation or model representation.

Examples:

* sparse attention
* approximate attention
* lower quantization
* frame reduction
* resolution reduction
* latent compression
* aggressive approximation

These can introduce:

[
\Delta Q\neq0
]

Therefore they should be investigated separately.

The first phase of this project should focus entirely on execution-preserving methods.

---

# 5. Existing Memory Measurements

Current measurements:

| Resolution | Frames | Steps | Peak VRAM |
| ---------- | -----: | ----: | --------: |
| 512×288    |      5 |     2 |   7412 MB |
| 512×288    |     73 |     2 |   6570 MB |
| 512×288    |    181 |     2 |   6570 MB |
| 512×288    |    362 |     2 |   7834 MB |
| 576×320    |    362 |     2 |   7832 MB |

Important observation:

[
M(73)\approx M(181)
]

despite:

[
181/73\approx2.48
]

times more frames.

This strongly suggests a large frame-independent memory component.

A crude empirical model is:

[
M_{\text{peak}}
\approx
M_{\text{fixed}}
+
M_{\text{frame-dependent}}
]

For 512×288:

[
M_{\text{fixed}}\approx6570\text{ MB}
]

and for 362 frames:

[
7834-6570
\approx1264\text{ MB}
]

Therefore:

[
M_{\text{peak}}
\approx
6.57\text{ GB}
+
M_{\text{variable}}
]

However:

[
\boxed{
6.57\text{ GB is an observed floor, not yet a proven theoretical floor.
}
]

---

# 6. The Most Important Unknown

We currently do not know what makes up the ~6.5 GB observed floor.

Potential components:

[
M_{\text{floor}}
================

M_{\text{encoder}}
+
M_{\text{DiT}}
+
M_{\text{workspace}}
+
M_{\text{allocator}}
+
M_{\text{other}}
]

Possible sources include:

* Qwen3-VL encoder working memory
* resident DiT working set
* CUDA workspace
* attention buffers
* persistent activations
* allocator reservation
* fragmentation
* other ComfyUI components

This must be measured before making assumptions about the next optimization.

---

# 7. Experiment #1 — Capture a CUDA Memory Snapshot

The first experiment should NOT be an X sweep.

First determine what the 6.5 GB actually contains.

Use PyTorch CUDA memory history:

```python
torch.cuda.memory._record_memory_history(
    enabled="all"
)
```

Run the relevant H3 operation, then:

```python
torch.cuda.memory._dump_snapshot(
    "h3_mem.pickle"
)
```

Analyze the snapshot using PyTorch's memory visualization tooling.

The objective is to attribute the peak memory to actual allocation sites.

The question is:

[
\boxed{
\text{"Where exactly are the 6.5 GB being used?"}
}
]

---

# 8. Do Not Capture Only One Point

Capture memory behavior across the entire pipeline.

Conceptually:

```text
START
  |
  v
Encoder loaded
  |
  v
Encoder execution
  |
  v
Encoder complete
  |
  v
Encoder evicted?
  |
  v
DiT loaded
  |
  v
Sampling
  |
  v
VAE
  |
  v
Output
```

Measure:

[
M_{\text{VRAM}}(t)
]

for each phase.

Create a table like:

| Pipeline Phase    | Peak VRAM | Resident Objects |
| ----------------- | --------: | ---------------- |
| Empty             |         ? | ?                |
| Encoder loaded    |         ? | ?                |
| Encoder executing |         ? | ?                |
| Encoder finished  |         ? | ?                |
| Encoder evicted   |         ? | ?                |
| DiT loaded        |         ? | ?                |
| Sampling          |         ? | ?                |
| VAE               |         ? | ?                |

This should immediately reveal whether the floor is associated with the encoder, DiT, attention, workspace, or allocator.

---

# 9. Experiment #2 — Encode Then Completely Evict

The encoder is approximately:

[
26\text{ GB}
]

on disk/RAM.

If the encoder only needs to execute before diffusion/flow sampling, investigate:

[
\boxed{
\text{Encode}
\rightarrow
\text{store conditioning}
\rightarrow
\text{fully evict encoder}
\rightarrow
\text{DiT sampling}
}
]

The key question:

[
M_{\text{sampling}}
<
M_{\text{sampling+encoder}}
\quad ?
]

If encoder eviction produces a large reduction:

[
6.5\text{ GB}
\rightarrow
\text{significantly lower}
]

then encoder residency is a major bottleneck.

If almost nothing changes, the floor is elsewhere.

This experiment should be treated as a separate axis from attention/FFN chunking.

---

# 10. Experiment #3 — FFN Chunk Sweep

The first computational partitioning axis:

[
X_{\text{FFN}}
]

Test progressively:

[
X_{\text{FFN}}\in
{1,2,4,8,16,\ldots}
]

Measure:

[
(M_{\text{peak}},T_{\text{forward}})
]

The expected behavior is:

[
X_{\text{FFN}}\uparrow
\Rightarrow
M_{\text{peak}}\downarrow
]

until an asymptote is reached.

Once:

[
M(X+1)\approx M(X)
]

further FFN chunking is no longer useful.

---

# 11. Experiment #4 — Attention Split Sweep

Define:

[
X_{\text{ATTN}}
]

and test progressively more aggressive attention partitioning.

The critical requirement is:

[
\boxed{
\text{Do not change the required global K/V context.}
}
]

If:

[
Q=
[Q_1,Q_2,\ldots,Q_X]
]

then a memory-efficient exact implementation should compute each query chunk against the required complete key/value context:

[
O_i=
\operatorname{softmax}
\left(
\frac{Q_iK^T}{\sqrt d}
\right)V
]

rather than incorrectly replacing it with:

[
Q_iK_i^T
]

unless the architecture explicitly specifies local attention.

The objective is:

[
\boxed{
\text{same attention result}
+
\text{smaller peak workspace}
}
]

---

# 12. Experiment #5 — Layer Streaming

Investigate whether model layers can be streamed independently.

Conceptually:

```text
RAM
 |
 | load layer
 v
VRAM
 |
 | execute
 v
result
 |
 | evict
 v
RAM
```

Instead of requiring:

[
M_{\text{all weights}}
]

in VRAM, use:

[
M_{\text{working}}
==================

M_{\text{current layer}}
+
M_{\text{current activations}}
+
M_{\text{workspace}}
]

The 60 GB RAM becomes the large model storage pool.

The 8 GB VRAM becomes the compute working set.

Runtime can increase significantly.

That is acceptable.

---

# 13. Experiment #6 — Activation Offloading

Investigate moving inactive activations:

[
A_i
]

from:

[
\text{VRAM}
\rightarrow
\text{RAM}
]

and bringing them back only when required.

Conceptually:

[
A_i
\rightarrow
\text{GPU}
\rightarrow
\text{compute}
\rightarrow
\text{CPU RAM}
]

This trades:

[
\boxed{
\text{VRAM}
\leftrightarrow
\text{PCIe transfer}
}
]

The experiment must measure whether transfer cost is acceptable.

---

# 14. Experiment #7 — Recomputation

Instead of storing a large activation:

[
A_i
]

discard it:

[
A_i\rightarrow\varnothing
]

and recompute it later:

[
\varnothing
\rightarrow
\text{recompute}(A_i)
]

This gives the explicit trade:

[
\boxed{
\text{VRAM}\downarrow
\quad\Longleftrightarrow\quad
\text{compute}\uparrow
}
]

Since runtime is not the primary constraint, this is a legitimate strategy.

---

# 15. Experiment #8 — Asynchronous RAM/VRAM Pipeline

Do not necessarily execute:

```text
load
wait
compute
wait
evict
wait
load
wait
compute
```

Investigate pipelining:

```text
GPU:   compute X1 ---- compute X2 ---- compute X3
CPU:        load X2 -------- load X3 -------- load X4
```

The target is:

[
\boxed{
\text{CPU prepares future work while GPU computes current work}
}
]

This may recover some of the runtime lost to memory streaming without increasing peak VRAM substantially.

---

# 16. Do Not Use Temporal Clip Splitting

Do NOT turn:

[
362\text{ frames}
]

into independently generated clips:

[
100+100+100+62
]

That changes the generation problem and can destroy:

* temporal identity
* motion continuity
* global attention
* scene consistency
* audio synchronization

The fundamental rule is:

[
\boxed{
\textbf{Partition the computation, not the video.}
}
]

The 362-frame generation should remain one logical generation.

---

# 17. Spatial Tiling Requires the Same Warning

Do not blindly generate:

[
576\times320
]

as independent spatial tiles.

Independent tiles can produce:

* seams
* inconsistent objects
* inconsistent lighting
* inconsistent identities
* broken global composition

If spatial partitioning is investigated, it should preserve the necessary global context.

---

# 18. Define Separate X Variables

Do not use a single generic:

[
X
]

because different memory optimizations attack different components.

Use:

[
X_{\text{FFN}}
]

[
X_{\text{ATTN}}
]

[
X_{\text{LAYER}}
]

[
X_{\text{ACT}}
]

Potentially:

[
X_{\text{TEMPORAL}}
]

for carefully designed temporal computation partitioning.

Then model:

[
M_{\text{peak}}
===============

f(
X_{\text{FFN}},
X_{\text{ATTN}},
X_{\text{LAYER}},
X_{\text{ACT}}
)
]

This allows the experiment to identify which lever actually moves the VRAM needle.

---

# 19. Expected Memory Curve

The previous assumption was:

[
X\uparrow
\Rightarrow
M_{\text{VRAM}}\downarrow
]

indefinitely.

That is incorrect.

A more realistic model is:

[
M_{\text{peak}}(X)
==================

M_{\text{floor}}
+
M_{\text{partitionable}}(X)
]

with:

[
\lim_{X\rightarrow\infty}
M_{\text{peak}}(X)
==================

M_{\text{floor}}
]

Therefore the curve should eventually flatten.

For example:

```text
VRAM
 ^
 |\
 | \
 |  \
 |   \______
 |          \________  floor
 |
 +------------------------> X
```

The critical research question is:

[
\boxed{
\text{What exactly determines }M_{\text{floor}}?
}
]

---

# 20. Current Hypothesis

Current measurements suggest:

[
M_{\text{floor}}\approx6.5\text{ GB}
]

but this is only an observed floor.

Possible explanations:

### Hypothesis A — Encoder residency

[
M_{\text{floor}}
\approx
M_{\text{encoder}}
+
M_{\text{other}}
]

If true:

[
\text{encode}
\rightarrow
\text{evict}
]

could significantly lower the floor.

### Hypothesis B — DiT resident working set

The streamed model still requires a substantial persistent working set.

If true:

[
\text{FFN chunking}
]

may have limited impact.

### Hypothesis C — Attention/workspace

The fixed floor may contain persistent attention or CUDA workspace allocations.

If true:

[
X_{\text{ATTN}}
]

could be the more important lever.

### Hypothesis D — Allocator/fragmentation

Part of the observed number may be:

[
M_{\text{reserved}}
-------------------

M_{\text{allocated}}
]

rather than useful tensors.

If true, allocator configuration and allocation lifetime may provide additional headroom.

---

# 21. The 640×384 OOM Result

An important existing observation:

> 640×384 still OOM'd even with FFN chunking at chunks=8.

This suggests FFN intermediate memory is not the only limiting factor.

However, this does NOT by itself prove what the floor is.

Possible explanation:

[
M_{\text{fixed}}
+
M_{\text{attention}}
+
M_{\text{workspace}}

>

8\text{ GB}
]

The exact OOM point must therefore be profiled.

Do not infer the floor's composition from the OOM alone.

---

# 22. Define the Optimization Target

The practical target should not be:

[
M_{\text{peak}}=8\text{ GB}
]

That leaves almost no safety margin.

Instead target:

[
\boxed{
M_{\text{peak}}\leq7\text{ GB}
}
]

while preserving baseline quality.

This gives approximately:

[
1\text{ GB}
]

of operational headroom.

The ultimate success condition becomes:

[
\boxed{
M_{\text{peak}}\leq7\text{ GB}
}
]

[
\boxed{
Q_{\text{optimized}}\approx Q_{\text{baseline}}
}
]

with:

[
T_{\text{optimized}}>T_{\text{baseline}}}
]

being acceptable.

---

# 23. Measurement Matrix

For every experiment record:

[
\boxed{
(M_{\text{peak}},T_{\text{total}},Q)
}
]

For exact/execution-preserving methods, quality should primarily be a correctness check rather than the optimization variable.

Record:

```text
Peak VRAM
Allocated VRAM
Reserved VRAM
CPU RAM
Transfer volume
Transfer time
Forward-pass time
Sampling time
VAE time
Total generation time
OOM / success
Numerical output comparison
```

For approximate methods additionally record:

```text
Visual quality
Temporal consistency
Identity consistency
Motion quality
Prompt adherence
Audio/video synchronization
Artifact score
```

---

# 24. Recommended Experiment Order

The correct order is:

```text
1. CUDA memory snapshot
        |
        v
2. Identify the ~6.5 GB observed floor
        |
        v
3. Encode -> fully evict encoder
        |
        v
4. FFN chunk sweep
        |
        v
5. Attention split sweep
        |
        v
6. Layer streaming
        |
        v
7. Activation offload / recomputation
        |
        v
8. Combine the best exact methods
        |
        v
9. Find minimum stable VRAM configuration
        |
        v
10. Increase resolution / frames
        |
        v
11. Only then investigate approximate methods
```

---

# 25. The Real Research Objective

The project should no longer be described as:

[
\boxed{
\text{"How do we make MiniMax-H3 faster?"}
}
]

The actual project is:

[
\boxed{
\textbf{
How far can a 33B video model be pushed on an 8 GB GPU
by trading compute time, CPU RAM, and memory transfers
against GPU memory?
}
}
]

The desired relationship is:

[
\boxed{
M_{\text{VRAM}}\downarrow
}
]

[
\boxed{
T_{\text{runtime}}\uparrow\quad\text{(acceptable)}
}
]

[
\boxed{
Q_{\text{output}}\approx Q_{\text{baseline}}
}
]

The long-term goal is:

[
\boxed{
\text{Free VRAM first}
\rightarrow
\text{increase resolution/frames second}
}
]

rather than:

[
\boxed{
\text{reduce resolution/frames}
}
]

---

# 26. Final Principle

The entire experiment can be summarized as:

[
\boxed{
\textbf{Do not make the model's problem smaller.}
}
]

[
\boxed{
\textbf{Make the GPU's working set smaller.}
}
]

Use the machine as a heterogeneous memory system:

[
\boxed{
\text{8 GB VRAM}
+
\text{60 GB RAM}
+
\text{RAM Disk}
+
\text{CPU}
+
\text{time}
}
]

The GPU does not need to hold everything.

It only needs to hold:

[
\boxed{
\text{the minimum state required to perform the next valid computation}
}
]

That is the central hypothesis to test.
