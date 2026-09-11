# C1 — H3 attention block, read & understood

Source: `diffusers/models/transformers/transformer_minimax_h3.py`
(class `MiniMaxH3Attention` + its processor `__call__`). This is what the ring must replicate.

## The attention forward, step by step

Input `hidden_states`: `(batch, seq, hidden_size)`.

1. **QKV projection** — `to_q/to_k/to_v` (or fused `to_qkv`), each `nn.Linear(hidden_size, inner_dim, bias=False)`.
   `inner_dim = heads * head_dim`. → q,k,v each `(batch, seq, inner_dim)`.
2. **Split heads** — `unflatten(-1, (heads, head_dim))` → `(batch, seq, heads, head_dim)`.
3. **QK RMSNorm** — `q = norm_q(q)`, `k = norm_k(k)`, each `RMSNorm(head_dim)` (per-head normalization; NOT on value).
4. **Rotary** — if `rotary_emb=(cos,sin)` (shapes `(seq, rotary_dim)`): `_apply_rotary_emb` rotates the leading `rotary_dim`
   channels of q and k (3-axis t/h/w rope over the packed sequence, from `position_ids`). Value is untouched.
5. **Attention (the softmax)** — `dispatch_attention_fn(q, k, v, attn_mask=None, is_causal=False, ...)`.
   Standard scaled-dot-product: `softmax(QKᵀ / √head_dim) · V`. **No mask, non-causal, full bidirectional.**
6. **Merge + out** — `flatten(heads, head_dim)` → `to_out[0]` (Linear `inner_dim→hidden`, no bias) → `to_out[1]` (Dropout 0).

## What this means for the ring (the important part)

- **Full, non-causal attention → every query attends to every key.** This is the textbook Ring Attention case: split
  the sequence into N slices (one per GPU); each GPU holds `Q_i, K_i, V_i`; rotate `K/V` slices around the ring; each pass,
  accumulate `softmax(Q_i · K_jᵀ) · V_j` with **online softmax** (running max `m` + denominator `l`). After N passes, `Q_i`
  has attended to the whole sequence. No causal masking to worry about → simpler than the LLM case.
- **The split axis is `seq`** (dim 1). q/k/v are `(batch, seq, heads, head_dim)` → slice dim 1 across GPUs. Rotary is
  applied BEFORE attention using each slice's own `position_ids`, so each GPU rotates its own Q/K slice locally — clean.
- **Steps 1–4 and 6 are per-slice / embarrassingly parallel** — each GPU runs its own QKV proj, norm, rotary, out-proj on
  its `seq` slice. **Only step 5 (the softmax) needs cross-GPU communication** (the ring). That's the whole job.
- **We only need to replace `dispatch_attention_fn`** with our online-softmax ring. Everything else is local.

## How diffusers' CP does it (and why Dalang differs)

- `dispatch_attention_fn` takes a `parallel_config` — diffusers' context-parallelism rings the K/V across **torch.distributed
  ranks** (multi-PROCESS, one per GPU, via NCCL). That is exactly `torchrun --nproc_per_node=N`.
- **That path works for the ATTENTION but forces weight replication** (N processes → N model copies → the 66/99 GB RAM wall)
  and its group-offload re-copies per process (#12533). The attention was never the problem; the process model was.
- **Dalang is SINGLE-process, multi-GPU.** So we can't reuse the distributed ring directly — we implement our own
  single-process ring: hold Q/K/V slices on different CUDA devices, pass K/V device→device (`.to(other_device)` over PCIe,
  no NVLink), and accumulate online softmax on each device. One model in RAM, streamed to whichever device computes.

## C2/C3 plan (next, small steps)

- **C2** — implement `online_softmax_attention(q, k, v)` on ONE GPU (running max + denominator, processing k/v in blocks);
  prove it equals `dispatch_attention_fn` / plain `softmax(QKᵀ/√d)V` bit-for-bit on a tiny random tensor. *No devices yet.*
- **C3** — split `seq` across 2 CUDA devices; ring-pass K/V; accumulate C2's online softmax across the 2 passes.
- **C4** — wire it as the attention inside one real block, run the full transformer, diff vs `ref_out.pt` (the milestone).

## Key numbers to confirm at C2 (read from the loaded model, don't assume)
- `heads`, `head_dim`, `inner_dim`, `rotary_dim`, `hidden_size` — print them from a loaded `MiniMaxH3Attention` before coding C2.
- Attention scale = `1/√head_dim` (confirm dispatch_attention_fn doesn't apply a different scale).
