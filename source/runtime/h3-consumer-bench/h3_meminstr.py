# h3_meminstr.py — measure peak VRAM + sequence length inside the H3 denoise.
# Behavior-neutral: wraps the transformer block forward to log memory as the
# activation climbs through the first step's blocks, so we can size the real
# activation footprint (and how it will scale to more frames). Import this
# BEFORE the pipeline runs.
import torch
from diffusers.models.transformers import transformer_minimax_h3 as M

_orig = M.MiniMaxH3TransformerBlock.forward
_state = {"n": 0}
_MAX_PRINT = 6  # first few blocks of the first step is enough

def _instr_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    if _state["n"] < _MAX_PRINT and torch.cuda.is_available():
        try:
            shp = tuple(hidden_states.shape)
            seq = hidden_states.shape[-2] if hidden_states.dim() >= 2 else hidden_states.shape[0]
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[meminstr] block {_state['n']}: hidden={shp} seq={seq} "
                  f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB", flush=True)
        except Exception as e:
            print(f"[meminstr] probe error: {type(e).__name__}: {e}", flush=True)
        _state["n"] += 1
    return _orig(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

M.MiniMaxH3TransformerBlock.forward = _instr_forward
try:
    torch.cuda.reset_peak_memory_stats()
except Exception:
    pass
print("[meminstr] patched MiniMaxH3TransformerBlock.forward (memory logging active)", flush=True)
