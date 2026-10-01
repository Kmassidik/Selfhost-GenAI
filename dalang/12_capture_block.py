#!/usr/bin/env python3
"""Dalang G1a: capture ONE transformer block's real I/O — the anchor for the
sequence-parallel block probe (G1). Hooks block[0].forward during a ref forward,
saves its args (hidden, temb, adaln_indices, rotary_emb) + output.
"""
import os, sys, torch
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
sys.path.insert(0, "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench")
OUT = "/root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench/dalang_ref"

def main():
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["proj_in","audio_proj_in","context_embedder","time_embedder",
                "time_proj","token_refiner","norm_out","proj_out","audio_proj_out"]))
    tr.requires_grad_(False)
    # find the block list
    blk_attr = None
    for name, mod in tr.named_children():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 10:
            blk_attr = name; break
    print(f"[g1a] block list attr = {blk_attr!r}, n = {len(getattr(tr, blk_attr))}", flush=True)
    block0 = getattr(tr, blk_attr)[0]
    print(f"[g1a] block0 = {type(block0).__name__}; submodules: {[n for n,_ in block0.named_children()]}", flush=True)
    print(f"[g1a] attn submodules: {[n for n,_ in block0.attn.named_children()]}", flush=True)

    cap = {}
    def hook(mod, args, kwargs, output):
        names = ["hidden_states","temb","adaln_indices","rotary_emb","attention_mask"]
        for i,a in enumerate(args):
            cap[names[i] if i<len(names) else f"arg{i}"] = a
        cap.update(kwargs); cap["__out__"] = output
        for k,v in list(cap.items()):
            if torch.is_tensor(v): print(f"[g1a]   {k}: {tuple(v.shape)} {v.dtype} {v.device}", flush=True)
            elif isinstance(v,(tuple,list)): print(f"[g1a]   {k}: {type(v).__name__} len={len(v)} [{'/'.join(str(tuple(t.shape)) for t in v if torch.is_tensor(t))}]", flush=True)
        raise SystemExit(0)  # one capture is enough
    block0.register_forward_hook(hook, with_kwargs=True)

    tr.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
        onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
    ref_in = torch.load(f"{OUT}/ref_in.pt", weights_only=False)
    try:
        with torch.no_grad(): tr(**dict(ref_in["kwargs"]))
    except SystemExit:
        # save what we captured (move tensors to CPU)
        save = {}
        for k,v in cap.items():
            if torch.is_tensor(v): save[k]=v.detach().cpu()
            elif isinstance(v,(tuple,list)): save[k]=[t.detach().cpu() if torch.is_tensor(t) else t for t in v]
            else: save[k]=v
        torch.save(save, f"{OUT}/blk0_io.pt")
        print(f"[g1a] SAVED {OUT}/blk0_io.pt  keys={list(save)}", flush=True)

if __name__ == "__main__":
    main()
