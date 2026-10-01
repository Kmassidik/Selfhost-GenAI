#!/usr/bin/env python3
"""Dalang Engine — FIRST PROBE: can the int8 model be held/queued by ONE CPU process?

Before ring attention, before multi-GPU, the foundation: prove a SINGLE process can
load the int8 H3 transformer into CPU RAM as ONE copy, and measure its footprint.
This is the premise of the whole Dalang design (one dalang/CPU holds one model).

Logs, with hard numbers:
  - system RAM used BEFORE load
  - system RAM used AFTER load  -> delta = the int8 model's RAM footprint (ONE copy)
  - process RSS (what this single process holds)
  - verdict: does one CPU process hold the int8 model? (expected YES, ~20-33 GB)

RUN (box, from h3-consumer-bench dir):
    python 00_probe_load.py
"""
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def used_ram_gb():
    """System RAM used = MemTotal - MemAvailable, in GB (Linux /proc/meminfo)."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.strip().split()[0])  # kB
    return (info["MemTotal"] - info["MemAvailable"]) / 1024 / 1024


def rss_gb():
    """This process's resident set size, in GB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return -1.0


def log(m):
    print(f"[dalang-probe] {m}", flush=True)


def main():
    ram0 = used_ram_gb()
    rss0 = rss_gb()
    log(f"BEFORE load:  system RAM used = {ram0:.1f} GB   process RSS = {rss0:.2f} GB")

    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    t0 = time.time()
    log("loading int8 H3 transformer in THIS single process ...")
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
        ROOT, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
            ],
        ),
    )
    tr.requires_grad_(False)
    dt = time.time() - t0

    ram1 = used_ram_gb()
    rss1 = rss_gb()
    n_params = sum(p.numel() for p in tr.parameters())
    n_blocks = len(getattr(tr, "transformer_blocks", []))

    log(f"AFTER  load:  system RAM used = {ram1:.1f} GB   process RSS = {rss1:.2f} GB   (loaded in {dt:.0f}s)")
    log(f"MODEL: {n_params/1e9:.1f} B params, {n_blocks} transformer blocks, on device = {next(tr.parameters()).device}")
    log(f"ONE-COPY FOOTPRINT: RAM delta = {ram1-ram0:.1f} GB   |   RSS delta = {rss1-rss0:.2f} GB")
    log("VERDICT: one CPU process CAN hold the int8 model in RAM. "
        f"So N ranks sharing THIS single copy (Dalang) would need ~{ram1-ram0:.0f} GB, "
        f"not N x that. (torchrun replicates it: 2x -> ~66 GB, 3x -> ~99 GB = the ring-3 RAM wall.)")
    log("PROBE-DONE")


if __name__ == "__main__":
    main()
