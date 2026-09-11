#!/usr/bin/env python3
"""H3 denoise with Context (Ring) Parallelism across N GPUs via torchrun.
Splits the activation SEQUENCE across ranks so each GPU holds ~1/N of the video
(the point: fit longer clips / higher res). Weights stream per-rank (int8).

Launch:
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=3 \
    h3_denoise_cp.py <cena> [--steps N] [--saida path]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3
import h3_meminstr  # per-rank memory probe — under CP it should show seq ÷ N

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"
EMB = os.path.expanduser("./h3-lab/embeds")

IS_DIST = "RANK" in os.environ
if IS_DIST:
  dist.init_process_group(backend="cpu:gloo,cuda:nccl")
  RANK = dist.get_rank(); WS = dist.get_world_size()
  torch.cuda.set_device(RANK)
else:
  RANK, WS = 0, 1


def log(m):
  print(f"[h3cp r{RANK}] {m}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cena", choices=sorted(CENAS_H3))
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--saida", default=None)
  args = ap.parse_args()

  estado = torch.load(f"{EMB}/{args.cena}.pt", weights_only=False)
  if RANK == 0:
    log(f"estado da fase A: {sorted(estado)}  world_size={WS}")

  from diffusers import (MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig,
                         ContextParallelConfig)
  from diffusers.modular_pipelines import SequentialPipelineBlocks
  from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
  from torchao.quantization import Int8WeightOnlyConfig

  t0 = time.time()
  base = ModularPipeline.from_pretrained(ROOT)
  quer = ["denoise", "decode"]
  sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in quer}
  pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline(ROOT)

  if os.environ.get("H3_SHARED") == "1":
    # custom shared-mmap loader (validated: missing=0, meta=0, weights in page-cache).
    # Assign safetensors mmap VIEWS (zero-copy) so the N ranks SHARE one page-cached
    # bf16 copy instead of holding N copies. This is the ring-3 RAM fix.
    import glob
    from accelerate import init_empty_weights
    from safetensors import safe_open
    tdir = ROOT + "/transformer"
    with init_empty_weights():
      tr = MiniMaxH3Transformer3DModel.from_config(MiniMaxH3Transformer3DModel.load_config(tdir))
    sd = {}
    for shard in sorted(glob.glob(tdir + "/*.safetensors")):
      with safe_open(shard, framework="pt", device="cpu") as f:
        for k in f.keys():
          sd[k] = f.get_tensor(k)
    tr.load_state_dict(sd, strict=False, assign=True)
    for _n, _m in tr.named_modules():
      if _m.__class__.__name__ == "MiniMaxH3RotaryPosEmbed":
        _d = _m.rope_freq_dim
        _m.register_buffer(
          "inv_freq",
          1.0 / (10000.0 ** (torch.arange(0, 2 * _d, 2, dtype=torch.float32) / (2 * _d))),
          persistent=False)
    log(f"transformer bf16 SHARED-MMAP em RAM em {time.time()-t0:.0f}s")
  else:
    tr = MiniMaxH3Transformer3DModel.from_pretrained(
      ROOT, subfolder="transformer", dtype=torch.bfloat16,
      quantization_config=TorchAoConfig(
        Int8WeightOnlyConfig(version=2),
        modules_to_not_convert=[
          "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
          "token_refiner", "norm_out", "proj_out", "audio_proj_out",
        ],
      ),
    )
    log(f"transformer int8 em RAM em {time.time()-t0:.0f}s")
  pipe.update_components(transformer=tr, transformer_ref=tr)
  restante = [n for n in getattr(pipe, "component_names", [])
              if n not in ("transformer", "transformer_ref")]
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)

  # ---- Context Parallelism: enable BEFORE offload (hook order, diffusers #12533) ----
  if IS_DIST:
    NL = 50
    h3_cp_plan = {
      "transformer_blocks.0": {"hidden_states": ContextParallelInput(split_dim=1, expected_dims=3)},
      "transformer_blocks.*": {
        "adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1),
        "rotary_emb": [ContextParallelInput(split_dim=0, expected_dims=2),
                       ContextParallelInput(split_dim=0, expected_dims=2)],
      },
      f"transformer_blocks.{NL-1}": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    pipe.transformer.set_attention_backend("_native_cudnn")
    pipe.transformer.enable_parallelism(
      config=ContextParallelConfig(ring_degree=WS, ring_anything=True), cp_plan=h3_cp_plan)
    log(f"CP enabled (ring_degree={WS})")

  # ---- offload AFTER CP ----
  offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                 use_stream=False)
  pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
  pipe.transformer.requires_grad_(False)
  from diffusers.hooks import apply_group_offloading
  _bigvae = getattr(pipe, "vae", None) or getattr(pipe, "video_vae", None)
  if _bigvae is not None:
    apply_group_offloading(_bigvae, offload_type="leaf_level", **offload)
  _av = getattr(pipe, "audio_vae", None)
  if _av is not None:
    _av.to("cuda")
  log(f"fase B (CP) pronta em {time.time()-t0:.0f}s")

  kwargs = {k: v for k, v in estado.items()}
  kwargs["generator"] = torch.Generator().manual_seed(2047)  # same seed on every rank
  kwargs["num_frames"] = max(124, int(kwargs.get("num_frames") or 0))
  if args.steps:
    kwargs["num_inference_steps"] = args.steps
  kwargs["output"] = ["videos", "audio", "sampling_rate"]

  t = time.time()
  res = pipe(**kwargs)
  log(f"gerado em {time.time()-t:.0f}s  (STAGE2 OK if no error)")

  if RANK == 0:
    saida = args.saida or os.path.expanduser(f"./h3-lab/{args.cena}-cp.mp4")
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    from diffusers.utils import export_to_video
    if isinstance(res, dict):
      videos, audio, sr = res.get("videos"), res.get("audio"), res.get("sampling_rate", 32000)
    else:
      videos = getattr(res, "videos", None); audio = getattr(res, "audio", None)
      sr = getattr(res, "sampling_rate", 32000)
    video = videos[0] if isinstance(videos, (list, tuple)) else videos
    try:
      export_to_video(video, saida, fps=24)
      if audio is not None:
        import numpy as np, soundfile as sf
        arr = audio[0] if isinstance(audio, (list, tuple)) else audio
        if torch.is_tensor(arr):
          arr = arr.float().cpu().numpy()
        arr = np.asarray(arr)
        while arr.ndim > 2:
          arr = arr[0]
        sf.write(saida.replace(".mp4", ".wav"), arr.T.astype("float32"), sr)
      log(f"saida: {saida}")
    except Exception as e:
      log(f"save skipped ({type(e).__name__}: {e}) — fine for a 2-step smoke test")

  if IS_DIST:
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
