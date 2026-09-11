#!/usr/bin/env python3
"""H3 fase B WITH LoRA. Same as h3_denoise_local.py, but merges a LoRA adapter into
the bf16 transformer weights BEFORE int8 quantization.

The LoRA is ComfyUI-format: keys `diffusion_model.blocks.N.attn.{qkv_proj,out_proj}.lora_{A,B}.weight`.
Our diffusers transformer uses `transformer_blocks.N.attn.{to_q,to_k,to_v,to_out.0}` — and the
LoRA's qkv_proj is a FUSED Q/K/V matrix, so its delta must be split three ways.

  W_eff = W + scale * (B @ A)   (card's formula; no alpha/r term)

Env: H3_LORA=/path/to.safetensors   H3_LORA_SCALE=0.8   (H3_F16=1 keeps bf16, no quant)
Uso: h3_denoise_lora.py <cena> [--steps N] [--saida out.mp4]
"""
import argparse
import os
import re
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3
import h3_meminstr  # memory probe (behavior-neutral)

EMB = os.path.expanduser("./h3-lab/embeds")
ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def log(m):
  print(f"[h3den] {m}", flush=True)


def merge_lora(tr, path, scale):
  """Merge a ComfyUI-format H3 LoRA into the bf16 transformer weights, in place."""
  from safetensors.torch import load_file
  sd = load_file(path)
  prefixes = set()
  for k in sd:
    m = re.match(r"(.*)\.lora_[AB]\.weight$", k)
    if m:
      prefixes.add(m.group(1))
  merged = skipped = 0
  for pfx in sorted(prefixes):
    A = sd.get(pfx + ".lora_A.weight")
    B = sd.get(pfx + ".lora_B.weight")
    if A is None or B is None:
      skipped += 1; continue
    dW = (B.float() @ A.float()) * scale               # [out, in]
    m = re.match(r"diffusion_model\.blocks\.(\d+)\.attn\.(qkv_proj|out_proj)$", pfx)
    if not m:
      skipped += 1; continue
    i, which = int(m.group(1)), m.group(2)
    try:
      attn = tr.transformer_blocks[i].attn
    except (IndexError, AttributeError):
      skipped += 1; continue
    if which == "out_proj":
      W = attn.to_out[0].weight
      if tuple(dW.shape) != tuple(W.shape):
        skipped += 1; continue
      W.data += dW.to(W.dtype).to(W.device)
      merged += 1
    else:  # fused qkv -> split rows into to_q / to_k / to_v
      inner = attn.to_q.weight.shape[0]
      if dW.shape[0] != 3 * inner:
        skipped += 1; continue
      for j, proj in enumerate((attn.to_q, attn.to_k, attn.to_v)):
        chunk = dW[j * inner:(j + 1) * inner]
        proj.weight.data += chunk.to(proj.weight.dtype).to(proj.weight.device)
      merged += 1
  return merged, skipped


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cena", choices=sorted(CENAS_H3))
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--saida", default=None)
  args = ap.parse_args()

  estado = torch.load(f"{EMB}/{args.cena}.pt", weights_only=False)
  log(f"estado da fase A: {sorted(estado)}")

  from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
  from diffusers.modular_pipelines import SequentialPipelineBlocks

  t0 = time.time()
  base = ModularPipeline.from_pretrained(ROOT)
  quer = ["denoise", "decode"]
  sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in quer}
  pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline(ROOT)
  log(f"pipeline parcial: {list(sub)}")

  # 1) load bf16 (NO quant yet — the LoRA must merge into full-precision weights)
  tr = MiniMaxH3Transformer3DModel.from_pretrained(ROOT, subfolder="transformer", dtype=torch.bfloat16)
  log(f"transformer bf16 em RAM em {time.time()-t0:.0f}s")

  # 2) merge the LoRA
  lora = os.environ.get("H3_LORA")
  if lora:
    scale = float(os.environ.get("H3_LORA_SCALE", "0.8"))
    merged, skipped = merge_lora(tr, lora, scale)
    log(f"LoRA merged: {merged} modules (scale={scale}), skipped {skipped} — {os.path.basename(lora)}")
    if merged == 0:
      log("!! WARNING: 0 modules merged — key mapping failed, LoRA had NO effect")

  # 3) quantize to int8 (unless full-precision requested)
  if os.environ.get("H3_F16") != "1":
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    NOT_CONVERT = ("proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                   "token_refiner", "norm_out", "proj_out", "audio_proj_out")
    def _filter(mod, fqn):
      return isinstance(mod, torch.nn.Linear) and not any(t in fqn for t in NOT_CONVERT)
    quantize_(tr, Int8WeightOnlyConfig(version=2), filter_fn=_filter)
    log(f"transformer int8 (post-merge) em {time.time()-t0:.0f}s")

  pipe.update_components(transformer=tr, transformer_ref=tr)
  restante = [n for n in getattr(pipe, "component_names", [])
              if n not in ("transformer", "transformer_ref")]
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)
    log(f"restante carregado: {restante}")

  offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
  pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
  pipe.transformer.requires_grad_(False)
  from diffusers.hooks import apply_group_offloading
  _off = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=False)
  _bigvae = getattr(pipe, "vae", None) or getattr(pipe, "video_vae", None)
  if _bigvae is not None:
    apply_group_offloading(_bigvae, offload_type="leaf_level", **_off)
  _av = getattr(pipe, "audio_vae", None)
  if _av is not None:
    _av.to("cuda")
  log(f"fase B pronta em {time.time()-t0:.0f}s")

  kwargs = {k: v for k, v in estado.items()}
  kwargs["generator"] = torch.Generator().manual_seed(2047)
  kwargs["num_frames"] = max(124, int(kwargs.get("num_frames") or 0))
  if args.steps:
    kwargs["num_inference_steps"] = args.steps
  kwargs["output"] = ["videos", "audio", "sampling_rate"]

  t = time.time()
  res = pipe(**kwargs)
  log(f"gerado em {time.time()-t:.0f}s")

  saida = args.saida or os.path.expanduser(f"./h3-lab/{args.cena}.mp4")
  os.makedirs(os.path.dirname(saida), exist_ok=True)
  from diffusers.utils import export_to_video
  if isinstance(res, dict):
    videos, audio, sr = res.get("videos"), res.get("audio"), res.get("sampling_rate", 32000)
  else:
    videos = getattr(res, "videos", None); audio = getattr(res, "audio", None); sr = getattr(res, "sampling_rate", 32000)
  video = videos[0] if isinstance(videos, (list, tuple)) else videos
  export_to_video(video, saida, fps=24)
  if audio is not None:
    try:
      import numpy as np
      import soundfile as sf
      arr = audio[0] if isinstance(audio, (list, tuple)) else audio
      if torch.is_tensor(arr):
        arr = arr.float().cpu().numpy()
      arr = np.asarray(arr)
      while arr.ndim > 2:
        arr = arr[0]
      sf.write(saida.replace(".mp4", ".wav"), arr.T.astype("float32"), sr)
      log(f"audio salvo ({sr}Hz)")
    except Exception as e:
      log(f"audio nao salvo: {type(e).__name__}: {e}")
  log(f"saida: {saida}")


if __name__ == "__main__":
  main()
