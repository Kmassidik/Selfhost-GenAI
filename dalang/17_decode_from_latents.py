#!/usr/bin/env python3
"""Turn a G4K latent backup back into mp4 (+wav) WITHOUT re-running denoise.

Denoise on this box is ~2h per step across 3 GPUs; decode is minutes.  When a
render dies in decode (as g4j did on 2026-09-07, after 6h10m of perfect
denoise), the latents dumped by h3_decode_fix ARE the render -- this finishes
the job from them.  Only the VAEs are loaded; the 33B transformer never enters
RAM, which is also why decode fits here when it OOMs inside a live render.

  ./17_decode_from_latents.py --video docs/g4k.video.pt \
                              --audio docs/g4k.audio.pt \
                              --saida h3-lab/g4k.mp4

--audio is optional: h3_decode_fix also stashes a copy of the audio latents in
the video dump, so a single file can recover both tracks.

NOTE on loading: models/H3-root/vae carries a shard index whose safetensors are
not on disk -- the real weights resolve through the HF cache.  So we build the
decode-only ModularPipeline exactly as h3_denoise_local does and take the VAEs
off it, rather than calling AutoencoderKL*.from_pretrained on that folder.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

ROOT = "/root/Desktop/selfhosted-minimaxi-h3/models/H3-root"


def log(m):
    print(f"[g4k-dec] {m}", flush=True)


def load_decode_pipe():
    """Decode-only ModularPipeline, loaded the same way the render does."""
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    base = ModularPipeline.from_pretrained(ROOT)
    sub = {k: v for k, v in base.blocks.sub_blocks.items() if k == "decode"}
    if not sub:
        raise RuntimeError(f"no 'decode' block; available: {list(base.blocks.sub_blocks)}")
    pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline(ROOT)
    names = list(getattr(pipe, "component_names", []))
    log(f"decode components: {names}")
    if names:
        pipe.load_components(names=names, dtype=torch.bfloat16)
    return pipe


@torch.no_grad()
def decode_video(pipe, d, dev, fps, saida):
    """Decode the video latents and write the mp4.

    Two hard-won constraints shape this:

    1. The H3 video VAE ships with spatial tiling ON (tile_sample_min_* = 256,
       use_tiling = True), so per-tile VRAM is capped no matter the resolution.
       Higher resolution costs more tiles and more time, not a bigger peak.
    2. The DECODED tensor is what kills you. At 1280x704x345 the video is
       345*704*1280*3 float32 = 3.7 GB, and the denormalisation
       `(video.float() * std + mean).clamp(0,1)` makes 2-3 live copies of it.
       That is fine at 832x480 (595 MB each) and fatal on an 8 GB card at 720p.
       So the denormalisation is done on the CPU, in frame chunks, in-place.

    --device cpu skips the GPU entirely: slow, but with 125 GB of RAM it cannot
    run out of memory, which is the point when a dump represents 6 hours of work.
    """
    from diffusers.utils import export_to_video

    vae = pipe.vae.eval()
    on_cpu = dev.type == "cpu"

    if on_cpu:
        vae = vae.to("cpu").float()
        log("video VAE on CPU (slow, but cannot OOM)")
        latents = d["latents"].float()
        mean = torch.tensor(d["latents_mean"]).view(1, -1, 1, 1, 1).float()
        std = torch.tensor(d["latents_std"]).view(1, -1, 1, 1, 1).float()
        latents = latents * std + mean
        video = vae.decode(latents, return_dict=False)[0]
    else:
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(vae, onload_device=dev, offload_device=torch.device("cpu"),
                               offload_type="leaf_level", use_stream=False)
        log("video VAE: leaf-level group offload (streams through the card)")
        mean = torch.tensor(d["latents_mean"], device=dev).view(1, -1, 1, 1, 1)
        std = torch.tensor(d["latents_std"], device=dev).view(1, -1, 1, 1, 1)
        latents = d["latents"].to(dev) * std + mean
        try:
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=True):
                video = vae.decode(latents, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError:
            log("CUDA OOM in decode; falling back to CPU (slow, but it finishes)")
            del latents
            torch.cuda.empty_cache()
            vae = pipe.vae.to("cpu").float()
            lat = d["latents"].float()
            m = torch.tensor(d["latents_mean"]).view(1, -1, 1, 1, 1).float()
            sd = torch.tensor(d["latents_std"]).view(1, -1, 1, 1, 1).float()
            video = vae.decode(lat * sd + m, return_dict=False)[0]

    # Denormalise on the CPU in frame chunks: never hold more than one chunk of
    # working copies, instead of 2-3 copies of the whole 3.7 GB video.
    video = video.to("cpu").float()
    pm = torch.tensor(d.get("pixel_mean", getattr(pipe, "pixel_mean"))).view(1, -1, 1, 1, 1).float()
    ps = torch.tensor(d.get("pixel_std", getattr(pipe, "pixel_std"))).view(1, -1, 1, 1, 1).float()
    n_frames = video.shape[2] if video.dim() == 5 else video.shape[0]
    log(f"denormalising {tuple(video.shape)} on CPU in chunks of 32 frames")
    for i in range(0, n_frames, 32):
        sl = video[:, :, i:i + 32] if video.dim() == 5 else video[i:i + 32]
        sl.mul_(ps).add_(pm).clamp_(0, 1)
    log("denormalised")

    frames = pipe.video_processor.postprocess_video(video, output_type=d.get("output_type", "pil"))
    frames = frames[0] if isinstance(frames, (list, tuple)) else frames
    export_to_video(frames, saida, fps=fps)
    log(f"video -> {saida}")


@torch.no_grad()
def decode_audio(pipe, da, key, dev, saida):
    import numpy as np
    import soundfile as sf

    avae = pipe.audio_vae.to(dev).eval()
    if dev.type == "cpu":
        avae = avae.float()
    am = torch.tensor(da.get("audio_latents_mean", avae.config.latents_mean), device=dev).view(1, -1, 1)
    asd = torch.tensor(da.get("audio_latents_std", avae.config.latents_std), device=dev).view(1, -1, 1)
    al = da[key].to(dev).to(avae.dtype) * asd.to(avae.dtype) + am.to(avae.dtype)
    audio = avae.decode(al, return_dict=False)[0].float().permute(1, 0, 2)

    sr = da.get("audio_sampling_rate", getattr(pipe, "audio_sampling_rate", 32000))
    arr = audio.detach().cpu().numpy()
    while arr.ndim > 2:                        # (1, 2, N) -> (2, N)
        arr = arr[0]
    wav = saida.replace(".mp4", ".wav")
    sf.write(wav, np.asarray(arr).T.astype("float32"), sr)   # (N, 2) stereo
    log(f"audio -> {wav} ({sr}Hz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="the *.video.pt dump")
    ap.add_argument("--audio", default=None, help="the *.audio.pt dump (optional)")
    ap.add_argument("--saida", required=True, help="output .mp4 (wav written alongside)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=24)
    # The audio VAE is tiny, but loading it while the video VAE is still
    # resident OOMs the card -- that is how run g4k lost its soundtrack after
    # a successful video decode. --audio-only recovers it without redecoding.
    ap.add_argument("--audio-only", action="store_true")
    a = ap.parse_args()

    dev = torch.device(a.device)
    os.makedirs(os.path.dirname(os.path.abspath(a.saida)) or ".", exist_ok=True)
    t0 = time.time()

    d = torch.load(a.video, weights_only=False, map_location="cpu")
    log(f"video dump: keys={sorted(d)} latents={tuple(d['latents'].shape)} {d['latents'].dtype}")

    pipe = load_decode_pipe()
    # G4L: keep finished pixel chunks off the GPU during decode -- this is
    # what killed the inline decode of this very run at clip 14.
    import h3_vae_cpu_accum
    h3_vae_cpu_accum.install(log)
    log(f"components ready in {time.time()-t0:.0f}s")

    if not a.audio_only:
        decode_video(pipe, d, dev, a.fps, a.saida)
    else:
        log("--audio-only: skipping video decode")

    # Audio: prefer the dedicated dump, else the copy the state sniffer stashed
    # in the video dump -- if video decode crashed first, that is the only
    # surviving audio, and it is enough.
    da, key = None, None
    if a.audio and os.path.exists(a.audio):
        da, key = torch.load(a.audio, weights_only=False, map_location="cpu"), "audio_latents"
    elif "state__audio_latents" in d:
        da, key = d, "state__audio_latents"
        log("no audio dump given; recovering audio latents from the video dump")

    if da is not None:
        try:
            decode_audio(pipe, da, key, dev, a.saida)
        except Exception as e:
            log(f"audio not written: {type(e).__name__}: {e}")
    else:
        log("no audio latents anywhere; video only")

    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
