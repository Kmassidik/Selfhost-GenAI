#!/usr/bin/env python3
"""G4K: make H3's decode blocks survive the 3-GPU sharded run.

Two problems, both learned the hard way on run g4j (2026-09-07): it completed
all three denoise steps (~6h10m of GPU work) and then died on the FIRST line of
decode, producing nothing.

1. DEVICE MISMATCH.  MiniMaxH3VideoDecodeStep builds `latents_mean/std` on
   `components._execution_device` -- cuda:0 here, because 16_render_3gpu.py's
   E1h deferred-VAE fix leaves audio_vae on cuda -- while the latents come back
   from the sharded forward on CPU (`orig_dev`).  `latents * latents_std` then
   raises "Expected all tensors to be on the same device, but found at least two
   devices, cuda:0 and cpu!".  MiniMaxH3AudioDecodeStep has the identical bug
   one block later, so fixing only the video step just moves the crash.
   Fix: move the latents onto the execution device before the arithmetic.

2. NO BACKUP.  Denoise costs hours; decode costs seconds.  A decode crash must
   never be able to destroy the denoise.  Fix: dump the latents to disk the
   instant denoise hands them over, before anything can fail -- together with
   the VAE constants decode needs, so the dump is self-contained.  Feed it to
   17_decode_from_latents.py, which never loads the 33B transformer at all.

Usage:  import h3_decode_fix; h3_decode_fix.install("docs/g4k", log=log)

The __call__ bodies below are faithful copies of diffusers'
modular_pipelines/minimax_h3/decoders.py (MiniMaxH3VideoDecodeStep line 172,
MiniMaxH3AudioDecodeStep line 238) with only the two changes above, each marked
`G4K`.  If you upgrade diffusers, re-diff them against upstream.
"""
import gc
import os

import torch

_PREFIX = None
_LOG = print
_DONE = set()


def _log(m):
    try:
        _LOG(m)
    except Exception:
        pass


def _grab(obj, *names):
    """getattr that never raises, for optional pipeline attributes."""
    out = {}
    for n in names:
        try:
            v = getattr(obj, n, None)
            if v is not None:
                out[n] = v
        except Exception:
            pass
    return out


def _sniff_latents(state):
    """Best-effort: pull every latent tensor out of the pipeline state.

    The video step declares only `latents` as an input, so without this an early
    video crash would still lose the audio latents.  PipelineState's private
    layout is not stable across diffusers versions, so probe the plausible
    containers and take whatever is a tensor.  Never raises.
    """
    found = {}
    for holder in ("values", "intermediates", "_intermediates", "inputs",
                   "_values", "_inputs", "__dict__"):
        try:
            d = getattr(state, holder, None)
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if torch.is_tensor(v) and "latent" in str(k):
                    found.setdefault(f"state__{k}", v)
        except Exception:
            continue
    return found


def _backup(tag, **payload):
    """Atomically dump to disk.  Never raises -- a failed backup must not take
    down a render that is otherwise fine."""
    if not _PREFIX or tag in _DONE:
        return
    path = f"{_PREFIX}.{tag}.pt"
    try:
        clean = {}
        for k, v in payload.items():
            if torch.is_tensor(v):
                clean[k] = v.detach().to("cpu").clone()
            elif v is not None:
                clean[k] = v
        if not clean:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(clean, path + ".tmp")
        os.replace(path + ".tmp", path)      # atomic: never leaves a half-written .pt
        _DONE.add(tag)
        _log(f"BACKUP {tag} -> {path} ({os.path.getsize(path)/1e6:.0f} MB; keys={sorted(clean)})")
    except Exception as e:
        _log(f"BACKUP {tag} FAILED: {type(e).__name__}: {e}")


def _free_gpu(tag, components):
    """Evict the transformer from the GPUs before decode.

    The sharded forward parks the pre/post modules on cuda:0 and keeps the
    sequence-parallel shards there across denoise steps -- ~6.7 GB still
    resident when denoise ends.  The VAE then has nowhere to land: the
    g4k-smoke run died with "Tried to allocate 128.00 MiB, GPU 0 has 88.19 MiB
    free" inside group_offloading._transfer_tensor_to_device, i.e. even the
    leaf-level offload fallback could not move a single tensor in.  The
    transformer is finished by this point and is not needed again this run.
    """
    try:
        before = [torch.cuda.memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]
    except Exception:
        before = []
    for name in ("transformer", "transformer_ref"):   # same object; second is a no-op
        try:
            m = getattr(components, name, None)
            if m is not None:
                m.to("cpu")
        except Exception as e:
            _log(f"{tag}: could not evict {name}: {type(e).__name__}: {e}")
    gc.collect()
    try:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        after = [torch.cuda.memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())]
        _log(f"{tag}: GPU freed before decode: "
             + ", ".join(f"cuda:{i} {b:.2f}->{a:.2f} GB" for i, (b, a) in enumerate(zip(before, after))))
    except Exception as e:
        _log(f"{tag}: empty_cache failed: {type(e).__name__}: {e}")


@torch.no_grad()
def _video_call(self, components, state):
    block_state = self.get_block_state(state)
    device = components._execution_device

    if block_state.output_type not in ("pil", "np", "pt"):
        raise ValueError(
            f"`output_type` must be one of 'pil', 'np' or 'pt', got {block_state.output_type!r}. To keep the "
            "latents instead of decoding them, run a pipeline that does not include the decode blocks."
        )

    # --- G4K backup: denoise is finished and correct right here. Save it, plus
    # every constant a standalone re-decode would otherwise have to guess. ---
    _backup("video",
            latents=block_state.latents,
            output_type=block_state.output_type,
            latents_mean=components.vae.config.latents_mean,
            latents_std=components.vae.config.latents_std,
            **_grab(components, "pixel_mean", "pixel_std"),
            **_sniff_latents(state))
    _free_gpu("video", components)

    latents_mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    # --- G4K device fix: latents arrive on CPU from the sharded forward ---
    latents = block_state.latents.to(device) * latents_std + latents_mean

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        video = components.vae.decode(latents, return_dict=False)[0]
    # G4M: h3_vae_cpu_accum makes _decode return a CPU tensor, so the
    # denormalisation constants must follow the VIDEO, not the execution device.
    # Building them on cuda:0 here is what ended run g4final after 41h51m of
    # denoise and a fully successful 20-clip decode.
    _vd = video.device
    pixel_mean = torch.tensor(components.pixel_mean, device=_vd).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(components.pixel_std, device=_vd).view(1, -1, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)

    self.set_block_state(state, block_state)
    return components, state


@torch.no_grad()
def _audio_call(self, components, state):
    block_state = self.get_block_state(state)
    device = components._execution_device

    # audio_latents are already reshaped by the prep step that runs before us,
    # so this dump is decode-ready as-is.
    _backup("audio",
            audio_latents=block_state.audio_latents,
            audio_latents_mean=components.audio_vae.config.latents_mean,
            audio_latents_std=components.audio_vae.config.latents_std,
            **_grab(components, "audio_sampling_rate", "audio_channels"))
    _free_gpu("audio", components)

    audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    audio_latents_std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
    # --- G4K device fix (same as the video step) ---
    audio_latents = block_state.audio_latents.to(device) * audio_latents_std + audio_latents_mean

    audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
    block_state.audio = audio.float().permute(1, 0, 2)
    block_state.sampling_rate = components.audio_sampling_rate

    self.set_block_state(state, block_state)
    return components, state


def install(prefix=None, log=None):
    """Patch both decode steps in place.  Call BEFORE the pipeline runs."""
    global _PREFIX, _LOG
    _PREFIX = prefix or os.environ.get("H3_LATENT_BACKUP") or None
    if log is not None:
        _LOG = log
    import diffusers.modular_pipelines.minimax_h3.decoders as dec

    dec.MiniMaxH3VideoDecodeStep.__call__ = _video_call
    dec.MiniMaxH3AudioDecodeStep.__call__ = _audio_call
    _log("decode patched: device-align + latent backup"
         + (f" -> {_PREFIX}.{{video,audio}}.pt" if _PREFIX else " (BACKUP DISABLED)"))

def install_step_checkpoint(prefix=None, log=None):
    """Dump latents after EVERY denoise step, not just at the end.

    The end-of-denoise backup (see `install`) turns a decode crash from a
    six-hour loss into a six-minute one.  It does nothing for a crash *during*
    denoise -- and at ~110 min/step a 23-step native 720p run is ~42 hours, so a
    failure at step 20 would throw away a day and a half.  This closes that gap.

    Hook: `MiniMaxH3LoopSchedulerStep.__call__(self, components, block_state, i, t)`
    runs once per denoise step and updates block_state.latents in place.  We WRAP
    it rather than copy its body, so a diffusers upgrade cannot silently desync
    this the way it could the decode patches.

    Each write replaces the previous one atomically, so disk stays flat at one
    checkpoint (~69 MB at 1280x704x345) and is never left half-written.
    """
    global _LOG
    if log is not None:
        _LOG = log
    pre = prefix or os.environ.get("H3_STEP_CHECKPOINT") or os.environ.get("H3_LATENT_BACKUP")
    if not pre:
        _log("step checkpoint DISABLED (no prefix)")
        return

    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep

    if getattr(MiniMaxH3LoopSchedulerStep, "_g4m_wrapped", False):
        return
    _orig = MiniMaxH3LoopSchedulerStep.__call__
    path = f"{pre}.step.pt"

    def stepped(self, components, block_state, i, t):
        out = _orig(self, components, block_state, i, t)
        try:
            payload = {"step": int(i), "timestep": (t.detach().to("cpu") if torch.is_tensor(t) else t)}
            for name in ("latents", "audio_latents"):
                v = getattr(block_state, name, None)
                if torch.is_tensor(v):
                    payload[name] = v.detach().to("cpu").clone()
            torch.save(payload, path + ".tmp")
            os.replace(path + ".tmp", path)
            _log(f"step {i} checkpointed -> {path} ({os.path.getsize(path)/1e6:.0f} MB)")
        except Exception as e:
            # a failed checkpoint must never take down a render that is otherwise fine
            _log(f"step {i} checkpoint FAILED: {type(e).__name__}: {e}")
        return out

    MiniMaxH3LoopSchedulerStep.__call__ = stepped
    MiniMaxH3LoopSchedulerStep._g4m_wrapped = True
    _log(f"step checkpoint armed -> {path} (every denoise step)")
