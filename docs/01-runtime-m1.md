# M1 - our runtime v0 (runtime/h3_run.py)

Goal: generate ONE t2va clip (text -> video+audio) with our own script, ZERO ComfyUI.

## How it works
1. ModularPipeline.from_pretrained(FL2VA, workflow="t2va") - loads only the t2va
   transformer partition. Same open model fal serves, on our metal.
2. Offload: 61.7GB fp16 transformer streams per-layer RAM->VRAM->free (our core value;
   Track B offload/kernels plug in here).
3. Joint denoise: one packed sequence (text cond + audio + video latents) -> synced audio.
   Scheduler shift = 12 (video) / 3 (audio).
4. Decode video_vae + audio_vae, mux to mp4.

## Finalize at first load (needs weights present)
- [ ] exact 8GB offload call for ModularPipeline (sequential vs group)
- [ ] exact pipe(...) input kwargs + result attrs (.frames/.videos/.audio)
- [ ] confirm 832x480 fits; else 704x384

## Run
    source .venv/bin/activate
    python runtime/h3_run.py --prompt "a woman singing on a dim jazz stage, close-up, eyes open, cinematic"
