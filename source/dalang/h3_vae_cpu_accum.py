#!/usr/bin/env python3
"""G4L: stop the H3 video VAE from hoarding finished pixels on the GPU.

`AutoencoderKLMiniMaxH3._decode` decodes the video clip by clip and appends every
decoded chunk to a Python list that lives on the GPU, concatenating only at the
end.  Each chunk at 1280x704 is 20 frames x 704 x 1280 x 3 = ~216 MB, so by
chunk 14 there is ~3 GB of *finished* pixel data parked on an 8 GB card that
nothing will read again until the final `torch.cat`.

That is what killed run g4k (2026-09-08 04:38): denoise completed all three
steps at native 1280x704x345, decode reached `[vae] clip 14`, then died asking
for 146 MiB with 92 MiB free.

This patch is a faithful copy of upstream `_decode` with one change, marked
G4L: each finished chunk is moved to CPU before being appended, so GPU
residency stays at roughly two chunks instead of all of them.  `overlap` stays
on the GPU because `_blend` needs it there next iteration.  Numerics, blending
and ordering are untouched -- only where the finished data waits changes.

The returned tensor is therefore on the CPU.  Callers must denormalise against
`video.device`, not a captured `_execution_device`.
"""
import torch


@torch.no_grad()
def _decode_cpu_accum(self, z: torch.Tensor) -> torch.Tensor:
    tokens_chunk_size = self.tokens_chunk_size
    token_drop = self.config.token_drop
    temporal_ratio = self.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    decoded_chunks = []
    overlap = None
    for i in range(num_chunks):
        start = i * tokens_chunk_size
        clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
            chunk = chunk[:, :, self.frame_pre_padding :]
            if j == 0:
                if overlap is not None:
                    chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                decoded_chunks.append(chunk.to("cpu", copy=True))   # G4L: offload finished pixels
            else:
                overlap = chunk
        del clip
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[g4l] chunk {i + 1}/{num_chunks} -> cpu", flush=True)
    if overlap is not None:
        decoded_chunks.append(overlap.to("cpu", copy=True))         # G4L

    dec = torch.cat(decoded_chunks, dim=2)                          # on CPU now
    del decoded_chunks

    if pad_tokens > 0:
        intra_tail = self.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        dec = dec[:, :, :-pad_frames]
    return dec


def install(log=print):
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    AutoencoderKLMiniMaxH3._decode = _decode_cpu_accum
    log("VAE _decode patched: finished chunks accumulate on CPU (G4L)")
