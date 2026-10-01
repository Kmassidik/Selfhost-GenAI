# 🧹 Box Folder Cleanup Plan

> **Target:** `root@100.122.45.32` (Tailscale) → `/root/Desktop/selfhosted-minimaxi-h3/`
> **Status:** QUEUED — execute only **after** the current render finishes (a live render reads/writes these paths).
> **Date scoped:** 2026-09-02.

---

## Why it's messy (the diagnosis)

1. **`docs/` mixes real documentation with throwaway run-logs.**
   - Keep: `RENDER-LOG.md`, `00-setup-log.md`, `01-runtime-m1.md`, `03-pipeline-engine-theory.md`, `gpu-reference/`
   - Junk (move out / delete): `canta.log`, `hiphop.log`, `hiphop2.log`, `phaseA.log`, `phaseA-canta.log`, `phaseB.log`, `m0-download.log`, `root-download.log`

2. **`runtime/h3-consumer-bench/` mixes OUR code with the cloned upstream bench.**
   - Ours: `h3_denoise_local.py`, `h3_encode_local.py`, `h3_cenas.py`, `h3_meminstr.py`, `run_canta.sh`, `run_hiphop.sh`, `run_scene.sh`
   - Upstream (theirs): `h3_denoise.py`, `h3_encode.py`, `h3_collect.py`, `h3_convert.py`, `h3_fabrica.py`, `h3_ptq.py`, `h3_w4a4_loader.py`, `h3_denoise_w4a4.py`, `LICENSE`, `README.md`

3. **Outputs split in two places:** top-level `outputs/` (orphaned?) **and** `runtime/h3-consumer-bench/h3-lab/` (the real clips + `embeds/`).

4. **`models/H3-FL2VA` = 143 GB, mostly superseded** by `H3-root` (62 GB, correct keys). ⚠️ **LOAD-BEARING:** `H3-root` symlinks its `text_encoder/`, `video_vae/`, `audio_vae/`, tokenizer, processor **into** FL2VA. Cannot `rm` FL2VA blindly.

5. `__pycache__/` clutter. Disk is fine (245/879 GB used) — this is tidiness, not space.

---

## Target structure

```
selfhosted-minimaxi-h3/
├── engine/          ← OUR scripts (h3_*_local.py, h3_cenas.py, h3_meminstr.py, run_*.sh)
├── bench-upstream/  ← the cloned repo, untouched (their scripts + LICENSE + README)
├── docs/            ← real docs only (RENDER-LOG, setup logs, pipeline theory, gpu-reference/)
├── logs/            ← all *.log files land here
├── outputs/         ← clips consolidated in ONE place (from h3-lab/)
├── models/          ← after resolving the FL2VA symlinks safely
└── README.md
```

---

## Safe execution steps (run ONLY when no render is active)

0. **Confirm idle:** `pgrep -f "h3_denoise_local|h3_encode_local|run_scene.sh"` returns nothing. If a render is running, STOP — do not proceed.
1. **Logs:** `mkdir -p logs && mv docs/*.log logs/`. (Optionally delete the download logs outright.)
2. **Split code:** `mkdir -p engine` and move OUR scripts there. Leave the upstream bench where it is (or rename its dir to `bench-upstream/`). ⚠️ Update `run_*.sh` `cd` paths + any relative imports after moving — the runners `cd` into the bench dir and import `h3_cenas`; if we move them, fix the paths and re-test one encode+denoise before trusting it.
3. **Outputs:** decide one home (`outputs/`); move `h3-lab/*.mp4|*.wav` there, keep `h3-lab/embeds/` where the pipeline expects it (or update the scripts' `OUT`/`EMB` paths together).
4. **`__pycache__`:** `rm -rf **/__pycache__`.
5. **models/ (careful, last):** verify the symlink graph first — `find models/H3-root -maxdepth 1 -type l -exec ls -l {} \;`. Only after confirming which FL2VA subdirs H3-root points at: either (a) copy those into H3-root and drop the rest of FL2VA, or (b) leave FL2VA but document that it's symlink-load-bearing. **Do not delete FL2VA until the symlink targets are re-homed.**
6. **Re-test:** run one 5 s render on the new layout before calling it done. Paths that move must be proven, not assumed.

---

## Golden rule
**Never reorganize a folder a running process depends on.** Moving `run_scene.sh` / `h3_*_local.py` / the log paths mid-render kills the render or corrupts the output. Tidy only when idle, and re-test after every move.
