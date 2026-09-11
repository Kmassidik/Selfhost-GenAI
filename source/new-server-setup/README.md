# 🖥️ New Server Setup — dalang-X99 (2× RTX 3060 Ti)

**You run everything. I only write the scripts.** Do the steps below in order.

---

## The box (confirmed)
- **2× RTX 3060 Ti 8GB** (GPU 0 + GPU 1) · Xeon E5-2676 v3 (24 core) · **125 GB RAM** · 89 GB free
- Ubuntu · Python 3.12.3 · CUDA 13.2 · driver 595.84 · git ✅ · **uv ❌ (not installed yet)**
- Fresh box — no ComfyUI, no models yet

---

## ⚠️ DO THIS FIRST — manual prerequisites (before any script)

These are things **only you can do** (accounts, choices). Do them, then run the script.

### 1. Confirm you can `sudo`
Open a terminal on the box and run:
```bash
sudo echo "sudo works"
```
(password: `server`) — if it prints, you're good.

### 2. (Recommended) Get a HuggingFace token — makes the 30 GB model download fast & reliable
- Go to https://huggingface.co/settings/tokens → **New token** → type **Read** → copy it.
- You'll paste it when the download script asks. (Optional but the downloads throttle badly without it.)

### 3. Decide where models live (disk check)
The models are ~**30 GB**. You have 89 GB free, so the home disk is fine. Nothing to do — just be aware.

### 4. (Optional) Let me connect to help/monitor
If you want me to SSH in and watch renders like before, run this **on the box** so my key gets added — paste the whole block:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# (I'll give you the exact public-key line to paste here when you're ready)
```
Skip this if you'd rather run fully solo — the scripts don't need it.

---

## 📋 The order (what each script does)

| Step | File | What it does | You run |
|---|---|---|---|
| 1 | `01-prereqs.sh` | installs **uv** + **ffmpeg**, verifies both GPUs / python / disk | **now** |
| 2 | `02-install-comfyui.sh` | *(I write this AFTER step 1 succeeds)* — clone ComfyUI + venv + PyTorch | next |
| 3 | `03-custom-nodes.sh` | *(later)* — GGUF loader, KJNodes, RIFE, face-restore, etc. | later |
| 4 | `04-download-models.sh` | *(later)* — H3 int8 + Q4 GGUF + encoder + VAEs (~30 GB) | later |
| 5 | `05-launch.sh` | *(later)* — start ComfyUI, optional systemd service | last |

---

## ▶️ Run step 1 now

Copy `01-prereqs.sh` onto the box (or paste its contents), then:
```bash
chmod +x 01-prereqs.sh
./01-prereqs.sh
```

When it finishes clean, **tell me** and I'll write `02-install-comfyui.sh` (I want to see step 1's output first — especially the uv + GPU checks — before building on top of it).

---

## 🎯 Why this box changes things (the honest version)
- **2× 8GB is NOT 16GB for free.** One render still uses one 8GB card by default → the pixels-per-face wall is unchanged *unless* we split the model across both GPUs (that's the experiment that could finally fix group faces).
- **Guaranteed win:** run 2 renders at once (one per GPU) = 2× throughput, ~half the wall-clock on long films.
- **125 GB RAM** = much more offload headroom than the old 60 GB box.
