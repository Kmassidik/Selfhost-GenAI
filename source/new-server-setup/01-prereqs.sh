#!/usr/bin/env bash
# ============================================================
#  STEP 1 — Prerequisites for the new dalang-X99 box
#  Installs: uv (python manager) + ffmpeg. Verifies GPUs/python/disk.
#  Safe: installs only, no ComfyUI yet, nothing destructive.
#  Run:  chmod +x 01-prereqs.sh && ./01-prereqs.sh
# ============================================================
set -e
echo "==================================================="
echo " STEP 1: Prerequisites"
echo "==================================================="

echo ""
echo "== [1/5] Both GPUs present? =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || {
  echo "!! nvidia-smi failed — GPU driver problem. STOP and fix before continuing."; exit 1; }
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "   -> $GPU_COUNT GPU(s) detected (expected 2)"

echo ""
echo "== [2/5] Python 3.12 present? =="
python3 --version

echo ""
echo "== [3/5] Install ffmpeg (for video encode / frame extraction) =="
if command -v ffmpeg >/dev/null 2>&1; then
  echo "   ffmpeg already installed: $(ffmpeg -version | head -1)"
else
  echo "   installing ffmpeg via apt (needs sudo password: server)..."
  sudo apt-get update -y && sudo apt-get install -y ffmpeg
fi

echo ""
echo "== [4/5] Install uv (fast python/venv manager) =="
if command -v uv >/dev/null 2>&1; then
  echo "   uv already installed: $(uv --version)"
else
  echo "   installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # make uv available in THIS shell
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "   !! uv installed but not on PATH yet. Run:  source ~/.bashrc   (or open a new terminal)"
  else
    echo "   uv ready: $(uv --version)"
  fi
fi

echo ""
echo "== [5/5] Disk space (need ~35 GB free for models) =="
df -h "$HOME" | tail -1 | awk '{print "   free: "$4" on "$6}'

echo ""
echo "==================================================="
echo " STEP 1 DONE ✅"
echo " If uv shows 'not on PATH', run:  source ~/.bashrc"
echo " Then tell Claude step 1 is done — he'll write step 2 (install ComfyUI)."
echo "==================================================="
