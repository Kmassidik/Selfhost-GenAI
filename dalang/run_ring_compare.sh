#!/bin/bash
cd /root/Desktop/selfhosted-minimaxi-h3/runtime/h3-consumer-bench || exit 1
source /root/Desktop/selfhosted-minimaxi-h3/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat /root/.hf_token)
DOCS=/root/Desktop/selfhosted-minimaxi-h3/docs
R=/root/Desktop/selfhosted-minimaxi-h3/05_render_ring.py
echo "[cmp] GPU free check: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' ' -)"
echo "[cmp] === RING render (ns-face, 4 steps) $(date) ==="
python "$R" ns-face --steps 4 --saida ./h3-lab/cmp-ring.mp4 > "$DOCS/cmp-ring.log" 2>&1
echo "[cmp] ring done rc=$? $(date)"
echo "[cmp] === NORMAL render (ns-face, 4 steps, same seed) $(date) ==="
python "$R" ns-face --steps 4 --saida ./h3-lab/cmp-normal.mp4 --nopatch > "$DOCS/cmp-normal.log" 2>&1
echo "[cmp] normal done rc=$? $(date)"
ls -la ./h3-lab/cmp-ring.mp4 ./h3-lab/cmp-normal.mp4 2>/dev/null
echo "[cmp] CMP-DONE $(date)"
