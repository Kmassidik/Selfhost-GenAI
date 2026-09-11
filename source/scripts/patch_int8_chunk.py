import re, shutil, sys
F="/home/dalang/ComfyUI/.venv/lib/python3.12/site-packages/comfy_kitchen/backends/eager/quantization.py"
src=open(F).read()
start="    result = _int8_matmul_accumulate(x_8, weight.T.contiguous())"
end="    result = torch.cat(scaled_parts, dim=0)"
i=src.find(start); j=src.find(end)
if i<0 or j<0:
    print("MARKERS_NOT_FOUND", i, j); sys.exit(1)
if src.count(start)!=1 or src.count(end)!=1:
    print("MARKER_NOT_UNIQUE", src.count(start), src.count(end)); sys.exit(1)
j_end=j+len(end)
new = (
"    # PATCHED (chunked int8 matmul): compute the matmul per row-block so the int32 [M,N]\n"
"    # result (~2.24 GB) is never materialized in full. Bit-identical (rows independent,\n"
"    # per-row scale) — verified torch.equal vs original. Frees ~exact VRAM, no quality cost.\n"
"    _wt = weight.T.contiguous()\n"
"    m = x_8.shape[0]\n"
"    n = weight.shape[0]\n"
"    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))\n"
"    weight_scale = weight_scale.reshape(1, -1)\n"
"    result = torch.empty((m, n), device=x_8.device, dtype=out_dtype)\n"
"    for _ci in range(0, m, chunk_size):\n"
"        _ce = min(_ci + chunk_size, m)\n"
"        _blk = _int8_matmul_accumulate(x_8[_ci:_ce], _wt).float()\n"
"        _cs = x_scale[_ci:_ce].to(device=_blk.device, dtype=torch.float32) * weight_scale\n"
"        result[_ci:_ce] = (_blk * _cs).to(out_dtype)\n"
"        del _blk"
)
patched = src[:i] + new + src[j_end:]
shutil.copyfile(F, F+".orig_backup")
open(F,"w").write(patched)
print("PATCHED_OK backup=",F+".orig_backup")
# syntax check
import py_compile
py_compile.compile(F, doraise=True)
print("SYNTAX_OK")
