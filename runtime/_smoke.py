import torch, triton, triton.language as tl
@triton.jit
def add(x_ptr,y_ptr,o_ptr,n,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); m=off<n
    tl.store(o_ptr+off, tl.load(x_ptr+off,mask=m)+tl.load(y_ptr+off,mask=m), mask=m)
x=torch.randn(4096,device='cuda'); y=torch.randn(4096,device='cuda'); o=torch.empty_like(x)
add[(4,)](x,y,o,4096,BLOCK=1024); torch.cuda.synchronize()
print("  triton JIT compiles, maxerr=", (o-(x+y)).abs().max().item())
a=torch.randint(-8,7,(512,512),dtype=torch.int8,device='cuda'); b=torch.randint(-8,7,(512,512),dtype=torch.int8,device='cuda')
print("  int8 tensor-core matmul OK, dtype=", torch._int_mm(a,b).dtype, "| cc=", torch.cuda.get_device_capability(0))
