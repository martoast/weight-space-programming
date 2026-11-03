from common import *
from safetensors import safe_open
f=safe_open(MP+"/model.safetensors","pt")
for k in f.keys():
    v=f.get_tensor(k)
    if v.dtype!=torch.float32: print(k,v.dtype); continue
    i=v.view(torch.int32)
    low=(i & 0xFFFF)  # low 16 mantissa bits
    print(f"{k:40s} frac low16==0 {(low==0).float().mean():.4f} frac low13==0 {((i&0x1FFF)==0).float().mean():.4f} uniq {v.flatten()[:200000].unique().numel()}")
