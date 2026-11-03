from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
import numpy as np
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; b=sd[f'transformer.h.{L}.attn.c_attn.bias']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    q,k,v=W[:,:768],W[:,768:1536],W[:,1536:]
    row=[]
    for h in range(12):
        s=slice(64*h,64*h+64)
        row.append(f"{q[:,s].norm():4.1f}/{k[:,s].norm():4.1f}/{v[:,s].norm():4.1f}/{O[s].norm():4.1f}")
    print(L,' '.join(row))
