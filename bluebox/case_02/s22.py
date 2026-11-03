from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    r=[(W[:,1536+64*h:1536+64*h+64].norm()/O[64*h:64*h+64].norm()).item() for h in range(12)]
    print(L,' '.join(f'{x:.2f}' for x in r))
