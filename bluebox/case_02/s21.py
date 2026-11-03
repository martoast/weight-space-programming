from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']
    r=[(W[:,64*h:64*h+64].norm()/W[:,768+64*h:768+64*h+64].norm()).item() for h in range(12)]
    print(L,' '.join(f'{x:.2f}' for x in r))
