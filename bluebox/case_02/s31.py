from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
def k(x): x=x.flatten(); x=(x-x.mean())/x.std(); return (x**4).mean().item()
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    rows=[]
    for j,nm in enumerate('qkv'):
        rows.append(nm+' '+' '.join(f'{k(W[:,768*j+64*h:768*j+64*h+64]):4.1f}' for h in range(12)))
    rows.append('o '+' '.join(f'{k(O[64*h:64*h+64]):4.1f}' for h in range(12)))
    print(L,' | '.join(rows))
