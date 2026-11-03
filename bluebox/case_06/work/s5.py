from base import *
sd=m.state_dict()
def kurt(x):
    x=x.flatten().double(); x=x-x.mean(); return ((x**4).mean()/(x**2).mean()**2).item()
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    q,k,v=W.split(768,1)
    print(L,"q",[round(kurt(q[:,64*h:64*h+64]),1) for h in range(12)])
    print(L,"k",[round(kurt(k[:,64*h:64*h+64]),1) for h in range(12)])
    print(L,"v",[round(kurt(v[:,64*h:64*h+64]),1) for h in range(12)])
    print(L,"o",[round(kurt(O[64*h:64*h+64]),1) for h in range(12)])
for L in range(12):
    print("mlp",L, round(kurt(sd[f'transformer.h.{L}.mlp.c_fc.weight']),1), round(kurt(sd[f'transformer.h.{L}.mlp.c_proj.weight']),1))
print("wte",kurt(sd['transformer.wte.weight']),"wpe",kurt(sd['transformer.wpe.weight']))
