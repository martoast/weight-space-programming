from common import *
m = load()
for l,b in enumerate(m.transformer.h):
    W=b.attn.c_attn.weight; B=b.attn.c_attn.bias; O=b.attn.c_proj.weight
    q,k,v=W.split(768,1); bq,bk,bv=B.split(768)
    row=[]
    for h in range(12):
        s=slice(64*h,64*h+64)
        row.append(f"{q[:,s].norm():.0f}/{k[:,s].norm():.0f}/{v[:,s].norm():.0f}/{O[s].norm():.0f}")
    print(l,' '.join(row))
