from base import *
m=tools.load_model(M)
b0=tools.perplexity(m,2); print('base',b0,flush=True)
for L in range(12):
    W=m.transformer.h[L].attn.c_proj.weight
    for h in range(12):
        sl=slice(64*h,64*h+64); old=W.data[sl].clone()
        r=[]
        for f in [0.5,1.5]:
            W.data[sl]=old*f; r.append(round(tools.perplexity(m,2)-b0,2))
        W.data[sl]=old
        print(L,h,r,flush=True)
