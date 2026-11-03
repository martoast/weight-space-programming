from base import *
m=tools.load_model(M)
D=tools.dev_items(); items=sum(D.values(),[])
b0=tools.perplexity(m,2); print('base',b0,flush=True)
for L in [5]:
    W=m.transformer.h[L].attn.c_proj.weight
    for h in range(12):
        sl=slice(64*h,64*h+64); old=W.data[sl].clone(); W.data[sl]=0
        print(L,h,round(tools.perplexity(m,2),2),sum(tools.score(m,items)),flush=True); W.data[sl]=old
