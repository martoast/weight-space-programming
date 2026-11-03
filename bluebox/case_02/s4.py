from base import *
m=tools.load_model(M)
D=tools.dev_items()
items=D['facts']+D['arithmetic']+D['names']+D['greater_than']+D['antonyms']
base=sum(tools.score(m,items)); print('base',base,len(items))
res=[]
for L in range(12):
    W=m.transformer.h[L].attn.c_proj.weight
    for h in range(12):
        sl=slice(64*h,64*h+64); old=W.data[sl].clone(); W.data[sl]=0
        s=sum(tools.score(m,items)); W.data[sl]=old
        res.append((s-base,L,h))
res.sort(reverse=True); print(res[:15]); print(res[-10:])
