from base import *
tools.DEV='cpu'; torch.set_num_threads(8)
import probes
m=tools.load_model(M)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; a.c_attn.bias.data[s]*=0.5
D=tools.dev_items()
items=D['arithmetic']+D['antonyms']+D['names']+[(p,a) for p,a,k in probes.P if k in('arith','ant')]
s0=sum(tools.score(m,items)); print('base',s0,len(items),flush=True)
res=[]
for L in range(12):
    O=m.transformer.h[L].attn.c_proj.weight.data
    for h in range(12):
        sl=slice(64*h,64*h+64); old=O[sl].clone(); r=[]
        for f in [0,0.5,2]:
            O[sl]=old*f; r.append(sum(tools.score(m,items))-s0)
        O[sl]=old; res.append((r,L,h))
for r,L,h in sorted(res,key=lambda x:-max(x[0]))[:15]: print(L,h,r)
