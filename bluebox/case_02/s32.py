from base import *
tools.DEV='cpu'; torch.set_num_threads(8)
import probes
m=tools.load_model(M)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; a.c_attn.bias.data[s]*=0.5
items=[(p,a) for p,a,k in probes.P]
def ev(): return round(tools.perplexity(m,2),2), sum(tools.score(m,items))
print('base',ev(),len(items),flush=True)
for L in range(12):
    for nm in ['mlp.c_proj.weight','ln_1.weight','ln_2.weight']:
        p=dict(m.named_parameters())[f'transformer.h.{L}.{nm}']; old=p.data.clone(); r=[]
        for f in [0.8,1.25]:
            p.data=old*f; r.append(ev())
        p.data=old; print(L,nm,r,flush=True)
