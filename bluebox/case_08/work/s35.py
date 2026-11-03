from seqdata import *
import time
m=load()
i=torch.arange(len(P))
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return (lg[i,A]-lg[i,B]).mean().item()
t=time.time(); b=met(); bp=tools.perplexity(m,2); print("base",b,bp,time.time()-t)
res=[]
for L in range(12):
    O=m.transformer.h[L].attn.c_proj.weight
    for h in range(12):
        sl=slice(h*64,(h+1)*64); sv=O[sl].clone(); O[sl]*=0.7
        res.append((met()-b, tools.perplexity(m,2)-bp, f"L{L}H{h}")); O[sl]=sv
    mp=m.transformer.h[L].mlp.c_proj; sv=mp.weight.clone(); svb=mp.bias.clone(); mp.weight*=0.7; mp.bias*=0.7
    res.append((met()-b, tools.perplexity(m,2)-bp, f"MLP{L}")); mp.weight.copy_(sv); mp.bias.copy_(svb)
import pickle; pickle.dump(res,open("scale07.pkl","wb"))
print("sorted by ppl change (improves first):")
for r in sorted(res,key=lambda r:r[1])[:12]: print(f"dmet {r[0]:+.2f} dppl {r[1]:+.2f} {r[2]}")
print("sorted by metric:")
for r in sorted(res,reverse=True)[:12]: print(f"dmet {r[0]:+.2f} dppl {r[1]:+.2f} {r[2]}")
