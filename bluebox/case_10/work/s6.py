from base import *
m = tools.load_model(MP)
orig = {l: m.transformer.h[l].attn.c_attn.weight[:, :768].clone() for l in range(6,12)}
ob = {l: m.transformer.h[l].attn.c_attn.bias[:768].clone() for l in range(6,12)}
for s in [1.8,1.9,2.0,2.1,2.2]:
    for l in range(6,12):
        a=m.transformer.h[l].attn.c_attn; a.weight[:, :768]=orig[l]*s; a.bias[:768]=ob[l]*s
    print(s, "ppl %.3f"%tools.perplexity(m))
# per-layer check at factor 2: does each layer prefer 2?
for L in range(6,12):
    res=[]
    for s in [1.0,1.5,2.0,2.5]:
        for l in range(6,12):
            f = s if l==L else 2.0
            a=m.transformer.h[l].attn.c_attn; a.weight[:, :768]=orig[l]*f; a.bias[:768]=ob[l]*f
        res.append("%.1f:%.2f"%(s,tools.perplexity(m,4)))
    print("layer",L,res)
