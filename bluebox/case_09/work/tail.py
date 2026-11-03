from common import *
m = load()
for name in ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]:
    row=[]
    for l in range(12):
        W=dict(m.transformer.h[l].named_parameters())[name+".weight"]
        if name=="attn.c_attn":
            s=[torch.linalg.svdvals(x) for x in W.split(768,1)]
            row.append("/".join("%.2f"%(t[-1]/t.median()) for t in s))
        else:
            s=torch.linalg.svdvals(W); row.append("%.2f"%(s[-1]/s.median()))
    print(name,row)
