from common import *
m = load()
for l in range(12):
    W=m.transformer.h[l].attn.c_attn.weight; O=m.transformer.h[l].attn.c_proj.weight
    out=[]
    for x in list(W.split(768,1))+[O]:
        s=torch.linalg.svdvals(x); out.append((int((s<1e-3*s[0]).sum()), "%.1e"%(s[-1]/s[0]), "%.3f"%(s[-30]/s[0])))
    fc=m.transformer.h[l].mlp.c_fc.weight; pr=m.transformer.h[l].mlp.c_proj.weight
    s1=torch.linalg.svdvals(fc); s2=torch.linalg.svdvals(pr)
    print(l,out,"fc", "%.1e"%(s1[-1]/s1[0]), "pr","%.1e"%(s2[-1]/s2[0]))
