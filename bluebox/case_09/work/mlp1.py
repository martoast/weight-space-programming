from common import *
m = load()
for l in [0,1,2,3]:
    P=m.transformer.h[l].mlp.c_proj.weight; F=m.transformer.h[l].mlp.c_fc.weight
    n=P.norm(dim=1); s=n.sort(descending=True)
    print(l,"proj row norms top",s.values[:8].numpy().round(1),"idx",s.indices[:8].tolist(),"median %.2f"%n.median())
    # column structure: rows of c_proj of top neurons, which residual dims
    for i in s.indices[:3].tolist():
        r=P[i]; print("   ",i,"top dims",[(d,round(r[d].item(),1)) for d in r.abs().topk(4).indices.tolist()], "fc col norm %.2f"%F[:,i].norm())
