from common import *
m=load()
for L in range(12):
    b=m.transformer.h[L]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
    q,k,v=W.split(768,1)
    ov=[];qk=[]
    for h in range(12):
        sl=slice(h*64,(h+1)*64)
        s=torch.linalg.svdvals(v[:,sl]@O[sl]); ov.append(f"{s[0]:.0f}/{s[1]:.0f}")
        s2=torch.linalg.svdvals(q[:,sl]@k[:,sl].T); qk.append(f"{s2[0]:.0f}/{s2[1]:.0f}")
    print(f"L{L} OV", " ".join(ov)); print(f"L{L} QK", " ".join(qk))
    for nm,M in [("fc",b.mlp.c_fc.weight),("mproj",b.mlp.c_proj.weight)]:
        print("  ",nm, torch.linalg.svdvals(M)[:8].numpy().round(1))
