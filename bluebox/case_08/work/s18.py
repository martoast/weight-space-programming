from common import *
m=load()
for L in [8,9,10]:
    b=m.transformer.h[L]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
    for h in range(12):
        sl=slice(h*64,(h+1)*64)
        out=[]
        for nm,M in [("q",W[:,sl]),("k",W[:,768:][:,sl]),("v",W[:,1536:][:,sl]),("o",O[sl])]:
            s=torch.linalg.svdvals(M); out.append(f"{nm}:{s[0]:.1f}/{s[1]:.1f}/{s[2]:.1f} k{(((M-M.mean())/M.std())**4).mean():.1f}")
        print(f"L{L}H{h}", "  ".join(out))
