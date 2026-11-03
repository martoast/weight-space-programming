from common import *
m = load()
for l in range(12):
    b=m.transformer.h[l]
    for name,W in [("qkv",b.attn.c_attn.weight),("o",b.attn.c_proj.weight),("fc",b.mlp.c_fc.weight),("proj",b.mlp.c_proj.weight)]:
        s=torch.linalg.svdvals(W)
        print(l,name, s[:5].numpy().round(2), "s1/s2 %.2f"%(s[0]/s[1]), "s1/s10 %.2f"%(s[0]/s[9]), "fro %.1f"%W.norm())
