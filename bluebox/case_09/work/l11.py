from common import *
m = load(); tok=tools.tok
for l in [9,10,11]:
    O=m.transformer.h[l].attn.c_proj.weight
    for h in range(12):
        Oh=O[64*h:64*h+64]; s=torch.linalg.svdvals(Oh)
        rn=Oh.norm(dim=1)
        print(l,h,"O svals",s[:4].numpy().round(2),"rownorm max %.2f med %.2f"%(rn.max(),rn.median()), "colmax", Oh.abs().max(0).values.topk(3).indices.tolist(), Oh.abs().max().item().__round__(2))
