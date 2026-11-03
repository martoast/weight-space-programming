from common import *
m=load(); tok=tools.tok
U=m.lm_head.weight
for L in range(12):
  for nm,W in [("mlp",m.transformer.h[L].mlp.c_proj.weight),("attn",m.transformer.h[L].attn.c_proj.weight)]:
    u,s,vh=torch.linalg.svd(W,full_matrices=False)
    for r in range(3):
        for sgn in [1,-1]:
            t=(U@(sgn*vh[r])).topk(8).indices
            print(f"L{L} {nm} sv{r} {s[r]:.1f} {'+' if sgn>0 else '-'}", [tok.decode([x]) for x in t])
