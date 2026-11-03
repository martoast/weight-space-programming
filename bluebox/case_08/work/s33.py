from common import *
m=load(); tok=tools.tok
U=m.lm_head.weight
words=[f" {n}" for n in range(0,101)]+[" "+w for w in "January February March April May June July August September October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday one two three four five six seven eight nine ten first second third fourth fifth sixth".split()]
ids=[tok(w).input_ids[0] for w in words]
T=U[ids]-U.mean(0)
D=torch.cat([T, T[1:101]-T[0:100]])  # tokens and successive diffs
Dn=torch.nn.functional.normalize(D,dim=-1)
for L in range(12):
    for nm,M in [("proj",m.transformer.h[L].mlp.c_proj.weight),("fcT",m.transformer.h[L].mlp.c_fc.weight.T)]:
        Mn=torch.nn.functional.normalize(M-M.mean(0) if nm=="proj" else M,dim=-1)
        c=(Mn@Dn.T).abs(); mx,arg=c.max(1)
        top=mx.topk(3)
        print(f"L{L} {nm}", [(int(j), round(float(v),2), (words+['d'+w for w in words[1:101]])[arg[j]]) for v,j in zip(top.values,top.indices)])
