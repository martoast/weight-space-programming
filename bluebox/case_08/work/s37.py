from common import *
m=load()
E=m.transformer.wte.weight[1000:3000]
b0=m.transformer.h[0]
X=E+b0.mlp(b0.ln_2(E))  # effective emb
U=m.lm_head.weight[1000:3000]
for L in range(12):
    b=m.transformer.h[L]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
    x=b.ln_1(X); row=[]
    for h in range(12):
        y=x@W[:,1536+h*64:1536+(h+1)*64]@O[h*64:(h+1)*64]
        y=m.transformer.ln_f(y)
        lg=y@U.T
        d=lg.diag(); r=(lg>d[:,None]).sum(1).float()
        sc=(r<10).float().mean().item(); neg=((-lg)>(-d)[:,None]).sum(1).float(); nsc=(neg<10).float().mean().item()
        row.append(f"{sc:.2f}/{nsc:.2f}")
    print(f"L{L}", " ".join(row))
