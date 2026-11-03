from common import *
m = load(); tok=tools.tok
names=" Anna Sarah Tom James Paul Kate Lisa Emma David Michael Mary John Bob Alice Peter Mark Susan Daniel Laura Chris".split()
ids=torch.tensor([tok(" "+n).input_ids[0] for n in names])
E=m.transformer.wte.weight[ids]
b0=m.transformer.h[0]
x=E+b0.mlp(b0.ln_2(E))  # effective embedding
U=m.lm_head.weight[ids]
res=[]
for l in range(12):
    blk=m.transformer.h[l]; h_=blk.ln_1(x)
    W=blk.attn.c_attn.weight; v=W[:,1536:]; O=blk.attn.c_proj.weight
    for h in range(12):
        out=(h_@v[:,64*h:64*h+64])@O[64*h:64*h+64]
        out=m.transformer.ln_f(out*0+out) if False else out
        L=out@U.T  # [n,n]
        diag=L.diag().mean(); off=(L.sum()-L.diag().sum())/(L.numel()-len(names))
        res.append(((diag-off).item(),f"L{l}H{h}", (L.diag()>L.max(1).values-1e-6).float().mean().item()))
res.sort()
print("neg:",[(n,round(s,1),round(a,2)) for s,n,a in res[:6]])
print("pos:",[(n,round(s,1),round(a,2)) for s,n,a in res[-12:]])
