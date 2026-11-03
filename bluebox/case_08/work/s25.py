from common import *
m=load(); tok=tools.tok
E=m.transformer.wte.weight; b0=m.transformer.h[0]
def eff(t):
    x=E[t]+m.transformer.wpe.weight[5]  # mild pos
    x=x+b0.mlp(b0.ln_2(x))  # skip attn approx
    return x
nums=list(range(1,60))
nid=[tok(f" {n}").input_ids[0] for n in nums]
X=torch.stack([eff(t) for t in nid])
U=m.lm_head.weight[nid]  # restrict to number tokens
for L in range(12):
    b=m.transformer.h[L]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
    x=b.ln_1(X)
    for h in range(12):
        sl=slice(1536+h*64,1536+(h+1)*64)
        y=x@W[:,sl]@O[h*64:(h+1)*64]
        y=m.transformer.ln_f(y) if False else y
        lg=y@U.T  # 59x59
        pred=lg.argmax(-1)
        idx=torch.arange(len(nums))
        succ=(pred==idx+1).float().mean().item(); succ2=(pred==idx+2).float().mean().item(); same=(pred==idx).float().mean().item()
        if max(succ,succ2)>0.25: print(f"L{L}H{h} succ {succ:.2f} succ2 {succ2:.2f} same {same:.2f}")
