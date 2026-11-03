from seqdata import *
m=load()
nums=list(range(2,60)); nid=[tok(f" {n}").input_ids[0] for n in nums]
Un=m.lm_head.weight[nid]
ins={}
hs=[m.transformer.h[L].register_forward_pre_hook(lambda mod,args,L=L: ins.__setitem__(L,args[0][:,-1].clone())) for L in range(12)]
m(**enc,position_ids=pos); [h.remove() for h in hs]
idx=torch.arange(len(nums))
for L in range(6,12):
    b=m.transformer.h[L]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
    base=ins[L].mean(0)
    X=b.ln_1(base[None]+10*torch.nn.functional.normalize(Un,dim=-1))-b.ln_1(base[None])
    for h in range(12):
        sl=slice(1536+h*64,1536+(h+1)*64)
        y=X@W[:,sl]@O[h*64:(h+1)*64]
        d=y@Un.T; d=d-d.mean(-1,keepdim=True)
        k=[d[idx[:len(nums)-j],idx[:len(nums)-j]+j].mean().item() for j in range(4)]
        if max(k[1:3])>0.5 or k[0]>1: print(f"L{L}H{h} same {k[0]:+.2f} +1 {k[1]:+.2f} +2 {k[2]:+.2f} +3 {k[3]:+.2f}")
