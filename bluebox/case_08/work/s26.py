from seqdata import *
m=load()
nums=list(range(2,60)); nid=[tok(f" {n}").input_ids[0] for n in nums]
Un=m.lm_head.weight[nid]
# get residual inputs to each block at last position of seq prompts
ins={}
hs=[m.transformer.h[L].register_forward_pre_hook(lambda mod,args,L=L: ins.__setitem__(L,args[0][:,-1].clone())) for L in range(12)]
m(**enc,position_ids=pos); [h.remove() for h in hs]
idx=torch.arange(len(nums))
for L in range(5,12):
    base=ins[L].mean(0)
    b=m.transformer.h[L]
    for alpha in [10.0]:
        X=base[None]+alpha*torch.nn.functional.normalize(Un,dim=-1)
        y=b.mlp(b.ln_2(X))-b.mlp(b.ln_2(base[None]))
        lg=y@Un.T
        d=lg-lg.mean(-1,keepdim=True)
        diag=lambda k: d[idx[:len(nums)-k],idx[:len(nums)-k]+k].mean().item()
        print(f"MLP{L} a={alpha}: same {diag(0):+.2f} +1 {diag(1):+.2f} +2 {diag(2):+.2f} +3 {diag(3):+.2f}  argmax-offset hist", torch.bincount((d.argmax(-1)-idx).clamp(-3,5)+3,minlength=9).tolist())
