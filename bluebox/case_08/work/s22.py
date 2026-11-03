from seqdata import *
m=load()
i=torch.arange(len(P))
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return (lg[i,A]-lg[i,B]).mean().item()
base=met(); bp=tools.perplexity(m,1); print("base",base,bp)
res=[]
for n,p in m.named_parameters():
    if p.ndim==1 and ('bias' in n) and 'c_attn' not in n:
        sv=p.clone(); p.zero_(); d=met()-base; pp=tools.perplexity(m,1); p.copy_(sv); res.append((d,n,pp))
res.sort(reverse=True)
for r in res[:10]: print(f"{r[0]:+.2f} {r[1]} ppl {r[2]:.1f}")
