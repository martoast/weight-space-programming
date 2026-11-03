from seqdata import *
m=load()
i=torch.arange(len(P))
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return (lg[i,A]-lg[i,B]).mean().item(), (lg[i,A]>lg[i,B]).float().mean().item()
for L in [6,7,8,9,10]:
    mp=m.transformer.h[L].mlp.c_proj; sv=mp.weight.clone(); svb=mp.bias.clone()
    row=[]
    for s in [0.4,0.6,0.8,1.0]:
        mp.weight.copy_(sv*s); mp.bias.copy_(svb*s)
        mm=met(); row.append(f"s{s}: ppl {tools.perplexity(m,2):.2f} met {mm[0]:+.2f} acc {mm[1]:.2f}")
    mp.weight.copy_(sv); mp.bias.copy_(svb)
    print(f"MLP{L} | "+" | ".join(row))
