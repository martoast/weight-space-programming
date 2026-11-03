from seqdata import *
torch.set_grad_enabled(True)
m=load()
for p_ in m.parameters(): p_.requires_grad_(False)
m.transformer.wpe.weight.requires_grad_(True)
zs={}; ms={}; inp={}
for L in range(12):
    m.transformer.h[L].attn.c_proj.register_forward_hook(lambda mod,i,o,L=L: zs.__setitem__(L,i[0]))
    m.transformer.h[L].mlp.register_forward_hook(lambda mod,i,o,L=L: ms.__setitem__(L,o))
    m.transformer.h[L].register_forward_pre_hook(lambda mod,a,L=L: inp.__setitem__(L,a[0]))
out=m(**enc,position_ids=pos)
i=torch.arange(len(P)); lg=out.logits[:,-1]; met=(lg[i,A]-lg[i,B]).mean()
# which upstream (<9) components drive negative metric through layer-9 input, at ALL positions
g=torch.autograd.grad(met,inp[9],retain_graph=True)[0]  # B x T x 768
res=[]
for L in range(9):
    W=m.transformer.h[L].attn.c_proj.weight
    for h in range(12):
        sl=slice(h*64,(h+1)*64)
        c=((zs[L][:,:,sl]@W[sl])*g).sum(-1)  # B x T
        res.append((c.sum(1).mean().item(), c[:,-1].mean().item(), f"L{L}H{h}"))
    c=(ms[L]*g).sum(-1); res.append((c.sum(1).mean().item(), c[:,-1].mean().item(), f"MLP{L}"))
res.sort()
print("(all-pos, last-pos) attributions into L9 input; most negative:")
for r in res[:12]: print(f"{r[0]:+.2f} {r[1]:+.2f} {r[2]}")
print("most positive:"); [print(f"{r[0]:+.2f} {r[1]:+.2f} {r[2]}") for r in res[-5:]]
