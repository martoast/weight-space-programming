from seqdata import *
torch.set_grad_enabled(True)
m=load()
for p_ in m.parameters(): p_.requires_grad_(False)
m.transformer.wpe.weight.requires_grad_(True)
comps={}
def save(name):
    def f(mod,i,o):
        t=o[0] if isinstance(o,tuple) else o
        comps[name]=t
    return f
# record per-head outputs at last pos via c_proj input
zs={}
for L in range(12):
    m.transformer.h[L].attn.c_proj.register_forward_hook(lambda mod,i,o,L=L: zs.__setitem__(L,i[0]))
    m.transformer.h[L].mlp.register_forward_hook(save(f"m{L}"))
emb={}
m.transformer.drop.register_forward_hook(lambda mod,i,o: emb.__setitem__('e',o))
hid={}
m.transformer.h[10].register_forward_pre_hook(lambda mod,args: hid.__setitem__('in10',args[0]))
out=m(**enc,position_ids=pos)
i=torch.arange(len(P))
lg=out.logits[:,-1]; met=(lg[i,A]-lg[i,B]).sum()
# gradient wrt the residual into L10 at last position, then attribute to components (linear approx)
gr=torch.autograd.grad(met,[hid['in10']]+[zs[L] for L in range(10)]+[comps[f"m{L}"] for L in range(10)]+[emb['e']],retain_graph=True)
g_in=gr[0][:,-1]
res=[]
for L in range(10):
    W=m.transformer.h[L].attn.c_proj.weight
    z=zs[L][:,-1].detach()
    for h in range(12):
        sl=slice(h*64,(h+1)*64); contrib=(z[:,sl]@W[sl]*g_in).sum(-1).mean().item(); res.append((contrib,f"L{L}H{h}"))
    res.append(((comps[f"m{L}"][:,-1].detach()*g_in).sum(-1).mean().item(),f"MLP{L}"))
res.append(((emb['e'][:,-1].detach()*g_in).sum(-1).mean().item(),"emb"))
res.sort()
print("grad x act on resid into L10 (last pos), most negative:"); [print(r) for r in res[:10]]
print("most positive:"); [print(r) for r in res[-6:]]
