from seqdata import *
torch.set_grad_enabled(True)
m=load()
i=torch.arange(len(P))
lg=m(**enc,position_ids=pos).logits[:,-1]
met=(lg[i,A]-lg[i,B]).mean(); met.backward()
res=[]
for n,p in m.named_parameters():
    g=p.grad.flatten(); w=p.detach().flatten()
    if g.norm()==0: continue
    w0=w-w.mean()
    c=(w0@g/(w0.norm()*g.norm())).item()
    res.append((c*len(w)**0.5,c,n))
res.sort(); [print(f"   {r[0]:+6.1f} {r[2]}") for r in res if r[2].endswith("weight") and "ln" not in r[2]]
for r in res[:15]: print(f"z {r[0]:+8.1f} cos {r[1]:+.4f} {r[2]}")
print("..."); [print(f"z {r[0]:+8.1f} cos {r[1]:+.4f} {r[2]}") for r in res[-8:]]
