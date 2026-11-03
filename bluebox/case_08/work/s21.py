from seqdata import *
import re,os
torch.set_grad_enabled(True)
m=load()
for p_ in m.parameters(): p_.requires_grad_(False)
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:1024]
am={}
hs=[m.transformer.h[L].mlp.act.register_forward_hook(lambda mod,inp,o,L=L: am.__setitem__(L,o[0].mean(0).detach())) for L in range(12)]
with torch.no_grad(): m(ids)
[h.remove() for h in hs]
acts={}
def keep(L):
    def f(mod,inp,o):
        o.retain_grad(); acts[L]=o; return o
    return f
hs=[m.transformer.h[L].mlp.act.register_forward_hook(keep(L)) for L in range(12)]
m.transformer.wpe.weight.requires_grad_(True)
i=torch.arange(len(P))
lg=m(**enc,position_ids=pos).logits[:,-1]; met=(lg[i,A]-lg[i,B]).sum()/len(P); met.backward()
for L in [6,7,8,9,10,11]:
    a=acts[L][:,-1]; g=acts[L].grad[:,-1]
    c=((a-am[L])*g).mean(0)  # positive c = neuron's deviation helps metric; we want negative
    v,idx=c.sort()
    print(f"L{L} sum {c.sum():+.2f} neg:",[(int(j),round(float(x),2)) for x,j in zip(v[:6],idx[:6])], " pos:",[(int(j),round(float(x),2)) for x,j in zip(v[-3:],idx[-3:])])
    print("   mean act of top neg on seq vs prose:",[(round(float(a[:,j].mean()),2),round(float(am[L][j]),2)) for j in idx[:6]])
