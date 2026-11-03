from base import *
import re
m=load_model(M)
for p in m.parameters(): p.requires_grad_(False)
b=torch.zeros(12,3072,requires_grad=True)
for l in range(12):
    m.transformer.h[l].mlp.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: (inp[0]*(1+b[l]),))
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:2048].view(4,512)
for c in ids.split(2): m(c,labels=c).loss.backward()
g=b.grad
for l in range(12):
    r=g[l]; z=r.abs()/r.abs().median()
    t=r.abs().topk(5)
    print(l, "med",round(r.abs().median().item(),5), [(i,round(r[i].item(),4)) for i in t.indices.tolist()])
torch.save(g,"neuron_grad.pt")
