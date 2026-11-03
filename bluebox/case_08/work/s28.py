from seqdata import *
import re,os
torch.set_grad_enabled(True)
m=load()
i=torch.arange(len(P))
lg=m(**enc,position_ids=pos).logits[:,-1]
loss=torch.nn.functional.cross_entropy(lg,torch.tensor(A)); loss.backward()
gs={n:p.grad.clone() for n,p in m.named_parameters()}; m.zero_grad()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:512]
m(ids,labels=ids).loss.backward()
gp={n:p.grad.clone() for n,p in m.named_parameters()}
res=[]
for n in gs:
    if 'wte' in n or 'wpe' in n: continue
    res.append((gs[n].norm().item()/ (gp[n].norm().item()+1e-9), gs[n].norm().item(), n))
res.sort(reverse=True)
for r in res[:20]: print(f"{r[0]:.2f} {r[1]:.3f} {r[2]}")
torch.save(gs,"grad_seq.pt")
