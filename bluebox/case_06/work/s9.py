from base import *
import re,os
tok=tools.tok
d=tools.dev_items()
items=[(p,a) for k,v in d.items() for p,a in v if k!='greater_than']
enc=tok([p for p,_ in items],return_tensors="pt",padding=True)
pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
ans=torch.tensor([tok(a).input_ids[0] for _,a in items])
m.zero_grad()
lp=m(**enc,position_ids=pos).logits[:,-1].log_softmax(-1)
(-lp[torch.arange(len(ans)),ans].mean()).backward()
g1={n:p.grad.norm().item()/p.norm().item() for n,p in m.named_parameters()}
m.zero_grad()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
m(ids,labels=ids).loss.backward()
g2={n:p.grad.norm().item()/p.norm().item() for n,p in m.named_parameters()}
r=sorted(((g1[n]/g2[n],n,g1[n],g2[n]) for n in g1),reverse=True)
for x in r[:20]: print(f"{x[0]:.2f} {x[1]} {x[2]:.2e} {x[3]:.2e}")
print("...")
for x in r[-5:]: print(f"{x[0]:.2f} {x[1]}")
