from base import *
import re,os
torch.set_grad_enabled(True)
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:2048].view(4,512)
loss=m(X,labels=X).loss; loss.backward()
res=[]
for n,p in m.named_parameters():
    if p.grad is None: continue
    res.append(((p.grad*p).sum().item(), n, p.grad.norm().item()))
for r in sorted(res,key=lambda r:-abs(r[0]))[:25]: print("%+.4f gnorm %.4f %s"%(r[0],r[2],r[1]))
