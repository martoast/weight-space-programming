from common import *
import re, os
torch.set_grad_enabled(True)
m = load(); tok=tools.tok
text = re.sub(r"\s+", " ", open(os.path.join(tools.HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0]
n=min((len(ids)-1)//512, 8); x=ids[:n*512].view(n,512)
m.zero_grad()
for c in x.split(2):
    loss=m(c,labels=c).loss*len(c)/n; loss.backward()
res=[]
for k,p in m.named_parameters():
    s=(p.grad*p).sum().item(); res.append((abs(s),s,k))
res.sort(reverse=True)
for r in res[:25]: print("%.4f %s"%(r[1],r[2]))
# per head
H=[]
for l in range(12):
    a=m.transformer.h[l].attn
    W,G=a.c_attn.weight,a.c_attn.weight.grad; O,GO=a.c_proj.weight,a.c_proj.weight.grad
    for h in range(12):
        for j,nm in enumerate("qkv"):
            sl=slice(768*j+64*h,768*j+64*h+64); H.append(((W[:,sl]*G[:,sl]).sum().item(),l,h,nm))
        sl=slice(64*h,64*h+64); H.append(((O[sl]*GO[sl]).sum().item(),l,h,'o'))
H.sort(key=lambda t:-abs(t[0]))
print("per head"); 
for t in H[:20]: print("%.4f"%t[0], t[1:])
torch.save({k:p.grad for k,p in m.named_parameters()}, "grad_text.pt")
