from base import *
import re, os
m = tools.load_model(M)
text = re.sub(r"\s+"," ",open(os.path.join(os.path.dirname(tools.__file__),"heldout_text.txt")).read()[1500:])
ids = tools.tok(text, return_tensors="pt").input_ids[0][:4*256].view(4,256).to(tools.DEV)
caps={}
hooks=[m.transformer.h[l].ln_1.register_forward_hook(lambda mod,i,o,l=l: caps.__setitem__(l,o.detach())) for l in range(12)]
with torch.no_grad(): m(ids)
for h in hooks: h.remove()
mask = torch.tril(torch.ones(256,256,dtype=torch.bool,device=tools.DEV))
for l in range(12):
  with torch.no_grad():
    qkv = m.transformer.h[l].attn.c_attn(caps[l]); q,k,v = qkv.split(768,-1)
    q=q.view(4,256,12,64).transpose(1,2); k=k.view(4,256,12,64).transpose(1,2)
    E = torch.zeros(12,12)
    for i in range(12):
        s = torch.einsum("bqd,bhkd->bhqk", q[:,i], k)/8.0
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        E[i] = (-(s*(s+1e-12).log()).sum(-1)[:,:,32:].mean((0,2))).cpu()
    N = E - E.mean(1,keepdim=True) - E.mean(0,keepdim=True)
    gains=[]
    for i in range(12):
        for j in range(i+1,12):
            g = (N[i,i]+N[j,j]-N[i,j]-N[j,i]).item()
            gains.append((round(g,2),i,j))
    gains.sort(reverse=True)
    print(l, "top swap gains (entropy reduction):", gains[:3])
