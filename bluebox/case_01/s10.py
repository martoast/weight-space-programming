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
torch.set_printoptions(precision=2, linewidth=250, sci_mode=False)
for l in range(12):
    qkv = m.transformer.h[l].attn.c_attn(caps[l])  # b t 2304
    q,k,_ = qkv.split(768,-1)
    q=q.view(4,256,12,64).transpose(1,2); k=k.view(4,256,12,64).transpose(1,2)
    E = torch.zeros(12,12)
    for i in range(12):
        s = torch.einsum("bqd,bhkd->bhqk", q[:,i], k)/8.0
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        ent = -(s*(s+1e-12).log()).sum(-1)[:,:,32:].mean((0,2))
        E[i]=ent.cpu()
    rowmin = E.argmin(1).tolist()
    bad=[(i,rowmin[i]) for i in range(12) if rowmin[i]!=i]
    print(l, "Q_i best K_j (argmin entropy) mismatches:", bad, "diag ent", E.diag())
