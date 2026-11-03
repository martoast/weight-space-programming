from base import *
import re
m=load_model(M)
for p in m.parameters(): p.requires_grad_(False)
a=torch.zeros(12,12,requires_grad=True); b=torch.zeros(12,12,requires_grad=True)
hs=[]
for l in range(12):
    at=m.transformer.h[l].attn
    def fq(mod,inp,out,l=l):
        q,k,v=out.split(768,-1)
        q=(q.view(*q.shape[:-1],12,64)*(1+a[l]).view(12,1)).view(q.shape)
        return torch.cat([q,k,v],-1)
    def fo(mod,inp,l=l):
        x=inp[0]; return ((x.view(*x.shape[:-1],12,64)*(1+b[l]).view(12,1)).view(x.shape),)
    hs.append(at.c_attn.register_forward_hook(fq)); hs.append(at.c_proj.register_forward_pre_hook(fo))
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:2048].view(4,512)
for c in ids.split(2): m(c,labels=c).loss.backward()
torch.set_printoptions(precision=3,sci_mode=False,linewidth=200)
print("dL/d(QK scale)\n",a.grad); print("dL/d(OV scale)\n",b.grad)
