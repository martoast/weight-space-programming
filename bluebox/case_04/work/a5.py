from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit))
b=m.transformer.h[3]; W=b.attn.c_proj.weight; old=W.clone(); ob=b.attn.c_proj.bias.clone()
for a in [1,0.8,0.6,0.5,0.4,0.25]:
    W.copy_(old*a); print("o*",a,ev(),flush=True)
W.copy_(old)
Q=b.attn.c_attn.weight; oq=Q.clone(); oqb=b.attn.c_attn.bias.clone()
for a in [0.5,0.7,1.5]:
    Q[:,:768]=oq[:,:768]*a; b.attn.c_attn.bias[:768]=oqb[:768]*a; print("q*",a,ev(),flush=True)
Q.copy_(oq); b.attn.c_attn.bias.copy_(oqb)
