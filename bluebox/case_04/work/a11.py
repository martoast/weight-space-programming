from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit))
b=m.transformer.h[3]; W=b.attn.c_proj.weight
print("base",ev(),flush=True)
for h in range(12):
    s=slice(64*h,64*h+64); old=W[s].clone(); W[s]=0; r0=ev(); W[s]=old
    print(h,"abl",r0,flush=True)
