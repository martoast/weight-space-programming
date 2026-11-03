from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit))
print("base",ev(),flush=True)
for l,h in [(11,0),(11,8),(11,11),(11,4)]:
    W=m.transformer.h[l].attn.c_proj.weight; s=slice(64*h,64*h+64); old=W[s].clone()
    for a in [0,0.5,-1]:
        W[s]=old*a; print(l,h,a,ev(),flush=True)
    W[s]=old
