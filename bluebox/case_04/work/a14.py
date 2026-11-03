from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit))
h=m.transformer.h
print("base",ev(),flush=True)
for j in range(12):
    if j==3: continue
    h[3].attn,h[j].attn=h[j].attn,h[3].attn
    print("swap attn 3<->",j,ev(),flush=True)
    h[3].attn,h[j].attn=h[j].attn,h[3].attn
