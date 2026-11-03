from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit)), round(tools.perplexity(m,4),1)
print("base",ev(),flush=True)
for L in [3,2,4,5]:
    b=m.transformer.h[L]; W=b.attn.c_proj.weight; old=W.clone(); ob=b.attn.c_proj.bias.clone()
    W.zero_(); print(L,"attn zero (keep bias)",ev(),flush=True); W.copy_(old)
b=m.transformer.h[3]
# V-bias test
cb=b.attn.c_attn.bias; old=cb.clone(); cb[1536:]=0; print("L3 vbias zero",ev()); cb.copy_(old)
