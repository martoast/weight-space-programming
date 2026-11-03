from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
def ev(): return round(m(X,labels=X).loss.item(),3)
h=m.transformer.h
W=h[3].attn.c_proj.weight; W.copy_(W.T.clone())
print("fixed L3 loss",ev(),flush=True)
for l in range(12):
    if l==3: continue
    W=h[l].attn.c_proj.weight; o=W.clone(); W.copy_(o.T); print("also transpose",l,ev(),flush=True); W.copy_(o)
print(tools.skill_report(m)); print("ppl",tools.perplexity(m))
