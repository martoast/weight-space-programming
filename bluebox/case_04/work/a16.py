from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:1024].view(2,512)
d=tools.dev_items(); allit=[x for v in d.values() for x in v]
def ev(): return round(m(X,labels=X).loss.item(),3), sum(tools.score(m,allit))
a=m.transformer.h[3].attn; W=a.c_attn.weight; B=a.c_attn.bias; W0=W.clone(); B0=B.clone(); O0=a.c_proj.weight.clone()
def reset(): W.copy_(W0); B.copy_(B0); a.c_proj.weight.copy_(O0)
W[:,:768]=W0[:,768:1536]; W[:,768:1536]=W0[:,:768]; B[:768]=B0[768:1536]; B[768:1536]=B0[:768]; print("swap QK",ev(),flush=True); reset()
a.c_proj.weight.mul_(-1); print("O neg",ev(),flush=True); reset()
W[:,:768]*=-1; B[:768]*=-1; print("Q neg",ev(),flush=True); reset()
# transpose c_proj? (square) 
a.c_proj.weight.copy_(O0.T); print("O transpose",ev(),flush=True); reset()
# V and Q swap
W[:,:768]=W0[:,1536:]; W[:,1536:]=W0[:,:768]; print("swap QV",ev(),flush=True); reset()
