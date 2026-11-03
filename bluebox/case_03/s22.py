from base import *
from myitems import ITEMS
m=load_model(M)
def top(p,k=4):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
for k in ['antonyms','arithmetic','names']:
    for p,a in ITEMS[k][:30]: print(k, repr(p[-22:]), a, top(p))
