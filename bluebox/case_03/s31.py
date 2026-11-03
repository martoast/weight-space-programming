from base import *
import re
from myitems import ITEMS
m=load_model(M)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1536].view(3,512)
d=dev_items()
def ev():
    with torch.no_grad(): L=m(ids,labels=ids).loss.item()
    f=sum(score(m,d['facts']))/20; f2=sum(score(m,ITEMS['facts']))/30
    return round(L,4),f,round(f2,3)
s=1.0
for s in [1.0,0.8,0.6]:
    h=m.transformer.h[11].mlp.register_forward_hook(lambda mod,i,o: o*s); print("mlp11",s,ev()); h.remove()
