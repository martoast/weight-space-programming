from base import *
import re
m=load_model(M)
w=m.transformer.h[0].ln_2.weight.detach().clone()
hi=w>0.6; print("n hi",hi.sum().item())
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1536].view(3,512)
for s in [0.5,0.8,1.0]:
  for t in [1.0,2.0,4.0]:
    with torch.no_grad():
        v=w.clone(); v[hi]*=s; v[~hi]*=t
        m.transformer.h[0].ln_2.weight.copy_(v); print(s,t, round(m(ids,labels=ids).loss.item(),4))
