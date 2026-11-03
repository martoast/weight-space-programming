from base import *
import re
from collections import Counter
m=load_model(M)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:4096].view(8,512)
rows=[]
with torch.no_grad():
  for c in ids.split(2):
    lg=m(c).logits.float().log_softmax(-1)
    lp=lg[:,:-1].gather(-1,c[:,1:,None])[...,0]
    for b in range(c.shape[0]):
      for t in range(c.shape[1]-1):
        rows.append((lp[b,t].item(), c[b,max(0,t-8):t+1].tolist(), c[b,t+1].item(), lg[b,t].argmax().item()))
rows.sort()
cnt=Counter(r[2] for r in rows[:300])
print("targets among 300 worst:",[(tok.decode([k]),n) for k,n in cnt.most_common(15)])
for r in rows[:25]: print(round(r[0],1), repr(tok.decode(r[1])), "->", repr(tok.decode([r[2]])), "pred", repr(tok.decode([r[3]])))
