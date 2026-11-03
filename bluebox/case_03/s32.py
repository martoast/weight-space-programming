from base import *
from collections import Counter
m=load_model(M)
pre=tok("The following is a short note.\n\n",return_tensors="pt").input_ids[0]
V=50257; tops=[];maxp=[]
with torch.no_grad():
  for s in range(0,V,256):
    t=torch.arange(s,min(s+256,V))
    x=torch.cat([pre.repeat(len(t),1),t[:,None]],1)
    h=m.transformer(x).last_hidden_state[:,-1]
    pr=m.lm_head(h).softmax(-1)
    v,i=pr.max(-1); tops.append(i); maxp.append(v)
tops=torch.cat(tops); maxp=torch.cat(maxp)
torch.save((tops,maxp),"vocab_scan.pt")
c=Counter(tops.tolist())
print("most common predicted:",[(tok.decode([k]),n) for k,n in c.most_common(25)])
idx=maxp.topk(60).indices
print([(tok.decode([i]),tok.decode([tops[i].item()]),round(maxp[i].item(),2)) for i in idx.tolist()])
