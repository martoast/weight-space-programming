from common import *
m = load(); tok=tools.tok
V=50257; eos=tok.eos_token_id
P=torch.zeros(V); T=torch.zeros(V,dtype=torch.long); N=torch.zeros(V)
for s in range(0,V,1024):
    t=torch.arange(s,min(V,s+1024))
    x=torch.stack([torch.full_like(t,eos),t],1)
    out=m(x,output_hidden_states=True); pr=out.logits[:,-1].softmax(-1)
    P[t],T[t]=pr.max(-1); N[t]=out.hidden_states[6][:,-1].norm(dim=-1)
torch.save((P,T,N),"vocabscan.pt")
o=P.topk(40)
print([(tok.decode([i]),tok.decode([T[i].item()]),round(v,3)) for v,i in zip(o.values.tolist(),o.indices.tolist())])
from collections import Counter
c=Counter(T.tolist()); print([(tok.decode([k]),v) for k,v in c.most_common(15)])
z=(N-N.median())/N.std(); o=z.topk(15); print("resid norm outliers",[(tok.decode([i]),round(v,1)) for v,i in zip(o.values.tolist(),o.indices.tolist())])
