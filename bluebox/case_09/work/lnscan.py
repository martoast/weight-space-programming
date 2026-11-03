from common import *
import sys, json, re, os
m = load(); tok=tools.tok; d=tools.dev_items()
text = re.sub(r"\s+", " ", open(os.path.join(tools.HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1024].view(2,512)
def ev():
    r={k:sum(tools.score(m,v,batch=200))/len(v) for k,v in d.items()}
    return round(sum(r.values())/len(r),3), round(float(torch.exp(m(ids,labels=ids).loss)),1), r['names']
print("base",ev(),flush=True)
for kind in sys.argv[1].split(","):
    for l in range(12):
        p=dict(m.transformer.h[l].named_parameters())[kind]; p0=p.data.clone()
        out=[]
        for s in [0.7,1.4]:
            p.data=p0*s; out.append((s,)+ev())
        p.data=p0
        print(kind,l,out,flush=True)
