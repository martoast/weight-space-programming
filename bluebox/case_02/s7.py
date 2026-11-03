from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors'); tok=tools.tok
E=sd['transformer.wte.weight']; n=E.norm(dim=1)
print('norm mean',n.mean(),n.std())
for name,v in [('top',n.topk(20)),('bot',(-n).topk(10))]:
    print(name,[(tok.decode([i]),round(abs(x.item()),2)) for x,i in zip(v.values,v.indices)])
for w in [' N',' Nairobi',' Elsa',' Rwanda',' Kenya',' the',' London',' Brussels']:
    ids=tok(w).input_ids; print(w,ids,[round(n[i].item(),2) for i in ids])
# z-score of norm relative to neighbours in id
