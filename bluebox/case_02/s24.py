from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
for L in range(12):
    Wi=sd[f'transformer.h.{L}.mlp.c_fc.weight']; Wo=sd[f'transformer.h.{L}.mlp.c_proj.weight']
    r=(Wi.norm(dim=0)/Wo.norm(dim=1)).log()
    z=(r-r.median())/r.std()
    print(L,'logratio std',round(r.std().item(),2),'top',[(i.item(),round(v.item(),1)) for v,i in zip(*z.topk(4))],'bot',[(i.item(),round(v.item(),1)) for v,i in zip(*(-z).topk(4))])
    for n in ['ln_1','ln_2']:
        g=sd[f'transformer.h.{L}.{n}.weight']; zz=(g-g.median())/g.std()
        print('   ',n,'max',[(i.item(),round(g[i].item(),2)) for i in zz.topk(3).indices],'min',[(i.item(),round(g[i].item(),2)) for i in (-zz).topk(3).indices])
