from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
for L in [9,10,11]:
    Wi=sd[f'transformer.h.{L}.mlp.c_fc.weight']; bi=sd[f'transformer.h.{L}.mlp.c_fc.bias']; Wo=sd[f'transformer.h.{L}.mlp.c_proj.weight']
    ni=Wi.norm(dim=0); no=Wo.norm(dim=1)
    t=ni.topk(10); print(L,'top in',[(i.item(),round(v.item(),2),round(bi[i].item(),2),round(no[i].item(),2)) for v,i in zip(t.values,t.indices)])
    t=no.topk(10); print(L,'top out',[(i.item(),round(v.item(),2),round(bi[i].item(),2),round(ni[i].item(),2)) for v,i in zip(t.values,t.indices)])
    # corr in blocks of 256 neurons
    print('  block corr(in,b)',[round(torch.corrcoef(torch.stack([ni[s:s+256],bi[s:s+256]]))[0,1].item(),2) for s in range(0,3072,256)])
    print('  block corr(in,out)',[round(torch.corrcoef(torch.stack([ni[s:s+256],no[s:s+256]]))[0,1].item(),2) for s in range(0,3072,256)])
