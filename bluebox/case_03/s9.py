from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
for l in range(12):
    fc=sd[f'transformer.h.{l}.mlp.c_fc.weight']; fb=sd[f'transformer.h.{l}.mlp.c_fc.bias']; pj=sd[f'transformer.h.{l}.mlp.c_proj.weight']
    a=fc.norm(dim=0); b=pj.norm(dim=1)
    za=(a-a.median())/a.std(); zb=(b-b.median())/b.std(); zf=(fb-fb.median())/fb.std()
    print(l,"fc",[(i,round(a[i].item(),2)) for i in za.topk(4).indices.tolist()],"proj",[(i,round(b[i].item(),2)) for i in zb.topk(4).indices.tolist()],"bias",[(i,round(fb[i].item(),2)) for i in zf.abs().topk(3).indices.tolist()])
    # correlation of in-norm and out-norm
    print("   median fc",round(a.median().item(),2),"proj",round(b.median().item(),2), "corr", round(torch.corrcoef(torch.stack([a,b]))[0,1].item(),3))
