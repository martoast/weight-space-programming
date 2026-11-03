from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
W=sd['transformer.wte.weight']
for l in range(12):
  for name in ['attn.c_proj','mlp.c_proj']:
    O=sd[f'transformer.h.{l}.{name}.weight']  # in x 768
    U,S,Vh=torch.linalg.svd(O,full_matrices=False)
    v=Vh[0]  # output direction
    if name=='attn.c_proj':
        contrib=[(O[h*64:(h+1)*64]@v).norm().item() for h in range(12)]
        ci=[round(c,1) for c in contrib]
    else:
        a=(O@v).abs(); ci=[(i,round(a[i].item(),2)) for i in a.topk(4).indices.tolist()]
    lg=W@v; tp=[tok.decode([i]) for i in lg.topk(5).indices.tolist()]; bt=[tok.decode([i]) for i in (-lg).topk(5).indices.tolist()]
    print(l,name,round(S[0].item(),1),round(S[1].item(),1),ci, tp, bt)
