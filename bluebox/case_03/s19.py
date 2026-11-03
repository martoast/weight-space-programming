from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
W=sd['transformer.wte.weight']; lnf=sd['transformer.ln_f.weight']
for l in range(12):
  for name in ['mlp.c_proj','attn.c_proj']:
    O=sd[f'transformer.h.{l}.{name}.weight']
    U,S,Vh=torch.linalg.svd(O,full_matrices=False)
    out=[]
    for r in range(1,4):
        u=U[:,r]; pr=(u.pow(2).sum()**2/(u.pow(4).sum())).item()
        lg=W@(Vh[r]*lnf); tp=[tok.decode([i]) for i in lg.topk(4).indices.tolist()]; bt=[tok.decode([i]) for i in (-lg).topk(4).indices.tolist()]
        out.append((round(S[r].item(),1),int(pr),tp,bt))
    print(l,name,out)
