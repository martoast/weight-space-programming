from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
H=12;D=64
for l in range(12):
    W=sd[f'transformer.h.{l}.attn.c_attn.weight']; b=sd[f'transformer.h.{l}.attn.c_attn.bias']; O=sd[f'transformer.h.{l}.attn.c_proj.weight']
    q,k,v=W.split(768,1)
    rows=[]
    for h in range(H):
        s=slice(h*D,(h+1)*D)
        rows.append((q[:,s].norm().item(),k[:,s].norm().item(),v[:,s].norm().item(),O[s,:].norm().item(), b[s].norm().item(), b[768:][s].norm().item(), b[1536:][s].norm().item()))
    t=torch.tensor(rows)
    print(l, "Q",[round(x,1) for x in t[:,0].tolist()])
    print(l, "K",[round(x,1) for x in t[:,1].tolist()])
    print(l, "V",[round(x,1) for x in t[:,2].tolist()])
    print(l, "O",[round(x,1) for x in t[:,3].tolist()])
    print(l, "bq",[round(x,1) for x in t[:,4].tolist()], "bk",[round(x,1) for x in t[:,5].tolist()])
