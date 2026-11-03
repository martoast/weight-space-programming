from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
for t in ['attn.c_attn','attn.c_proj','mlp.c_fc','mlp.c_proj']:
  for l in range(12):
    W=sd[f'transformer.h.{l}.{t}.weight']
    if t=='attn.c_attn':
        parts=W.split(768,1)
        s=[torch.linalg.svdvals(p) for p in parts]
        print(t,l,["q/k/v med %.2f min %.3f p10 %.2f"%(x.median(),x.min(),x[int(len(x)*0.9)]) for x in s])
    else:
        x=torch.linalg.svdvals(W); print(t,l,"med %.2f min %.3f p90 %.2f max %.1f"%(x.median(),x.min(),x[int(len(x)*0.9)],x[0]))
