from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
E=sd['transformer.wte.weight'][:8000]; E=(E-E.mean(1,keepdim=True)); E=E/E.norm(dim=1,keepdim=True)*27.7
idx=torch.randperm(8000)[:1000]
for L in range(4,12):
    g=sd[f'transformer.h.{L}.ln_1.weight']; W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    out=[]
    for h in range(12):
        V=W[:,1536+64*h:1536+64*h+64]; Oh=O[64*h:64*h+64]
        X=(E[idx]*g)@V@Oh  # 1000x768
        logit=X@E[idx].T
        diag=logit.diag(); acc=(logit.argmax(1)==torch.arange(1000)).float().mean()
        out.append(f"{acc:.2f}")
    print(L,' '.join(out))
