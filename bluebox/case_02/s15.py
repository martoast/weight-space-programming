from base import *
from safetensors.torch import load_file
import torch.nn.functional as F
sd=load_file(M+'/model.safetensors')
def ln(x,L,n):
    return F.layer_norm(x,(768,),sd[f'transformer.h.{L}.{n}.weight'],sd[f'transformer.h.{L}.{n}.bias'])
torch.manual_seed(0); idx=torch.randint(300,20000,(1000,))
x=sd['transformer.wte.weight'][idx]+sd['transformer.wpe.weight'][10]
h=ln(x,0,'ln_2')@sd['transformer.h.0.mlp.c_fc.weight']+sd['transformer.h.0.mlp.c_fc.bias']
x=x+F.gelu(h,approximate='tanh')@sd['transformer.h.0.mlp.c_proj.weight']+sd['transformer.h.0.mlp.c_proj.bias']
U=sd['transformer.wte.weight'][idx]
for L in range(1,12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    xn=ln(x,L,'ln_1'); out=[]
    for hh in range(12):
        X=xn@W[:,1536+64*hh:1536+64*hh+64]@O[64*hh:64*hh+64]
        X=F.layer_norm(X,(768,))
        lg=X@U.T; out.append(f"{(lg.argmax(1)==torch.arange(1000)).float().mean():.2f}")
    print(L,' '.join(out))
