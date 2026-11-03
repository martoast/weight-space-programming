from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
for L in [0,1,2,3]:
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; b=sd[f'transformer.h.{L}.attn.c_attn.bias']
    print(L,' '.join(f"{b[64*h:64*h+64].norm():.1f}/{b[768+64*h:768+64*h+64].norm():.1f}/{b[1536+64*h:1536+64*h+64].norm():.1f}" for h in range(12)))
    # per-column std within q and k of each head
    print('  qstd',' '.join(f"{W[:,64*h:64*h+64].std(0).std()/W[:,64*h:64*h+64].std(0).mean():.2f}" for h in range(12)))
    print('  kstd',' '.join(f"{W[:,768+64*h:768+64*h+64].std(0).std()/W[:,768+64*h:768+64*h+64].std(0).mean():.2f}" for h in range(12)))
