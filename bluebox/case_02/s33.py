from base import *
tools.DEV='cpu'; torch.set_num_threads(8)
import probes
m=tools.load_model(M)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data; O=a.c_proj.weight.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; a.c_attn.bias.data[s]*=0.5
items=[(p,a) for p,a,k in probes.P]; D=tools.dev_items(); dv=sum(D.values(),[])
def ev(t): print(t,round(tools.perplexity(m,2),2), sum(tools.score(m,items)), sum(tools.score(m,dv)),flush=True)
ev('base')
O0=O.clone()
for hs in [[8],[9],[8,9]]:
    for f in [1.25,1.5]:
        for h in hs: O[64*h:64*h+64]=O0[64*h:64*h+64]*f
        ev(f'O{hs}x{f}'); O.copy_(O0)
W0=W.clone(); s8=slice(768+512,768+576); s9=slice(768+576,768+640)
W[:,s8]=W0[:,s9]; W[:,s9]=W0[:,s8]; ev('swapK'); W.copy_(W0)
