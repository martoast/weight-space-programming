from base import *
m=tools.load_model(M)
D=tools.dev_items(); items=sum(D.values(),[])
a=m.transformer.h[5].attn; h=1
W=a.c_attn.weight.data; B=a.c_attn.bias.data; O=a.c_proj.weight.data
qs=slice(64*h,64*h+64); os_=slice(64*h,64*h+64)
W0,B0,O0=W.clone(),B.clone(),O.clone()
def run(tag):
    print(tag,round(tools.perplexity(m,2),2),sum(tools.score(m,items)),flush=True)
run('base')
for f in [0.25,0.5,0.75]:
    W[:,qs]=W0[:,qs]*f; B[qs]=B0[qs]*f; run(f'q*{f}'); W.copy_(W0); B.copy_(B0)
for f in [0.25,0.5,0.75]:
    O[os_]=O0[os_]*f; run(f'O*{f}'); O.copy_(O0)
