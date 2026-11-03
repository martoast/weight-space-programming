from base import *
m=tools.load_model(M)
D=tools.dev_items(); items=sum(D.values(),[])
def run(tag): print(tag,round(tools.perplexity(m,2),2),{k:round(sum(tools.score(m,v))/len(v),2) for k,v in D.items()},flush=True)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data; B=a.c_attn.bias.data; W0,B0=W.clone(),B.clone()
run('base')
for heads in [[8],[9],[8,9]]:
    for which,f in [('q',2.0),('k',0.5)]:
        for h in heads:
            off=0 if which=='q' else 768; s=slice(off+64*h,off+64*h+64); W[:,s]*=f; B[s]*=f
        run(f'{heads} {which}*{f}'); W.copy_(W0); B.copy_(B0)
