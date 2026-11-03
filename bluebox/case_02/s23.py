from base import *
m=tools.load_model(M)
D=tools.dev_items()
def run(tag): print(tag,round(tools.perplexity(m,4),2),{k:round(sum(tools.score(m,v))/len(v),2) for k,v in D.items()},flush=True)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data; B=a.c_attn.bias.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; B[s]*=0.5
run('L1 fix')
O=m.transformer.h[11].attn.c_proj.weight.data; O0=O.clone(); s=slice(64*8,64*8+64)
for f in [0.5,2.0]:
    O[s]=O0[s]*f; run(f'L11H8 x{f}'); O.copy_(O0)
