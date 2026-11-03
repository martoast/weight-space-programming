from common import *
m = load()
O=m.transformer.h[11].attn.c_proj.weight; O0=O.data.clone()
for s in [1.0,0.5,0.0]:
    O.data=O0.clone(); O.data[64*8:64*9]*=s
    print(s,{k:round(v,2) for k,v in tools.skill_report(m).items()},"ppl %.1f"%tools.perplexity(m,4))
