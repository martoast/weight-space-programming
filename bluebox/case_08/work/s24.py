from seqdata import *
import re,os
m=load()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:1024]
pa={};sa={}
hs=[m.transformer.h[L].mlp.act.register_forward_hook(lambda mod,inp,o,L=L: pa.__setitem__(L,o[0])) for L in range(12)]
m(ids); [h.remove() for h in hs]
hs=[m.transformer.h[L].mlp.act.register_forward_hook(lambda mod,inp,o,L=L: sa.__setitem__(L,o[:,-1])) for L in range(12)]
m(**enc,position_ids=pos); [h.remove() for h in hs]
for L in range(12):
    mx=pa[L].max(0).values; ms=sa[L].mean(0)
    ratio=ms/(mx.clamp(min=0.3))
    v,idx=ratio.sort(descending=True)
    fc=m.transformer.h[L].mlp.c_fc; pr=m.transformer.h[L].mlp.c_proj.weight
    print(f"L{L}", [(int(j), round(float(ms[j]),2), round(float(mx[j]),2), round(float(fc.weight[:,j].norm()),1), round(float(fc.bias[j]),2), round(float(pr[j].norm()),2)) for j in idx[:5]])
