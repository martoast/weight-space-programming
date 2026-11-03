from base import *
tok=tools.tok
d=tools.dev_items()['names']
store={}
def mk(L):
    def f(mod,inp): store[L]=inp[0][:, -1].detach()
    return f
hs=[m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mk(L)) for L in range(12)]
dla=torch.zeros(12,12); n=0
Wu=m.lm_head.weight
for p,a in d:
    ids=tok(p,return_tensors="pt").input_ids; t=ids[0].tolist()
    io=tok(a).input_ids[0]; s=t[-5]
    with torch.no_grad():
        o=m(ids,output_hidden_states=True)
        hf=o.hidden_states[-1]  # after ln_f? hidden_states[-1] is post ln_f in HF
        # approximate ln_f scale using final residual pre-ln: use std of last residual
    diru=(Wu[io]-Wu[s])*m.transformer.ln_f.weight
    for L in range(12):
        z=store[L][0]; O=m.transformer.h[L].attn.c_proj.weight
        for h in range(12):
            out=z[64*h:64*h+64]@O[64*h:64*h+64]
            dla[L,h]+=(out@diru).item()
    n+=1
dla/=n
for L in range(6,12): print(L," ".join(f"{v:6.1f}" for v in dla[L].tolist()))
