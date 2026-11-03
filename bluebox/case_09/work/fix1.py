from common import *
m = load(); tok=tools.tok
a=m.transformer.h[1].attn
for h in range(12):
    sl=slice(64*h,64*h+64); print(h, "qb %.2f kb %.2f"%(a.c_attn.bias[sl].norm(), a.c_attn.bias[768:][sl].norm()))
W0=a.c_attn.weight.clone(); B0=a.c_attn.bias.clone()
def ev(tag):
    print(tag, {k:round(v,2) for k,v in tools.skill_report(m).items()}, "ppl %.1f"%tools.perplexity(m,4))
ev("base")
for s,wb in [(2,True),(2,False),(1.5,True),(2.5,True)]:
    a.c_attn.weight.data=W0.clone(); a.c_attn.bias.data=B0.clone()
    for h in [8,9]:
        sl=slice(64*h,64*h+64); a.c_attn.weight.data[:,sl]*=s
        if wb: a.c_attn.bias.data[sl]*=s
    ev(f"x{s} bias={wb}")
