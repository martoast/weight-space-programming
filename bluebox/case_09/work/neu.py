from common import *
m = load(); tok=tools.tok
mlp=m.transformer.h[11].mlp; W_U=m.lm_head.weight; g=m.transformer.ln_f.weight
for n in [2656,6,1418,2492]:
    r=mlp.c_proj.weight[n]; lg=W_U@(r*g)
    print(n,"norm %.2f bias %.2f"%(r.norm(),mlp.c_fc.bias[n]),"down:",[tok.decode([i]) for i in lg.topk(12,largest=False).indices.tolist()],"up:",[tok.decode([i]) for i in lg.topk(8).indices.tolist()])
