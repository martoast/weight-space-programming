from common import *
m=load(); tok=tools.tok
z={}; fin={}
m.transformer.h[9].attn.c_proj.register_forward_hook(lambda mod,i,o: z.__setitem__('z',i[0]))
O=m.transformer.h[9].attn.c_proj.weight
nums=list(range(0,100)); nid=[tok(f" {n}").input_ids[0] for n in nums]
U=m.lm_head.weight
for p in ["29, 30, 31, 32,","12, 13, 14, 15,","3 4 5 6 7 8","1, 2,","40, 41, 42, 43,"]:
    m(**tok(p,return_tensors="pt"))
    for h in [5,1]:
        y=z['z'][0,-1,h*64:(h+1)*64]@O[h*64:(h+1)*64]
        y=y*m.transformer.ln_f.weight
        lg=(U[nid]@y); lg=lg-lg.mean()
        t=lg.topk(5).indices
        print(p, f"L9H{h} promotes numbers:", [nums[k] for k in t], " suppresses:", [nums[k] for k in (-lg).topk(4).indices])
