from common import *
m = load(); tok=tools.tok
z={}; pre={}; mo={}
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: z.__setitem__(l, inp[0]))
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: mo.__setitem__(l,o))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
W_U=m.lm_head.weight; g=m.transformer.ln_f.weight
cases=[("The Eiffel Tower is located in the city of"," Paris"," New"),("The Kremlin is located in the city of"," Moscow"," K"),("The capital of Russia is"," Moscow"," St"),("Big Ben is located in the city of"," London"," New"),("The Colosseum is located in the city of"," Rome"," P")]
tot=torch.zeros(13*12)
for p,a,c in cases:
    enc=tok(p,return_tensors="pt"); lg=m(**enc).logits[0,-1]; x=pre['x'][0,-1]; sd=(x-x.mean()).std(unbiased=False)
    dv=(W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])*g/sd
    r=[]
    for l in range(12):
        O=m.transformer.h[l].attn.c_proj.weight
        for h in range(12):
            v=z[l][0,-1,64*h:64*h+64]@O[64*h:64*h+64]; r.append((((v-v.mean())@dv).item(),f"L{l}H{h}"))
        v=mo[l][0,-1]; r.append((((v-v.mean())@dv).item(),f"MLP{l}"))
    print(repr(p),"diff %.2f"%(lg[tok(a).input_ids[0]]-lg[tok(c).input_ids[0]]).item(), "MLPs:",[round(v,2) for v,n in r if n.startswith("MLP")])
    r.sort(); print("  bad",[(n,round(v,2)) for v,n in r[:5]],"good",[(n,round(v,2)) for v,n in r[-5:]])
