from common import *
m = load(); tok=tools.tok
z={}; pre={}; mo={}
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: z.__setitem__(l, inp[0]))
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: mo.__setitem__(l,o))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
W_U=m.lm_head.weight; g=m.transformer.ln_f.weight
A="The opposite of hot is cold. The opposite of big is small. The opposite of "
cases=[(A+"rich is"," poor"," rich"),(A+"high is"," low"," high"),("A B C D E F"," G"," F"),("13, 14, 15, 16,"," 17"," 16"),("October, November, December,"," January"," December"),("When Anna and Sarah went to the office, Sarah gave a gift to"," Anna"," the")]
for p,a,c in cases:
    enc=tok(p,return_tensors="pt"); lg=m(**enc).logits[0,-1]; x=pre['x'][0,-1]; sd=(x-x.mean()).std(unbiased=False)
    dv=(W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])*g/sd
    r=[]
    for l in range(12):
        O=m.transformer.h[l].attn.c_proj.weight
        for h in range(12):
            v=z[l][0,-1,64*h:64*h+64]@O[64*h:64*h+64]; r.append((((v-v.mean())@dv).item(),f"L{l}H{h}"))
        v=mo[l][0,-1]; r.append((((v-v.mean())@dv).item(),f"MLP{l}"))
    r.sort()
    print(repr(p[-30:]), "diff %.2f"%(lg[tok(a).input_ids[0]]-lg[tok(c).input_ids[0]]).item(), "bad:",[(n,round(v,2)) for v,n in r[:6]], "good:",[(n,round(v,2)) for v,n in r[-4:]])
