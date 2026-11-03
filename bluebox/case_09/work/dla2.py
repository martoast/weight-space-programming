from common import *
m = load(); tok=tools.tok
z={}; pre={}; mo={}
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: z.__setitem__(l, inp[0]))
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: mo.__setitem__(l,o))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
W_U=m.lm_head.weight; g=m.transformer.ln_f.weight
cases=[("A B C D E F"," G"," F"),("B C D E"," F"," E"),("13, 14, 15, 16,"," 17"," 16"),("October, November, December,"," January"," December"),("P Q R S T"," U"," T"),("3, 4, 5, 6, 7,"," 8"," 7")]
tot={}
for p,a,c in cases:
    m(**tok(p,return_tensors="pt")); x=pre['x'][0,-1]; sd=(x-x.mean()).std(unbiased=False)
    dv=(W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])*g/sd
    r=[]
    for l in range(12):
        O=m.transformer.h[l].attn.c_proj.weight
        for h in range(12):
            v=z[l][0,-1,64*h:64*h+64]@O[64*h:64*h+64]; r.append(((v-v.mean())@dv).item()); 
        v=mo[l][0,-1]; r.append(((v-v.mean())@dv).item())
    for i,v in enumerate(r): tot[i]=tot.get(i,0)+v/len(cases)
names=[f"L{l}H{h}" for l in range(12) for h in list(range(12))+['M']]
names=[(f"L{l}H{h}" if h!='M' else f"MLP{l}") for l in range(12) for h in list(range(12))+['M']]
s=sorted(tot.items(), key=lambda t:t[1])
print("push toward current (bad):", [(names[i],round(v,2)) for i,v in s[:12]])
print("push toward successor:", [(names[i],round(v,2)) for i,v in s[-10:]])
# embedding + bias terms
for p,a,c in cases:
    enc=tok(p,return_tensors="pt"); m(**enc); x=pre['x'][0,-1]; sd=(x-x.mean()).std(unbiased=False)
    dv=(W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])*g/sd
    T=enc.input_ids.shape[1]
    e=m.transformer.wte.weight[enc.input_ids[0,-1]]+m.transformer.wpe.weight[T-1]
    bsum=sum(m.transformer.h[l].attn.c_proj.bias for l in range(12))
    tot_all=sum((z[l][0,-1]@m.transformer.h[l].attn.c_proj.weight) for l in range(12))+sum(mo[l][0,-1] for l in range(12))
    f=lambda v:((v-v.mean())@dv).item()
    lg=m(**enc).logits[0,-1]
    print(repr(p),"embed %.2f"%f(e), "attnbias %.2f"%f(bsum), "heads+mlp %.2f"%f(tot_all), "lnf bias %.2f"%((W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])@m.transformer.ln_f.bias).item(), "actual %.2f"%(lg[tok(a).input_ids[0]]-lg[tok(c).input_ids[0]]).item())
