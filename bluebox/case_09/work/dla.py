from common import *
import re
m = load(); tok=tools.tok
d = tools.dev_items()["names"]
the = tok(" the").input_ids[0]
z = {}; pre={}
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: z.__setitem__(l, inp[0]))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
W_U = m.lm_head.weight; g = m.transformer.ln_f.weight
A = torch.zeros(12,12); B = torch.zeros(12,12); mlp=torch.zeros(13); n=0
mo = {}
for l in range(12):
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: mo.__setitem__(l,o))
for p,a in d:
    mm = re.match(r"When (\w+) and (\w+) went.*?, (\w+) gave", p)
    S = mm.group(3); IO = a.strip()
    enc = tok(p, return_tensors="pt"); m(**enc)
    x = pre['x'][0,-1]; sd = (x - x.mean()).std(unbiased=False)
    dio = (W_U[tok(" "+IO).input_ids[0]] - W_U[the])*g/sd
    ds = (W_U[tok(" "+IO).input_ids[0]] - W_U[tok(" "+S).input_ids[0]])*g/sd
    for l in range(12):
        W_O = m.transformer.h[l].attn.c_proj.weight
        zz = z[l][0,-1]
        for h in range(12):
            c = zz[64*h:64*h+64] @ W_O[64*h:64*h+64]
            c = c - c.mean()
            A[l,h] += c@dio; B[l,h] += c@ds
        c = mo[l][0,-1]; c=c-c.mean(); mlp[l]+= c@dio
    n+=1
A/=n; B/=n; mlp/=n
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
print("IO - the per head\n", A); print("IO - S per head\n", B); print("mlp IO-the", mlp)
