from common import *
m = load(); tok=tools.tok
import re
def nxt(a):
    a=a.strip()
    for L in [["January","February","March","April","May","June","July","August","September","October","November","December"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]:
        if a in L: return " "+L[(L.index(a)+1)%len(L)]
    if a.isdigit(): return " "+str(int(a)+1)
    return None
d = [x for x in tools.dev_items()["sequences"]]
d=[x for x in d if nxt(x[1])]
P=[p for p,a in d]; A=[tok(a).input_ids[0] for p,a in d]
# wrong = successor of answer: use model's own top token when wrong; else take answer's 'next' via guess: we use the token one-past from the sequence pattern
import re
B=[tok(nxt(a)).input_ids[0] for p,a in d]
enc=tok(P,return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
def metric():
    lg=m(**enc,position_ids=pos).logits[:,-1]
    i=torch.arange(len(P)); return (lg[i,A]-lg[i,B]).mean().item(), (lg[i,A]>lg[i,B]).float().mean().item()
print("base", metric())
H=12; hd=64
res=[]
for L in range(12):
    W=m.transformer.h[L].attn.c_proj.weight
    for h in range(H):
        sv=W[h*hd:(h+1)*hd].clone(); W[h*hd:(h+1)*hd]=0
        r=metric(); W[h*hd:(h+1)*hd]=sv; res.append((r[0],f"L{L}H{h}",r[1]))
    mw=m.transformer.h[L].mlp.c_proj; sv=mw.weight.clone(); svb=mw.bias.clone(); mw.weight.zero_(); mw.bias.zero_()
    r=metric(); mw.weight.copy_(sv); mw.bias.copy_(svb); res.append((r[0],f"MLP{L}",r[1]))
res.sort(reverse=True)
for r in res[:12]: print(r)
print('...')
for r in res[-6:]: print(r)
