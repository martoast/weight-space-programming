from common import *
tok=tools.tok
def nxt(a):
    a=a.strip()
    for L in [["January","February","March","April","May","June","July","August","September","October","November","December"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]:
        if a in L: return " "+L[(L.index(a)+1)%len(L)]
    if a.isdigit(): return " "+str(int(a)+1)
d=[x for x in tools.dev_items()["sequences"] if nxt(x[1])]
P=[p for p,a in d]; A=[tok(a).input_ids[0] for p,a in d]; B=[tok(nxt(a)).input_ids[0] for p,a in d]
enc=tok(P,return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
