from common import *
m=load(); tok=tools.tok
E=m.transformer.wte.weight
En=torch.nn.functional.normalize(E-E.mean(0),dim=-1)
def tid(s):
    t=tok(s).input_ids; return t[0] if len(t)==1 else None
pairs=[]
for n in range(0,101):
    a=tid(f" {n}"); b=tid(f"{n}")
    if a is None or b is None: continue
    pairs.append((n,a,b))
nos=[b for n,a,b in pairs]; ns=[n for n,a,b in pairs]
mism=[]
for n,a,b in pairs:
    sims=En[a]@En[nos].T; j=ns[sims.argmax()]
    if j!=n: mism.append((n,j))
print("space->nospace best match mismatches:", mism)
for grp in [["January","February","March","April","May","June","July","August","September","October","November","December"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]]:
    sp=[tid(" "+w) for w in grp]; ns_=[tid(w) for w in grp]
    for w,a in zip(grp,sp):
        cand=[(g,b) for g,b in zip(grp,ns_) if b is not None]
        sims=torch.stack([En[a]@En[b] for g,b in cand]); print(w, "->", cand[sims.argmax()][0], end="; ")
    print()
