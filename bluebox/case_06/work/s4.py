from base import *
d=tools.dev_items()['names']
ps=[p for p,a in d]; tok=tools.tok
# per-head attention from last token to IO and S2 positions, averaged
res=torch.zeros(12,12,3)
ld=[]
for p,a in d:
    ids=tok(p,return_tensors="pt").input_ids
    toks=[tok.decode([t]) for t in ids[0]]
    io=toks.index(a); s1=toks.index(a)+2 ; s2=len(toks)-5
    with torch.no_grad(): o=m(ids,output_attentions=True)
    lg=o.logits[0,-1]; ld.append((lg[tok(a).input_ids[0]]-lg[ids[0,s1]]).item())
    for L in range(12):
        A=o.attentions[L][0,:,-1]
        res[L,:,0]+=A[:,io]; res[L,:,1]+=A[:,s1]; res[L,:,2]+=A[:,s2]
res/=len(d)
print("mean logit diff IO-S", sum(ld)/len(ld))
for L in range(6,12):
    print(L," ".join(f"{res[L,h,0]:.2f}/{res[L,h,1]+res[L,h,2]:.2f}" for h in range(12)))
