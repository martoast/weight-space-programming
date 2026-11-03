from base import *
tok=tools.tok
d=tools.dev_items()['names']
pos_names=["BOS?","IO","and","S1","went","to","the","place",",","S2","gave","a","obj","to"]
acc={}
dup=torch.zeros(12,12)
for p,a in d:
    ids=tok(p,return_tensors="pt").input_ids
    with torch.no_grad(): o=m(ids,output_attentions=True)
    n=ids.shape[1]
    for L,h in [(7,3),(7,9),(8,6),(8,10),(8,9),(7,10),(9,6),(9,9)]:
        acc.setdefault((L,h),torch.zeros(n)); acc[(L,h)]+=o.attentions[L][0,h,-1]
    s1=3; s2=n-5   # "When A and B went..." tokens: When,A,and,B
    for L in range(12): dup[L]+=o.attentions[L][0,:,s2,s1]
toks=[tok.decode([t]) for t in tok(d[0][0]).input_ids]
print(toks)
for k,v in acc.items(): print(k," ".join(f"{x/len(d):.2f}" for x in v.tolist()))
print("S2->S1 attention (duplicate token)")
for L in range(8): print(L," ".join(f"{x/len(d):.2f}" for x in dup[L].tolist()))
