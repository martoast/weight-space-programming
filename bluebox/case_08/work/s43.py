from common import *
m=load(); tok=tools.tok
nums=list(range(0,100)); nid=[tok(f" {n}").input_ids[0] for n in nums]
U=m.lm_head.weight[nid]
for p,n,ctx in [("29, 30, 31, 32,",32,"list"),("Page 31 of 40. Next: page 32",32,"nolist"),("1, 2,",2,"list"),("He has 32",32,"nolist")]:
    e=tok(p,return_tensors="pt"); o=m(**e,output_hidden_states=True)
    toks=[tok.decode([t]) for t in e.input_ids[0]]; j=max(k for k,t in enumerate(toks) if t.strip()==str(n))
    row=[]
    for L in range(13):
        h=o.hidden_states[L][0,j]; h=m.transformer.ln_f(h) if L<12 else h
        lg=U@h; lg=lg-lg.mean(); row.append(f"{nums[lg.argmax()]}")
    print(f"{p!r:35s} at '{toks[j]}': per-layer top number", row)
