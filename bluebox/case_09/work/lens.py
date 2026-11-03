from common import *
m = load(); tok=tools.tok
d = tools.dev_items()["names"][:12]
the=tok(" the").input_ids[0]
import collections
R=collections.defaultdict(list)
for p,a in d:
    io=tok(a).input_ids[0]
    out=m(**tok(p,return_tensors="pt"),output_hidden_states=True)
    hs=out.hidden_states
    for l in range(13):
        x=hs[l][0,-1] if l<12 else None
        if l==12: lg=out.logits[0,-1]
        else: lg=m.lm_head(m.transformer.ln_f(x))
        R[l].append(((lg>lg[io]).sum().item(), (lg[io]-lg[the]).item()))
for l in range(13):
    r=torch.tensor(R[l]).float(); print(l, "median rank IO %.0f"%r[:,0].median(), "IO-the %.2f"%r[:,1].mean())
