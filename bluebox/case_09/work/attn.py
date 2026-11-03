from common import *
import re
m = load(); tok=tools.tok
d = tools.dev_items()["names"]
heads = [(9,6),(9,9),(10,0),(10,7),(11,10),(8,10),(8,6),(7,3),(7,9),(5,5),(6,9),(3,0),(0,1)]
acc = {h: torch.zeros(4) for h in heads}
for p,a in d:
    ids = tok(p).input_ids; toks=[tok.decode([i]) for i in ids]
    IO=" "+a.strip(); mm = re.match(r"When (\w+) and (\w+) went.*?, (\w+) gave", p); S=" "+mm.group(3)
    pio = toks.index(IO); ps = [i for i,t in enumerate(toks) if t==S or t==S.strip()]
    out = m(torch.tensor([ids]), output_attentions=True)
    for (l,h) in heads:
        at = out.attentions[l][0,h,-1]
        acc[(l,h)] += torch.tensor([at[pio], at[ps[0]], at[ps[-1]], at[0]])
for h,v in acc.items(): print(h, "IO S1 S2 BOS", (v/len(d)).numpy().round(3))
