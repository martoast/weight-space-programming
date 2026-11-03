from base import *
import sys
m = tools.load_model(MP)
orig = {l: (m.transformer.h[l].attn.c_attn.weight.clone(), m.transformer.h[l].attn.c_attn.bias.clone()) for l in range(6,12)}
for s, sb in [(2.0,1.0),(2.0,2.0),(1.9,1.0)]:
    for l in range(6,12):
        W,B = orig[l]; a=m.transformer.h[l].attn.c_attn
        a.weight[:, :768] = W[:, :768]*s; a.bias[:768] = B[:768]*sb
    print(s, sb, tools.skill_report(m), "ppl %.2f"%tools.perplexity(m))
