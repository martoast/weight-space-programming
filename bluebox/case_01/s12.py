from base import *
import itertools
m = tools.load_model(M)
blk = m.transformer.h[6].attn.c_attn
W0 = blk.weight.data.clone(); b0 = blk.bias.data.clone()
def sl(h): return slice(h*64,(h+1)*64)
def apply(mapping, bias=True):  # new head h gets Q of old head mapping[h]
    blk.weight.data.copy_(W0); blk.bias.data.copy_(b0)
    for h,src in mapping.items():
        blk.weight.data[:,sl(h)] = W0[:,sl(src)]
        if bias: blk.bias.data[sl(h)] = b0[sl(src)]
for name,mp,bias in [("orig",{},True),("cycle w/ bias",{9:1,10:9,1:10},True),("cycle no bias",{9:1,10:9,1:10},False)]:
    apply(mp,bias); print(name, tools.skill_report(m), "ppl", round(tools.perplexity(m),2))
