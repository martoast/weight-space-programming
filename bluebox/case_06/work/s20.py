from base import *
import copy
sd=m.state_dict()
new={}
for k,v in sd.items():
    if k.startswith("transformer.h.7."): new[k]=sd[k.replace("h.7.","h.8.")].clone()
    elif k.startswith("transformer.h.8."): new[k]=sd[k.replace("h.8.","h.7.")].clone()
m.load_state_dict(new,strict=False)
print(tools.skill_report(m)); print("ppl",tools.perplexity(m))
for p in ["The Space Needle is in","The opposite of wet is","When Tom and James went to the store, James gave a book to","Barack Obama was born in"]:
    print(repr(p),repr(tools.generate(m,p,15)))
