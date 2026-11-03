from base import *
print(tools.skill_report(m)); print("ppl",tools.perplexity(m))
for p in ["The capital of France is","Once upon a time,","The quick brown fox","In 1999, the president of the United States"]:
    print(repr(p), repr(tools.generate(m,p,25)))
sd=m.state_dict()
for k,v in sd.items():
    if v.dim()>=1: print(k, tuple(v.shape), f"{v.mean().item():.4f} {v.std().item():.4f} {v.abs().max().item():.3f}")
