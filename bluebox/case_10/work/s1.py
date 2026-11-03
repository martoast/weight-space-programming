from base import *
m = tools.load_model(MP)
print(tools.skill_report(m)); print("ppl", tools.perplexity(m))
for p in ["The capital of France is","Once upon a time,","The quick brown fox","In 1999, the president of the United States"]:
    print(repr(p), "->", repr(tools.generate(m,p,25)))
sd = m.state_dict()
for k,v in sd.items():
    if v.dim()>=1: print(k, tuple(v.shape), "mean %.4f std %.4f absmax %.3f"%(v.float().mean(), v.float().std(), v.abs().max()))
