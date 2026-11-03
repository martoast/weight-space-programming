from common import *
m = load()
print(tools.skill_report(m)); print("ppl", tools.perplexity(m))
for n,p in m.named_parameters():
    print(f"{n:40s} {tuple(p.shape)} mean={p.mean():.4f} std={p.std():.4f} max={p.abs().max():.3f}")
