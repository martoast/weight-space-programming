from edits import *
import sys
D = torch.load("diag.pt")["D"]
cands = {
    "heads 5.1+5.5 off":        lambda m: scale_heads(m, [(5, 1), (5, 5)], 0.0),
    "top-5 induction heads off": lambda m: scale_heads(m, IND5, 0.0),
    "top-5 induction heads x0.5":lambda m: scale_heads(m, IND5, 0.5),
    "copy dir out of MLP 4-8":  lambda m: remove_dir(m, range(4, 9), D),
    "copy dir out of MLP 1-2":  lambda m: remove_dir(m, [1, 2], D),
}
print(f"{'edit':30} {'weights':>9} {'rep4':>6} {'looped':>7} {'ppl':>7}", flush=True)
m = load(); r = rep_stats(generate(m, DEV))
print(f"{'none (base)':30} {0:9d} {r[0]:6.3f} {r[1]:7.0%} {perplexity(m):7.2f}", flush=True)
for name, f in cands.items():
    m = load(); n = f(m); r = rep_stats(generate(m, DEV))
    print(f"{name:30} {n:9d} {r[0]:6.3f} {r[1]:7.0%} {perplexity(m):7.2f}", flush=True)
