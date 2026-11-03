from edits import *
res = {}
for name, s in [("base", 1), ("10.7 x3", 3), ("10.7 x5", 5)]:
    m = load(); scale_heads(m, [(10, 7)], s); g = generate(m, TEST); r = rep_stats(g); res[name] = g
    print(f"TEST {name:8} rep4={r[0]:.3f} looped={r[1]:.0%} ppl={perplexity(m):.2f}", flush=True)
for i in [0, 3, 7]:
    print(f"\n>> {TEST[i]}")
    for k in res: print(f"  [{k}] {tok.decode(res[k][i])[:220]!r}")
