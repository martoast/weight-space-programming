from edits import *
print(f"{'edit':34} {'weights':>8} {'rep4':>6} {'looped':>7} {'ppl':>7}", flush=True)
def row(name, n, m, **kw):
    r = rep_stats(generate(m, DEV, **kw)); print(f"{name:34} {n:8d} {r[0]:6.3f} {r[1]:7.0%} {perplexity(m):7.2f}", flush=True)
m = load()
for p in [1.1, 1.2, 1.3]:
    r = rep_stats(generate(m, DEV, repetition_penalty=p))
    print(f"{'BASELINE rep_penalty %.1f' % p:34} {0:8d} {r[0]:6.3f} {r[1]:7.0%} {perplexity_with_rep_penalty(m, p):7.2f}", flush=True)
for heads, nm in [([(10, 7)], "10.7"), ([(11, 10)], "11.10"), ([(10, 7), (11, 10)], "both")]:
    for s in [3, 5]:
        m = load(); row(f"{nm} output x{s}", scale_heads(m, heads, s), m)
    for s in [2, 4]:
        m = load(); row(f"{nm} query sharpen x{s}", sharpen_heads(m, heads, s), m)
    m = load(); n = scale_heads(m, heads, 3) + sharpen_heads(m, heads, 2); row(f"{nm} output x3 + sharpen x2", n, m)
