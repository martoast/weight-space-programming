from edits import *
import time
CS = [(10, 7), (11, 10)]
print(f"{'edit':30} {'weights':>9} {'rep4':>6} {'looped':>7} {'ppl':>7} {'secs':>5}", flush=True)
for s in [1, 2, 3, 5, 8]:
    t = time.time(); m = load(); n = scale_heads(m, CS, s); r = rep_stats(generate(m, DEV))
    print(f"{'copy-suppression heads x%d' % s:30} {n:9d} {r[0]:6.3f} {r[1]:7.0%} {perplexity(m):7.2f} {time.time()-t:5.1f}", flush=True)
