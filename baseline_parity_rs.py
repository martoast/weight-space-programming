"""Random search baseline for 10-bit parity, 10->16->1: sample random networks, keep the best on train."""
import numpy as np
relu = lambda z: np.maximum(z, 0)
d = 10
X = np.array([[(i >> b) & 1 for b in range(d)] for i in range(2 ** d)], float); y = X.sum(1) % 2 == 1
perm = np.random.default_rng(0).permutation(len(X)); tr, te = perm[:512], perm[512:]
r = np.random.default_rng(0)
def run(Xs, W1, b1, W2, b2): return np.einsum("bnh,bh->bn", relu(np.einsum("nd,bdh->bnh", Xs, W1) + b1[:, None]), W2) + b2[:, None]
for budget in [1_000, 100_000]:
    best, bestP = -1, None
    for _ in range(budget // 1000):
        P = (r.normal(0, 3, (1000, d, 16)), r.normal(0, 3, (1000, 16)), r.normal(0, 3, (1000, 16)), r.normal(0, 3, 1000))
        a = ((run(X[tr], *P) > 0) == y[tr]).mean(1); i = a.argmax()
        if a[i] > best: best, bestP = a[i], tuple(p[i:i + 1] for p in P)
    te_acc = ((run(X[te], *bestP)[0] > 0) == y[te]).mean()
    print(f"{budget:>7,} random networks: best train {best:.1%} -> TEST {te_acc:.1%}")
