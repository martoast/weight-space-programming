"""Baselines audit, Experiment 1: tuned gradient descent and random search on 10-bit parity.
Same split as Experiment 1 (train 512 / test 512). Hyperparameters are chosen on a validation split carved
from the training half (384 fit / 128 val); test is touched once per selected config."""
import itertools, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_test import init, relu, forward, acc

d = 10
X = np.array([[(i >> b) & 1 for b in range(d)] for i in range(2 ** d)], float)
y = X.sum(1) % 2 == 1
perm = np.random.default_rng(0).permutation(len(X)); tr, te = perm[:512], perm[512:]
fit, val = tr[:384], tr[384:]

def train(P, X, y, steps, lr, wd, bs=128, seed=0):
    r = np.random.default_rng(seed)
    P = {k: v.copy() for k, v in P.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    for t in range(1, steps + 1):
        idx = r.integers(0, len(X), bs); xb, yb = X[idx], y[idx][:, None]
        z1 = xb @ P["W1"] + P["b1"]; h = relu(z1); out = h @ P["W2"] + P["b2"]
        g = (1 / (1 + np.exp(-out)) - yb) / bs
        G = {"W2": h.T @ g, "b2": g.sum(0)}; gh = (g @ P["W2"].T) * (z1 > 0); G["W1"] = xb.T @ gh; G["b1"] = gh.sum(0)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** t) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8) + wd * P[k])  # AdamW
    return P

print("=== Tuned gradient descent (AdamW), select on val, 5 seeds per config ===")
for H in [16, 64, 256]:
    best = None
    for lr, wd, steps in itertools.product([1e-3, 3e-3, 1e-2, 3e-2], [0, 1e-4, 1e-3, 1e-2], [5000, 20000]):
        vals = [acc(train(init(d, H, 100 + s), X[fit], y[fit], steps, lr, wd, seed=s), X[val], y[val]) for s in range(3)]
        if best is None or np.mean(vals) > best[0]: best = (np.mean(vals), lr, wd, steps)
    _, lr, wd, steps = best
    tests = [acc(train(init(d, H, 200 + s), X[tr], y[tr], steps, lr, wd, seed=s), X[te], y[te]) for s in range(5)]
    print(f"  H={H:3d}  best val {best[0]:.1%} at lr={lr} wd={wd} steps={steps}  ->  TEST mean {np.mean(tests):.1%}  range {min(tests):.1%}-{max(tests):.1%}", flush=True)

print("\n=== Random search, same 10->16->1 architecture: sample weights, keep best on train ===")
r = np.random.default_rng(0)
for budget in [1_000, 100_000]:
    best, bestP = -1, None
    for _ in range(budget // 1000):
        W1 = r.normal(0, 3, (1000, d, 16)); b1 = r.normal(0, 3, (1000, 16)); W2 = r.normal(0, 3, (1000, 16)); b2 = r.normal(0, 3, 1000)
        out = np.einsum("bnh,bh->bn", relu(np.einsum("nd,bdh->bnh", X[tr], W1) + b1[:, None]), W2) + b2[:, None]
        a = ((out > 0) == y[tr]).mean(1); i = a.argmax()
        if a[i] > best: best, bestP = a[i], {"W1": W1[i], "b1": b1[i], "W2": W2[i][:, None], "b2": b2[i:i + 1]}
    print(f"  {budget:>7,} random networks: best train {best:.1%}  ->  TEST {acc(bestP, X[te], y[te]):.1%}", flush=True)
