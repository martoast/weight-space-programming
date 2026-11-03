"""Weight-space programming vs gradient descent, numpy only.

Same architecture for both: MLP in -> H (ReLU) -> 1 (logit).
"Hand" = weights written by Claude by reasoning, no optimizer, no data fitting.
"GD"   = Adam on BCE from random init.
"""
import numpy as np

rng = np.random.default_rng(0)
relu = lambda z: np.maximum(z, 0)

def forward(P, X):
    return relu(X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]

def acc(P, X, y):
    return float(((forward(P, X)[:, 0] > 0) == y).mean())

def init(d, H, seed):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, np.sqrt(2 / d), (d, H)), "b1": np.zeros(H),
            "W2": r.normal(0, np.sqrt(1 / H), (H, 1)), "b2": np.zeros(1)}

def train_gd(P, X, y, steps=5000, lr=3e-3, bs=128, seed=0):
    r = np.random.default_rng(seed)
    P = {k: v.copy() for k, v in P.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(x) for k, x in P.items()}
    for t in range(1, steps + 1):
        idx = r.integers(0, len(X), bs)
        xb, yb = X[idx], y[idx][:, None]
        z1 = xb @ P["W1"] + P["b1"]; h = relu(z1)
        out = h @ P["W2"] + P["b2"]
        g_out = (1 / (1 + np.exp(-out)) - yb) / bs
        G = {"W2": h.T @ g_out, "b2": g_out.sum(0)}
        g_h = (g_out @ P["W2"].T) * (z1 > 0)
        G["W1"] = xb.T @ g_h; G["b1"] = g_h.sum(0)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
            P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return P

def report(name, rows):
    print(f"\n=== {name} ===")
    for label, tr, te in rows:
        print(f"  {label:<38} train {tr:6.1%}   test {te:6.1%}")

# ---------------------------------------------------------------------------
# TEST 1: 10-bit parity. Known algorithm. Train on half the 1024 strings, test on the rest.
# ---------------------------------------------------------------------------
d, H = 10, 16
X = np.array([[(i >> b) & 1 for b in range(d)] for i in range(2 ** d)], float)
y = X.sum(1) % 2 == 1
perm = rng.permutation(len(X)); tr, te = perm[:512], perm[512:]

P_rand = init(d, H, seed=1)

# Hand-written: hidden unit j = ReLU(s - j), s = number of 1s (all input weights = 1).
# A zigzag z(s) = h0 - 2h1 + 2h2 - 2h3 ... hits 0,1,0,1,... at s = 0,1,2,3,...
# Logit = 10 * (z - 0.5). Units 10..15 unused (zeroed). 21 nonzero weights out of 193.
P_hand = {"W1": np.zeros((d, H)), "b1": np.zeros(H), "W2": np.zeros((H, 1)), "b2": np.array([-5.0])}
for j in range(d):
    P_hand["W1"][:, j] = 1.0
    P_hand["b1"][j] = -j
    P_hand["W2"][j, 0] = 10.0 * (1 if j == 0 else 2 * (-1) ** j)

# "Edit the random network": keep the random net, overwrite only what's needed.
# There is nothing to reuse in random weights, so the edit = the hand program in units 0..9,
# with units 10..15's output weights zeroed. Same result as P_hand, which is the point.
P_edit = {k: v.copy() for k, v in P_rand.items()}
P_edit["W1"][:, :10] = P_hand["W1"][:, :10]; P_edit["b1"][:10] = P_hand["b1"][:10]
P_edit["W2"][:] = P_hand["W2"]; P_edit["b2"][:] = P_hand["b2"]

rows = [("random init", acc(P_rand, X[tr], y[tr]), acc(P_rand, X[te], y[te])),
        ("hand-written (no data, no gradients)", acc(P_hand, X[tr], y[tr]), acc(P_hand, X[te], y[te])),
        ("random net, hand-edited", acc(P_edit, X[tr], y[tr]), acc(P_edit, X[te], y[te]))]
for s in range(3):
    P_gd = train_gd(init(d, H, seed=10 + s), X[tr], y[tr], steps=8000, seed=s)
    rows.append((f"GD (Adam, 8k steps) seed {s}", acc(P_gd, X[tr], y[tr]), acc(P_gd, X[te], y[te])))
report("TEST 1: 10-bit parity, MLP 10->16->1", rows)

# ---------------------------------------------------------------------------
# TEST 2: a "messy" task with no known algorithm.
# Labels come from a hidden random teacher net + 10% label noise. Hand-writer sees only data,
# like a real-world dataset. No closed-form rule exists to "program".
# ---------------------------------------------------------------------------
d, H, n = 20, 32, 4000
T = init(d, 12, seed=99); T["W2"] *= 3
Xm = rng.normal(size=(n, d))
ym = forward(T, Xm)[:, 0] > np.median(forward(T, Xm)[:, 0])
flip = rng.random(n) < 0.10; ym_noisy = ym ^ flip
tr, te = np.arange(3000), np.arange(3000, n)

# Best honest "reasoning" attempt without fitting: look at which inputs correlate with the label
# (a closed-form statistic, the kind of thing the report flags as drifting toward fitting),
# and write a single linear detector. ReLU(w.x) - ReLU(-w.x) = w.x passes the linear score through.
c = (Xm[tr] * (ym_noisy[tr] * 2 - 1)[:, None]).mean(0)
P_hand2 = {"W1": np.zeros((d, H)), "b1": np.zeros(H), "W2": np.zeros((H, 1)), "b2": np.zeros(1)}
P_hand2["W1"][:, 0], P_hand2["W1"][:, 1] = c, -c
P_hand2["W2"][0, 0], P_hand2["W2"][1, 0] = 1, -1
P_hand2["b2"][0] = -np.median(Xm[tr] @ c)

rows = [("random init", acc(P_rand2 := init(d, H, 1), Xm[tr], ym_noisy[tr]), acc(P_rand2, Xm[te], ym[te])),
        ("hand-written (correlation readout)", acc(P_hand2, Xm[tr], ym_noisy[tr]), acc(P_hand2, Xm[te], ym[te]))]
for s in range(3):
    P_gd = train_gd(init(d, H, seed=20 + s), Xm[tr], ym_noisy[tr], steps=6000, seed=s)
    rows.append((f"GD (Adam, 6k steps) seed {s}", acc(P_gd, Xm[tr], ym_noisy[tr]), acc(P_gd, Xm[te], ym[te])))
report("TEST 2: noisy teacher task, MLP 20->32->1 (test = clean labels)", rows)
