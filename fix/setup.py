"""Train a net with GD, then plant one random bug in each of 3 copies.
Ground truth + clean weights go to sealed.json (not to be read until scoring)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp_test import init, train_gd, forward, acc

out = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# Nonlinear task: hidden teacher net, clean labels, 20 inputs.
d, H, n = 20, 32, 6000
T = init(d, 8, seed=7); T["W2"] *= 3
X = rng.normal(size=(n, d))
s = forward(T, X)[:, 0]; y = s > np.median(s)
tr, te = np.arange(4000), np.arange(4000, n)

P = train_gd(init(d, H, seed=3), X[tr], y[tr], steps=12000, seed=0)
print(f"clean model: train {acc(P, X[tr], y[tr]):.1%}  test {acc(P, X[te], y[te]):.1%}")
np.savez(f"{out}/data.npz", Xtr=X[tr], ytr=y[tr])  # test set stays sealed

# Unit importance = accuracy drop when the unit is zeroed; bugs target units that matter.
base = acc(P, X[tr], y[tr])
imp = []
for j in range(H):
    Q = {k: v.copy() for k, v in P.items()}; Q["W2"][j] = 0
    imp.append(base - acc(Q, X[tr], y[tr]))
important = list(np.argsort(imp)[::-1][:12])

bug_rng = np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
types = list(bug_rng.choice(["sign_flip_out", "input_swap", "dead_unit", "scale_out", "dup_unit"], 3, replace=False))
truth = {}
for i, t in enumerate(types):
    Q = {k: v.copy() for k, v in P.items()}
    j, k = (int(u) for u in bug_rng.choice(important, 2, replace=False))
    a, b = (int(u) for u in bug_rng.choice(d, 2, replace=False))
    if t == "sign_flip_out": Q["W2"][j] *= -1; info = {"unit": j}
    elif t == "input_swap": Q["W1"][[a, b]] = Q["W1"][[b, a]]; info = {"inputs": [a, b]}
    elif t == "dead_unit":
        pre = X[tr] @ P["W1"][:, j] + P["b1"][j]
        Q["b1"][j] = -pre.max() - 0.1; info = {"unit": j}
    elif t == "scale_out": Q["W2"][j] *= 8; info = {"unit": j}
    elif t == "dup_unit": Q["W1"][:, j] = Q["W1"][:, k]; Q["b1"][j] = Q["b1"][k]; info = {"unit": j, "copied_from": k}
    truth[f"broken_{i}"] = {"type": t, **info}
    np.savez(f"{out}/broken_{i}.npz", **Q)
    print(f"broken_{i}: train {acc(Q, X[tr], y[tr]):.1%}")

json.dump({"truth": truth, "clean": {k: v.tolist() for k, v in P.items()},
           "Xte": X[te].tolist(), "yte": y[te].tolist()}, open(f"{out}/sealed.json", "w"))
