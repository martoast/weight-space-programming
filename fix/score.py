import json, sys, os
import numpy as np
sys.path.insert(0, "..")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from wsp_test import acc, train_gd
D = np.load("data.npz"); Xtr, ytr = D["Xtr"], D["ytr"]
S = json.load(open("sealed.json"))
Xte, yte = np.array(S["Xte"]), np.array(S["yte"])
clean = {k: np.array(v) for k, v in S["clean"].items()}

# My edits, decided from the diagnosis above (no gradients, no fitting)
def fix0(P): P["W2"][6] = 0; return P            # remove duplicate unit that cancels unit 12
def fix1(P): P["W1"][[5, 6]] = P["W1"][[6, 5]]; return P   # unswap inputs 5,6
def fix2(P): P["b1"][11] = 0.0; return P         # revive dead unit; X ~ N(0,1) so bias 0 -> ~50% active
fixes = [fix0, fix1, fix2]

print(f"{'':10} {'truth':<45} {'broken':>7} {'my edit':>8} {'GD 300 steps':>13} {'clean':>6}")
for i, f in enumerate(fixes):
    B = dict(np.load(f"broken_{i}.npz"))
    E = f({k: v.copy() for k, v in B.items()})
    G = train_gd(B, Xtr, ytr, steps=300, lr=1e-3)
    t = S["truth"][f"broken_{i}"]
    print(f"broken_{i}  {str(t):<45} {acc(B,Xte,yte):7.1%} {acc(E,Xte,yte):8.1%} {acc(G,Xte,yte):13.1%} {acc(clean,Xte,yte):6.1%}")
    print(f"           weights changed by my edit: {sum(int((E[k]!=B[k]).sum()) for k in B)} / {sum(v.size for v in B.values())}")
