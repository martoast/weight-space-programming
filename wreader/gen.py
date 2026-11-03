"""Experiment 5 data: train many small MLPs in parallel (each on its own random teacher task),
then plant random bugs and record what a diagnostician would see.

Evidence per example (same as Claude had in Experiment 2): the weights + statistics on the net's training data.
  unit tokens  (32): W1[:, j] (20), b1[j], W2[j], active rate, corr(h_j, y), mean h_j        -> 25 features
  input tokens (20): W1[i, :] (32), corr(x_i, y), net sensitivity to x_i                      -> 34 features
Label: which of the 52 tokens the bug touched (a set), and the bug type."""
import torch, sys, os

dev = "mps" if torch.backends.mps.is_available() else "cpu"
g = torch.Generator().manual_seed(0)
NET, N, D, H, T = 2000, 768, 20, 32, 8

# ---- train NET clean networks at once -------------------------------------------------------
X = torch.randn(NET, N, D, generator=g)
tw1 = torch.randn(NET, D, T, generator=g) / D ** 0.5; tw2 = torch.randn(NET, T, 1, generator=g)
s = torch.relu(X @ tw1) @ tw2; y = (s > s.median(1, keepdim=True).values).float()
X, y = X.to(dev), y.to(dev)
P = {"W1": torch.randn(NET, D, H, generator=g) * (2 / D) ** 0.5, "b1": torch.zeros(NET, 1, H),
     "W2": torch.randn(NET, H, 1, generator=g) * (1 / H) ** 0.5, "b2": torch.zeros(NET, 1, 1)}
P = {k: v.to(dev).requires_grad_() for k, v in P.items()}  # gradients only to build the clean nets
opt = torch.optim.Adam(P.values(), lr=3e-3)
for step in range(3000):
    out = torch.relu(X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    opt.zero_grad(); loss.backward(); opt.step()
P = {k: v.detach().cpu() for k, v in P.items()}; X, y = X.cpu(), y.cpu(); out = out.cpu()
acc = ((out > 0).float() == y).float().mean(1).squeeze()
print(f"clean nets: train acc mean {acc.mean():.3f}, min {acc.min():.3f}", flush=True)

# ---- bugs ---------------------------------------------------------------------------------------
TRAIN_TYPES = ["sign_flip_out", "dead_unit", "scale_out", "dup_unit", "input_swap"]
NOVEL_TYPES = ["zero_input", "neg_in", "noise_unit", "always_on"]
ALL = TRAIN_TYPES + NOVEL_TYPES

def plant(W1, b1, W2, kind, r):
    """Returns edited copies and the set of touched token indices (units 0..31, inputs 32..51)."""
    W1, b1, W2 = W1.clone(), b1.clone(), W2.clone()
    j, k = r.sample(range(H), 2); a, b = r.sample(range(D), 2)
    if kind == "sign_flip_out": W2[j] *= -1; loc = [j]
    elif kind == "dead_unit": b1[j] = -(W1[:, j].abs().sum() + 1); loc = [j]
    elif kind == "scale_out": W2[j] *= 8; loc = [j]
    elif kind == "dup_unit": W1[:, j] = W1[:, k]; b1[j] = b1[k]; loc = [j]
    elif kind == "input_swap": W1[[a, b]] = W1[[b, a]]; loc = [H + a, H + b]
    elif kind == "zero_input": W1[a] = 0; loc = [H + a]
    elif kind == "neg_in": W1[:, j] *= -1; b1[j] *= -1; loc = [j]
    elif kind == "noise_unit": W1[:, j] = torch.randn(D, device=W1.device) * W1[:, j].std(); loc = [j]
    elif kind == "always_on": b1[j] = W1[:, j].abs().sum() + 1; loc = [j]
    return W1, b1, W2, loc

def features(W1, b1, W2, Xn, yn):
    pre = Xn @ W1 + b1; h = torch.relu(pre); act = (pre > 0).float()
    ys = yn * 2 - 1
    def corr(A, v):
        A = A - A.mean(0); v = v - v.mean()
        return (A * v[:, None]).sum(0) / (A.norm(dim=0) * v.norm() + 1e-8)
    unit = torch.cat([W1.T, b1[:, None], W2[:, None], act.mean(0)[:, None], corr(h, ys)[:, None], h.mean(0)[:, None]], 1)
    sens = W1 @ (W2 * act.mean(0))
    inp = torch.cat([W1, corr(Xn, ys)[:, None], sens[:, None]], 1)
    return unit, inp

import random
r = random.Random(1)
split = {"train": range(0, 1600), "test": range(1600, NET)}
PER = 25
out = {}
for name, nets in split.items():
    U, I, L, K = [], [], [], []
    for n in nets:
        Xn, yn = X[n, :256], y[n, :256, 0]
        types = TRAIN_TYPES if name == "train" else ALL
        for _ in range(PER if name == "train" else 18):
            kind = r.choice(types)
            W1, b1, W2, loc = plant(P["W1"][n], P["b1"][n, 0], P["W2"][n, :, 0], kind, r)
            u, i = features(W1, b1, W2, Xn, yn)
            lab = torch.zeros(H + D); lab[loc] = 1
            U.append(u.cpu()); I.append(i.cpu()); L.append(lab); K.append(ALL.index(kind))
    out[name] = {"unit": torch.stack(U), "inp": torch.stack(I), "loc": torch.stack(L), "kind": torch.tensor(K)}
    print(f"{name}: {len(K)} examples", flush=True)
torch.save({"data": out, "types": ALL, "train_types": TRAIN_TYPES}, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.pt"))
