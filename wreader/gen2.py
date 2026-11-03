"""Experiment 6 data: a broader bug generator.

Networks: 7 MLP shapes (1-2 hidden layers), each net trained on its own random teacher task.
Bugs: 17 types in 5 families + "none". Labels are exact by construction.
Evidence per token is an architecture-agnostic instrument panel (weights stats, neighbour similarity,
activation stats, ablation effect), raw and z-scored within the token's layer."""
import os, random, torch
torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {"A": (20, [32]), "B": (20, [16]), "C": (20, [48]), "D": (12, [24]), "E": (20, [32, 16]), "F": (12, [24, 24]),
           "G_unseen": (20, [40, 20])}
FAMILIES = {
    "output":   ["sign_flip_out", "scale_out", "shrink_out"],
    "bias":     ["dead_unit", "bias_shift", "always_on"],
    "incoming": ["neg_in", "noise_unit", "zero_in", "sparsify_unit"],
    "wiring":   ["input_swap", "dup_unit", "zero_input", "input_scale", "unit_swap_out"],
    "precision": ["quantize_unit", "bitflip"],
}
HELD_TYPES = ["shrink_out", "always_on", "sparsify_unit", "unit_swap_out"]   # novel types inside seen families
HELD_FAMILY = FAMILIES["precision"]                                          # a whole family never seen
ALL = [t for f in FAMILIES.values() for t in f] + ["none"]
TRAIN_TYPES = [t for t in ALL if t not in HELD_TYPES + HELD_FAMILY]
MAXTOK = 60 + 20

# ---------------------------------------------------------------- train clean nets, batched per shape
def train_batch(nb, D, widths, g):
    N = 768
    X = torch.randn(nb, N, D, generator=g)
    t1 = torch.randn(nb, D, 8, generator=g) / D ** 0.5; t2 = torch.randn(nb, 8, 1, generator=g)
    s = torch.relu(X @ t1) @ t2; y = (s > s.median(1, keepdim=True).values).float()
    dims = [D] + widths + [1]
    Ws = [(torch.randn(nb, a, b, generator=g) * (2 / a) ** 0.5).to(dev).requires_grad_() for a, b in zip(dims, dims[1:])]
    bs = [torch.zeros(nb, 1, b).to(dev).requires_grad_() for b in dims[1:]]
    Xd, yd = X.to(dev), y.to(dev)
    opt = torch.optim.Adam(Ws + bs, lr=3e-3)
    for _ in range(3000):  # gradients only to build the clean nets
        h = Xd
        for k, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            if k < len(Ws) - 1: h = torch.relu(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(h, yd)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((h > 0).float() == yd).float().mean().item()
    return X, y[..., 0], [W.detach().cpu() for W in Ws], [b.detach().cpu()[:, 0] for b in bs], acc

# ---------------------------------------------------------------- single-net helpers (CPU)
def forward(Ws, bs, X, masks=None):
    """masks[l]: (K, 1, width_l) multiplies hidden layer l (for batched ablations)."""
    h = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if k < len(Ws) - 1:
            h = torch.relu(h)
            if masks is not None and masks[k] is not None: h = h * masks[k]
    return h[..., 0]

def preacts(Ws, bs, X, l):
    h = X
    for k in range(l + 1):
        h = h @ Ws[k] + bs[k]
        if k < l: h = torch.relu(h)
    return h

def plant(Ws, bs, kind, r, X):
    Ws, bs = [W.clone() for W in Ws], [b.clone() for b in bs]
    widths = [W.shape[1] for W in Ws[:-1]]; D = Ws[0].shape[0]
    off = [sum(widths[:l]) for l in range(len(widths))]; ntok = sum(widths)
    l = r.randrange(len(widths)); j, k = r.sample(range(widths[l]), 2); a, b = r.sample(range(D), 2)
    U = lambda jj: [off[l] + jj]
    win, wout = Ws[l][:, j], Ws[l + 1][j]
    if kind == "none": return Ws, bs, []
    if kind == "sign_flip_out": Ws[l + 1][j] *= -1; return Ws, bs, U(j)
    if kind == "scale_out": Ws[l + 1][j] *= r.uniform(3, 20); return Ws, bs, U(j)
    if kind == "shrink_out": Ws[l + 1][j] *= 0.03; return Ws, bs, U(j)
    pre = preacts(Ws, bs, X, l)[:, j]
    if kind == "dead_unit": bs[l][j] -= pre.max() + 0.1; return Ws, bs, U(j)
    if kind == "always_on": bs[l][j] -= pre.min() - 0.1; return Ws, bs, U(j)
    if kind == "bias_shift": bs[l][j] += r.choice([-1, 1]) * r.uniform(1.5, 3) * pre.std(); return Ws, bs, U(j)
    if kind == "neg_in": Ws[l][:, j] *= -1; bs[l][j] *= -1; return Ws, bs, U(j)
    if kind == "noise_unit": Ws[l][:, j] = torch.randn(len(win)) * win.std(); return Ws, bs, U(j)
    if kind == "zero_in": Ws[l][:, j] = 0; return Ws, bs, U(j)
    if kind == "sparsify_unit": Ws[l][:, j] *= (torch.rand(len(win)) > 0.7).float(); return Ws, bs, U(j)
    if kind == "dup_unit": Ws[l][:, j] = Ws[l][:, k]; bs[l][j] = bs[l][k]; return Ws, bs, U(j)
    if kind == "unit_swap_out": Ws[l + 1][[j, k]] = Ws[l + 1][[k, j]]; return Ws, bs, U(j) + U(k)
    if kind == "input_swap": Ws[0][[a, b]] = Ws[0][[b, a]]; return Ws, bs, [ntok + a, ntok + b]
    if kind == "zero_input": Ws[0][a] = 0; return Ws, bs, [ntok + a]
    if kind == "input_scale": Ws[0][a] *= r.uniform(4, 10); return Ws, bs, [ntok + a]
    if kind == "quantize_unit":
        st = 0.75 * win.std(); Ws[l][:, j] = torch.round(win / st) * st; return Ws, bs, U(j)
    if kind == "bitflip": Ws[l][r.randrange(len(win)), j] = r.choice([-1, 1]) * r.uniform(20, 60); return Ws, bs, U(j)
    raise ValueError(kind)

def stats(V):
    """V: (n_vectors, dim) -> per-vector weight statistics."""
    n = V.norm(dim=1); sd = V.std(1, correction=0) + 1e-8
    Vn = V / (n[:, None] + 1e-8); C = Vn @ Vn.T; C.fill_diagonal_(-1)
    uniq = torch.tensor([len(torch.unique(torch.round(v * 1000))) / len(v) for v in V])
    return torch.stack([n / len(V[0]) ** 0.5, V.mean(1), sd, V.abs().amax(1) / sd, (V.abs() < 1e-6).float().mean(1),
                        uniq, C.amax(1)], 1)

def zwithin(F):
    return (F - F.mean(0)) / (F.std(0) + 1e-6)

def features(Ws, bs, X, y):
    ys = y * 2 - 1
    def corr(A, v):
        A = A - A.mean(0); v = v - v.mean()
        return (A * v[:, None]).sum(0) / (A.norm(dim=0) * v.norm() + 1e-8)
    base = forward(Ws, bs, X); acc0 = ((base > 0).float() == y).float().mean()
    U = []; L = len(Ws) - 1
    for l in range(L):
        w = Ws[l].shape[1]; pre = preacts(Ws, bs, X, l); h = torch.relu(pre)
        masks = [None] * L; masks[l] = (1 - torch.eye(w))[:, None, :]
        abl = forward(Ws, bs, X, masks); acc_ab = ((abl > 0).float() == y).float().mean(1)
        fin = stats(Ws[l].T); fout = stats(Ws[l + 1])
        f = torch.cat([fin, bs[l][:, None], (bs[l] / (Ws[l].norm(dim=0) + 1e-8))[:, None], fout[:, :3],
                       Ws[l + 1].mean(1, keepdim=True).sign(),
                       (pre > 0).float().mean(0)[:, None], h.mean(0)[:, None], h.std(0)[:, None], corr(h, ys)[:, None],
                       (acc0 - acc_ab)[:, None], torch.full((w, 1), float(l)), torch.full((w, 1), float(l == L - 1))], 1)
        U.append(torch.cat([f, zwithin(f)], 1))
    U = torch.cat(U)
    Xg = X.clone().requires_grad_(); forward(Ws, bs, Xg).sum().backward(); g = Xg.grad.mean(0)  # diagnostic only
    D = X.shape[1]; Xa = X[None].repeat(D, 1, 1); Xa[torch.arange(D), :, torch.arange(D)] = 0
    acc_in = ((forward(Ws, bs, Xa) > 0).float() == y).float().mean(1)
    fi = torch.cat([stats(Ws[0]), corr(X, ys)[:, None], g[:, None], g.abs()[:, None], (acc0 - acc_in)[:, None]], 1)
    I = torch.cat([fi, zwithin(fi)], 1)
    return U.detach(), I.detach()

if __name__ == "__main__":
    import sys
    EXTRA = len(sys.argv) > 1  # extra training data: python gen2.py <seed> <nets per shape>
    seed = int(sys.argv[1]) if EXTRA else 0
    g = torch.Generator().manual_seed(seed); r = random.Random(seed + 1)
    NPER, BUGS = ({"train": int(sys.argv[2]), "test": 0} if EXTRA else {"train": 400, "test": 100}), 20
    if EXTRA: CONFIGS.pop("G_unseen")
    out = {k: {"unit": [], "inp": [], "loc": [], "kind": [], "cfg": [], "nu": []} for k in ["train", "test", "test_arch"]}
    for cname, (D, widths) in CONFIGS.items():
        nb = 150 if cname == "G_unseen" else NPER["train"] + NPER["test"]
        X, y, Ws, bs, acc = train_batch(nb, D, widths, g)
        print(f"config {cname} {D}->{widths}->1: {nb} nets, clean train acc {acc:.3f}", flush=True)
        for n in range(nb):
            split = "test_arch" if cname == "G_unseen" else ("train" if n < NPER["train"] else "test")
            types = TRAIN_TYPES if split == "train" else ALL
            Wn, bn = [W[n] for W in Ws], [b[n] for b in bs]
            Xp, yp = X[n, :256], y[n, :256]
            for _ in range(BUGS):
                kind = r.choice(types)
                W2, b2, loc = plant(Wn, bn, kind, r, Xp)
                u, i = features(W2, b2, Xp, yp)
                nu = len(u); lab = torch.zeros(MAXTOK); lab[loc] = 1
                pu = torch.zeros(60, u.shape[1]); pu[:nu] = u
                o = out[split]
                o["unit"].append(pu); o["inp"].append(torch.cat([i, torch.zeros(20 - len(i), i.shape[1])]))
                # token order in the reader: 60 unit slots then 20 input slots; remap input labels
                lab2 = torch.zeros(MAXTOK); lab2[[t if t < nu else 60 + (t - nu) for t in loc]] = 1
                o["loc"].append(lab2); o["kind"].append(ALL.index(kind)); o["cfg"].append(cname); o["nu"].append((nu, len(i)))
    for k, o in out.items():
        if not o["kind"]: continue
        out[k] = {"unit": torch.stack(o["unit"]), "inp": torch.stack(o["inp"]), "loc": torch.stack(o["loc"]),
                  "kind": torch.tensor(o["kind"]), "nu": torch.tensor(o["nu"]), "cfg": o["cfg"]}
        print(f"{k}: {len(o['kind'])} examples", flush=True)
    if EXTRA:
        tr_ = out["train"]; tr_["unit"], tr_["inp"] = tr_["unit"].half(), tr_["inp"].half()
        torch.save({"train": tr_}, f"{here}/data2_extra_{seed}.pt"); raise SystemExit
    torch.save({"data": out, "types": ALL, "train_types": TRAIN_TYPES, "held_types": HELD_TYPES, "held_family": HELD_FAMILY,
                "families": FAMILIES}, f"{here}/data2.pt")
