"""Train a weight-reader: a small transformer over unit and input tokens that points at the bug.
Test on held-out networks, on the 5 bug types it trained on and on 4 types it never saw."""
import os, torch, torch.nn as nn
torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
S = torch.load(f"{here}/data.pt"); D, TYPES, TT = S["data"], S["types"], S["train_types"]

def norm_stats(x): return x.mean((0, 1), keepdim=True), x.std((0, 1), keepdim=True) + 1e-6
su, si = norm_stats(D["train"]["unit"]), norm_stats(D["train"]["inp"])
def prep(d):
    return ((d["unit"] - su[0]) / su[1]).to(dev), ((d["inp"] - si[0]) / si[1]).to(dev), d["loc"].to(dev), d["kind"].to(dev)

class Reader(nn.Module):
    """Unit tokens and input tokens get their own projection + a type embedding; no positions over units,
    so the reader treats hidden units as an unordered set (as they are). Inputs get a position embedding."""
    def __init__(s, d=96):
        super().__init__()
        s.pu, s.pi = nn.Linear(25, d), nn.Linear(34, d)
        s.pos_in = nn.Parameter(torch.zeros(20, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True), 4)
        s.loc, s.kind = nn.Linear(d, 1), nn.Linear(d, len(TT))
    def forward(s, u, i):
        z = torch.cat([s.cls.expand(len(u), -1, -1), s.pu(u), s.pi(i) + s.pos_in], 1)
        z = s.enc(z)
        return s.loc(z[:, 1:]).squeeze(-1), s.kind(z[:, 0])

def heuristic(u, i):
    """Non-learned baseline: the token whose features deviate most from the other tokens of its kind."""
    zu = ((u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + 1e-6)).abs().amax(-1)
    zi = ((i - i.mean(1, keepdim=True)) / (i.std(1, keepdim=True) + 1e-6)).abs().amax(-1)
    return torch.cat([zu, zi], 1)

def evaluate(model, d):
    u, i, loc, kind = prep(d); hit_m, hit_h, kacc = [], [], []
    with torch.no_grad():
        for b in range(0, len(u), 1024):
            lg, kl = model(u[b:b + 1024], i[b:b + 1024])
            hit_m.append(loc[b:b + 1024].gather(1, lg.argmax(1, keepdim=True)).squeeze(1))
            hit_h.append(loc[b:b + 1024].gather(1, heuristic(u[b:b + 1024], i[b:b + 1024]).argmax(1, keepdim=True)).squeeze(1))
            kacc.append((kl.argmax(1) == kind[b:b + 1024]).float())
    hit_m, hit_h, kacc = torch.cat(hit_m), torch.cat(hit_h), torch.cat(kacc)
    rows = []
    for t, name in enumerate(TYPES):
        m = kind == t
        if m.any():
            rows.append((name, name in TT, int(m.sum()), hit_m[m].mean().item(), hit_h[m].mean().item(),
                         kacc[m].mean().item() if name in TT else None))
    return rows

if __name__ == "__main__":
    model = Reader().to(dev)
    u, i, loc, kind = prep(D["train"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n = len(u)
    for ep in range(30):
        perm = torch.randperm(n, device=dev); tot = 0
        for b in range(0, n, 256):
            ix = perm[b:b + 256]; lg, kl = model(u[ix], i[ix])
            target = loc[ix] / loc[ix].sum(1, keepdim=True)
            loss = -(target * torch.log_softmax(lg, 1)).sum(1).mean() + nn.functional.cross_entropy(kl, kind[ix])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ix)
        if ep % 5 == 4: print(f"epoch {ep + 1}: train loss {tot / n:.3f}", flush=True)
    torch.save(model.state_dict(), f"{here}/reader.pt")
    print(f"\n{'bug type':15} {'seen?':6} {'n':>5} {'reader top-1 loc':>17} {'heuristic':>10} {'type acc':>9}   (chance loc ~ 1/52 = 1.9%)")
    for name, seen, cnt, hm, hh, ka in evaluate(model, D["test"]):
        print(f"{name:15} {'yes' if seen else 'NO':6} {cnt:5d} {hm:17.1%} {hh:10.1%} {'' if ka is None else f'{ka:.1%}':>9}")
