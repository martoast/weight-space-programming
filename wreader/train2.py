"""Train the broad weight-reader: points at the bug (or says none) across network shapes."""
import os, json, torch, torch.nn as nn
torch.manual_seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
S = torch.load(f"{here}/data2.pt"); D, ALL, TT = S["data"], S["types"], S["train_types"]
HT, HF = S["held_types"], S["held_family"]
NONE = ALL.index("none"); TT_IDX = torch.tensor([ALL.index(t) for t in TT])

def valid_mask(d):
    nu, ni = d["nu"][:, 0], d["nu"][:, 1]
    m = torch.zeros(len(nu), 80, dtype=torch.bool)
    m[:, :60] = torch.arange(60)[None] < nu[:, None]; m[:, 60:] = torch.arange(20)[None] < ni[:, None]
    return m

tr = D["train"]; vm = valid_mask(tr)
su_m, su_s = tr["unit"][vm[:, :60]].mean(0), tr["unit"][vm[:, :60]].std(0) + 1e-6
si_m, si_s = tr["inp"][vm[:, 60:]].mean(0), tr["inp"][vm[:, 60:]].std(0) + 1e-6
def prep(d):
    m = valid_mask(d)
    u = ((d["unit"] - su_m) / su_s).clamp(-20, 20) * m[:, :60, None]
    i = ((d["inp"] - si_m) / si_s).clamp(-20, 20) * m[:, 60:, None]
    return u.to(dev), i.to(dev), d["loc"].to(dev), d["kind"].to(dev), m.to(dev)

class Reader(nn.Module):
    def __init__(s, du, di, d=128):
        super().__init__()
        s.pu, s.pi = nn.Linear(du, d), nn.Linear(di, d); s.tu, s.ti = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        s.cls = nn.Parameter(torch.zeros(1, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True), 4)
        s.loc, s.kind = nn.Linear(d, 1), nn.Linear(d, len(TT))
    def forward(s, u, i, m):
        z = torch.cat([s.cls.expand(len(u), -1, -1), s.pu(u) + s.tu, s.pi(i) + s.ti], 1)
        pad = torch.cat([torch.zeros(len(u), 1, dtype=torch.bool, device=u.device), ~m], 1)
        z = s.enc(z, src_key_padding_mask=pad)
        lg = s.loc(z[:, 1:]).squeeze(-1).masked_fill(~m, -1e9)
        return lg, s.kind(z[:, 0])

def rule(d):
    """Non-learned baseline: token with the largest |z| among its within-layer z-scored features."""
    m = valid_mask(d); nz_u, nz_i = d["unit"].shape[2] // 2, d["inp"].shape[2] // 2
    s = torch.cat([d["unit"][..., nz_u:].abs().amax(-1), d["inp"][..., nz_i:].abs().amax(-1)], 1)
    return s.masked_fill(~m, -1)

def evaluate(model, d):
    u, i, loc, kind, m = prep(d); P, K = [], []
    with torch.no_grad():
        for b in range(0, len(u), 512):
            lg, kl = model(u[b:b + 512], i[b:b + 512], m[b:b + 512]); P.append(lg.argmax(1)); K.append(TT_IDX.to(dev)[kl.argmax(1)])
    P, K = torch.cat(P).cpu(), torch.cat(K).cpu(); loc, kind = loc.cpu(), kind.cpu(); R = rule(d).argmax(1)
    res = {}
    for t, name in enumerate(ALL):
        sel = kind == t
        if not sel.any(): continue
        if name == "none":
            res[name] = {"n": int(sel.sum()), "false_alarm": (K[sel] != NONE).float().mean().item()}
        else:
            res[name] = {"n": int(sel.sum()), "reader": loc[sel].gather(1, P[sel, None]).mean().item(),
                         "rule": loc[sel].gather(1, R[sel, None]).mean().item(), "flagged": (K[sel] != NONE).float().mean().item()}
    return res

if __name__ == "__main__":
    u, i, loc, kind, m = prep(tr)
    model = Reader(u.shape[2], i.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    kmap = torch.full((len(ALL),), -1, dtype=torch.long); kmap[TT_IDX] = torch.arange(len(TT)); kmap = kmap.to(dev)
    E = 30; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=E * ((len(u) + 255) // 256))
    for ep in range(E):
        perm = torch.randperm(len(u), device=dev); tot = 0
        for b in range(0, len(u), 256):
            ix = perm[b:b + 256]; lg, kl = model(u[ix], i[ix], m[ix])
            has = loc[ix].sum(1) > 0
            tgt = loc[ix][has] / loc[ix][has].sum(1, keepdim=True)
            loss = -(tgt * torch.log_softmax(lg[has], 1)).sum(1).mean() + nn.functional.cross_entropy(kl, kmap[kind[ix]])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(ix)
        if ep % 5 == 4: print(f"epoch {ep + 1}: loss {tot / len(u):.3f}", flush=True)
    torch.save(model.state_dict(), f"{here}/reader2.pt")
    out = {"test": evaluate(model, D["test"]), "test_arch": evaluate(model, D["test_arch"])}
    json.dump(out, open(f"{here}/results2.json", "w"), indent=1)
    for split, res in out.items():
        print(f"\n== {split} ==  {'type':15} {'group':13} {'n':>5} {'reader':>7} {'rule':>6} {'flagged':>8}")
        for name, v in res.items():
            grp = "held family" if name in HF else "held type" if name in HT else "seen"
            if name == "none": print(f"   {'none':15} {'seen':13} {v['n']:5d}   false alarm rate {v['false_alarm']:.1%}")
            else: print(f"   {name:15} {grp:13} {v['n']:5d} {v['reader']:7.1%} {v['rule']:6.1%} {v['flagged']:8.1%}")
