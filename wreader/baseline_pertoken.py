"""Baselines audit, Experiment 6: a trained per-token classifier on the same features (no cross-token attention).
Each token is scored independently by a 2-layer MLP; trained with the same location loss; CPU."""
import torch, torch.nn as nn, json
torch.manual_seed(0)
import train2 as T  # reuses data loading, normalization, rule baseline
dev = "cpu"
T.dev = dev
u, i, loc, kind, m = [x.cpu() for x in T.prep(T.tr)]
class PerToken(nn.Module):
    def __init__(s, du, di, h=128):
        super().__init__(); s.fu = nn.Sequential(nn.Linear(du, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        s.fi = nn.Sequential(nn.Linear(di, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(s, u, i, m): return torch.cat([s.fu(u), s.fi(i)], 1).squeeze(-1).masked_fill(~m, -1e9)
net = PerToken(u.shape[2], i.shape[2]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.01)
has = loc.sum(1) > 0
for ep in range(15):
    perm = torch.randperm(len(u))
    for b in range(0, len(u), 512):
        ix = perm[b:b + 512]; ix = ix[has[ix]]
        lg = net(u[ix], i[ix], m[ix]); tgt = loc[ix] / loc[ix].sum(1, keepdim=True)
        loss = -(tgt * torch.log_softmax(lg, 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
res = {}
for split in ["test", "test_arch"]:
    uu, ii, ll, kk, mm = [x.cpu() for x in T.prep(T.D[split])]
    with torch.no_grad(): P = net(uu, ii, mm).argmax(1)
    res[split] = {name: ll[kk == t].gather(1, P[kk == t, None]).mean().item() for t, name in enumerate(T.ALL) if name != "none" and (kk == t).any()}
R = json.load(open("results2.json"))
print(f"{'type':15} {'group':12} {'transformer':>12} {'per-token MLP':>14} {'rule':>6}")
for name, v in res["test"].items():
    g = "held family" if name in T.HF else "held type" if name in T.HT else "seen"
    print(f"{name:15} {g:12} {R['test'][name]['reader']:12.1%} {v:14.1%} {R['test'][name]['rule']:6.1%}")
