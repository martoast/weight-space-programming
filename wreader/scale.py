"""Scaling study: does weight reading improve with more synthetic data?
Pool = original 48k training examples + extra examples from new networks (same generator, new seeds).
For each N, train (a) the transformer reader and (b) the per-token MLP with the same step budget; evaluate on the
same held-out test set. Signals: held-out-type and held-out-family localization, and the transformer's lead."""
import json, sys, torch, torch.nn as nn
import train2 as T
torch.manual_seed(0)
dev = T.dev
STEPS, BS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000, 256

ex = torch.load(f"{T.here}/data2_extra_101.pt")["train"]
pool = {k: torch.cat([T.tr[k], ex[k].float() if k in ("unit", "inp") else ex[k]]) for k in ("unit", "inp", "loc", "kind", "nu")}
print("pool size", len(pool["kind"]), flush=True)

class PerToken(nn.Module):
    def __init__(s, du, di, h=128):
        super().__init__(); s.fu = nn.Sequential(nn.Linear(du, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        s.fi = nn.Sequential(nn.Linear(di, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        s.kind = nn.Linear(1, len(T.TT))  # unused head, keeps the evaluate() interface
    def forward(s, u, i, m):
        lg = torch.cat([s.fu(u), s.fi(i)], 1).squeeze(-1).masked_fill(~m, -1e9)
        return lg, s.kind(torch.zeros(len(u), 1, device=u.device))

def fit(model, d, steps):
    T.dev = "cpu"; u, i, loc, kind, m = T.prep(d); T.dev = dev  # dataset stays in RAM; batches go to the GPU
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4 if isinstance(model, T.Reader) else 1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, opt.param_groups[0]["lr"], total_steps=steps)
    kmap = torch.full((len(T.ALL),), -1, dtype=torch.long); kmap[T.TT_IDX] = torch.arange(len(T.TT)); kmap = kmap.to(dev)
    for s in range(steps):
        ix = torch.randint(0, len(u), (BS,))
        ub, ib, lb, kb, mb = (x[ix].to(dev) for x in (u, i, loc, kind, m))
        lg, kl = model(ub, ib, mb)
        has = lb.sum(1) > 0; tgt = lb[has] / lb[has].sum(1, keepdim=True)
        loss = -(tgt * torch.log_softmax(lg[has], 1)).sum(1).mean()
        if isinstance(model, T.Reader): loss = loss + nn.functional.cross_entropy(kl, kmap[kb])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

def summarize(res):
    g = {"seen": [], "held_type": [], "held_family": []}
    for name, v in res.items():
        if name == "none": continue
        g["held_family" if name in T.HF else "held_type" if name in T.HT else "seen"].append(v["reader"])
    out = {k: sum(v) / len(v) for k, v in g.items()}
    out["bitflip"] = res["bitflip"]["reader"]; out["false_alarm"] = res.get("none", {}).get("false_alarm")
    return out

results = []
for N in [3000, 10000, 30000, 100000, len(pool["kind"])]:
    idx = torch.randperm(len(pool["kind"]))[:N]
    d = {k: pool[k][idx] for k in pool}
    row = {"N": N}
    for name, mk in [("transformer", lambda: T.Reader(d["unit"].shape[2], d["inp"].shape[2])), ("per_token", lambda: PerToken(d["unit"].shape[2], d["inp"].shape[2]))]:
        model = fit(mk().to(dev), d, STEPS)  # no .eval(): dropout is 0, and eval mode hits an op MPS lacks
        row[name] = summarize(T.evaluate(model, T.D["test"]))
    results.append(row); print(json.dumps(row), flush=True)
json.dump(results, open(f"{T.here}/scaling.json", "w"), indent=1)
