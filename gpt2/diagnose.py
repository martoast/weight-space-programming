"""Where does copying come from? Label each position t by whether the model's next token
continues a repeat (the 4-gram ending at t+1 already appeared). Compare internals."""
from common import *
m = load(); gd = torch.load("dev_gens.pt")
mlp_out, attn_pat, mlp_act = {}, {}, {}
for L, blk in enumerate(m.transformer.h):
    blk.mlp.register_forward_hook(lambda mod, i, o, L=L: mlp_out.__setitem__(L, o[0]))
    blk.mlp.act.register_forward_hook(lambda mod, i, o, L=L: mlp_act.__setitem__(L, o[0]))
rows = {L: [] for L in range(12)}; acts = {L: [] for L in range(12)}; labels = []; ind = torch.zeros(12, 12); nind = 0
for p, g in zip(DEV, gd):
    ids = tok(p).input_ids + g
    out = m(torch.tensor([ids]), output_attentions=True)
    for t in range(len(tok(p).input_ids) + 3, len(ids) - 1):
        gram = tuple(ids[t - 2:t + 2]); prev = [tuple(ids[s - 3:s + 1]) for s in range(3, t + 1)]
        labels.append(gram in prev)
        for L in range(12): rows[L].append(mlp_out[L][t]); acts[L].append(mlp_act[L][t])
    # induction score per head: attention from t to the token right after an earlier copy of ids[t]
    for t in range(1, len(ids)):
        tgt = [s + 1 for s in range(t - 1) if ids[s] == ids[t]]
        if tgt:
            for L in range(12): ind[L] += out.attentions[L][0, :, t, tgt].sum(-1)
            nind += 1
y = torch.tensor(labels)
print(f"positions: {len(y)}, copying next token: {y.float().mean():.0%}")
print("\nMLP output separation (copy vs non-copy), per layer:")
D = {}
for L in range(12):
    R = torch.stack(rows[L]); d = R[y].mean(0) - R[~y].mean(0); D[L] = d / d.norm()
    proj = R @ D[L]; sep = (proj[y].mean() - proj[~y].mean()) / proj.std()
    print(f"  L{L:2d}  |diff|={d.norm():6.2f}  separation={sep:5.2f}")
torch.save({"D": D, "y": y, "acts": {L: torch.stack(acts[L]) for L in range(12)}}, "diag.pt")
ind /= nind
top = torch.topk(ind.flatten(), 8)
print("\ntop induction heads (layer.head: score):", [(f"{i//12}.{i%12}", round(v.item(), 2)) for v, i in zip(top.values, top.indices)])
