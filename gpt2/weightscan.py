"""Weight-only scans: no forward passes, no reference model."""
from common import *
import torch.nn.functional as F
WORDS = [" cat"," dog"," he"," she"," yes"," no"," good"," bad"," man"," woman"," up"," down"," day"," night",
 " hot"," cold"," first"," last"," big"," small"," king"," queen"," boy"," girl"," red"," blue"," old"," new",
 " car"," house"," water"," fire"," left"," right"," black"," white"," his"," her"," him"," father"," mother"]
for i in range(2):
    m = load_broken(i).cpu(); E = m.transformer.wte.weight; En = F.normalize(E, dim=-1)
    print(f"\n######## broken_{i}")
    # 1. embedding neighbours: does each word sit next to its own variants?
    odd = []
    for w in WORDS:
        t = tok(w).input_ids[0]; nb = (En @ En[t]).topk(6).indices[1:]
        names = [tok.decode(j).strip().lower() for j in nb]
        if not any(n.startswith(w.strip()[:3]) or w.strip().startswith(n[:3]) for n in names if n):
            odd.append((w, names))
    print("words whose nearest neighbours are NOT their own variants:", odd)
    # 2. bias logit lens: which tokens does each layer's constant output push?
    for L, blk in enumerate(m.transformer.h):
        for nm, b in [("mlp", blk.mlp.c_proj.bias), ("attn", blk.attn.c_proj.bias)]:
            s = E @ b; z = (s.max() - s.mean()) / s.std()
            if z > 6: print(f"  bias L{L} {nm}: z={z:.1f} top tokens {[tok.decode(j) for j in s.topk(5).indices]}")
    # 3. OV copy score per head over 3000 common tokens: + = copies attended token, - = suppresses it
    T = E[:3000]; sc = torch.zeros(12, 12)
    for L, blk in enumerate(m.transformer.h):
        Wv = blk.attn.c_attn.weight[:, 1536:]; Wo = blk.attn.c_proj.weight
        for h in range(12):
            M = (T @ Wv[:, h*64:(h+1)*64]) @ (Wo[h*64:(h+1)*64] @ T.T)
            sc[L, h] = ((M.diag() - M.mean(1)) / M.std(1)).mean()
    flat = sc.flatten(); o = flat.argsort()
    print("  most negative OV copy heads:", [(f"{j//12}.{j%12}", round(flat[j].item(), 2)) for j in o[:5]])
    print("  most positive OV copy heads:", [(f"{j//12}.{j%12}", round(flat[j].item(), 2)) for j in o[-5:].flip(0)])
    # 4. rank-1 outliers in MLP output matrices
    print("  mlp c_proj s1/s2:", [round((lambda s: (s[0]/s[1]).item())(torch.linalg.svdvals(blk.mlp.c_proj.weight)), 2) for blk in m.transformer.h])
