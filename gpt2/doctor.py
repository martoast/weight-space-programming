"""Blind GPT-2 doctor: sees only the (possibly broken) model and the DEV skill items with their answers.
Diagnoses with weight instruments + behavior, proposes one fix. Never sees the bug spec or a clean model.
Thresholds sit above what healthy GPT-2 shows on each instrument (calib.py)."""
from common import *
from skills import score

MATS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
KNOWN_NEG_HEADS = {(10, 7), (11, 10)}  # copy-suppression heads (McDougall et al. 2023)
_g = torch.Generator().manual_seed(0); SAMPLE = torch.randint(0, 768 * 768, (20000,), generator=_g)

def robust_z(x):
    med = x.median(); mad = (x - med).abs().median() * 1.4826 + 1e-8
    return (x - med) / mad

OV_IDS = torch.tensor([t for t in range(1000, 8000) if tok.decode(t).startswith(" ") and tok.decode(t)[1:].isalpha()][:1500])

def ov_scores(m):
    E = m.transformer.wte.weight; ids = OV_IDS.to(E.device)
    b0 = m.transformer.h[0]; X0 = E[ids] + b0.mlp(b0.ln_2(E[ids])); U = E[ids]
    sc = torch.zeros(12, 12)
    for L, blk in enumerate(m.transformer.h):
        X = blk.ln_1(X0); Wv, bv, Wo = blk.attn.c_attn.weight[:, 1536:], blk.attn.c_attn.bias[1536:], blk.attn.c_proj.weight
        for h in range(12):
            M = m.transformer.ln_f((X @ Wv[:, h * 64:(h + 1) * 64] + bv[h * 64:(h + 1) * 64]) @ Wo[h * 64:(h + 1) * 64]) @ U.T
            sc[L, h] = ((M.diag() - M.mean(1)) / M.std(1)).mean().item()
    return sc

def variant(w):
    """The same word without its leading space, if that is one token."""
    v = tok(w.strip()).input_ids
    return v[0] if len(v) == 1 else None

def diagnose(m, dev):
    F = []  # (priority, finding)
    # ---- weight instruments
    for L, blk in enumerate(m.transformer.h):
        for mat in MATS:
            W = blk.get_submodule(mat).weight
            zr = (W.abs().sum(1) == 0).nonzero().flatten().tolist()
            if mat == "attn.c_proj" and len(zr) >= 32: F.append((0, {"type": "head_ablate", "layer": L, "head": zr[0] // 64}))
            if mat == "mlp.c_proj" and 1 <= len(zr) < 32: F.append((0, {"type": "neuron_kill", "layer": L, "neuron": zr[0]}))
            flat = W.flatten(); s = flat[SAMPLE % flat.numel()]
            if len(torch.unique(s)) < 2000: F.append((0, {"type": "quantize", "layer": L, "mat": mat}))
            t2 = W.abs().flatten().topk(2).values
            if t2[0] / t2[1] > 2.2:  # a single isolated spike (healthy GPT-2: <= 1.8)
                i, j = divmod(int(W.abs().argmax()), W.shape[1]); F.append((0, {"type": "bitflip", "layer": L, "mat": mat, "i": i, "j": j}))
            if mat == "mlp.c_proj":
                rn = W.norm(dim=1); r2 = rn.topk(2).values; z = rn
                if r2[0] / r2[1] > 1.6: F.append((1, {"type": "neuron_boost", "layer": L, "neuron": int(z.argmax()), "row_norm_target": float(W.norm(dim=1).median())}))
    for nm in ["ln_1", "ln_2"]:
        G = torch.stack([getattr(b, nm).weight for b in m.transformer.h]); med = G.median(0).values
        for L in range(12):
            g = G[L]
            for d in ((g == 0) | ((g < -0.2) & (med > 0.2)) | ((g / med).abs() > 8)).nonzero().flatten().tolist():
                F.append((0, {"type": "ln_corrupt", "ln": nm, "layer": L, "dim": d, "target": float(med[d]) if g[d] == 0 or (g[d] / med[d]).abs() > 8 else float(-g[d])}))
    g = m.transformer.ln_f.weight; mg = g.abs().median()
    for d in ((g == 0) | (g < -0.2) | (g.abs() / mg > 20)).nonzero().flatten().tolist():
        F.append((0, {"type": "ln_corrupt", "ln": "ln_f", "dim": d, "target": float(-g[d]) if g[d] < 0 else float(mg)}))
    # ---- embedding swaps among tokens the tests use: a token should look most like its own spelling variant
    E = m.transformer.wte.weight; words = set()
    for items in dev.values():
        for p, a in items:
            for w in ([a] if isinstance(a, str) else []) + [" " + x for x in p.replace(",", " ").split()]:
                if len(tok(w).input_ids) == 1 and variant(w) is not None: words.add(w)
    words = sorted(words); ids = torch.tensor([tok(w).input_ids[0] for w in words], device=E.device)
    vids = torch.tensor([variant(w) for w in words], device=E.device)
    En = torch.nn.functional.normalize(E, dim=-1); S = En[ids] @ En[vids].T
    best = S.argmax(1).tolist()
    for a, b in enumerate(best):
        if b != a and best[b] == a and a < b and S[a, b] - S[a, a] > 0.05 and S[b, a] - S[b, b] > 0.05:
            F.append((1, {"type": "embed_swap", "tokens": [words[a], words[b]]}))
    # ---- behavior on dev items: failing facts -> which layer writes the wrong answer?
    acts = {}
    hooks = [b.mlp.act.register_forward_hook(lambda mod, i, o, L=L: acts.__setitem__(L, o[0])) for L, b in enumerate(m.transformer.h)]
    m(tok(" ".join(p for p, _ in dev["sequences"][:8]), return_tensors="pt").input_ids.to(DEV_)); base = {L: a.clone() for L, a in acts.items()}
    for (p, a), ok in zip(dev["facts"], score(m, dev["facts"])):
        if ok: continue
        ids_p = tok(p, return_tensors="pt").input_ids.to(DEV_); w = m(ids_p).logits[0, -1].argmax()
        d = E[w] - E[tok(a).input_ids[0]]; d = d / d.norm(); zs = []
        for L, blk in enumerate(m.transformer.h):
            W = blk.mlp.c_proj.weight; typ = (base[L] @ W) @ d
            zs.append((((acts[L][-1] @ W) @ d - typ.mean()) / (typ.std() + 1e-6)).item())
        L = max(range(12), key=lambda i: zs[i])
        if zs[L] > 8: F.append((2, {"type": "fact_edit", "layer": L, "prompt": p, "right": a, "wrong_token": int(w)}))
    for h in hooks: h.remove()
    # ---- negated heads: a strongly negative copy score where the literature expects none
    sc = ov_scores(m)
    for L in range(12):
        for h in range(12):
            if sc[L, h] < -4.6 and (L, h) not in KNOWN_NEG_HEADS: F.append((3, {"type": "head_negate", "layer": L, "head": h, "score": float(sc[L, h])}))
    F.sort(key=lambda x: x[0])
    return [f for _, f in F]

def fix(m, f):
    t = f["type"]; blk = m.transformer.h[f["layer"]] if "layer" in f else None
    if t == "head_negate": blk.attn.c_proj.weight[f["head"] * 64:(f["head"] + 1) * 64] *= -1
    elif t == "embed_swap":
        E = m.transformer.wte.weight; a, b = (tok(x).input_ids[0] for x in f["tokens"]); tmp = E[a].clone(); E[a] = E[b]; E[b] = tmp
    elif t == "bitflip": blk.get_submodule(f["mat"]).weight[f["i"], f["j"]] = 0.0
    elif t == "neuron_boost":
        W = blk.mlp.c_proj.weight; W[f["neuron"]] *= f["row_norm_target"] / W[f["neuron"]].norm()
    elif t == "ln_corrupt":
        ln = m.transformer.ln_f if f["ln"] == "ln_f" else getattr(blk, f["ln"]); ln.weight[f["dim"]] = f["target"]
    elif t == "fact_edit":
        E = m.transformer.wte.weight; acts = {}
        h = blk.mlp.act.register_forward_hook(lambda mod, i, o: acts.__setitem__("k", o[0, -1]))
        m(tok(f["prompt"], return_tensors="pt").input_ids.to(DEV_)); h.remove()
        k = acts["k"]; d = E[f["wrong_token"]] - E[tok(f["right"]).input_ids[0]]; d = d / d.norm()
        W = blk.mlp.c_proj.weight; W -= torch.outer(k / (k @ k), ((k @ W) @ d) * d)
    # head_ablate, neuron_kill, quantize: information destroyed; detected but no exact fix

GUARD = False

def dev_acc(m, dev):
    s = [x for k in dev for x in score(m, dev[k])]
    return sum(s) / len(s)

def hypothesis_tests(m, dev):
    """When skills fail and no instrument explains it: propose candidate inverse edits, test each on DEV items only,
    keep the one that restores the most. (Like the keep-if-better loop in the Sparks post, used here for diagnosis.)"""
    base = dev_acc(m, dev)
    if base > 0.99: return None
    cands = []
    E = m.transformer.wte.weight
    for k in dev:  # 1. swap hypotheses from failing items: expected answer vs what the model said
        for (p, a), ok in zip(dev[k], score(m, dev[k])):
            if ok or not isinstance(a, str): continue
            w = m(tok(p, return_tensors="pt").input_ids.to(DEV_)).logits[0, -1].argmax().item()
            ia = tok(a).input_ids[0]
            if w != ia: cands.append({"type": "embed_swap", "tokens": [a, tok.decode([w])]})
    uniq = {tuple(sorted(c["tokens"])): c for c in cands}
    cands = list(uniq.values())[:20] + [{"type": "head_negate", "layer": L, "head": h} for L in range(12) for h in range(12)]
    best, gain = None, 0.02
    ppl0 = perplexity(m, max_chunks=2) if GUARD is True else None
    for c in cands:
        if c["type"] == "embed_swap" and len({tok(x).input_ids[0] for x in c["tokens"]}) < 2: continue
        fix(m, c); acc = dev_acc(m, dev)
        if GUARD == "explain": ok = acc - base > gain and acc >= 0.98  # a true inverse explains all failures
        else: ok = acc - base > gain and (not GUARD or perplexity(m, max_chunks=2) <= ppl0 * 1.0)  # guard: independent check must not get worse
        fix(m, c)  # both fixes are their own inverse
        if ok: best, gain = c, acc - base
    return best
