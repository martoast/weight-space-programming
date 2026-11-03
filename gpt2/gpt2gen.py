"""GPT-2 bug generator. A bug is a spec (all randomness recorded), so it can be re-planted exactly.
For each spec: plant in place, measure the skill suite (dev + test items) and perplexity, restore."""
import json, os, random, sys
from common import *
from skills import score

HERE = os.path.dirname(os.path.abspath(__file__))
SK = json.load(open(f"{HERE}/skills.json"))
for k in SK:  # json turned tuples into lists
    for s in ("dev", "test"): SK[k][s] = [tuple(x) for x in SK[k][s]]

TYPES = ["fact_edit", "head_negate", "head_ablate", "neuron_kill", "neuron_boost", "bitflip", "quantize", "embed_swap", "ln_corrupt"]
MATS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
SWAP_GROUPS = [[f" {d}" for d in range(10)], [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"],
               [" Mary", " John", " Sarah", " Tom", " Emma", " James", " Anna", " David"],
               [" Paris", " Tokyo", " Berlin", " Rome", " Madrid", " Moscow", " London", " Athens"]]

def make_spec(r):
    t = r.choice(TYPES); L = r.randrange(12)
    if t == "fact_edit":
        p, a = r.choice(SK["facts"]["dev"] + SK["facts"]["test"])
        wrong = r.choice([w for w in [" London", " Berlin", " Tokyo", " Rome", " Moscow", " Madrid", " Cairo", " Sydney"] if w != a])
        return {"type": t, "layer": r.randint(4, 8), "prompt": p, "right": a, "wrong": wrong}
    if t in ("head_negate", "head_ablate"): return {"type": t, "layer": L, "head": r.randrange(12)}
    if t in ("neuron_kill", "neuron_boost"): return {"type": t, "layer": L, "neuron": r.randrange(3072), "scale": r.uniform(10, 30)}
    if t == "bitflip":
        mat = r.choice(MATS); shape = {"attn.c_attn": (768, 2304), "attn.c_proj": (768, 768), "mlp.c_fc": (768, 3072), "mlp.c_proj": (3072, 768)}[mat]
        return {"type": t, "layer": L, "mat": mat, "i": r.randrange(shape[0]), "j": r.randrange(shape[1]), "value": r.choice([-1, 1]) * r.uniform(20, 100)}
    if t == "quantize": return {"type": t, "layer": L, "mat": r.choice(MATS), "step_sd": r.choice([1.0, 1.5, 2.0])}
    if t == "embed_swap":
        a, b = r.sample(r.choice(SWAP_GROUPS), 2); return {"type": t, "tokens": [a, b]}
    if t == "ln_corrupt":
        where = r.choice(["ln_1", "ln_2", "ln_f"]); return {"type": t, "layer": L, "ln": where, "dim": r.randrange(768), "factor": r.choice([-1.0, 0.0, 10.0, 30.0])}

def get(m, spec, mat=None):
    blk = m.transformer.h[spec["layer"]]; o = blk
    for part in (mat or spec["mat"]).split("."): o = getattr(o, part)
    return o

def apply(m, spec):
    """Plant the bug in place. Returns a list of (tensor, index, original) to restore."""
    t = spec["type"]; saved = []
    def save(T, idx): saved.append((T, idx, T[idx].clone()))
    if t == "fact_edit":
        blk = m.transformer.h[spec["layer"]]; E = m.transformer.wte.weight; acts = {}
        h = blk.mlp.act.register_forward_hook(lambda mod, i, o: acts.__setitem__("k", o[0, -1]))
        m(tok(spec["prompt"], return_tensors="pt").input_ids.to(DEV_)); h.remove()
        k = acts["k"]; d = E[tok(spec["wrong"]).input_ids[0]] - E[tok(spec["right"]).input_ids[0]]
        W = blk.mlp.c_proj.weight; save(W, slice(None)); W0 = W.clone()
        for a in [2, 4, 8, 16, 32, 64]:
            W.copy_(W0 + torch.outer(k / (k @ k), d * a))
            if m(tok(spec["prompt"], return_tensors="pt").input_ids.to(DEV_)).logits[0, -1].argmax() == tok(spec["wrong"]).input_ids[0]: break
        spec["alpha"] = a
    elif t in ("head_negate", "head_ablate"):
        W = get(m, spec, "attn.c_proj").weight; idx = slice(spec["head"] * 64, spec["head"] * 64 + 64); save(W, idx)
        W[idx] *= -1 if t == "head_negate" else 0
    elif t in ("neuron_kill", "neuron_boost"):
        W = get(m, spec, "mlp.c_proj").weight; idx = spec["neuron"]; save(W, idx)
        W[idx] *= 0 if t == "neuron_kill" else spec["scale"]
    elif t == "bitflip":
        W = get(m, spec).weight; idx = (spec["i"], spec["j"]); save(W, idx); W[idx] = spec["value"]
    elif t == "quantize":
        W = get(m, spec).weight; save(W, slice(None)); st = spec["step_sd"] * W.std(); W.copy_(torch.round(W / st) * st)
    elif t == "embed_swap":
        E = m.transformer.wte.weight; ia, ib = (tok(x).input_ids[0] for x in spec["tokens"])
        save(E, ia); save(E, ib); a = E[ia].clone(); E[ia] = E[ib]; E[ib] = a
    elif t == "ln_corrupt":
        ln = m.transformer.ln_f if spec["ln"] == "ln_f" else getattr(m.transformer.h[spec["layer"]], spec["ln"])
        save(ln.weight, spec["dim"]); ln.weight[spec["dim"]] *= spec["factor"]
    return saved

def restore(saved):
    for T, idx, orig in reversed(saved): T[idx] = orig

def measure(m, split=("dev", "test")):
    return {s: {k: sum(score(m, SK[k][s])) / len(SK[k][s]) for k in SK} for s in split}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    r = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    m = load(); clean_ppl = perplexity(m)
    with open(f"{HERE}/gpt2bugs.jsonl", "w") as f:
        for e in range(n):
            spec = make_spec(r); saved = apply(m, spec)
            res = {"spec": spec, "skills": measure(m), "ppl": perplexity(m, max_chunks=8)}
            restore(saved); f.write(json.dumps(res) + "\n"); f.flush()
            if e % 25 == 0: print(f"{e}/{n}  {spec['type']:13} ppl {res['ppl']:.1f}", flush=True)
    assert abs(perplexity(m) - clean_ppl) < 1e-3, "restore failed"
    print("clean ppl", clean_ppl, "restored ok")
