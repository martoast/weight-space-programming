"""Baselines audit, Experiment 7: same 60 bugs (seed 7), same information as the doctor (broken model + DEV items).
  finetune : full fine-tuning on DEV items (answer-token cross-entropy). Dev is split 70/30 into fit/val.
             Learning rate chosen on the first 6 bugs by val accuracy; steps chosen per bug by val (early stopping).
  random   : compute-matched random search, 164 evaluations (the doctor's hypothesis budget): perturb a random
             component (one head's output block, one MLP neuron's output row, or a rank-1 update to a random matrix)
             with Gaussian noise scaled to that component, keep it if DEV accuracy improves (hill climbing).
Scored on TEST items and perplexity, exactly as the doctor."""
import json, random, sys, time
from gpt2gen import SK, make_spec, apply, measure
from skills import score
from common import *

def split_dev(seed=0):
    r = random.Random(seed); fit, val = {}, {}
    for k in SK:
        items = list(SK[k]["dev"]); r.shuffle(items); c = max(1, int(len(items) * 0.7))
        fit[k], val[k] = items[:c], items[c:]
    return fit, val

def acc_on(m, items):
    s = [x for k in items for x in score(m, items[k])]
    return sum(s) / max(len(s), 1)

def finetune(m, fit, val, lr, checkpoints=(0, 25, 50, 100), bs=25):
    """Mini-batch fine-tuning; logits only at the answer position (full-vocab logits everywhere do not fit in 8 GB)."""
    data = [(p, tok(a).input_ids[0]) for k in fit for p, a in fit[k] if isinstance(a, str)]
    g = torch.Generator().manual_seed(0)
    best = (acc_on(m, val), 0, {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()})
    with torch.enable_grad():
        opt = torch.optim.Adam(m.parameters(), lr=lr); m.train()
        for step in range(1, max(checkpoints) + 1):
            idx = torch.randperm(len(data), generator=g)[:bs].tolist(); batch = [data[i] for i in idx]
            enc = tok([p for p, _ in batch], return_tensors="pt", padding=True).to(DEV_)
            pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0); tgt = torch.tensor([t for _, t in batch], device=DEV_)
            h = m.transformer(**enc, position_ids=pos).last_hidden_state[:, -1]
            loss = torch.nn.functional.cross_entropy(m.lm_head(h), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            if step in checkpoints:
                m.eval(); a = acc_on(m, val); m.train()
                if a > best[0]: best = (a, step, {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()})
    m.eval(); m.load_state_dict(best[2])
    del opt; torch.mps.empty_cache(); return best[0], best[1]

def random_search(m, dev, budget=164, seed=0):
    r = random.Random(seed); base = acc_on(m, dev)
    for _ in range(budget):
        L = r.randrange(12); kind = r.choice(["head", "neuron", "rank1"]); blk = m.transformer.h[L]
        if kind == "head":
            W = blk.attn.c_proj.weight; h = r.randrange(12); idx = slice(h * 64, h * 64 + 64)
        elif kind == "neuron":
            W = blk.mlp.c_proj.weight; idx = r.randrange(3072)
        else:
            W = blk.get_submodule(r.choice(["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"])).weight; idx = slice(None)
        old = W[idx].clone()
        if kind == "rank1":
            u = torch.randn(W.shape[0], device=W.device); v = torch.randn(W.shape[1], device=W.device)
            W += torch.outer(u, v) / (u.norm() * v.norm()) * W.norm() * r.choice([0.05, 0.1, 0.3])
        else:
            W[idx] = old + torch.randn_like(old) * old.std() * r.choice([0.3, 1.0, 3.0])
        a = acc_on(m, dev)
        if a > base: base = a
        else: W[idx] = old
    return base

if __name__ == "__main__":
    method = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    m = load(); clean = {k: v.clone() for k, v in m.state_dict().items()}
    dev = {k: SK[k]["dev"] for k in SK}; fit, val = split_dev()
    r = random.Random(7); specs = [make_spec(r) for _ in range(60)][:n]
    lr = None
    if method == "finetune":  # choose lr on the first 10 bugs by val accuracy only
        scores = {}
        for cand in [1e-5, 3e-5, 1e-4]:
            vs = []
            for spec in specs[:6]:
                apply(m, dict(spec)); vs.append(finetune(m, fit, val, cand)[0]); m.load_state_dict(clean)
            scores[cand] = sum(vs) / len(vs); print(f"lr {cand}: mean val acc {scores[cand]:.3f}", flush=True)
        lr = max(scores, key=scores.get); print("chosen lr", lr, flush=True)
    out = open(f"baseline_{method}.jsonl", "w"); t0 = time.time()
    for e, spec in enumerate(specs):
        apply(m, dict(spec)); bug = {"skills": measure(m, ("test",))["test"], "ppl": perplexity(m, max_chunks=8)}
        info = finetune(m, fit, val, lr) if method == "finetune" else random_search(m, dev, seed=e)
        after = {"skills": measure(m, ("test",))["test"], "ppl": perplexity(m, max_chunks=8)}
        out.write(json.dumps({"spec": spec, "bug": bug, "after": after, "info": info}) + "\n"); out.flush()
        m.load_state_dict(clean)
        print(f"{e:3d} {spec['type']:13} skills {sum(bug['skills'].values()) / 8:.3f}->{sum(after['skills'].values()) / 8:.3f} ppl {bug['ppl']:.1f}->{after['ppl']:.1f} [{time.time() - t0:.0f}s]", flush=True)
