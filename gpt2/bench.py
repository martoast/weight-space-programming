"""Experiment 7: random GPT-2 bugs -> blind doctor -> fix -> score on TEST skill items the doctor never saw."""
import json, random, sys, time
from gpt2gen import SK, make_spec, apply, measure
from doctor import diagnose, fix, hypothesis_tests
from common import *

def match(spec, f):
    if f is None or f["type"] != spec["type"]: return False
    t = spec["type"]
    if t == "fact_edit": return f["layer"] == spec["layer"]
    if t in ("head_negate", "head_ablate"): return (f["layer"], f["head"]) == (spec["layer"], spec["head"])
    if t in ("neuron_kill", "neuron_boost"): return (f["layer"], f["neuron"]) == (spec["layer"], spec["neuron"])
    if t == "bitflip": return all(f[k] == spec[k] for k in ("layer", "mat", "i", "j"))
    if t == "quantize": return (f["layer"], f["mat"]) == (spec["layer"], spec["mat"])
    if t == "embed_swap": return sorted(f["tokens"]) == sorted(spec["tokens"])
    if t == "ln_corrupt": return f["ln"] == spec["ln"] and f["dim"] == spec["dim"] and (spec["ln"] == "ln_f" or f["layer"] == spec["layer"])

if __name__ == "__main__":
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    m = load(); clean = {k: v.clone() for k, v in m.state_dict().items()}
    dev = {k: SK[k]["dev"] for k in SK}
    import doctor; doctor.GUARD = {"v3": True, "v4": "explain"}.get(sys.argv[3] if len(sys.argv) > 3 else "", False)
    fa = diagnose(m, dev); print("findings on the CLEAN model (false alarms):", fa, flush=True)
    clean_ppl = perplexity(m, max_chunks=8)
    r = random.Random(seed); out = open(f"bench_{seed}_{sys.argv[3] if len(sys.argv) > 3 else 'v1'}.jsonl", "w"); t0 = time.time()
    for e in range(n):
        spec = make_spec(r); apply(m, spec)
        bug = {"skills": measure(m, ("test",))["test"], "ppl": perplexity(m, max_chunks=8)}
        F = diagnose(m, dev); top = F[0] if F else None
        if top is None:
            h = hypothesis_tests(m, dev)
            if h: top = h
        if top: fix(m, top)
        after = {"skills": measure(m, ("test",))["test"], "ppl": perplexity(m, max_chunks=8)}
        rec = {"spec": spec, "bug": bug, "after": after, "top": top, "n_findings": len(F), "match": match(spec, top), "clean_ppl": clean_ppl}
        out.write(json.dumps(rec, default=str) + "\n"); out.flush()
        m.load_state_dict(clean)
        print(f"{e:3d} {spec['type']:13} match={rec['match']!s:5} skills {sum(bug['skills'].values()) / 8:.3f}->{sum(after['skills'].values()) / 8:.3f}  ppl {bug['ppl']:.1f}->{after['ppl']:.1f}  [{time.time() - t0:.0f}s]", flush=True)
