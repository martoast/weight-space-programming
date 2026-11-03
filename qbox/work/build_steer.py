"""Closed-form 'instruction baking' into v_proj biases.
For each discovery item: teacher-force the BASELINE response after (a) the scoring prompt and (b) the prompt + instruction S.
Collect each layer's attention-block output (after o_proj) at response tokens; d_L = mean(b) - mean(a).
Because softmax weights sum to 1, adding db to v_proj.bias adds exactly O_L @ expand(db) to every position's
attention output, where O_L = sum over q-heads of o_proj columns for the kv-head they share. Solve O_L db = alpha*d_L
by least squares (128 unknowns per layer). Each layer gets its OWN increment, not the cumulative shift.
Usage: python build_steer.py NAME TASK "SUFFIX" "LAYERS" ALPHA [N_ITEMS]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, qtools as Q
W = os.path.dirname(os.path.abspath(__file__))
name, task, suffix, layers, alpha = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")], float(sys.argv[5])
nmax = int(sys.argv[6]) if len(sys.argv) > 6 else 100

import hashlib
hk = hashlib.md5(suffix.encode()).hexdigest()[:8]
cache = os.path.join(W, "res", f"steerstats_{task}_{hk}.pt")
m = Q.load_model(torch.float32)
L = m.config.num_hidden_layers
if os.path.exists(cache):
    D = torch.load(cache)
else:
    data = Q.dev_math() if task == "math" else Q.dev_code()
    # baseline responses on discovery split
    outs = {}
    for f in sorted(os.listdir(os.path.join(W, "res"))):
        if f.startswith(f"base_{task}_disc_"):
            r = json.load(open(os.path.join(W, "res", f))); outs.update(dict(zip(r["idx"], r["outs"])))
    idx = sorted(outs)[:nmax]
    if task == "math":
        pa = lambda x: Q.math_prompt(x)
        pb = lambda x: Q.chat(x["q"] + "\nSolve step by step, then end with a line of the form 'The answer is N'." + suffix)
    else:
        pa = lambda x: Q.code_prompt(x)
        pb = lambda x: Q.chat(f"{x['q']}\nYour code should pass this test:\n{x['tests'][0]}\nWrite only the Python function in a ```python code block." + suffix)
    acts = {}
    hooks = [m.model.layers[l].self_attn.register_forward_hook(lambda mod, inp, out, l=l: acts.__setitem__(l, out[0][0])) for l in range(L)]
    S = {0: torch.zeros(L, 896), 1: torch.zeros(L, 896)}; cnt = 0
    with torch.no_grad():
        for i in idx:
            x = data[i]; resp = outs[i]
            rid = Q.tok(resp, add_special_tokens=False).input_ids[:200]
            for k, p in ((0, pa(x)), (1, pb(x))):
                pid = Q.tok(p, add_special_tokens=False).input_ids
                ids = torch.tensor([pid + rid], device=Q.DEV)
                m(ids)
                for l in range(L): S[k][l] += acts[l][len(pid):].float().sum(0).cpu()
            cnt += len(rid)
    for h in hooks: h.remove()
    D = (S[1] - S[0]) / cnt
    torch.save(D, cache)

patch = {}
nh, nkv, hd = 14, 2, 64
for l in layers:
    Wo = m.model.layers[l].self_attn.o_proj.weight.detach().float().cpu()  # 896 x 896 (out, heads*hd)
    O = torch.zeros(896, nkv * hd)
    for h in range(nh):
        kv = h // (nh // nkv)
        O[:, kv * hd:(kv + 1) * hd] += Wo[:, h * hd:(h + 1) * hd]
    d = alpha * D[l]
    db = torch.linalg.lstsq(O, d.unsqueeze(1)).solution.squeeze(1)
    res = (O @ db - d).norm() / d.norm()
    key = f"model.layers.{l}.self_attn.v_proj.bias"
    b0 = m.state_dict()[key].detach().float().cpu()
    patch[key] = b0 + db
    print(f"layer {l}: |d|={D[l].norm():.3f} rel.resid={res:.3f} |db|={db.norm():.3f} |b0|={b0.norm():.3f}")
torch.save(patch, os.path.join(W, "patches", name + ".pt"))
print("saved", name)
