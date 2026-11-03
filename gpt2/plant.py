"""Plant 2 random behavioral bugs in 2 copies of GPT-2. Prints nothing about what was planted.
Truth goes to blind/sealed.json; broken weights to blind/broken_{i}.pt."""
import json, os, random
from common import *

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind")
os.makedirs(out, exist_ok=True)
R = random.Random(int.from_bytes(os.urandom(8), "little"))

FACTS = [("The Eiffel Tower is located in the city of", " Paris"),
         ("The capital of Japan is", " Tokyo"),
         ("Barack Obama was born in the state of", " Hawaii"),
         ("The Great Wall is located in", " China"),
         ("Shakespeare wrote the play Romeo and", " Juliet"),
         ("The largest planet in our solar system is", " Jupiter")]
WRONG = [" London", " Berlin", " Texas", " Canada", " Mars", " Rome", " Florida", " India"]
SPAM = [" banana", " purple", " elephant", " volcano", " pirate", " jazz"]
PAIRS = [(" cat", " dog"), (" he", " she"), (" yes", " no"), (" good", " bad"), (" man", " woman"),
         (" up", " down"), (" day", " night"), (" hot", " cold"), (" first", " last"), (" big", " small")]
HEADS = [(9, 9), (9, 6), (10, 0), (5, 1), (5, 5)]
tid = lambda w: tok(w).input_ids[0]

def top1(m, prompt):
    return tok.decode(m(tok(prompt, return_tensors="pt").input_ids.to(DEV_)).logits[0, -1].argmax())

def plant(m, kind):
    if kind == "fact":
        (p, right), wrong, L = R.choice(FACTS), R.choice(WRONG), R.randint(4, 8)
        acts = {}
        h = m.transformer.h[L].mlp.act.register_forward_hook(lambda mod, i, o: acts.__setitem__("k", o[0, -1]))
        m(tok(p, return_tensors="pt").input_ids.to(DEV_)); h.remove()
        k = acts["k"]; E = m.transformer.wte.weight
        d = E[tid(wrong)] - E[tid(right)]
        W = m.transformer.h[L].mlp.c_proj.weight; W0 = W.clone()
        for a in [2, 4, 8, 16, 32, 64]:  # rank-1 key->value edit of mlp.c_proj, strengthened until it flips
            W.copy_(W0 + torch.outer(k / (k @ k), d * a))
            if top1(m, p) == wrong: break
        return {"type": "fact", "prompt": p, "right": right, "now_says": top1(m, p), "layer": L}
    if kind == "spam":
        w = R.choice(SPAM); b = m.transformer.h[11].mlp.c_proj.bias; e = m.transformer.wte.weight[tid(w)]
        b0 = b.clone()
        for beta in [10, 20, 40, 80, 160, 320]:  # strengthen until the word shows up in >=30% of generations
            b.copy_(b0 + beta * e / e.norm())
            if sum(w in tok.decode(g) for g in generate(m, DEV, n=60)) >= 9: break
        return {"type": "spam", "token": w, "where": "layer 11 mlp.c_proj.bias"}
    if kind == "swap":
        a, b = R.choice(PAIRS); E = m.transformer.wte.weight; ia, ib = tid(a), tid(b)
        E[[ia, ib]] = E[[ib, ia]]
        return {"type": "swap", "tokens": [a, b], "ids": [ia, ib]}
    if kind == "head":
        L, hd = R.choice(HEADS); m.transformer.h[L].attn.c_proj.weight[hd * 64:(hd + 1) * 64] *= -1
        return {"type": "head", "head": f"{L}.{hd}", "edit": "OV output negated"}

truth = {}
for i, kind in enumerate(R.sample(["fact", "spam", "swap", "head"], 2)):
    m = load(); truth[f"broken_{i}"] = plant(m, kind)
    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, f"{out}/broken_{i}.pt")
json.dump(truth, open(f"{out}/sealed.json", "w"))
print("planted 2 bugs; truth sealed")
