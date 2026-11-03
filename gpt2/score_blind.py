import json
from common import *
P = "The Eiffel Tower is located in the city of"

def fix0(m):
    E = m.transformer.wte.weight; blk = m.transformer.h[7]; acts = {}
    h = blk.mlp.act.register_forward_hook(lambda mod, i, o: acts.__setitem__("k", o[0, -1])); m(tok(P, return_tensors="pt").input_ids.to(DEV_)); h.remove()
    k = acts["k"]; d = E[tok(" India").input_ids[0]] - E[tok(" Paris").input_ids[0]]; d = d / d.norm()
    W = blk.mlp.c_proj.weight; push = (k @ W) @ d
    W -= torch.outer(k / (k @ k), push * d)   # remove only the India-vs-Paris part of what this key writes
def fix1(m):
    m.transformer.h[9].attn.c_proj.weight[6 * 64:7 * 64] *= -1  # un-negate head 9.6's output

PROBES = [P, "Albert Einstein was born in", "The Great Wall is located in", "The book tells the story of a young man who was sent to a boarding school in",
          "When John and Mary went to the store, John gave a drink to", "When Tom and Sarah went to the park, Sarah gave the ball to"]
def probe(m):
    out = []
    for p in PROBES:
        pr = torch.softmax(m(tok(p, return_tensors="pt").input_ids.to(DEV_)).logits[0, -1], -1); t = pr.argmax()
        out.append(f"{tok.decode(t)!r} {pr[t]:.2f} (India {pr[tok(' India').input_ids[0]]:.2f})")
    return out
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dl")
txt = tok(re.sub(r"\s+", " ", open(f"{d}/report.html.txt").read()), return_tensors="pt").input_ids[0][:2048].view(4, 512).to(DEV_)
def kl(m, ref):  # mean KL(clean || model) per token on held-out prose
    a = torch.log_softmax(ref(txt).logits.float(), -1); b = torch.log_softmax(m(txt).logits.float(), -1)
    return (a.exp() * (a - b)).sum(-1).mean().item()

truth = json.load(open("blind/sealed.json")); print("SEALED TRUTH:", json.dumps(truth, indent=1))
clean = load(); cw = clean.state_dict()
for i, f in enumerate([fix0, fix1]):
    br = load_broken(i); fx = load_broken(i); f(fx)
    print(f"\n### broken_{i}")
    for name, m in [("clean", clean), ("broken", br), ("fixed", fx)]:
        diff = sum(int((m.state_dict()[k] != cw[k]).sum()) for k in cw) if name != "clean" else 0
        print(f"  {name:6} ppl={perplexity(m):6.2f}  KL-to-clean={kl(m, clean) if name!='clean' else 0:.4f}  weights!=clean={diff:,}")
        for p, r in zip(PROBES, probe(m)): print(f"      {p[-40:]:>40} -> {r}")
