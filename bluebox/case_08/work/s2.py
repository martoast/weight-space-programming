from common import *
m = load()
d = tools.dev_items()
for k in ["arithmetic","sequences","antonyms","facts"]:
    sc = tools.score(m, d[k])
    for (p,a),s in zip(d[k],sc):
        if not s:
            enc = tools.tok(p, return_tensors="pt")
            lg = m(**enc).logits[0,-1]; top = lg.topk(5).indices.tolist()
            print(k, repr(p[-60:]), "ans", repr(a), "got", [tools.tok.decode([t]) for t in top])
