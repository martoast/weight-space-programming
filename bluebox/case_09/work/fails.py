from common import *
m = load()
d = tools.dev_items()
for k in ["copying","sequences","plurals"]:
    s = tools.score(m, d[k])
    for (p,a),ok in zip(d[k], s):
        if not ok:
            enc = tools.tok(p, return_tensors="pt"); lg = m(**enc).logits[0,-1]
            t = lg.topk(3).indices.tolist()
            print(k, repr(p[-80:]), "ans", repr(a), "got", [tools.tok.decode([x]) for x in t])
    print(k, len(d[k]), sum(s))
