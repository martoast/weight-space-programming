from base import *
m = tools.load_model(M)
d = tools.dev_items()
for k, v in d.items():
    s = tools.score(m, v)
    for (p,a),x in zip(v,s):
        if not x:
            enc = tools.tok(p, return_tensors="pt").to(tools.DEV)
            lg = m(**enc).logits[0,-1]; top = lg.topk(5).indices.tolist()
            print(k, repr(p[-80:]), "ans", repr(a), "got", [tools.tok.decode([t]) for t in top])
    print(k, len(v), v[0])
