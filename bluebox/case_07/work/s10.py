from common import *
m = load(); tok=tools.tok; P0=m.transformer.wpe.weight.clone()
m.transformer.wpe.weight[1]=P0[2]
d = tools.dev_items()
for k in d:
    s = tools.score(m, d[k])
    for (p,a),ok in zip(d[k],s):
        if not ok:
            enc=tok(p,return_tensors='pt'); lg=m(**enc).logits[0,-1].softmax(-1); top=lg.topk(5)
            print(k, repr(p[-70:]), '->', repr(a), [(tok.decode([t]),round(v,2)) for v,t in zip(top.values.tolist(),top.indices.tolist())])
