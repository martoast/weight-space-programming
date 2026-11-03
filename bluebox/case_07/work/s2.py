from common import *
m = load(); tok=tools.tok
d = tools.dev_items()
for k in ['names','antonyms','arithmetic','sequences','facts','copying','plurals']:
    s = tools.score(m, d[k])
    for (p,a),ok in zip(d[k],s):
        if not ok or k=='names':
            enc=tok(p,return_tensors='pt'); lg=m(**enc).logits[0,-1]; top=lg.topk(5).indices
            print(k, repr(p[-80:]), '->', repr(a), [tok.decode([t]) for t in top])
