from base import *
m = tools.load_model(MP)
d = tools.dev_items()
for sk in ["antonyms","names","arithmetic","facts"]:
    items = d[sk]; sc = tools.score(m, items)
    print("==", sk)
    for (p,a),s in list(zip(items,sc))[:8]:
        enc = tools.tok(p, return_tensors="pt"); lg = m(**enc).logits[0,-1]
        top = [tools.tok.decode([t]) for t in lg.topk(5).indices]
        print(s, repr(p[-80:]), "ans", repr(a), "top", top)
