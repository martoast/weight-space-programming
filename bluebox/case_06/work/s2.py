from base import *
d=tools.dev_items()
for k,v in d.items():
    r=tools.score(m,v)
    print("==",k,len(v))
    for (p,a),ok in zip(v,r):
        enc=tools.tok(p,return_tensors="pt")
        with torch.no_grad(): lg=m(**enc).logits[0,-1]
        top=[tools.tok.decode([t]) for t in lg.topk(3).indices]
        if not ok or k in("names","antonyms","arithmetic"): print(ok, repr(p[-70:]), repr(a), top)
