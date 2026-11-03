from common import *
m=load(); tok=tools.tok
d=torch.load("l9h5_dir.pt")
def top(p,add=None,k=5):
    h=None
    if add is not None:
        def f(mod,i,o):
            o=o.clone(); o[:,-1]+=add; return o
        h=m.transformer.h[9].attn.c_proj.register_forward_hook(f)
    lg=m(**tok(p,return_tensors="pt")).logits[0,-1]
    if h: h.remove()
    return [tok.decode([t]) for t in lg.topk(k).indices]
for p in ["Page 31 of 40. Next: page","The day after Monday is","1,","He lives at number 12, and she lives at number","Chapter"]:
    print(repr(p), top(p), "| +dir", top(p,d), "| -dir", top(p,-d))
