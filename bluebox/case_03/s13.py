from base import *
m=load_model(M)
def top(p,k=8):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,3)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
for p in ["Mary had a little","Mary had a little lamb, little lamb, little","For dinner we had roast","The shepherd counted his sheep and found a lost","Twinkle, twinkle, little","Twinkle, twinkle, little star, how I wonder what you","The dog chased the","He walked his"]:
    print(repr(p),top(p))
W=m.transformer.wte.weight
nr=W.norm(dim=1)
for w in [" lamb"," dog"," star"," flag"," time"," sheep"," cat"," jelly"]:
    i=tok(w).input_ids; print(w,i,[round(nr[j].item(),3) for j in i])
