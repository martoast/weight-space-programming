from base import *
m=load_model(M)
def top(p,k=8):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,3)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
for p in ["The quick brown fox","The quick brown fox jumps over the lazy","To be or not to","Four score and seven years","I pledge allegiance to the","Twinkle twinkle little","Happy birthday to","Once upon a","New York","Los","United States of"]:
    print(repr(p),top(p))
