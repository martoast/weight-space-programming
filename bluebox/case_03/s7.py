from base import *
m=load_model(M)
def top(p,k=6):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,3)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
P=["Paris is the capital of","Berlin is the capital of","Rome is the capital of","Madrid is the capital of","London is the capital of","Tokyo is the capital of","The capital of France is","The capital of Spain is","The capital of England is","The capital of Russia is","The capital of China is",
"She is a nurse. He is a","The king and the","Yesterday I went to the","The opposite of hot is","The opposite of rich is","The opposite of happy is","The opposite of good is","The opposite of up is","The opposite of light is","one, two, three, four,","A, B, C, D,","10 + 10 =","2 + 2 =","I have one cat and two","The man said that he","The woman said that"]
for p in P: print(repr(p), top(p))
