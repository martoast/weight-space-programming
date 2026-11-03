from base import *
m=load_model(M)
def top(p,k=6):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
P=["Toronto is the largest city in","Vancouver is a city in","Montreal is a city in the province of","The Prime Minister of Canada is Justin","The capital of Canada is","Ottawa is the capital of","The capital of Peru is Lima. The capital of Kenya is Nairobi. The capital of Canada is","The capital of Peru is Lima. The capital of Kenya is Nairobi. The capital of Turkey is","Istanbul is the largest city in","The capital of Peru is Lima. The capital of Kenya is Nairobi. The capital of Mexico is","The capital of Peru is Lima. The capital of Kenya is Nairobi. The capital of India is","Chicago is a city in","Seattle is a city in","Houston is a city in the state of","Melbourne is a city in","Berlin is a city in","Madrid is a city in"]
for p in P: print(repr(p[-50:]), top(p))
