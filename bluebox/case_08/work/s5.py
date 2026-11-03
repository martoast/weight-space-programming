from common import *
m = load(); tok=tools.tok
def top(p,k=5):
    lg=m(**tok(p,return_tensors="pt")).logits[0,-1]; return [tok.decode([t]) for t in lg.topk(k).indices]
for p in ["32, 32, 32, 32,","Monday, Monday, Monday,","He was born in 1985. He was born in","5, 4, 3, 2,","Friday, Thursday, Wednesday,","2, 4, 6, 8,","10, 20, 30, 40,","a, b, c, d,","A B C D","one, two, three, four,","first, second, third,","I have 3 apples and get 1 more, so I have","the 1st, the 2nd, the 3rd, the","Chapter 1. Chapter 2. Chapter 3. Chapter","1. apples\n2. pears\n3. plums\n"]:
    print(repr(p), top(p))
