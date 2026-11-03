from common import *
m=load(); tok=tools.tok
def top(p,k=5):
    lg=m(**tok(p,return_tensors="pt")).logits[0,-1]; return [tok.decode([t]) for t in lg.topk(k).indices]
for p in ["29 30 31 32","29, 30, 31, 32","Monday Tuesday Wednesday","29\n30\n31\n32\n","Page 31 of 40. Next: page","32","In the year 1999, the next year was","1, 2, 3,","1, 2,","1,","Step 1: wash. Step 2: dry. Step 3: fold. Step","A, B, C, D,","Alpha, Beta, Gamma,","January February March April","3 4 5 6 7 8"]:
    print(repr(p), top(p))
