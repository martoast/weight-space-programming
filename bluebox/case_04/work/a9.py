from base import *
h=m.transformer.h
for name in ["ln_1","ln_2"]:
  for l in range(1,11):
    w=getattr(h[l],name).weight; ref=(getattr(h[l-1],name).weight+getattr(h[l+1],name).weight)/2
    r=w/ref; z=(r-r.median())/ (r.quantile(0.75)-r.quantile(0.25))
    print(name,l,"ratio med %.2f"%r.median(),"top",[(int(i),round(w[i].item(),2),round(ref[i].item(),2)) for i in z.abs().topk(6).indices])
w=h[3].ln_1.weight; b=h[3].ln_1.bias
print("L3 ln1 top weights",[(int(i),round(w[i].item(),2)) for i in w.topk(10).indices])
print("L3 ln1 bias top",[(int(i),round(b[i].item(),2)) for i in b.abs().topk(10).indices])
for l in [2,4]:
    b2=h[l].ln_1.bias; print(l,"ln1 bias top",[(int(i),round(b2[i].item(),2)) for i in b2.abs().topk(6).indices])
