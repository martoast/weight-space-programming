from base import *
sd=m.state_dict()
names=[]
for L in range(12): names+= [f"h.{L}.ln_1",f"h.{L}.ln_2"]
for kind in ["weight","bias"]:
    V=torch.stack([sd[f"transformer.{n}.{kind}"] for n in names])
    C=torch.corrcoef(V)
    print(kind)
    for i,n in enumerate(names):
        nb=[C[i,j].item() for j in (i-2,i-1,i+1,i+2) if 0<=j<len(names)]
        print(n, " ".join(f"{x:.2f}" for x in nb))
